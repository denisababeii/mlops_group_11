import logging
import os
import random
from contextlib import nullcontext
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import wandb
from dotenv import load_dotenv
from google.cloud import storage
from omegaconf import DictConfig
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid

from mlops_group_11.data import poster_dataset
from mlops_group_11.model import create_timm_model, save_model

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed():
    """Initialize distributed training environment.
    
    Supports both CPU (gloo) and GPU (nccl) backends.
    """
    # Get distributed training parameters from environment
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize process group if distributed
    if world_size > 1:
        # Auto-detect backend based on GPU availability
        if torch.cuda.is_available():
            backend = "nccl"  # GPU backend
            torch.cuda.set_device(local_rank)
        else:
            backend = "gloo"  # CPU backend
        
        dist.init_process_group(
            backend=backend,
            init_method="env://",
        )
        logger.info(f"Initialized DDP with {backend} backend: rank {rank}/{world_size}, local_rank {local_rank}")
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def upload_to_gcs(local_path: str, gcs_path: str, bucket_name: str = "mlops-group11-data"):
    """Upload file to GCS bucket."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded {gcs_path} to GCS")
    except Exception as e:
        logger.warning(f"Failed to upload to GCS: {e}")


@hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train the model with Distributed Data Parallel support (CPU and GPU compatible)"""

    # Distributed training
    rank, local_rank, world_size = setup_distributed()
    is_main_process = rank == 0  # Only rank 0 does logging/saving
    
    # Set seed for reproducibility (different seed per rank for data augmentation diversity)
    set_seed(cfg.get("seed", 42) + rank)

    # Get unique job ID
    job_id = os.environ.get("CLOUD_ML_JOB_ID", wandb.util.generate_id())

    # Initialize W&B only on main process
    if is_main_process:
        run = wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT"),
            name=f"train-{cfg.model.name}-{job_id}",
            config={
                "lr": cfg.hyperparameters.lr,
                "batch_size": cfg.hyperparameters.batch_size,
                "epochs": cfg.hyperparameters.epochs,
                "prob_threshold": cfg.hyperparameters.prob_threshold,
                "model_name": cfg.model.name,
                "world_size": world_size,
                "distributed": world_size > 1,
            },
        )

        # Override config with W&B sweep values (if running sweep)
        if wandb.config.get("lr"):
            cfg.hyperparameters.lr = wandb.config.lr
        if wandb.config.get("batch_size"):
            cfg.hyperparameters.batch_size = wandb.config.batch_size
        if wandb.config.get("epochs"):
            cfg.hyperparameters.epochs = wandb.config.epochs
        if wandb.config.get("model_name"):
            cfg.model.name = wandb.config.model_name
        if wandb.config.get("prob_threshold"):
            cfg.hyperparameters.prob_threshold = wandb.config.prob_threshold

    logger.info(f"Starting training on rank {rank}/{world_size}")

    # Set device for this process
    if world_size > 1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Rank {rank} using device: {device}")

    # Create model
    if is_main_process:
        logger.info("Creating model")
    model = create_timm_model(
        name=cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
    ).to(device)

    # Wrap model with DDP if using distributed training
    if world_size > 1:
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)  # CPU - no device_ids
        logger.info(f"Wrapped model with DDP on rank {rank}")

    # Load data
    if is_main_process:
        logger.info("Loading data")
    try:
        train_dataset, val_dataset, _ = poster_dataset(Path(cfg.data.processed_path))
    except FileNotFoundError as e:
        logger.error(f"Processed data not found: {e}")
        logger.error("Run preprocessing first: invoke preprocess-data")
        cleanup_distributed()
        return

    # Create distributed samplers
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.get("seed", 42)
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    # Create data loaders with distributed samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.hyperparameters.batch_size,
        sampler=train_sampler,  # Use sampler instead of shuffle when distributed
        shuffle=(train_sampler is None),  # Only shuffle if no sampler
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.hyperparameters.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    if is_main_process:
        logger.info(f"Train dataset size: {len(train_dataset)}")
        if world_size > 1:
            logger.info(f"Samples per process: {len(train_dataset) // world_size}")

    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)

    # Tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(cfg.hyperparameters.epochs):
        if is_main_process:
            logger.info(f"Epoch {epoch + 1}/{cfg.hyperparameters.epochs}")

        # Set epoch for distributed sampler (important for proper shuffling across epochs)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        do_profile = bool(cfg.profiling.enabled) and (epoch < cfg.profiling.num_epochs) and is_main_process
        max_batches = cfg.profiling.num_batches if do_profile else None

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        profile_ctx = (
            profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=tensorboard_trace_handler(cfg.profiling.log_dir),
            )
            if do_profile
            else nullcontext()
        )

        with profile_ctx as prof:
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            epoch_train_correct = 0
            n_train = 0

            scores_list, targets_list = [], []
            for batch_idx, (images, labels) in enumerate(train_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                images = images.to(device)
                labels = labels.to(device).float()

                optimizer.zero_grad()

                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                probs = torch.sigmoid(logits)
                preds = (probs > cfg.hyperparameters.prob_threshold).float()

                batch_accuracy = (preds == labels).float().mean().item()
                
                # Log only from main process
                if is_main_process:
                    global_step = epoch * len(train_loader) + batch_idx
                    wandb.log(
                        {"train/batch_loss": loss.item(), "train/batch_accuracy": batch_accuracy},
                        step=global_step,
                    )

                scores_list.append(probs.detach().cpu())
                targets_list.append(labels.detach().cpu().float())

                epoch_train_loss += loss.item() * labels.numel()
                epoch_train_correct += (preds == labels).sum().item()
                n_train += labels.numel()

                if do_profile:
                    prof.step()

                if is_main_process and (batch_idx + 1) % 10 == 0:
                    logger.info(f"Batch {batch_idx + 1}/{len(train_loader)}: Loss: {loss.item():.4f}")

                    # Plot input images
                    grid = make_grid(images[:16].detach().cpu(), nrow=4, normalize=True)
                    wandb.log({"images_grid": wandb.Image(grid, caption="Train batch grid")})

                    # Plot histogram of the gradients
                    grads = torch.cat(
                        [p.grad.detach().flatten().cpu() for p in model.parameters() if p.grad is not None], 0
                    )
                    wandb.log({"gradients": wandb.Histogram(grads)})

            # Aggregate metrics across all processes
            if world_size > 1:
                # Convert to tensors for all_reduce
                train_loss_tensor = torch.tensor([epoch_train_loss], device=device)
                train_correct_tensor = torch.tensor([epoch_train_correct], device=device)
                n_train_tensor = torch.tensor([n_train], device=device)
                
                # Sum across all processes
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(n_train_tensor, op=dist.ReduceOp.SUM)
                
                epoch_train_loss = train_loss_tensor.item()
                epoch_train_correct = train_correct_tensor.item()
                n_train = n_train_tensor.item()

            train_loss = epoch_train_loss / n_train
            train_accuracy = epoch_train_correct / n_train
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        n_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()

                logits = model(images)
                loss = criterion(logits, labels)

                probs = torch.sigmoid(logits)
                preds = (probs > cfg.hyperparameters.prob_threshold).float()

                epoch_val_loss += loss.item() * labels.numel()
                epoch_val_correct += (preds == labels).sum().item()
                n_val += labels.numel()

        # Aggregate validation metrics across all processes
        if world_size > 1:
            val_loss_tensor = torch.tensor([epoch_val_loss], device=device)
            val_correct_tensor = torch.tensor([epoch_val_correct], device=device)
            n_val_tensor = torch.tensor([n_val], device=device)
            
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_val_tensor, op=dist.ReduceOp.SUM)
            
            epoch_val_loss = val_loss_tensor.item()
            epoch_val_correct = val_correct_tensor.item()
            n_val = n_val_tensor.item()

        val_loss = epoch_val_loss / n_val
        val_accuracy = epoch_val_correct / n_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if do_profile and is_main_process:
            logger.info("Profiling results for epoch:")
            logger.info(prof.key_averages().table(sort_by="cpu_time_total"))

        if is_main_process:
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_accuracy,
                    "val/loss": val_loss,
                    "val/accuracy": val_accuracy,
                },
                step=epoch + 1,
            )

            # Save the best model (only main process)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Unwrap DDP model before saving
                model_to_save = model.module if world_size > 1 else model
                save_model(
                    model_to_save,
                    Path(cfg.paths.best_model_file),
                    metadata={
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "val_acc": val_accuracy,
                        "train_loss": train_loss,
                        "train_acc": train_accuracy,
                    },
                )
                logger.info(f"Saved best model with val loss: {best_val_loss:.4f}")

            # Periodic checkpointing
            if (epoch + 1) % cfg.logging.save_frequency == 0:
                checkpoint_path = Path(cfg.paths.checkpoint_file)
                model_to_save = model.module if world_size > 1 else model
                save_model(
                    model_to_save,
                    checkpoint_path,
                    metadata={
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_acc": train_accuracy,
                    },
                )
                logger.info(f"Saved checkpoint at epoch {epoch + 1}")
                # Upload checkpoint to GCS
                upload_to_gcs(str(checkpoint_path), f"models/checkpoint_epoch{epoch + 1}_{job_id}.pth")

        # Wait for all processes before continuing to next epoch
        if world_size > 1:
            dist.barrier()

        if is_main_process:
            scores_list = torch.cat(scores_list, 0)
            targets_list = torch.cat(targets_list, 0)

            plt.figure()
            for class_id in range(scores_list.shape[1]):
                y_true = targets_list[:, class_id].numpy()
                y_score = scores_list[:, class_id].numpy()

                if y_true.min() == y_true.max():
                    continue

                RocCurveDisplay.from_predictions(
                    y_true,
                    y_score,
                    name=f"class {class_id}",
                )

            wandb.log({"roc_curves": wandb.Image(plt.gcf())})
            plt.close()

    if is_main_process:
        logger.info("Training complete")

        preds = (scores_list > cfg.hyperparameters.prob_threshold).int().numpy()
        labels = targets_list.int().numpy()

        final_accuracy = accuracy_score(labels, preds)
        final_precision = precision_score(labels, preds, average="samples", zero_division=0)
        final_recall = recall_score(labels, preds, average="samples", zero_division=0)
        final_f1 = f1_score(labels, preds, average="samples", zero_division=0)

        # Save final model
        final_model_path = Path(cfg.paths.checkpoint_file)
        model_to_save = model.module if world_size > 1 else model
        save_model(
            model_to_save,
            final_model_path,
            metadata={
                "epoch": cfg.hyperparameters.epochs,
                "train_loss": train_losses[-1],
                "train_acc": train_accuracies[-1],
                "status": "training_completed",
            },
        )
        logger.info(f"Saved final model to {final_model_path}")

        # Upload final model to GCS
        upload_to_gcs(str(final_model_path), f"models/final_model_{job_id}.pth")
        upload_to_gcs(str(cfg.paths.best_model_file), f"models/best_model_{job_id}.pth")

        # Save model as wandb artifact
        artifact = wandb.Artifact(
            name=cfg.model.name,
            type="model",
            metadata={
                "accuracy": final_accuracy,
                "precision": final_precision,
                "recall": final_recall,
                "f1": final_f1,
            },
        )
        artifact.add_file(str(final_model_path))
        run.log_artifact(artifact)

        # Plot training curves
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(train_losses, label="Train Loss")
        axs[0].plot(val_losses, label="Val Loss")
        axs[0].set_title("Loss Curve")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(train_accuracies, label="Train Acc")
        axs[1].plot(val_accuracies, label="Val Acc")
        axs[1].set_title("Accuracy Curve")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()
        axs[1].grid(True)

        report_path = Path(cfg.paths.reports_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(report_path, dpi=300, bbox_inches="tight")
        wandb.log({"training_curves": wandb.Image(fig)})
        logger.info(f"Saved training curves to {report_path}")

        # Upload plots to GCS
        upload_to_gcs(str(report_path), f"reports/training_curves_{job_id}.png")

        plt.close()

        # Finish W&B run
        wandb.finish()

    # Clean up distributed training
    cleanup_distributed()


if __name__ == "__main__":
    train()