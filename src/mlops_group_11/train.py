import logging
import os
import random
from contextlib import nullcontext
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
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
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from mlops_group_11.data import poster_dataset
from mlops_group_11.model import create_timm_model, save_model

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# To ensure there is reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def upload_to_gcs(local_path: str, gcs_path: str, bucket_name: str = "mlops-group11-data"):
    """Upload file to GCS bucket."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded to gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        logger.warning(f"Failed to upload to GCS: {e}")


@hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train the model (with checkpointing)"""

    # Seed is set for reproducibility
    set_seed(cfg.get("seed", 42))

    job_id = os.environ.get("CLOUD_ML_JOB_ID", wandb.util.generate_id())

    run = wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        name = f"train-{cfg.model.name}-{job_id}",
        config={
            "lr": cfg.hyperparameters.lr,
            "batch_size": cfg.hyperparameters.batch_size,
            "epochs": cfg.hyperparameters.epochs,
        },
    )

    logger.info("Starting training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    logger.info("Creating model")
    model = create_timm_model(
        name=cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
    ).to(device)

    # Load data
    logger.info("Loading data")
    try:
        train_dataset, _, _ = poster_dataset(Path(cfg.data.processed_path))
    except FileNotFoundError as e:
        logger.error(f"Processed data not found: {e}")
        logger.error("Run preprocessing first: invoke preprocess-data")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.hyperparameters.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")

    # Setup training
    criterion = nn.BCEWithLogitsLoss()  # Use float32 labels
    optimizer = optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)

    # Tracking
    train_losses = []
    train_accuracies = []

    # Training loop
    for epoch in range(cfg.hyperparameters.epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.hyperparameters.epochs}")

        do_profile = bool(cfg.profiling.enabled) and (epoch < cfg.profiling.num_epochs)

        # Choose how many batches to run this epoch (only reduced when profiling)
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
                global_step = epoch * len(train_loader) + batch_idx
                wandb.log({"train_loss": loss.item(), "train_accuracy": batch_accuracy}, step=global_step)

                scores_list.append(probs.detach().cpu())
                targets_list.append(labels.detach().cpu().float())

                epoch_train_loss += loss.item() * labels.numel()
                epoch_train_correct += (preds == labels).sum().item()
                n_train += labels.numel()

                if do_profile:
                    prof.step()

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Batch {batch_idx + 1}/{len(train_loader)}: Loss: {loss.item():.4f}")

                    # Plot input images
                    grid = make_grid(images[:16].detach().cpu(), nrow=4, normalize=True)  # (C, H, W)
                    wandb.log({"images_grid": wandb.Image(grid, caption="Train batch grid")})

                    # Plot histogram of the gradients
                    grads = torch.cat(
                        [p.grad.detach().flatten().cpu() for p in model.parameters() if p.grad is not None], 0
                    )
                    wandb.log({"gradients": wandb.Histogram(grads)})

            train_loss = epoch_train_loss / n_train
            train_accuracy = epoch_train_correct / n_train
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

        if do_profile:
            logger.info("Profiling results for epoch:")
            logger.info(prof.key_averages().table(sort_by="cpu_time_total"))

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")

        # Periodic checkpointing
        if (epoch + 1) % cfg.logging.save_frequency == 0:
            save_model(
                model,
                Path(cfg.paths.checkpoint_file),
                metadata={
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_accuracy,
                },
            )
            logger.info(f"Saved checkpoint at epoch {epoch + 1}")

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

    logger.info("Training complete")

    preds = (scores_list > cfg.hyperparameters.prob_threshold).int().numpy()
    labels = targets_list.int().numpy()

    final_accuracy = accuracy_score(labels, preds)
    final_precision = precision_score(labels, preds, average="samples", zero_division=0)
    final_recall = recall_score(labels, preds, average="samples", zero_division=0)
    final_f1 = f1_score(labels, preds, average="samples", zero_division=0)

    # Save final model
    final_model_path = Path(cfg.paths.checkpoint_file)
    save_model(
        model,
        final_model_path,
        metadata={
            "epoch": cfg.hyperparameters.epochs,
            "train_loss": train_losses[-1],
            "train_acc": train_accuracies[-1],
            "status": "training_completed",
        },
    )
    logger.info(f"Saved final model to {cfg.paths.checkpoint_file}")

    # Upload final model to GCS
    upload_to_gcs(str(final_model_path), f"models/final_model_{job_id}.pth")

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
    artifact.add_file(str(Path(cfg.paths.checkpoint_file)))
    run.log_artifact(artifact)

    # Plot training curves
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(train_losses, label="Train Loss")
    axs[0].set_title("Loss Curve")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(train_accuracies, label="Train Acc")
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

    # Upload to GCP
    upload_to_gcs(str(report_path), f"reports/training_curves_{job_id}.png")

    plt.close()

    # Finish the W&B (check if needed or not)
    wandb.finish()

if __name__ == "__main__":
    train()
