import logging
import os
from contextlib import nullcontext
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import wandb

from dotenv import load_dotenv
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


@hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train the model (with checkpointing)"""
    run = wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        config={
            "lr": cfg.hyperparameters.lr,
            "batch_size": cfg.hyperparameters.batch_size,
            "epochs": cfg.hyperparameters.epochs,
            "prob_threshold": cfg.hyperparameters.prob_threshold,
            "model_name": cfg.model.name,
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
        train_dataset, val_dataset, _ = poster_dataset(Path(cfg.data.processed_path))
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.hyperparameters.batch_size,
        shuffle=False,
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
    val_losses = []
    val_accuracies = []
    best_val_loss = float("inf")

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

        val_loss = epoch_val_loss / n_val
        val_accuracy = epoch_val_correct / n_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if do_profile:
            logger.info("Profiling results for epoch:")
            logger.info(prof.key_averages().table(sort_by="cpu_time_total"))

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

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model,
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
    save_model(
        model,
        Path(cfg.paths.checkpoint_file),
        metadata={
            "epoch": cfg.hyperparameters.epochs,
            "train_loss": train_losses[-1],
            "train_acc": train_accuracies[-1],
            "status": "training_completed",
        },
    )
    logger.info(f"Saved final model to {cfg.paths.checkpoint_file}")

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

    plt.close()


if __name__ == "__main__":
    train()
