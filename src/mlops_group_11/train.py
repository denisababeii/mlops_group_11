""" Model training Module """

import logging
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from google.cloud import storage
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader

from mlops_group_11.data import poster_dataset
from mlops_group_11.model import create_timm_model, save_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def upload_to_gcs(local_path: str, gcs_path: str, bucket_name: str = "mlops-group11-data"):
    """Upload file to GCS bucket."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logger.info(f"✅ Uploaded {gcs_path} to GCS")
    except Exception as e:
        logger.warning(f"⚠️ Failed to upload to GCS: {e}")


@hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train the model (validation and checkpointing)"""
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
        train_dataset, val_dataset, _ = poster_dataset(
            Path(cfg.data.processed_path)
        )
    except FileNotFoundError as e:
        logger.error(f"Processed data not found: {e}")
        logger.error("Run preprocessing first: invoke preprocess-data")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.hyperparameters.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.hyperparameters.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(cfg.hyperparameters.epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.hyperparameters.epochs}")
        
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        n_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs > cfg.hyperparameters.prob_threshold).float()

            epoch_train_loss += loss.item() * labels.numel()
            epoch_train_correct += (preds == labels).sum().item()
            n_train += labels.numel()

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Batch {batch_idx + 1}/{len(train_loader)}: "
                    f"Loss: {loss.item():.4f}"
                )

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
                labels = labels.to(device)

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

        scheduler.step(val_loss)

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model,
                Path(cfg.paths.best_model_file),
                metadata={
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'val_acc': val_accuracy,
                    'train_loss': train_loss,
                    'train_acc': train_accuracy,
                }
            )
            logger.info(f"Saved best model with val loss: {best_val_loss:.4f}")

        # Periodic checkpointing
        if (epoch + 1) % cfg.logging.save_frequency == 0:
            save_model(
                model,
                Path(cfg.paths.checkpoint_file),
                metadata={
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_accuracy,
                    'val_acc': val_accuracy,
                }
            )
            logger.info(f"Saved checkpoint at epoch {epoch + 1}")

    logger.info("Training complete")

    # Get job ID for unique naming
    job_id = os.environ.get("CLOUD_ML_JOB_ID", "local")

    # Save final model
    save_model(
        model,
        Path(cfg.paths.checkpoint_file),
        metadata={
            'epoch': cfg.hyperparameters.epochs,
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
            'train_acc': train_accuracies[-1],
            'val_acc': val_accuracies[-1],
            'best_val_loss': best_val_loss,
            'status': 'training_completed'
        }
    )
    logger.info(f"Saved final model to {cfg.paths.checkpoint_file}")
    
    # Upload models to GCS
    upload_to_gcs(str(cfg.paths.checkpoint_file), f"models/model_{job_id}.pth")
    upload_to_gcs(str(cfg.paths.best_model_file), f"models/best_model_{job_id}.pth")

    # Plot training curves
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(val_losses, label='Val Loss')
    axs[0].set_title("Loss Curve")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(train_accuracies, label='Train Acc')
    axs[1].plot(val_accuracies, label='Val Acc')
    axs[1].set_title("Accuracy Curve")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    report_path = Path(cfg.paths.reports_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(report_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved training curves to {report_path}")
    
    # Upload plots to GCS
    upload_to_gcs(str(report_path), f"reports/training_curves_{job_id}.png")

    plt.close()

if __name__ == "__main__":
    train()