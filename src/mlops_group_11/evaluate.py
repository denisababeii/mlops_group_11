"""Model evaluation Module"""

import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from mlops_group_11.data import poster_dataset
from mlops_group_11.model import create_timm_model, load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs", version_base=None)
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model."""
    logger.info("Starting evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading model...")
    checkpoint_path = Path(cfg.paths.checkpoint_file)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model(
        model_name=cfg.model.name, checkpoint_path=checkpoint_path, num_classes=cfg.model.num_classes, device=device
    )
    model.eval()
    logger.info("Model loaded successfully")

    # Load data
    logger.info("Loading data")
    try:
        _, _, test_dataset = poster_dataset(Path(cfg.data.processed_path))
    except FileNotFoundError as e:
        logger.error(f"Processed data not found: {e}")
        logger.error("Run preprocessing first: invoke preprocess-data")
        raise

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.hyperparameters.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > cfg.hyperparameters.prob_threshold).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    hamming = hamming_loss(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Log results
    logger.info("Evaluation Results (Multi-Label Classification):")
    logger.info(f"Hamming Loss: {hamming:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    logger.info("Detailed Classification Report:")
    logger.info(classification_report(all_labels, all_preds))

    # Save metrics to file
    metrics_file = Path("reports") / "evaluation_metrics.txt"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w") as f:
        f.write("Evaluation Results (Multi-Label Classification):\n")
        f.write(f"Hamming Loss: {hamming:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds))

    logger.info(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    evaluate()
