import os

import hydra
import torch
from model import create_timm_model
from omegaconf import DictConfig

# from data import movie_posters # TBD: Import training set here


@hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_timm_model()
    model.load_state_dict(torch.load(cfg.hyperparameters.checkpoint_file, map_location=device))
    model.to(device)

    _, test_set = []  # TBD: Add training set here
    testloader = torch.utils.data.DataLoader(test_set, batch_size=cfg.hyperparameters.batch_size, shuffle=True)

    model.eval()
    correct: int = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > cfg.hyperparameters.prob_threshold).int()

            correct += (preds == labels).sum().item()

    print(f"Test accuracy: {correct / len(testloader.dataset)}")


if __name__ == "__main__":
    evaluate()
