import os
from typing import Any

import hydra
import matplotlib.pyplot as plt
import torch
from model import create_timm_model
from omegaconf import DictConfig
from torch import nn, optim

# from data import movie_posters # TBD: Import training set here


@hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs")
def train(cfg: DictConfig) -> None:
    """Train the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_timm_model().to(device)
    train_set, _ = []  # TBD: Add training set here

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.hyperparameters.batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), cfg.hyperparameters.lr)

    train_loss: list[float] = []
    train_accuracy: list[float] = []

    for _ in range(cfg.hyperparameters.epochs):
        model.train()
        epoch_loss: float = 0.0
        epoch_correct: int = 0
        n: int = 0

        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs > cfg.hyperparameters.prob_threshold).int()

            epoch_loss += loss.item() * labels.numel()
            epoch_correct += (preds == labels).sum().item()
            n += labels.numel()

        train_loss.append(epoch_loss / n)
        train_accuracy.append(epoch_correct / n)

    print("Training complete")

    torch.save(model.state_dict(), cfg.hyperparameters.checkpoint_file)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(train_loss)
    axs[0].set_title("Train loss")
    axs[1].plot(train_accuracy)
    axs[1].set_title("Train accuracy")
    fig.savefig(cfg.hyperparameters.reports_file)


if __name__ == "__main__":
    train()
