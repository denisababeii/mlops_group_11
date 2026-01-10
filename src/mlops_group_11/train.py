import matplotlib.pyplot as plt
import torch
import typer
from model import create_timm_model
from torch import nn, optim

from data import movie_posters

app = typer.Typer()


@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train the model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_timm_model().to(device)
    train_set, _ = movie_posters()

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    train_loss, train_accuracy = [], []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        n = 0

        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

            epoch_loss += loss.item() * labels.numel()
            epoch_correct += (preds == labels).sum().item()
            n += labels.numel()

        train_loss.append(epoch_loss / n)
        train_accuracy.append(epoch_correct / n)

    print("Training complete")

    torch.save(model.state_dict(), "models/model.pth")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(train_loss)
    axs[0].set_title("Train loss")
    axs[1].plot(train_accuracy)
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    typer.run(train)
