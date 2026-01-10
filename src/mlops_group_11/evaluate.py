import torch
import typer
from model import create_timm_model

from data import movie_posters

app = typer.Typer()


@app.command()
def evaluate(model_checkpoint: str = "models/model.pth") -> None:
    """Evaluate a trained model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_timm_model()
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.to(device)

    _, test_set = movie_posters()
    testloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

            correct += (preds == labels).sum().item()

    print(f"Test accuracy: {correct / len(testloader.dataset)}")


if __name__ == "__main__":
    typer.run(evaluate)
