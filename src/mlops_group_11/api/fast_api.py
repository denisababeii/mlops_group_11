import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from prometheus_client import Counter, make_asgi_app
from torchvision import transforms

from mlops_group_11.model import load_model

# Configuration from environment variables

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "24"))
MODEL_NAME = os.getenv("MODEL_NAME", "csatv2_21m.sw_r512_in1k")

CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", "models/best_model.pth"))  # where the trained weights are stored
LABELS_PATH = Path(os.getenv("LABELS_PATH", "data/processed/label_names.json"))  # where the label names JSON are stored

DEFAULT_THRESHOLD = float(os.getenv("PROB_THRESHOLD", "0.5"))
DEFAULT_TOPK = int(os.getenv("TOPK", "5"))

# Same preprocessing as in your dataset pipeline (ImageNet stats)
_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

prediction_error_counter = Counter("prediction_error", "Number of prediction errors")
health_error_counter = Counter("health_error", "Number of health errors")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device, _labels

    # --- Startup ---
    # Initialize device once so that downstream code can rely on it.
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Welcome to the Movie Poster Genre Inference API! 🎬")
    print("Upload a poster and discover its genres. 🍿")

    yield  # <-- application runs while paused here

    # --- Shutdown ---
    # Explicitly release model and associated resources to free memory (incl. GPU).
    if _model is not None:
        _model = None

    if _labels is not None:
        _labels = None

    if _device is not None and torch.cuda.is_available():
        # Clear CUDA cache to free GPU memory used by the model, if any.
        torch.cuda.empty_cache()

    _device = None
    print("👋 Au revoir!")


app = FastAPI(
    title="MLOps Group 11 - Poster Genre Inference API",
    lifespan=lifespan,
)

app.mount("/metrics", make_asgi_app())


# Loaded once (cached)
_model: torch.nn.Module | None = None
_device: torch.device | None = None
_labels: list[str] | None = None


def _load_labels(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"label_names.json not found at: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_loaded() -> None:
    """Load model + labels once."""
    global _model, _device, _labels
    if _model is not None and _device is not None and _labels is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _model = load_model(
        model_name=MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        num_classes=NUM_CLASSES,
        device=_device,
    )
    _model.eval()

    _labels = _load_labels(LABELS_PATH)


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        _ensure_loaded()
        return {
            "status": "ok",
            "device": str(_device),
            "model_name": MODEL_NAME,
            "checkpoint_path": str(CHECKPOINT_PATH),
            "num_labels": len(_labels or []),
        }
    except Exception as e:
        health_error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e)) from e


MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024  # Default 10MB in bytes


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = DEFAULT_THRESHOLD,
    topk: int = DEFAULT_TOPK,
) -> dict[str, Any]:
    try:
        _ensure_loaded()
        assert _model is not None and _device is not None and _labels is not None

        # Read upload -> PIL
        content = await file.read()

        # Validate file size
        if len(content) > MAX_FILE_SIZE:
            return {"error": f"File size exceeds maximum allowed size of {MAX_FILE_SIZE // (1024 * 1024)}MB"}
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read uploaded file as an image: {str(e)}")

        # Preprocess
        input_tensor = _transform(img).unsqueeze(0).to(_device)  # (1,3,H,W)

        # Predict (multi-label: sigmoid)
        with torch.no_grad():
            logits = _model(input_tensor)  # (1, num_classes)
            probs = torch.sigmoid(logits)[0]  # (num_classes,)

        probs_list = probs.detach().cpu().tolist()

        # Clamp topk
        topk = max(1, min(int(topk), len(probs_list)))

        # Top-k results
        top_indices = sorted(range(len(probs_list)), key=lambda i: probs_list[i], reverse=True)[:topk]
        top_items = [{"label": _labels[i], "probability": float(probs_list[i])} for i in top_indices]

        # All labels above threshold
        predicted = [
            {"label": _labels[i], "probability": float(p)} for i, p in enumerate(probs_list) if p >= float(threshold)
        ]

        return {
            "filename": file.filename,
            "threshold": float(threshold),
            "predicted": predicted,
            "topk": top_items,
        }
    except Exception as e:
        prediction_error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e)) from e
