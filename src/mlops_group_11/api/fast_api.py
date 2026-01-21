import io
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from google.cloud import storage
from PIL import Image
from prometheus_client import Counter, Histogram, make_asgi_app
from torchvision import transforms

from mlops_group_11.model import load_model

# Configuration from environment variables
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "24"))
MODEL_NAME = os.getenv("MODEL_NAME", "csatv2_21m.sw_r512_in1k")

# GCS configuration
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", "mlops-group11")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "mlops-group11-data")
GCS_MODEL_PATH = os.getenv("GCS_MODEL_PATH", "models/best_model_8070986887763853312.pth")
GCS_LABELS_PATH = os.getenv("GCS_LABELS_PATH", "data/processed/label_names.json")
GCS_PREDICTIONS_PATH = os.getenv("GCS_PREDICTIONS_PATH", "predictions_database.csv")

# Local cache paths (where to store downloaded files)
CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", f"models/{GCS_MODEL_PATH.split('/')[-1]}"))
CHECKPOINT_LABELS_PATH = Path(os.getenv("LABELS_PATH", "data/processed/label_names.json"))

# Default prediction parameters
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
request_latency = Histogram("prediction_latency_seconds", "Prediction latency in seconds")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device, _labels

    # Initialize device once so that downstream code can rely on it.
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Welcome to the Movie Poster Genre Inference API! 🎬")
    print("Upload a poster and discover its genres. 🍿")

    yield

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


def _compute_image_statistics(image: Image.Image) -> dict[str, float]:
    """Compute statistics on the input image (before preprocessing).

    Args:
        image: PIL Image object

    Returns:
        Dictionary with mean, std, min, max pixel values
    """
    import numpy as np

    # Convert PIL Image to numpy array, then to torch tensor
    img_array = torch.tensor(np.array(image), dtype=torch.float32)

    return {
        "mean": float(img_array.mean().item()),
        "std": float(img_array.std().item()),
        "min": float(img_array.min().item()),
        "max": float(img_array.max().item()),
    }


def add_to_local_database(
    now: str,
    filename: str,
    predicted: list[dict[str, Any]],
    threshold: float,
    image_stats: dict[str, float],
) -> None:
    """Add prediction and image statistics to database."""
    genres_str = "|".join([f"{g['label']}:{g['probability']:.4f}" for g in predicted])
    stats_str = f"{image_stats['mean']:.4f},{image_stats['std']:.4f},{image_stats['min']:.4f},{image_stats['max']:.4f}"

    with open("prediction_database.csv", "a") as file:
        file.write(f"{now},{filename},{genres_str},{threshold},{stats_str}\n")


def add_to_gcs_database(
    timestamp: str,
    filename: str,
    predicted: list[dict[str, Any]],
    topk: list[dict[str, Any]],
    threshold: float,
    image_stats: dict[str, float],
) -> None:
    """Append prediction data as a CSV row to a single file in GCS."""
    try:
        # Format data like the local CSV: timestamp,filename,genres_str,threshold,stats_str
        genres_str = "|".join([f"{g['label']}:{g['probability']:.4f}" for g in predicted])
        stats_str = (
            f"{image_stats['mean']:.4f},{image_stats['std']:.4f},{image_stats['min']:.4f},{image_stats['max']:.4f}"
        )

        csv_line = f"{timestamp},{filename},{genres_str},{threshold},{stats_str}\n"

        client = storage.Client(project=GCS_PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_PREDICTIONS_PATH)

        # Read existing content (if any), append, and write back
        try:
            existing = blob.download_as_text()
        except Exception:
            existing = ""

        blob.upload_from_string(existing + csv_line, content_type="text/csv")
        print(f"Prediction appended to gs://{GCS_BUCKET_NAME}/{GCS_PREDICTIONS_PATH}")
    except Exception as e:
        print(f"Warning: Could not save prediction to GCS: {e}")


app.mount("/metrics", make_asgi_app())


# Loaded once (cached)
_model: torch.nn.Module | None = None
_device: torch.device | None = None
_labels: list[str] | None = None


def _download_labels_from_gcs() -> None:
    """Download label file from GCS bucket to local cache."""
    if CHECKPOINT_LABELS_PATH.exists():
        print(f"Using cached labels from: {CHECKPOINT_LABELS_PATH}")
        return

    try:
        from google.cloud import storage

        print(f"Downloading labels from gs://{GCS_BUCKET_NAME}/{GCS_LABELS_PATH}...")
        client = storage.Client(project=GCS_PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_LABELS_PATH)

        CHECKPOINT_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(CHECKPOINT_LABELS_PATH))
        print(f"Labels downloaded successfully to: {CHECKPOINT_LABELS_PATH}")
    except Exception as e:
        print(f"Warning: Could not download labels from GCS: {e}")
        if CHECKPOINT_LABELS_PATH.exists():
            print(f"Falling back to local labels at: {CHECKPOINT_LABELS_PATH}")
        else:
            raise FileNotFoundError(
                f"Labels not found. Failed to download from GCS and no local file exists at {CHECKPOINT_LABELS_PATH}"
            ) from e


def _load_labels(path: Path) -> list[str]:
    if not path.exists():
        _download_labels_from_gcs()

    if not path.exists():
        raise FileNotFoundError(f"label_names.json not found at: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _download_model_from_gcs() -> None:
    """Download model from GCS bucket to local cache."""
    if CHECKPOINT_PATH.exists():
        print(f"Using cached model from: {CHECKPOINT_PATH}")
        return

    try:
        from google.cloud import storage

        print(f"Downloading model from gs://{GCS_BUCKET_NAME}/{GCS_MODEL_PATH}...")
        client = storage.Client(project=GCS_PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_MODEL_PATH)

        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(CHECKPOINT_PATH))
        print(f"Model downloaded successfully to: {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"Warning: Could not download model from GCS: {e}")
        if CHECKPOINT_PATH.exists():
            print(f"Falling back to local path: {CHECKPOINT_PATH}")
        else:
            raise FileNotFoundError(
                f"Model not found. Failed to download from GCS and no local file exists at {CHECKPOINT_PATH}"
            ) from e


def _ensure_loaded() -> None:
    """Load model + labels once."""
    global _model, _device, _labels
    if _model is not None and _device is not None and _labels is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download model from GCS if not available locally
    _download_model_from_gcs()

    _model = load_model(
        model_name=MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        num_classes=NUM_CLASSES,
        device=_device,
    )
    _model.eval()

    _labels = _load_labels(CHECKPOINT_LABELS_PATH)


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
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> dict[str, Any]:
    with request_latency.time():
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

            # Compute image statistics before preprocessing
            image_stats = _compute_image_statistics(img)

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
                {"label": _labels[i], "probability": float(p)}
                for i, p in enumerate(probs_list)
                if p >= float(threshold)
            ]

            now = datetime.now().isoformat()
            # For local prediction database storage
            # background_tasks.add_task(
            #     add_to_local_database, now, file.filename, predicted, top_items, threshold, image_stats
            # )
            background_tasks.add_task(
                add_to_gcs_database, now, file.filename, predicted, top_items, threshold, image_stats
            )

            return {
                "filename": file.filename,
                "threshold": float(threshold),
                "predicted": predicted,
                "topk": top_items,
            }
        except Exception as e:
            prediction_error_counter.inc()
            raise HTTPException(status_code=500, detail=str(e)) from e
