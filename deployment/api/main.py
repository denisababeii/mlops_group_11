"""FastAPI backend for movie poster genre prediction."""

import io
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from PIL import Image
from prometheus_client import Counter, Histogram, make_asgi_app
from torchvision import transforms

from mlops_group_11.model import load_model

# CONFIGURATION

# Model configuration
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "24"))
MODEL_NAME = os.getenv("MODEL_NAME", "csatv2_21m.sw_r512_in1k")

# GCS configuration
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", "mlops-group-11")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "mlops-group11-data")
GCS_MODEL_PATH = os.getenv("GCS_MODEL_PATH", "models/best_model_8070986887763853312.pth")
GCS_LABELS_PATH = os.getenv("GCS_LABELS_PATH", "data/processed/label_names.json")
GCS_PREDICTIONS_PATH = os.getenv("GCS_PREDICTIONS_PATH", "predictions_database.csv")

# Local cache paths
CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", f"models/{GCS_MODEL_PATH.split('/')[-1]}"))
CHECKPOINT_LABELS_PATH = Path(os.getenv("LABELS_PATH", "data/processed/label_names.json"))

# Prediction parameters
DEFAULT_THRESHOLD = float(os.getenv("PROB_THRESHOLD", "0.5"))
DEFAULT_TOPK = int(os.getenv("TOPK", "5"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024  # MB to bytes

# IMAGE PREPROCESSING

# ImageNet normalization (same as training)
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

# METRICS

prediction_counter = Counter("prediction_total", "Total number of predictions")
prediction_error_counter = Counter("prediction_error", "Number of prediction errors")
health_error_counter = Counter("health_error", "Number of health check errors")
request_latency = Histogram("prediction_latency_seconds", "Prediction latency in seconds")

# GLOBAL STATE

_model: torch.nn.Module | None = None
_device: torch.device | None = None
_labels: list[str] | None = None

# LIFESPAN MANAGEMENT

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global _model, _device, _labels

    # Initialize device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Welcome to the Movie Poster Genre Inference API! 🎬")
    print("Upload a poster and discover its genres. 🍿")

    yield

    # Cleanup on shutdown
    print("Shutting down...")
    
    if _model is not None:
        del _model
        _model = None

    if _labels is not None:
        del _labels
        _labels = None

    if _device is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    _device = None
    print("👋 Au revoir!")


# FASTAPI APP

app = FastAPI(
    title="MLOps Group 11 - Poster Genre Inference API",
    description="Multi-label genre classification for movie posters",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins (change for security)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics
app.mount("/metrics", make_asgi_app())

# HELPER FUNCTIONS

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


def _download_from_gcs(gcs_path: str, local_path: Path) -> None:
    """Download file from GCS bucket to local cache.
    
    Args:
        gcs_path: Path in GCS bucket
        local_path: Local path to save file
    """
    if local_path.exists():
        print(f"Using cached file: {local_path}")
        return

    try:
        print(f"Downloading from gs://{GCS_BUCKET_NAME}/{gcs_path}...")
        client = storage.Client(project=GCS_PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        print(f"Downloaded to: {local_path}")
        
    except Exception as e:
        print(f"Download failed: {e}")
        if local_path.exists():
            print(f"Falling back to cached file: {local_path}")
        else:
            raise FileNotFoundError(
                f"Could not download {gcs_path} and no local cache exists at {local_path}"
            ) from e


def _load_labels(path: Path) -> list[str]:
    """Load genre labels from JSON file.
    
    Args:
        path: Path to label_names.json
        
    Returns:
        List of genre label names
    """
    if not path.exists():
        _download_from_gcs(GCS_LABELS_PATH, path)

    if not path.exists():
        raise FileNotFoundError(f"Labels not found at: {path}")
    
    with open(path, encoding="utf-8") as f:
        labels = json.load(f)
    
    print(f"Loaded {len(labels)} genre labels")
    return labels


def _load_model_from_checkpoint() -> None:
    """Load model from GCS checkpoint."""
    global _model, _device, _labels
    
    if _model is not None and _labels is not None:
        return

    # Ensure device is set
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download model from GCS
    _download_from_gcs(GCS_MODEL_PATH, CHECKPOINT_PATH)

    # Load model
    print(f"Loading model on {_device}...")
    _model = load_model(
        model_name=MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        num_classes=NUM_CLASSES,
        device=_device,
    )
    _model.eval()
    print("Model loaded")

    # Load labels
    _labels = _load_labels(CHECKPOINT_LABELS_PATH)


def add_to_gcs_database(
    timestamp: str,
    filename: str,
    predicted: list[dict[str, Any]],
    threshold: float,
    image_stats: dict[str, float],
) -> None:
    """Append prediction data as a CSV row to GCS.
    
    Args:
        timestamp: ISO format timestamp
        filename: Original filename
        predicted: List of predictions above threshold
        threshold: Probability threshold used
        image_stats: Image statistics dictionary
    """
    try:
        # Format data as CSV row
        genres_str = "|".join([f"{g['label']}:{g['probability']:.4f}" for g in predicted])
        stats_str = (
            f"{image_stats['mean']:.4f},"
            f"{image_stats['std']:.4f},"
            f"{image_stats['min']:.4f},"
            f"{image_stats['max']:.4f}"
        )

        csv_line = f"{timestamp},{filename},{genres_str},{threshold},{stats_str}\n"

        # Append to GCS file
        client = storage.Client(project=GCS_PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_PREDICTIONS_PATH)

        # Read existing content (if any), append, and write back
        try:
            existing = blob.download_as_text()
        except Exception:
            existing = ""

        blob.upload_from_string(existing + csv_line, content_type="text/csv")
        print(f"Prediction saved to gs://{GCS_BUCKET_NAME}/{GCS_PREDICTIONS_PATH}")
        
    except Exception as e:
        print(f"Could not save prediction to GCS: {e}")

# API ENDPOINTS

@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Movie Poster Genre Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint.
    
    Returns:
        Health status and model information
    """
    try:
        _load_model_from_checkpoint()
        
        return {
            "status": "healthy",
            "device": str(_device),
            "model_name": MODEL_NAME,
            "checkpoint_path": str(CHECKPOINT_PATH),
            "num_classes": NUM_CLASSES,
            "num_labels": len(_labels or []),
            "gcs_bucket": GCS_BUCKET_NAME,
        }
    except Exception as e:
        health_error_counter.inc()
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}") from e


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = DEFAULT_THRESHOLD,
    topk: int = DEFAULT_TOPK,
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> dict[str, Any]:
    """Predict genres for an uploaded movie poster.
    
    Args:
        file: Uploaded image file
        threshold: Probability threshold for predictions (default: 0.5)
        topk: Number of top predictions to return (default: 5)
        background_tasks: FastAPI background tasks
        
    Returns:
        Prediction results with genres above threshold and top-k predictions
    """
    with request_latency.time():
        try:
            # Ensure model is loaded
            _load_model_from_checkpoint()
            assert _model is not None and _device is not None and _labels is not None

            # Read and validate file
            content = await file.read()

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds maximum of {MAX_FILE_SIZE // (1024 * 1024)}MB"
                )

            # Load image
            try:
                img = Image.open(io.BytesIO(content)).convert("RGB")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image file: {str(e)}"
                ) from e

            # Compute image statistics (before preprocessing)
            image_stats = _compute_image_statistics(img)

            # Preprocess image
            input_tensor = _transform(img).unsqueeze(0).to(_device)  # (1, 3, H, W)

            # Predict (multi-label classification with sigmoid)
            with torch.no_grad():
                logits = _model(input_tensor)  # (1, num_classes)
                probs = torch.sigmoid(logits)[0]  # (num_classes,)

            probs_list = probs.detach().cpu().tolist()

            # Validate and clamp topk
            topk = max(1, min(int(topk), len(probs_list)))

            # Get top-k predictions
            top_indices = sorted(
                range(len(probs_list)),
                key=lambda i: probs_list[i],
                reverse=True
            )[:topk]
            
            top_items = [
                {"label": _labels[i], "probability": float(probs_list[i])}
                for i in top_indices
            ]

            # Get all predictions above threshold
            predicted = [
                {"label": _labels[i], "probability": float(p)}
                for i, p in enumerate(probs_list)
                if p >= float(threshold)
            ]

            # Save prediction to GCS (in background)
            timestamp = datetime.now().isoformat()
            background_tasks.add_task(
                add_to_gcs_database,
                timestamp,
                file.filename or "unknown",
                predicted,
                threshold,
                image_stats
            )

            # Increment success counter
            prediction_counter.inc()

            return {
                "filename": file.filename,
                "threshold": float(threshold),
                "predicted": predicted,
                "topk": top_items,
                "image_stats": image_stats,
                "timestamp": timestamp,
            }

        except HTTPException:
            raise
        except Exception as e:
            prediction_error_counter.inc()
            raise HTTPException(status_code=500, detail=str(e)) from e

# MAIN

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8080"))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )