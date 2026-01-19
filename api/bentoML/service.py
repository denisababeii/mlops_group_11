from __future__ import annotations

from pathlib import Path

import bentoml
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


@bentoml.service(
    workers=1,
)
class PosterClassifierService:
    """Movie poster genre classifier service using PyTorch model."""

    def __init__(self) -> None:
        """Initialize the service and load the TorchScript model immediately."""
        # Model configuration
        self.model_name = "csatv2_21m.sw_r512_in1k"
        self.num_classes = 24
        
        # Try to load TorchScript model first (faster), fallback to PyTorch
        torchscript_path = Path(__file__).parent.parent.parent / "models" / "model_scripted.pt"
        checkpoint_path = Path(__file__).parent.parent.parent / "models" / "best_model.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model immediately (not lazy)
        if torchscript_path.exists():
            print(f"Loading TorchScript model from {torchscript_path}...")
            self.model = torch.jit.load(str(torchscript_path), map_location=self.device)
            self.model.eval()
            print("TorchScript model loaded successfully! (Faster inference)")
        else:
            print(f"TorchScript model not found. Loading PyTorch model from {checkpoint_path}...")
            print("Tip: Run 'uv run python scripts/convert_to_torchscript.py' for faster inference")
            
            import sys
            src_path = Path(__file__).parent.parent.parent / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            from mlops_group_11.model import load_model
            
            self.model = load_model(
                model_name=self.model_name,
                checkpoint_path=checkpoint_path,
                num_classes=self.num_classes,
                device=self.device,
            )
            self.model.eval()
            print("PyTorch model loaded successfully!")

        # Default threshold for predictions
        self.threshold = 0.5

        # Define image preprocessing transform (same as used during training)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Genre labels (24 genres, excluding N/A)
        self.genre_labels = [
            "Action",
            "Adventure",
            "Animation",
            "Biography",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Family",
            "Fantasy",
            "Film-Noir",
            "History",
            "Horror",
            "Music",
            "Musical",
            "Mystery",
            "News",
            "Romance",
            "Sci-Fi",
            "Short",
            "Sport",
            "Thriller",
            "War",
            "Western",
        ]

    def _predict_internal(self, image: np.ndarray) -> dict[str, list[str] | list[float]]:
        """Internal prediction method without BentoML decorators."""
        import time
        start = time.time()
        
        # Convert numpy array to PIL Image
        if image.dtype == np.uint8:
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            pil_image = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")

        # Apply transform
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        preprocess_time = time.time() - start

        # Make prediction
        inference_start = time.time()
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        inference_time = time.time() - inference_start

        # Get genres above threshold
        predicted_indices = np.where(probabilities >= self.threshold)[0]
        predicted_genres = [self.genre_labels[i] for i in predicted_indices]

        total_time = time.time() - start
        print(f"Timing - Preprocess: {preprocess_time:.2f}s, Inference: {inference_time:.2f}s, Total: {total_time:.2f}s")

        return {
            "genres": predicted_genres,
            "probabilities": probabilities.tolist(),
        }

    @bentoml.api(route="/predict")
    def predict(self, image: np.ndarray) -> dict[str, list[str] | list[float]]:
        """Predict the genres of a single movie poster image.

        Args:
            image: Input image as numpy array (H, W, C) with values in [0, 255].

        Returns:
            Dictionary containing:
                - "genres": List of predicted genre names.
                - "probabilities": List of probabilities for each genre.
        """
        return self._predict_internal(image)
    
    @bentoml.api(route="/predict_base64")
    def predict_base64(self, image_base64: str) -> dict[str, list[str] | list[float]]:
        """Predict from base64 encoded image (bypasses some BentoML overhead).

        Args:
            image_base64: Base64 encoded image string.

        Returns:
            Dictionary containing genres and probabilities.
        """
        # Decode base64 to image
        image_bytes = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(pil_image)
        
        return self._predict_internal(image_array)

    @bentoml.api(route="/predict_from_file")
    def predict_from_file(self, image_path: str, threshold: float = 0.5) -> dict[str, list[str] | list[float]]:
        """Predict genres from an image file path (single image, supports custom threshold).

        Args:
            image_path: Path to the image file.
            threshold: Probability threshold for genre classification (default: 0.5).

        Returns:
            Dictionary containing:
                - "genres": List of predicted genre names.
                - "probabilities": List of probabilities for each genre.
        """
        # Load image from file
        pil_image = Image.open(image_path).convert("RGB")

        # Convert to numpy array
        image_array = np.array(pil_image)

        # Temporarily set threshold
        original_threshold = self.threshold
        self.threshold = threshold

        # Get prediction
        result = self._predict_internal(image_array)

        # Restore original threshold
        self.threshold = original_threshold

        return result
