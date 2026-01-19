from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from urllib.request import urlopen

import bentoml
import numpy as np
import onnxruntime as ort
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms


@bentoml.service(
    workers=4,
)
class PosterClassifierService:
    """Movie poster genre classifier service using ONNX model.
    Usage:
       Run the service with `bentoml serve service:PosterClassifierService`.
    """

    def __init__(self) -> None:
        """Initialize the service and load the ONNX model."""
        # Load Hydra configuration
        config_dir = str(Path(__file__).parent.parent.parent.parent / "configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            self.cfg: DictConfig = compose(config_name="config")
        
        # Model configuration from Hydra
        self.model_name = self.cfg.model.name
        self.num_classes = self.cfg.model.num_classes
        self.threshold = self.cfg.hyperparameters.prob_threshold
        
        print(f"Loaded configuration from {config_dir}")
        print(f"Model: {self.model_name}, Classes: {self.num_classes}, Threshold: {self.threshold}")
        
        # Load ONNX model
        onnx_path = Path(__file__).parent.parent.parent.parent / "models" / "model.onnx"
        
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. "
                "Please run 'uv run scripts/experimental_convert_to_onnx.py' first."
            )
        
        # Create ONNX Runtime inference session
        # Use CPUExecutionProvider or CUDAExecutionProvider based on availability
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"ONNX model loaded successfully!")
        print(f"Available providers: {self.session.get_providers()}")
        print(f"Input: {self.input_name}, Output: {self.output_name}")

        # Define image preprocessing transform from config
        image_size = self.cfg.data.image_size
        if self.cfg.data.use_imagenet_normalization:
            mean = self.cfg.data.imagenet_mean
            std = self.cfg.data.imagenet_std
        else:
            mean = self.cfg.data.default_mean
            std = self.cfg.data.default_std
        
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        # Load genre labels from JSON file
        label_names_path = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "label_names.json"
        
        if not label_names_path.exists():
            raise FileNotFoundError(
                f"Label names file not found at {label_names_path}. "
                "Please ensure the processed data exists."
            )
        
        with open(label_names_path, 'r') as f:
            self.genre_labels = json.load(f)
        
        print(f"Loaded {len(self.genre_labels)} genre labels from {label_names_path}")


    def _predict_internal(self, image: np.ndarray) -> dict[str, list[str] | list[float]]:
        """Internal prediction method."""
        start = time.time()
        
        # Convert numpy array to PIL Image
        if image.dtype == np.uint8:
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            pil_image = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")

        # Apply transform
        image_tensor = self.transform(pil_image).unsqueeze(0)
        preprocess_time = time.time() - start

        # Make prediction with ONNX Runtime
        inference_start = time.time()
        
        # Convert to numpy array (ONNX expects numpy input)
        input_numpy = image_tensor.numpy()
        
        # Run inference
        logits = self.session.run([self.output_name], {self.input_name: input_numpy})[0]
        
        # Apply sigmoid to get probabilities
        probabilities = 1 / (1 + np.exp(-logits[0]))  # sigmoid
        
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

    @bentoml.api(route="/predict_from_url")
    def predict_from_url(self, image_url: str, threshold: float = 0.5) -> dict[str, list[str] | list[float]]:
        """Predict genres from an image URL (supports custom threshold).

        Args:
            image_url: URL of the image to classify.
            threshold: Probability threshold for genre classification (default: 0.5).

        Returns:
            Dictionary containing:
                - "genres": List of predicted genre names.
                - "probabilities": List of probabilities for each genre.
        """
        try:
            # Download image from URL
            with urlopen(image_url) as response:
                image_data = response.read()
            
            # Load image from bytes
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
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
            
        except Exception as e:
            raise ValueError(f"Failed to load image from URL: {str(e)}")
