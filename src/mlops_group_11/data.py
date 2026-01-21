"Data loading and preprocessing for movie poster dataset."

import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import typer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms


class MyDataset(Dataset):
    """Raw movie poster dataset (multi-label classification).

    Args:
        data_path: Path to the dataset folder containing `train.csv` and `Images/`.
        image_size: Size to which images are resized (default: 224).
        use_imagenet_norm: Apply ImageNet normalization to images (default: True).
        exclude_genres: List of genre names to exclude (default: ['N/A']).

    Attributes:
        data_path: Path to the dataset folder.
        image_size: Image dimensions.
        df: DataFrame containing image IDs and labels.
        genre_columns: List of genre label columns.
        transform: Image transformation pipeline.

    """

    def __init__(
        self, data_path: Path, image_size: int = 224, use_imagenet_norm: bool = True, exclude_genres: list = None
    ) -> None:
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.use_imagenet_norm = use_imagenet_norm

        self.df = pd.read_csv(self.data_path / "train.csv")

        # Filter out N/A and other unwanted columns
        base_exclude = ["Id", "Genre"]
        if exclude_genres is None:
            exclude_genres = ["N/A"]  # Default: exclude N/A

        exclude_columns = base_exclude + exclude_genres

        # Genre columns = everything except excluded ones
        self.genre_columns = self.df.columns.drop(exclude_columns)

        # Log information
        print(f"Using {len(self.genre_columns)} genres: {list(self.genre_columns)}")
        if "N/A" in self.df.columns and "N/A" in exclude_columns:
            print("'N/A' column found and excluded")

        # Build transformation pipeline
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

        # Transformation pipeline
        if use_imagenet_norm:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                    std=[0.229, 0.224, 0.225],
                )
            )
        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.df)

    # Gets image and its label(s) at specific index
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image and its labels at index `idx`.

        Args:
            idx: Sample index.

        Returns:
            A tuple of (image tensor, target tensor)
            - image tensor: shape (3, H, W), float32
            - label_tensor: shape (num_genres,), float32 (hot encoded)
        """
        row = self.df.iloc[idx]

        img_path = self.data_path / "Images" / f"{row['Id']}.jpg"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        target = torch.tensor(row[self.genre_columns].astype(int).values, dtype=torch.float32)

        return image, target

    def preprocess(
        self,
        output_folder: Path,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        Args:
            output_folder: Path to save processed data.
            train_split: Proportion of data for training set (Default: 0.8).
            val_split: Proportion of data for validation set (Default: 0.1).
            test_split: Proportion of data for test set (Default: 0.1).
            seed: Random seed for reproducibility (Default: 42).
        """

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # Validate splits
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, (
            f"Splits must sum to 1.0, got {train_split + val_split + test_split}"
        )

        # Split dataset
        # Train + Val and Test split
        train_df, temp_df = train_test_split(
            self.df,
            test_size=(val_split + test_split),
            random_state=seed,
        )

        # Split Temp into Val and Test
        val_size_ratio = val_split / (val_split + test_split)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size_ratio),
            random_state=seed,
        )

        print("Loading and processing splits...")
        train_images, train_targets = self._load_split(train_df)
        val_images, val_targets = self._load_split(val_df)
        test_images, test_targets = self._load_split(test_df)

        # Normalize if not using ImageNet normalization
        if not self.use_imagenet_norm:
            print("Applying dataset normalization...")
            train_images = self._normalize_dataset(train_images)
            val_images = self._normalize_dataset(val_images)
            test_images = self._normalize_dataset(test_images)

        # Save the splits
        print("Saving processed data...")
        torch.save(train_images, output_folder / "train_images.pt")
        torch.save(train_targets, output_folder / "train_targets.pt")
        torch.save(val_images, output_folder / "val_images.pt")
        torch.save(val_targets, output_folder / "val_targets.pt")
        torch.save(test_images, output_folder / "test_images.pt")
        torch.save(test_targets, output_folder / "test_targets.pt")

        # Metadata saving
        metadata = {
            "genre_names": list(self.genre_columns),
            "train_size": len(train_images),
            "val_size": len(val_images),
            "test_size": len(test_images),
            "image_size": self.image_size,
            "use_imagenet_norm": self.use_imagenet_norm,
        }

        with open(output_folder / "label_names.json", "w") as f:
            json.dump(list(self.genre_columns), f)

        with open(output_folder / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("Preprocessing complete.")
        print(f"Train samples: {len(train_images)}")
        print(f"Validation samples: {len(val_images)}")
        print(f"Test samples: {len(test_images)}")

    def _load_split(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            df: DataFrame for the split.

        Returns:
            A tuple of (images tensor, targets tensor)
        """

        images, targets = [], []

        for _, row in df.iterrows():
            img_path = self.data_path / "Images" / f"{row['Id']}.jpg"

            if not img_path.exists():
                print(f"Warning: {img_path} not found, skipping")
                continue

            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)

            target = torch.tensor(row[self.genre_columns].astype(int).values, dtype=torch.float32)

            images.append(image)
            targets.append(target)

        return torch.stack(images), torch.stack(targets)

    def _normalize_dataset(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize dataset images to zero mean and unit variance.

        Args:
            images: Tensor of shape (N, C, H, W).

        Returns:
            Normalized images tensor.

        """
        return (images - images.mean()) / images.std()


def download_data(raw_dir: Path) -> None:
    """Download and extract the movie poster dataset from Kaggle.

    Args:
        raw_dir: Path to the directory where raw data will be stored.

    Requires:
        - Kaggle API installed and configured.
        - Kaggle package installed.

    """

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if (raw_dir / "train.csv").exists() and (raw_dir / "Images").exists():
        print("Raw data already present, skipping download")
        return

    print("Downloading dataset from Kaggle...")

    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "raman77768/movie-classifier",
                "-p",
                str(raw_dir),
            ],
            check=True,
        )

    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print("Make sure Kaggle API is configured (~/.kaggle/kaggle.json)")
        raise

    # Unzip zip files
    print("Extracting dataset...")
    for zip_file in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(raw_dir)
        zip_file.unlink()

    # Clean up directory structure
    normalize_raw_structure(raw_dir)
    print("Download complete.")


def normalize_raw_structure(raw_dir: Path) -> None:
    """Clean up the extracted dataset structure.

    Moves files from `Multi_Label_dataset/` to `raw_dir/` and removes empty folders.

    Args:
        raw_dir: Path to the raw data directory.

    """
    raw_dir = Path(raw_dir)
    dataset_dir = raw_dir / "Multi_Label_dataset"

    if not dataset_dir.exists():
        return

    # Move train.csv
    src_csv = dataset_dir / "train.csv"
    dst_csv = raw_dir / "train.csv"
    if src_csv.exists() and not dst_csv.exists():
        shutil.move(str(src_csv), str(dst_csv))

    # Move Images/
    src_images = dataset_dir / "Images"
    dst_images = raw_dir / "Images"
    if src_images.exists() and not dst_images.exists():
        shutil.move(str(src_images), str(dst_images))

    # Remove empty folder
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)


def run_data_pipeline(
    data_path: Path = typer.Argument("data/raw", help="Path to the raw data directory"),
    output_folder: Path = typer.Argument("data/processed", help="Path to save processed data"),
    image_size: int = typer.Option(224, help="Image size for resizing"),
    use_imagenet_norm: bool = typer.Option(True, help="Use ImageNet normalization"),
    exclude_genres: str = typer.Option("N/A", help="Comma-separated genres to exclude"),
    train_split: float = typer.Option(0.8, help="Proportion of training data"),
    val_split: float = typer.Option(0.1, help="Proportion of validation data"),
    test_split: float = typer.Option(0.1, help="Proportion of test data"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Run the complete data pipeline: download, preprocess, and split data.

    This function:
    1. Downloads the raw dataset from Kaggle if not already present.
    2. Loads raw images and labels.
    3. Applies transformations (resizing, normalization).
    4. Splits the dataset into training, validation, and test sets.
    5. Saves the processed datasets and metadata.

    """

    print("Starting data pipeline...")

    download_data(data_path)

    # Parse excluded genres
    exclude_list = [g.strip() for g in exclude_genres.split(",") if g.strip()]

    dataset = MyDataset(
        data_path, image_size=image_size, use_imagenet_norm=use_imagenet_norm, exclude_genres=exclude_list
    )

    dataset.preprocess(output_folder, train_split=train_split, val_split=val_split, test_split=test_split, seed=seed)

    print("Data pipeline complete.")

    dataset_statistics(output_folder)


def poster_dataset(processed_path: Path = Path("data/processed")) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Load the processed movie poster dataset.

    Args:
        processed_path: Path to the processed data directory (.pt files).

    Returns:
        A tuple of (train_dataset, val_dataset, test_dataset), each being a `TensorDataset`.

    Raises:
        FileNotFoundError: If processed data files are not found.

    """
    processed_path = Path(processed_path)

    # Check files exist
    required_files = [
        "train_images.pt",
        "train_targets.pt",
        "val_images.pt",
        "val_targets.pt",
        "test_images.pt",
        "test_targets.pt",
    ]

    for file_name in required_files:
        filepath = processed_path / file_name
        if not filepath.exists():
            raise FileNotFoundError(f"Processed data not found: {filepath}\nRun preprocessing pipeline first")

    # Load data
    train_images = torch.load(processed_path / "train_images.pt")
    train_targets = torch.load(processed_path / "train_targets.pt")
    val_images = torch.load(processed_path / "val_images.pt")
    val_targets = torch.load(processed_path / "val_targets.pt")
    test_images = torch.load(processed_path / "test_images.pt")
    test_targets = torch.load(processed_path / "test_targets.pt")

    # Return 3 datasets
    return (
        TensorDataset(train_images, train_targets),
        TensorDataset(val_images, val_targets),
        TensorDataset(test_images, test_targets),
    )


def dataset_statistics(datadir: str = "data/processed") -> None:
    """Compute dataset statistics."""

    datadir = Path(datadir)

    train_ds, val_ds, test_ds = poster_dataset(datadir)

    print("=== Dataset statistics ===\n")

    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")

    image, target = train_ds[0]
    print(f"\nImage shape: {image.shape}")
    print(f"Target shape: {target.shape}")

    # Load genre names
    with open(datadir / "label_names.json") as f:
        genre_names = json.load(f)

    num_genres = len(genre_names)
    print(f"Number of genres: {num_genres}")

    images = torch.stack([train_ds[i][0] for i in range(min(25, len(train_ds)))])

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    axes = axes.flatten()

    for ax, img in zip(axes, images):
        img = img.permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img)
        ax.axis("off")

    # Save sample images
    plt.tight_layout()
    plt.savefig("sample_images.png")
    plt.close()

    train_targets = torch.stack([t for _, t in train_ds])
    train_dist = train_targets.sum(dim=0)

    plt.figure(figsize=(12, 4))
    plt.bar(range(num_genres), train_dist)
    plt.xticks(range(num_genres), genre_names, rotation=90)
    plt.ylabel("Count")
    plt.title("Train genre distribution")
    plt.tight_layout()

    # Save genre distribution
    plt.savefig("train_genre_distribution.png")
    plt.close()

    # Average number of labels per image
    avg_labels = train_targets.sum(dim=1).float().mean()
    print(f"\nAverage genres per image (train): {avg_labels:.2f}")


if __name__ == "__main__":
    typer.run(run_data_pipeline)
