from pathlib import Path
import typer
from torch.utils.data import Dataset, TensorDataset
import json
import torch
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
import subprocess
import zipfile
import shutil

class MyDataset(Dataset):
    """Raw movie poster dataset (multi-label)."""

    def __init__(self, data_path: Path, image_size: int = 224):
        self.data_path = data_path
        self.image_size = image_size

        self.df = pd.read_csv(data_path / "train.csv")

        # Genre columns = everything except Id and Genre strings
        self.genre_columns = self.df.columns.drop(["Id", "Genre"])

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    # Gets image and its label(s) at specific index
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.data_path / "Images" / f"{row['Id']}.jpg"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        target = torch.tensor(
            row[self.genre_columns].astype(float).values,
            dtype=torch.float32
        )

        return image, target

    def preprocess(self, output_folder: Path):
        output_folder.mkdir(parents=True, exist_ok=True)

        # Split dataset
        # Maybe we don't need the tran_test_split for this, the data already seems shuffled
        train_df, test_df = train_test_split(
            self.df,
            test_size=0.1,
            random_state=42,
        )

        train_images, train_targets = self._load_split(train_df)
        test_images, test_targets = self._load_split(test_df)

        train_images = normalize(train_images)
        test_images = normalize(test_images)

        torch.save(train_images, output_folder / "train_images.pt")
        torch.save(train_targets, output_folder / "train_targets.pt")
        torch.save(test_images, output_folder / "test_images.pt")
        torch.save(test_targets, output_folder / "test_targets.pt")

        # Save genre names (maybe we want to use them later)
        with open(output_folder / "label_names.json", "w") as f:
            json.dump(list(self.genre_columns), f)
        
        print(f"Train samples: {len(train_images)}")
        print(f"Test samples: {len(test_images)}")


    def _load_split(self, df):
        images, targets = [], []

        for _, row in df.iterrows():
            img_path = self.data_path / "Images" / f"{row['Id']}.jpg"

            if not img_path.exists():
                print(f"Warning: {img_path} not found, skipping")
                continue

            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)

            target = torch.tensor(
                row[self.genre_columns].astype(float).values,
                dtype=torch.float32
            )

            images.append(image)
            targets.append(target)

        return torch.stack(images), torch.stack(targets)


def download_data(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)

    if (raw_dir / "train.csv").exists() and (raw_dir / "Images").exists():
        print("Raw data already present, skipping download")
        return

    # Hardcoded the dataset name here, not sure if that is the best approach
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "raman77768/movie-classifier",
            "-p", str(raw_dir),
        ],
        check=True,
    )

    # Unzip zip files
    for zip_file in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(raw_dir)
        zip_file.unlink()

    normalize_raw_structure(raw_dir)


def normalize(images: torch.Tensor) -> torch.Tensor:
    return (images - images.mean()) / images.std()

# This only works for the specific kaggle dataset structure
# It is to clean up the structure after extraction
def normalize_raw_structure(raw_dir: Path):
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
    shutil.rmtree(dataset_dir)


def run_data_pipeline(data_path: Path, output_folder: Path, image_size: int = 224):
    download_data(data_path)
    dataset = MyDataset(data_path, image_size=image_size)
    dataset.preprocess(output_folder)


def poster_dataset():
    train_images = torch.load("data/processed/train_images.pt")
    train_targets = torch.load("data/processed/train_targets.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_targets = torch.load("data/processed/test_targets.pt")

    return (
        TensorDataset(train_images, train_targets),
        TensorDataset(test_images, test_targets),
    )

if __name__ == "__main__":
    typer.run(run_data_pipeline)
