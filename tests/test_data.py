from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from mlops_group_11.data import MyDataset, poster_dataset


def _make_dummy_raw_dataset(root: Path, n: int = 20, num_genres: int = 3) -> None:
    """
    Helper: create a tiny fake dataset on disk.

    It creates:
      root/train.csv
      root/Images/0.jpg ... (n-1).jpg

    The CSV contains the columns expected by your MyDataset implementation:
      - Id (used to find images)
      - Genre (excluded internally)
      - N/A (excluded by default)
      - genre_0 ... genre_{num_genres-1} as binary labels
    """
    images_dir = root / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create tiny RGB images (content doesn't matter, only that PIL can open them)
    for i in range(n):
        img = Image.new("RGB", (64, 64))
        img.save(images_dir / f"{i}.jpg")

    genre_cols = [f"genre_{k}" for k in range(num_genres)]
    rows = []
    for i in range(n):
        row = {"Id": i, "Genre": "dummy", "N/A": 0}

        # simplified multi-genre pattern
        for k, col in enumerate(genre_cols):
            row[col] = 1 if (i + k) % 2 == 0 else 0
        rows.append(row)

    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)


#   Test function for poster_dataset preprocessed files check
def test_poster_dataset_raises_if_processed_files_missing(tmp_path: Path) -> None:
    """
    Data test: verify `poster_dataset` fails if processed files are missing.

    Corresponds to: poster_dataset() in mlops_group_11.data

    Indicates that users must run preprocessing before loading.
    """
    processed = tmp_path / "processed"
    processed.mkdir()

    with pytest.raises(FileNotFoundError) as excinfo:  # expecting FileNotFoundError
        poster_dataset(processed)

    # Check that the error message guides the user to run preprocessing
    assert "Run preprocessing pipeline first" in str(
        excinfo.value
    ), "Expected FileNotFoundError to instruct the user to run preprocessing."


# This function tests MyDataset.__len__ and MyDataset.__getitem__


@pytest.mark.parametrize("image_size", [32, 64])
def test_mydataset_len_and_getitem_shapes(tmp_path: Path, image_size: int) -> None:
    """
    Data test + parametrization requirement:
    - Check __len__ matches CSV rows.
    - Check __getitem__ returns (image_tensor, target_tensor) with correct shapes.

    Corresponds to:
      - MyDataset.__len__
      - MyDataset.__getitem__
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_dummy_raw_dataset(raw, n=10, num_genres=4)

    ds = MyDataset(data_path=raw, image_size=image_size, use_imagenet_norm=False)

    assert len(ds) == 10, "Dataset length should equal number of CSV rows."

    x, y = ds[0]
    assert isinstance(x, torch.Tensor), "Image should be a torch.Tensor"
    assert isinstance(y, torch.Tensor), "Target should be a torch.Tensor"

    # Your transform makes images (3, H, W)
    assert x.shape == (3, image_size, image_size), "Image tensor should have shape (3, image_size, image_size)."
    assert x.dtype == torch.float32, "Image tensor should be float32."

    # Target vector length equals number of label columns (num_genres)
    assert y.shape == (4,), "Target tensor should have shape (num_genres,)."
    assert y.dtype == torch.float32, "Target tensor should be float32."
    assert torch.all((y == 0) | (y == 1)), "Targets should be binary 0/1."


# Test function for MyDataset.preprocess output files and loader sanity check


def test_preprocess_creates_expected_outputs(tmp_path: Path) -> None:
    """
    Data test: verify preprocess() creates the exact files that the rest
    of the project depends on.

    Corresponds to: MyDataset.preprocess + poster_dataset
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_dummy_raw_dataset(raw, n=30, num_genres=5)

    processed = tmp_path / "processed"

    ds = MyDataset(data_path=raw, image_size=32, use_imagenet_norm=False)
    ds.preprocess(output_folder=processed, train_split=0.8, val_split=0.1, test_split=0.1, seed=123)

    expected_pt = [
        "train_images.pt",
        "train_targets.pt",
        "val_images.pt",
        "val_targets.pt",
        "test_images.pt",
        "test_targets.pt",
    ]
    for fname in expected_pt:
        assert (processed / fname).exists(), f"Expected {fname} to be created by preprocess()."

    assert (processed / "label_names.json").exists(), "Expected label_names.json to be created."
    assert (processed / "metadata.json").exists(), "Expected metadata.json to be created."

    # Sanity check: loader can read what preprocess created
    train_ds, val_ds, test_ds = poster_dataset(processed)
    assert len(train_ds) > 0, "Expected non-empty train dataset."
    assert len(val_ds) > 0, "Expected non-empty val dataset."
    assert len(test_ds) > 0, "Expected non-empty test dataset."


# Test function for MyDataset.preprocess invalid splits check


def test_preprocess_invalid_splits_raises(tmp_path: Path) -> None:
    """
    Data test: verify preprocess() rejects invalid split ratios.

    Corresponds to: the assertion in MyDataset.preprocess that splits sum to 1.0
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_dummy_raw_dataset(raw, n=10, num_genres=2)

    processed = tmp_path / "processed"
    ds = MyDataset(data_path=raw, image_size=32, use_imagenet_norm=False)

    with pytest.raises(AssertionError):
        ds.preprocess(output_folder=processed, train_split=0.8, val_split=0.1, test_split=0.2)
