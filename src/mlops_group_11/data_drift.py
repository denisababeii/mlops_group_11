import json
from pathlib import Path

import pandas as pd
import torch
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.legacy.report import Report
from evidently.legacy.test_suite import TestSuite
from evidently.legacy.tests import (
    TestColumnsType,
    TestNumberOfColumns,
    TestNumberOfEmptyColumns,
    TestNumberOfEmptyRows,
    TestNumberOfMissingValues,
)


def load_reference_data(
    train_images_path: str = "data/processed/train_images.pt",
    train_targets_path: str = "data/processed/train_targets.pt",
    labels_path: str = "data/processed/label_names.json",
) -> dict:
    """Load reference data from training set for drift comparison."""
    train_images = torch.load(train_images_path)
    train_targets = torch.load(train_targets_path)

    with open(labels_path) as f:
        labels = json.load(f)

    train_images_float = train_images.float()
    image_stats = {
        "mean": float(train_images_float.mean().item()),
        "std": float(train_images_float.std().item()),
        "min": float(train_images_float.min().item()),
        "max": float(train_images_float.max().item()),
    }

    return {
        "images": train_images,
        "targets": train_targets,
        "labels": labels,
        "num_samples": len(train_images),
        "image_stats": image_stats,
    }


def load_current_data(
    prediction_database_path: str = "prediction_database.csv",
) -> pd.DataFrame:
    """Load current predictions from the prediction database."""
    df = pd.read_csv(
        prediction_database_path,
        names=["timestamp", "filename", "genres", "threshold", "image_mean", "image_std", "image_min", "image_max"],
        header=None,
    )
    return df


def standardize_dataframes(
    reference_data: dict,
    current_data: pd.DataFrame,
) -> tuple:
    """Standardize reference and current data to have the same columns.
    Converts both to have the same genre columns plus image statistics.
    """
    # Get label names
    labels = reference_data["labels"]
    image_stat_cols = ["image_mean", "image_std", "image_min", "image_max"]

    # Convert reference targets to DataFrame (binary multi-label format)
    reference_df = pd.DataFrame(
        reference_data["targets"].numpy(),
        columns=labels,
    )

    # Compute image statistics for each training image
    train_images = reference_data["images"].float()
    image_stats_data = []
    for img in train_images:
        image_stats_data.append(
            {
                "image_mean": float(img.mean().item()),
                "image_std": float(img.std().item()),
                "image_min": float(img.min().item()),
                "image_max": float(img.max().item()),
            }
        )

    # Add image statistics columns
    for stat_col in image_stat_cols:
        reference_df[stat_col] = [stats[stat_col] for stats in image_stats_data]

    # Parse current data
    genre_data = []
    for idx, genres_str in enumerate(current_data["genres"]):
        row = {label: 0.0 for label in labels}

        # Parse the genres string and fill in the probabilities
        if pd.notna(genres_str) and genres_str:
            genre_pairs = genres_str.split("|")
            for pair in genre_pairs:
                if ":" in pair:
                    genre_name, probability = pair.split(":")
                    genre_name = genre_name.strip()
                    if genre_name in row:
                        row[genre_name] = float(probability)

        # Add image statistics
        for stat_col in image_stat_cols:
            row[stat_col] = current_data[stat_col].iloc[idx]

        genre_data.append(row)

    # Create current DataFrame with same columns as reference
    current_df = pd.DataFrame(genre_data)

    # Ensure columns are in the same order (image stats first, then genres)
    all_cols = image_stat_cols + labels
    current_df = current_df[[col for col in all_cols if col in current_df.columns]]
    reference_df = reference_df[[col for col in all_cols if col in reference_df.columns]]

    reference_df = reference_df.astype(float)
    current_df = current_df.astype(float)

    return reference_df, current_df


# Load reference and current data
reference_data = load_reference_data()
current_data = load_current_data()

# Standardize dataframes
reference_data_standardized, current_data_standardized = standardize_dataframes(reference_data, current_data)

report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
report.run(reference_data=reference_data_standardized, current_data=current_data_standardized)
report.save_html("outputs/data_drift_report.html")

data_test = TestSuite(
    tests=[
        TestNumberOfMissingValues(),
        TestNumberOfColumns(),
        TestNumberOfEmptyRows(),
        TestNumberOfEmptyColumns(),
        TestColumnsType(),
    ]
)
data_test.run(reference_data=reference_data_standardized, current_data=current_data_standardized)
result = data_test.as_dict()
print("All tests passed: ", result["summary"]["all_passed"])
