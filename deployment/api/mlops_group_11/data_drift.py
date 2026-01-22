"""Data drift detection module for movie poster predictions."""

import json
import os
from io import StringIO
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
from google.cloud import storage

# GCS configuration
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", "mlops-group-11")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "mlops-group11-data")
GCS_PREDICTIONS_PATH = os.getenv("GCS_PREDICTIONS_PATH", "predictions_database.csv")

# Local cache paths
LOCAL_DATA_DIR = Path("data/processed")
LOCAL_OUTPUTS_DIR = Path("outputs")


def download_from_gcs(gcs_path: str, local_path: Path) -> None:
    """Download file from GCS to local path.

    Args:
        gcs_path: Path in GCS bucket (e.g., 'data/processed/train_images.pt')
        local_path: Local filesystem path to save the file
    """
    if local_path.exists():
        print(f"Using cached file: {local_path}")
        return

    try:
        print(f"Downloading {gcs_path} from GCS...")
        client = storage.Client(project=GCS_PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)

        # Create parent directories if they don't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the file
        blob.download_to_filename(str(local_path))
        print(f"Downloaded {gcs_path} to {local_path}")

    except Exception as e:
        raise FileNotFoundError(f"Could not download {gcs_path} from gs://{GCS_BUCKET_NAME}/{gcs_path}: {e}") from e


def load_reference_data(
    train_images_gcs: str = "data/processed/train_images.pt",
    train_targets_gcs: str = "data/processed/train_targets.pt",
    labels_gcs: str = "data/processed/label_names.json",
) -> dict:
    """Load reference data from training set for drift comparison.

    Downloads data from GCS if not already cached locally.

    Args:
        train_images_gcs: GCS path to training images tensor
        train_targets_gcs: GCS path to training targets tensor
        labels_gcs: GCS path to label names JSON

    Returns:
        Dictionary containing training data and statistics
    """
    # Define local paths
    local_images = LOCAL_DATA_DIR / "train_images.pt"
    local_targets = LOCAL_DATA_DIR / "train_targets.pt"
    local_labels = LOCAL_DATA_DIR / "label_names.json"

    # Download from GCS if needed
    print("Loading reference data...")
    download_from_gcs(train_images_gcs, local_images)
    download_from_gcs(train_targets_gcs, local_targets)
    download_from_gcs(labels_gcs, local_labels)

    # Load from local cache
    print("Loading tensors into memory...")
    train_images = torch.load(local_images)
    train_targets = torch.load(local_targets)

    with open(local_labels) as f:
        labels = json.load(f)

    # Compute overall image statistics
    train_images_float = train_images.float()
    image_stats = {
        "mean": float(train_images_float.mean().item()),
        "std": float(train_images_float.std().item()),
        "min": float(train_images_float.min().item()),
        "max": float(train_images_float.max().item()),
    }

    print(f"Loaded {len(train_images)} reference samples")
    print(f"Labels: {len(labels)} genres")
    print(f"Image stats: mean={image_stats['mean']:.2f}, std={image_stats['std']:.2f}")

    return {
        "images": train_images,
        "targets": train_targets,
        "labels": labels,
        "num_samples": len(train_images),
        "image_stats": image_stats,
    }


def load_current_data_from_gcs(
    predictions_gcs_path: str = None,
) -> pd.DataFrame:
    """Load current predictions from GCS.

    Args:
        predictions_gcs_path: GCS path to predictions CSV file

    Returns:
        DataFrame with prediction data, or empty DataFrame if no data exists
    """
    if predictions_gcs_path is None:
        predictions_gcs_path = GCS_PREDICTIONS_PATH

    try:
        print(f"Loading current predictions from gs://{GCS_BUCKET_NAME}/{predictions_gcs_path}...")
        client = storage.Client(project=GCS_PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(predictions_gcs_path)

        # Download as text
        content = blob.download_as_text()

        # Parse CSV
        df = pd.read_csv(
            StringIO(content),
            names=["timestamp", "filename", "genres", "threshold", "image_mean", "image_std", "image_min", "image_max"],
            header=None,
        )

        print(f"Loaded {len(df)} prediction records")
        return df

    except Exception as e:
        print(f"Warning: Could not load predictions from GCS: {e}")
        print("Returning empty DataFrame (no drift analysis will be performed)")

        # Return empty DataFrame with correct structure
        return pd.DataFrame(
            columns=[
                "timestamp",
                "filename",
                "genres",
                "threshold",
                "image_mean",
                "image_std",
                "image_min",
                "image_max",
            ]
        )


def standardize_dataframes(
    reference_data: dict,
    current_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize reference and current data to have the same columns.

    Converts both to have the same genre columns plus image statistics.

    Args:
        reference_data: Dictionary with training data
        current_data: DataFrame with prediction data

    Returns:
        Tuple of (reference_df, current_df) with matching columns
    """
    # Get label names
    labels = reference_data["labels"]
    image_stat_cols = ["image_mean", "image_std", "image_min", "image_max"]

    print("Standardizing dataframes...")

    # Convert reference targets to DataFrame (binary multi-label format)
    reference_df = pd.DataFrame(
        reference_data["targets"].numpy(),
        columns=labels,
    )

    # Compute image statistics for each training image
    train_images = reference_data["images"].float()
    image_stats_data = []

    print(f"Computing image statistics for {len(train_images)} reference images...")
    for img in train_images:
        image_stats_data.append(
            {
                "image_mean": float(img.mean().item()),
                "image_std": float(img.std().item()),
                "image_min": float(img.min().item()),
                "image_max": float(img.max().item()),
            }
        )

    # Add image statistics columns to reference
    for stat_col in image_stat_cols:
        reference_df[stat_col] = [stats[stat_col] for stats in image_stats_data]

    # Parse current data
    genre_data = []
    print(f"Parsing {len(current_data)} prediction records...")

    for idx, row in current_data.iterrows():
        genres_str = row["genres"]

        # Initialize row with zeros for all genres
        parsed_row = {label: 0.0 for label in labels}

        # Parse the genres string and fill in the probabilities
        # Format: "Action:0.85|Comedy:0.62|Drama:0.71"
        if pd.notna(genres_str) and genres_str:
            genre_pairs = genres_str.split("|")
            for pair in genre_pairs:
                if ":" in pair:
                    genre_name, probability = pair.split(":", 1)
                    genre_name = genre_name.strip()
                    if genre_name in parsed_row:
                        try:
                            parsed_row[genre_name] = float(probability)
                        except ValueError:
                            print(f"Warning: Could not parse probability '{probability}' for genre '{genre_name}'")

        # Add image statistics from current data
        for stat_col in image_stat_cols:
            parsed_row[stat_col] = row[stat_col]

        genre_data.append(parsed_row)

    # Create current DataFrame with same columns as reference
    current_df = pd.DataFrame(genre_data)

    # Ensure columns are in the same order (image stats first, then genres)
    all_cols = image_stat_cols + labels
    current_df = current_df[[col for col in all_cols if col in current_df.columns]]
    reference_df = reference_df[[col for col in all_cols if col in reference_df.columns]]

    # Ensure both are float type
    reference_df = reference_df.astype(float)
    current_df = current_df.astype(float)

    print(f"Reference data shape: {reference_df.shape}")
    print(f"Current data shape: {current_df.shape}")

    return reference_df, current_df


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: Path = None,
) -> None:
    """Generate and save drift analysis report.

    Args:
        reference_df: Reference data DataFrame
        current_df: Current data DataFrame
        output_path: Path to save HTML report
    """
    if output_path is None:
        output_path = LOCAL_OUTPUTS_DIR / "data_drift_report.html"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating drift report...")
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(),
        ]
    )

    report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    report.save_html(str(output_path))
    print(f"Drift report saved to: {output_path}")


def run_drift_tests(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict:
    """Run data quality tests.

    Args:
        reference_df: Reference data DataFrame
        current_df: Current data DataFrame

    Returns:
        Dictionary with test results
    """
    print("Running data quality tests...")

    data_test = TestSuite(
        tests=[
            TestNumberOfMissingValues(),
            TestNumberOfColumns(),
            TestNumberOfEmptyRows(),
            TestNumberOfEmptyColumns(),
            TestColumnsType(),
        ]
    )

    data_test.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    result = data_test.as_dict()

    all_passed = result["summary"]["all_passed"]
    print(f"{'✓' if all_passed else '✗'} All tests passed: {all_passed}")

    return result


def upload_report_to_gcs(
    local_path: Path,
    gcs_path: str = None,
) -> None:
    """Upload drift report to GCS.

    Args:
        local_path: Local path to the report file
        gcs_path: GCS path to upload to (optional)
    """
    if gcs_path is None:
        gcs_path = f"reports/drift/{local_path.name}"

    try:
        print(f"Uploading report to gs://{GCS_BUCKET_NAME}/{gcs_path}...")
        client = storage.Client(project=GCS_PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)

        blob.upload_from_filename(str(local_path))
        print(f"Report uploaded to gs://{GCS_BUCKET_NAME}/{gcs_path}")

    except Exception as e:
        print(f"Warning: Could not upload report to GCS: {e}")


def main():
    """Main function to run drift detection pipeline."""
    print("=" * 60)
    print("MOVIE POSTER DATA DRIFT DETECTION")
    print("=" * 60)

    # Load reference and current data
    try:
        reference_data = load_reference_data()
    except Exception as e:
        print(f"✗ Error loading reference data: {e}")
        return 1

    try:
        current_data = load_current_data_from_gcs()
    except Exception as e:
        print(f"✗ Error loading current data: {e}")
        return 1

    # Check if we have current data
    if len(current_data) == 0:
        print("No current predictions found. Skipping drift analysis.")
        print("  Make some predictions first using the /predict endpoint")
        return 0

    # Standardize dataframes
    try:
        reference_df, current_df = standardize_dataframes(reference_data, current_data)
    except Exception as e:
        print(f"Error standardizing dataframes: {e}")
        return 1

    # Generate drift report
    try:
        report_path = LOCAL_OUTPUTS_DIR / "data_drift_report.html"
        generate_drift_report(reference_df, current_df, report_path)

        # Upload to GCS
        upload_report_to_gcs(report_path)

    except Exception as e:
        print(f"Error generating drift report: {e}")
        return 1

    # Run data quality tests
    try:
        test_results = run_drift_tests(reference_df, current_df)

        # Save test results
        results_path = LOCAL_OUTPUTS_DIR / "drift_test_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, "w") as f:
            json.dump(test_results, f, indent=2)

        print(f"Test results saved to: {results_path}")

        # Upload test results to GCS
        upload_report_to_gcs(results_path, "reports/drift/drift_test_results.json")

    except Exception as e:
        print(f"Error running drift tests: {e}")
        return 1

    print("DRIFT DETECTION COMPLETE")

    return 0


if __name__ == "__main__":
    import sys

    exit_code = main()
    sys.exit(exit_code)
