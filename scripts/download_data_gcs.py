"""Download processed data from GCS bucket."""

import sys
from pathlib import Path

from google.cloud import storage


def download_data():
    """Download all processed data files from GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket("mlops-group11-data")

        # Files to download
        files = [
            "processed/train_images.pt",
            "processed/train_targets.pt",
            "processed/val_images.pt",
            "processed/val_targets.pt",
            "processed/test_images.pt",
            "processed/test_targets.pt",
            "processed/label_names.json",
            "processed/metadata.json",
        ]

        print("Downloading data from GCS...")
        for file_path in files:
            blob = bucket.blob(file_path)
            local_path = Path("data") / file_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"  Downloading {file_path}...")
            blob.download_to_filename(str(local_path))

        print("All data downloaded successfully!")
        return 0

    except Exception as e:
        print(f"Error downloading data: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(download_data())
