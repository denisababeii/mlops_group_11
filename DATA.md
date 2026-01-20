# Data Setup Instructions

Created with the help of Claude :D

## For New Team Members

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/mlops_group_11.git
cd mlops_group_11
```

### 2. Install Dependencies
```bash
uv sync
```

### 3. Authenticate to GCP
```bash
gcloud auth login
gcloud config set project mlops-group-11
gcloud auth application-default login
```

### 4. Pull Data from DVC
```bash
# Download all data from GCS bucket
uv run dvc pull
```

This will create the `data/` folder with:
- `data/raw/Images/` - Raw movie poster images
- `data/raw/train.csv` - Training labels
- `data/processed/` - Preprocessed PyTorch tensors

### 5. Verify Data
```bash
ls data/raw/
ls data/processed/
```

## Data Structure
```
data/
├── raw/
│   ├── Images/           # Movie poster images (7254 JPG files)
│   └── train.csv         # Labels for multi-label classification
└── processed/
    ├── train_images.pt   # Preprocessed training images
    ├── train_targets.pt  # Training labels (float32)
    ├── val_images.pt     # Validation images
    ├── val_targets.pt    # Validation labels
    ├── test_images.pt    # Test images
    ├── test_targets.pt   # Test labels
    ├── label_names.json  # Genre names
    └── metadata.json     # Dataset metadata
```

## Updating Data

If you modify or add data:
```bash
# Re-track with DVC
uv run dvc add data/

# Commit changes
git add data.dvc
git commit -m "Update dataset - describe changes"

# Tag new version
git tag -a "data-v2.0" -m "Description of changes"

# Push to GCS and GitHub
uv run dvc push
git push origin main --tags
```

## Reverting to Previous Data Version
```bash
# Checkout old version
git checkout data-v1.0

# Pull corresponding data
uv run dvc checkout

# Return to latest
git checkout main
uv run dvc checkout
```
