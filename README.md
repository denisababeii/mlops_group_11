# mlops_group_11

## Machine Learning Goal
The goal of this project is to develop and train a machine learning model capable of classifying movie genres based on movie poster images.

## Dataset
The dataset used in this project consists of a collection of movie poster images and a CSV file that maps image IDs to their corresponding genres. Although the original image repository contains 7,867 poster images, we only use the 7,254 images that are referenced in the CSV file, to have the images' associated genre labels.

In total, the dataset includes 24 distinct genre classifications, as well as an additional “N/A” category for posters whose genre information is missing or unspecified. This diversity of labels presents a multi-class classification problem and allows for a comprehensive evaluation of the model’s performance across a wide range of genres.

The poster images will be resized, normalized and used to train the model for multi-class classification across the 25 genre labels.

## Modelling
We will leverage a pretrained model from the PyTorch Image Models (timm) library and fine-tune it for this specific task. We are planning to try various models and decide the final option based on the performance. Some models we are considering are timm/csatv2_21m.sw_r512_in1k, timm/resnet50.a1_in1k, timm/resnet32ts.ra2_in1k.

## Training
To train and evaluate the model, the dataset will be split into training and testing subsets. 90% of the data will be used for training the model, while the remaining 10% will be reserved for testing.

## Tools & Frameworks
We have used [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

We are planning to use PyTorch Image Models (timm) for modelling with a pretrained model, Hydra for configuration, Weights & Biases for logging and visualizations, Docker for containerization.

## Project structure
The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       ├── cml-data.yaml
│       ├── docker-building.yaml
│       ├── linting.yaml
│       ├── pre-commit-update.yaml
│       └── tests.yaml
├── configs/                  # Configuration files
│   ├── cloudbuild.yaml
│   ├── config.yaml
│   └── sweep.yaml
├── data/                     # Data directory (not git tracked)
│   ├── processed
│   └── raw
├── deployment/               # Deployment directory (adapted to use GCP)
│   └── api/
│       ├── mlops_group_11/
│       │   ├── __init__.py
│       │   ├── data_drift.py
│       │   └── frontend.py
│       ├── api.dockerfile
│       ├── drift.dockerfile
│       ├── frontend.dockerfile
│       └── main.py
├── dockerfiles/              # Dockerfiles
│   ├── api.dockerfile
│   ├── evaluate.dockerfile
│   ├── frontend.dockerfile
│   ├── train_vertex.dockerfile
│   └── train.dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
│   ├── model.onnx
│   └── model.pth
├── notebooks/                # Jupyter notebooks
│   └── inspect_data.ipynb
├── reports/                  # Reports
│   ├── figures/
│   └── README.md
├── src/                      # Source code
│   ├── mlops_group_11/
│   │   ├── api/
│   │   │   ├── bentoML/
│   │   │   │   ├── client.py
│   │   │   │   └── service.py
│   │   │   └── fast_api.py
│   │   ├── __init__.py
│   │   ├── adaptive_avg_pool_2d_custom.py
│   │   ├── data_drift.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── frontend.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_training.py
├── .dockerignore
├── .dvcignore
├── .gcloudignore
├── .gitignore
├── .pre-commit-config.yaml
├── cloudbuild.yaml
├── data.dvc
├── DATA.md                   # Data setup documentation
├── docker-compose.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── tasks.py
├── uv.lock
└── WANDB_SWEEP_GUIDE.md      # W&B documentation
```

## Running the Application

### Option 1: Using Docker Compose

The easiest way to run both the backend API and frontend together:

```bash
# Build and start both services
docker compose up --build

# Or run in detached mode
docker compose up -d --build

# Stop the services
docker compose down
```

The services will be locally available at:
- Backend API: http://localhost:8000
- Frontend: http://localhost:8001

### Option 2: Running Containers Standalone

#### Backend API

```bash
# Build the backend image
docker build -t backend -f dockerfiles/api.dockerfile .

# Run the backend container
docker run --rm -p 8000:8000 backend

# Or with custom port
docker run --rm -p 8080:8080 -e "PORT=8080" backend
```

#### Frontend

Build the frontend image:

```bash
docker build -t frontend -f dockerfiles/frontend.dockerfile .
```

Run the frontend container (platform-specific commands below):

**macOS/Windows:**
```bash
docker run --rm -p 8001:8001 -e "BACKEND=http://host.docker.internal:8000" frontend
```

**Linux:**
```bash
docker run --rm -p 8001:8001 -e "BACKEND=http://172.17.0.1:8000" frontend
```

> **Note:** When running standalone, the frontend needs to connect to the backend. Use `host.docker.internal` on macOS/Windows or `172.17.0.1` on Linux to access services running on the host machine.
