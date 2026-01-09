# mlops_group_11

The goal of this project is to develop and train a machine learning model capable of classifying movie genres based on movie poster images. 

The dataset used in this project consists of a collection of movie poster images and a CSV file that maps image IDs to their corresponding genres. Although the original image repository contains 7,867 poster images, we only use the 7,254 images that are referenced in the CSV file, to have the images' associated genre labels. 

In total, the dataset includes 24 distinct genre classifications, as well as an additional “N/A” category for posters whose genre information is missing or unspecified. This diversity of labels presents a multi-class classification problem and allows for a comprehensive evaluation of the model’s performance across a wide range of genres.

To train and evaluate the model, the dataset will be split into training and testing subsets. 90% of the data will be used for training the model, while the remaining 10% will be reserved for testing.

The poster images will be resized, normalized and used to train the model for multi-class classification across the 25 genre labels. We will leverage a pretrained model from the PyTorch Image Models (timm) library and fine-tune it for this specific task.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
