"""Initialize W&B hyperparameter sweep."""

import os
import sys

import wandb
import yaml
from dotenv import load_dotenv

load_dotenv()

# Load sweep config from file
with open("configs/sweep.yaml", "r") as f:
    sweep_config = yaml.safe_load(f)

# Create sweep
try:
    sweep_id = wandb.sweep(
        sweep_config,
        project=os.getenv("WANDB_PROJECT", "mlops_group_11"),
        entity=os.getenv("WANDB_ENTITY"),
    )

    print(f"Sweep created successfully!")
    print(f"Sweep ID: {sweep_id}")
    print(
        f"Run agents with: wandb agent {os.getenv('WANDB_ENTITY')}/{os.getenv('WANDB_PROJECT', 'mlops_group_11')}/{sweep_id}"
    )

except Exception as e:
    print(f"Error creating sweep: {e}")
    sys.exit(1)
