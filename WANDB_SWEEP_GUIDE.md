# W&B Hyperparameter Sweep Guide

Complete guide for running hyperparameter sweeps on Vertex AI with Weights & Biases integration. Created with the help of Claude :D

## Prerequisites

- GCP project configured (mlops-group-11)
- gcloud CLI authenticated
- W&B account created
- Repository cloned and dependencies installed (`uv sync`)

---

## Step-by-Step Instructions

### Step 1: Configure W&B Credentials

Create your local `.env` file with W&B credentials:

```bash
WANDB_API_KEY=your_wandb_api_key_here
WANDB_ENTITY=your_wandb_username
WANDB_PROJECT=mlops_group_11
```

**Where to find credentials:**
- API Key: https://wandb.ai/authorize
- Entity: Your W&B username (visible at top-right on wandb.ai)
- Project: Use `mlops_group_11`

---

### Step 2: Initialize the Sweep

Load your credentials and create the sweep:
```bash
# Load environment variables
source .env

# Initialize sweep (creates sweep on W&B)
uv run python scripts/create_sweep.py
```

**Output example:**
```
✅ Sweep created successfully!
Sweep ID: abc123xyz456
Run agents with: wandb agent your-entity/mlops_group_11/abc123xyz456
```

**⚠️ Important:** Copy the Sweep ID from the output - you'll need it in the next step!

---

### Step 3: Create Job Configuration

Create the Vertex AI job configuration file with your credentials:
```bash
# Set your sweep ID (replace with actual ID from Step 2)
SWEEP_ID="abc123xyz456"

# Create job config file
cat > sweep_job_config.yaml << EOF
workerPoolSpecs:
  - machineSpec:
      machineType: e2-standard-8
    replicaCount: 1
    containerSpec:
      imageUri: europe-west1-docker.pkg.dev/mlops-group-11/mlops-group11-images/train:latest
      env:
        - name: WANDB_API_KEY
          value: "${WANDB_API_KEY}"
        - name: WANDB_ENTITY
          value: "${WANDB_ENTITY}"
        - name: WANDB_PROJECT
          value: "${WANDB_PROJECT}"
        - name: WANDB_SWEEP_ID
          value: "${SWEEP_ID}"
EOF
```

**Verify the config was created correctly:**
```bash
# Check that variables expanded (should show actual values, not ${...})
grep "value:" sweep_job_config.yaml
```

Should show:
```yaml
value: "your_actual_api_key"
value: "your_username"
value: "mlops_group_11"
value: "abc123xyz456"
```

---

### Step 4: Rebuild Docker Image

Build the Docker image with updated code:
```bash
gcloud builds submit --config=cloudbuild.yaml --timeout=30m .
```

**Expected time:** 5-15 minutes

**What this does:**
- Uploads source code to Google Cloud Build
- Builds Docker image on Google's servers
- Pushes to Artifact Registry automatically

---

### Step 5: Submit Sweep Trials

Launch 10 parallel training jobs (sweep trials):
*NOTE: it cost $10 USD and cheaper options can be considered

```bash
for i in {1..10}; do
  gcloud ai custom-jobs create \
    --region=europe-west1 \
    --display-name=sweep-trial-$i \
    --config=sweep_job_config.yaml
done
```

**What happens:**
- Creates 10 Vertex AI training jobs
- Each tests different hyperparameter combinations
- W&B Bayesian optimization picks parameters
- Jobs may run sequentially or in parallel (based on quotas)

**Expected time:** +2 Hours (all trials complete)

---

### Step 6: Monitor Results

#### **W&B Dashboard (Primary)**

View real-time sweep progress and results in W&B page.

**What you'll see:**
- Parallel coordinates plot (parameter importance)
- Table ranking all trials by validation loss
- Best hyperparameters highlighted
- Training curves for each trial
- Real-time metrics as jobs run

#### **Vertex AI Console (Job Status)**

Monitor job execution status:
```
Vertex AI -- Training
```

**What you'll see:**
- Job status (Running/Succeeded/Failed)
- Execution logs
- Resource usage

---

### Step 7: Download Best Models

After sweep completes, download the best performing models (can also be seen in Google Cloud Buckets):
```bash
# List all models saved during sweep
gsutil ls gs://mlops-group11-data/models/

# Download best models
gsutil cp gs://mlops-group11-data/models/best_model_*.pth models/

# Download training curves
gsutil cp gs://mlops-group11-data/reports/training_curves_*.png reports/figures/
```

---


## Troubleshooting

### Common Issues

**Issue: "entity not found" error**
```bash
# Check your WANDB_ENTITY in .env
# Should be your username (white name), not organization name
cat .env
```

**Issue: "permission denied" error**
```bash
# Project doesn't exist on W&B
# Create it at: https://wandb.ai/home
# Or change WANDB_PROJECT in .env to existing project
```

**Issue: Variables not expanding in yaml**
```bash
# Make sure you ran: source .env
# Verify variables loaded:
echo $WANDB_API_KEY
echo $WANDB_ENTITY

# If empty, run: source .env
# Then recreate sweep_job_config.yaml
```

**Issue: Jobs failing immediately**
```bash
# Check logs for specific error
gcloud ai custom-jobs stream-logs <JOB_NAME> --region=europe-west1

# Common causes:
# - Invalid WANDB_API_KEY (check .env)
# - Docker image not rebuilt (run Step 4 again)
# - Sweep ID incorrect (check Step 2 output)
```

**Issue: Only some jobs running**
```bash
# Normal! GCP has concurrent job limits (5-10 jobs)
# Remaining jobs will start as others finish
# All will complete eventually
```

---

## Sweep Configuration

The sweep parameters are defined in `configs/sweep.yaml`:
```yaml
method: bayes                    # Bayesian optimization
metric:
  goal: minimize
  name: val/loss                # Optimize validation loss

parameters:
  hyperparameters.lr:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1
  
  hyperparameters.batch_size:
    values: [16, 32, 64]
  
  hyperparameters.epochs:
    values: [5, 10, 15]
  
  hyperparameters.prob_threshold:
    distribution: uniform
    min: 0.1
    max: 0.9
  
  model.name:
    values: ["csatv2_21m.sw_r512_in1k", "resnet50", "efficientnet_b0"]

run_cap: 10                      # Maximum 10 trials
```

To modify sweep parameters, edit `configs/sweep.yaml` before Step 2.

---
