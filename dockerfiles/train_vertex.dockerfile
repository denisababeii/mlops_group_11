# dockerfiles/train_vertex.dockerfile
# To build: docker build -f dockerfiles/train_vertex.dockerfile . -t train-vertex:latest
# To run: docker run --name vertex-train train-vertex:latest

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system dependencies (including git for DVC)
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# Copy source code and configs
COPY src src/
COPY configs configs/
COPY models models/
COPY reports reports/

WORKDIR /

# Set UV environment
ENV UV_LINK_MODE=copy

# Install dependencies
RUN uv sync --locked --no-cache --no-install-project && \
    uv add google-cloud-storage --no-cache

# Support both regular training and sweep agents
ENTRYPOINT ["sh", "-c", "\
    dvc pull && \
    if [ -n \"$WANDB_SWEEP_ID\" ]; then \
      uv run wandb agent ${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_SWEEP_ID} --count 1; \
    else \
      uv run src/mlops_group_11/train.py; \
    fi"]
