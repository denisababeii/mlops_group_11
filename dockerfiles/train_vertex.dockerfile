# To build: docker build -f dockerfiles/train_vertex.dockerfile . -t train-vertex:latest
# To run: docker run --name vertex-train train-vertex:latest

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system dependencies (including git for DVC)
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
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

# Copy DVC configuration
COPY .dvc/ .dvc/
COPY data.dvc data.dvc

WORKDIR /

# Set UV environment
ENV UV_LINK_MODE=copy

# Install project dependencies + DVC
RUN uv sync --locked --no-cache --no-install-project && \
    uv add dvc dvc-gs --no-cache

# Entry point: pull data from DVC, then train
# Important note, a shell is needed for "dvc pull" to be run, so it is necessary
# to use the "sh", "-c"

ENTRYPOINT ["sh", "-c", "uv run dvc pull && uv run src/mlops_group_11/train.py"]
