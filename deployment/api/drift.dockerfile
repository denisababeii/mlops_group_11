# Dockerfile for data drift monitoring Cloud Run Job
# To build: docker build -f deployment/api/drift.dockerfile . -t movie-poster-drift:latest
# To run locally: docker run --env-file .env movie-poster-drift:latest

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# Install uv for faster package installation
RUN pip install --no-cache-dir uv

# Copy source code
COPY src/ src/
COPY deployment/api/mlops_group_11/data_drift.py data_drift.py

# Create necessary directories
RUN mkdir -p data/processed outputs

# Install Python dependencies
RUN uv pip install --system --no-cache -e .

# Install additional drift monitoring dependencies
RUN uv pip install --system --no-cache \
    evidently \
    google-cloud-storage

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the data drift script
CMD ["python", "data_drift.py"]
