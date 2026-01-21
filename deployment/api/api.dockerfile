# Dockerfile for API deployment to Cloud Run
# To build: docker build -f deployment/api/Dockerfile . -t movie-poster-api:latest
# To run locally: docker run -p 8080:8080 --env-file .env movie-poster-api:latest

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# Install uv for faster package installation
RUN pip install --no-cache-dir uv

# Copy source code
COPY src/ src/
COPY deployment/api/main.py deployment/api/main.py

# Create necessary directories
RUN mkdir -p models data/processed outputs

# Install Python dependencies
RUN uv pip install --system --no-cache -e .

# Install additional API dependencies
RUN uv pip install --system --no-cache \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pillow \
    prometheus-client \
    google-cloud-storage

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD exec uvicorn deployment.api.main:app --host 0.0.0.0 --port ${PORT}
