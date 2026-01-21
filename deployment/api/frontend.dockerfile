# Dockerfile for Streamlit frontend deployment to Cloud Run
# To build: docker build -f deployment/api/frontend.dockerfile . -t movie-poster-frontend:latest
# To run locally: docker run -p 8501:8501 --env-file .env movie-poster-frontend:latest

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY deployment/api/mlops_group_11/frontend.py frontend.py

# Install uv for faster package installation
RUN pip install --no-cache-dir uv

# Install Streamlit and dependencies
RUN uv pip install --system --no-cache \
    streamlit \
    pandas \
    requests \
    google-cloud-run

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

# Expose Streamlit port
EXPOSE 8501

# Create Streamlit config
RUN mkdir -p /root/.streamlit
RUN echo '[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
[browser]\n\
gatherUsageStats = false\n\
' > /root/.streamlit/config.toml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD streamlit run frontend.py --server.address=0.0.0.0 --server.port=${PORT}
