FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY mlops_group_11/ ./mlops_group_11/

# Copy model and labels
COPY best_model.pth ./models/best_model.pth
COPY label_names.json ./data/processed/label_names.json

# Environment variables
ENV CHECKPOINT_PATH=/app/models/best_model.pth
ENV LABELS_PATH=/app/data/processed/label_names.json
ENV MODEL_NAME=csatv2_21m.sw_r512_in1k
ENV NUM_CLASSES=24
ENV IMAGE_SIZE=224
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run FastAPI
CMD uvicorn main:app --host 0.0.0.0 --port 8080
