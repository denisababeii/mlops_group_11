#!/bin/bash
# Start BentoML service with increased timeout

cd "$(dirname "$0")"

# Set environment variables for BentoML timeouts
export BENTOML_API_WORKERS=1
export BENTOML_TIMEOUT=300

# Start the service with timeout flag
uv run --project ../.. bentoml serve service:PosterClassifierService \
    --port 3001 \
    --timeout 300 \
    --api-workers 1
