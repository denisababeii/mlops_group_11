FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/

RUN --mount=type=cache,id=uv-cache-api,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --only-group api

COPY src /app/src/
COPY models /app/models/
COPY data /app/data/

ENV PORT=8000

EXPOSE 8000
CMD ["sh", "-c", "uv run uvicorn src.mlops_group_11.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
