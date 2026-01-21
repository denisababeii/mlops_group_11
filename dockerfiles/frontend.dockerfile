FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/

RUN --mount=type=cache,id=uv-cache-frontend,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --only-group frontend

COPY src /app/src/

ENV PORT=8501
# BACKEND should be set at runtime via -e BACKEND=... or docker-compose

EXPOSE 8501
CMD ["sh", "-c", "uv run streamlit run src/mlops_group_11/frontend.py --server.port ${PORT:-8501} --server.address=0.0.0.0"]
