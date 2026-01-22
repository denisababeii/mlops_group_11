FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/

RUN --mount=type=cache,id=uv-cache-frontend,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --only-group frontend

COPY src /app/src/

EXPOSE 8080

ENTRYPOINT ["sh", "-c", "uv run streamlit run src/mlops_group_11/frontend.py --server.port ${PORT:-8080} --server.address=0.0.0.0"]
