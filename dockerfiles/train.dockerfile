# To build the image run 'docker build -f dockerfiles/train.dockerfile . -t train:latest' in the mlops_group_11 directory
# To run inside the container run 'docker run --name container-name train:latest'

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src src/
COPY models models/
COPY reports reports/
COPY configs configs/

WORKDIR /
ENV UV_LINK_MODE=copy
RUN uv sync --locked --no-cache --no-install-project
# RUN --mount=type=cache,target=/root/.cache/uv uv sync # Comment above and uncomment this to use cache
ENTRYPOINT ["uv", "run", "src/mlops_group_11/train.py"]