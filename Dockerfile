FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/hf-cache \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Pinned uv binary (spec §30 tooling), used inside the container only.
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /usr/local/bin/

WORKDIR /app
RUN git config --system --add safe.directory /app

COPY pyproject.toml uv.lock* ./
ARG UID=1000
ARG GID=1000
RUN if [ -f uv.lock ]; then uv sync --frozen --no-install-project; else uv sync --no-install-project; fi \
    && mkdir -p /opt/hf-cache /opt/uv-cache \
    && chown -R ${UID}:${GID} /opt/venv /opt/hf-cache /opt/uv-cache \
    && git config --system --add safe.directory '*'

# The container runs as the host user (uid 1000, see docker-compose.yml). uv needs a writable cache.
ENV UV_CACHE_DIR=/opt/uv-cache HOME=/tmp

CMD ["bash"]
