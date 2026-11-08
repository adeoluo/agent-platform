# syntax=docker/dockerfile:1

# ---- builder ----
FROM python:3.15-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends git gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# ann-vector-search-library compiles its Rust core from source during the sync.
COPY --from=rust:1-slim /usr/local/cargo /usr/local/cargo
COPY --from=rust:1-slim /usr/local/rustup /usr/local/rustup
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo PATH=/usr/local/cargo/bin:$PATH
WORKDIR /application
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

# ---- runtime ----
FROM python:3.15-slim AS runtime
RUN useradd --create-home --uid 1000 application && mkdir /data && chown application /data
WORKDIR /application
COPY --from=builder --chown=application:application /application /application
ENV PATH="/application/.venv/bin:$PATH" PYTHONUNBUFFERED=1 \
    AGENT_PLATFORM_DB=/data/platform.db
USER application
VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')" || exit 1
CMD ["agent-platform", "serve", "--host", "0.0.0.0", "--port", "8080"]
