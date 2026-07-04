# CloudDrift — Production Docker Image
#
# Single image used by both the FastAPI (api) and Streamlit (dashboard)
# services in compose.yml. The command differs per service; everything
# else — Python version, dependencies, source code — is identical.
#
# UV is used as the sole package manager, consistent with local development.
# uv sync --frozen installs exactly the versions recorded in uv.lock.
#
# Model artifacts are NOT included in this image. They are volume-mounted
# from the host by compose.yml at container startup. The /ready endpoint
# confirms all artifacts loaded before accepting traffic.

FROM python:3.13-slim

# Metadata
LABEL maintainer="howlingwolf77"
LABEL description="CloudDrift — Cloud Infrastructure Anomaly Detector"
LABEL version="1.0.0"

# Install system dependencies required by the Python packages:
#   curl   — used by the compose.yml health check (GET /health)
#   build-essential — required by some Python packages during uv sync
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv from its official distribution image.
# Pinning to a specific version ensures reproducible builds.
COPY --from=ghcr.io/astral-sh/uv:0.5.26 /uv /bin/uv

# Set the working directory. All subsequent COPY and CMD instructions
# are relative to this path inside the container.
WORKDIR /app

# Copy dependency files first — Docker caches this layer.
# If pyproject.toml and uv.lock haven't changed, Docker reuses the
# cached dependency installation layer on subsequent builds.
COPY pyproject.toml uv.lock ./

# Install all production dependencies using the locked versions.
# --frozen: error if uv.lock is out of sync with pyproject.toml
# --no-dev: skip dev dependencies (pytest, ruff, etc.)
# --no-install-project: don't install the project itself yet
# The pytorch-cpu index in pyproject.toml is used automatically.
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source code
COPY src/          ./src/
COPY api/          ./api/
COPY dashboard/    ./dashboard/
COPY .streamlit/   ./.streamlit/

# Expose both service ports. compose.yml decides which one to use per service.
EXPOSE 8000 8501

# No CMD here — overridden per service in compose.yml
