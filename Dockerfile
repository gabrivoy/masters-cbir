# Single image for both the API and the Streamlit frontend; the compose file
# picks the entrypoint per service. Built on uv for reproducible installs.
FROM python:3.13-slim

# System deps: OpenCV/torch need libGL and glib at runtime for image decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (cached layer) using only the lockfiles.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Then the project source.
COPY cbir ./cbir
COPY README.md ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Default command runs the API; the compose app service overrides it.
EXPOSE 8100 8501
CMD ["cbir", "api", "--host", "0.0.0.0", "--port", "8100", "--device", "cpu"]
