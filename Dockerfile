# syntax=docker/dockerfile:1.6

FROM pytorchlightning/pytorch_lightning:2.3.3-py3.10-torch2.0-cuda11.8.0

# Basic runtime hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=0

WORKDIR /workspace

# Create an unprivileged user (the base image is often root by default)
# RUN useradd -m -u 1000 appuser

# Copy only requirements first to maximize layer caching
COPY requirements.txt /workspace/requirements.txt

# Install Python deps (BuildKit cache mount speeds up rebuilds)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /workspace/requirements.txt

# Copy the rest of the project
# COPY . /workspace

# Make workspace owned by the non-root user
# RUN chown -R appuser:appuser /workspace

# USER appuser

# Default command is intentionally minimal; override in docker run / compose
CMD ["python", "-c", "import torch; print('torch', torch.__version__)"]
