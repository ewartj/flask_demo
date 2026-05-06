# ── Stage 1: compile TypeScript ───────────────────────────────────────────────
FROM node:20-slim AS frontend

WORKDIR /build
COPY corp-ca.crt /usr/local/share/ca-certificates/corp-ca.crt
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt

COPY app/static/ts/ ./ts/
RUN npx --yes esbuild ts/main.ts --outfile=main.js --bundle --minify


# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.12-slim

# ── Corporate CA cert ─────────────────────────────────────────────────────────
COPY corp-ca.crt /usr/local/share/ca-certificates/corp-ca.crt
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN pip install --no-cache-dir uv

WORKDIR /app

# ── 1. Install Python dependencies (cached layer — reruns only if pyproject.toml changes)
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# ── 2. Copy application source
COPY . .

# ── 3. Drop in the compiled JS from the frontend stage
COPY --from=frontend /build/main.js ./app/static/js/main.js

# ── 4. Pre-download HuggingFace demo models so first request is instant
RUN uv run python scripts/download_models.py

EXPOSE 5000

# 1 worker + threads: avoids duplicating ~1 GB of model weights per worker.
CMD ["uv", "run", "gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "120", \
     "analysis_app:app"]
