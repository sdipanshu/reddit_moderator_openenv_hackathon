FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/reddit_mod_env
WORKDIR /app/reddit_mod_env

# Install Python dependencies (openenv-core is required per hackathon spec)
RUN pip install --no-cache-dir \
    openenv-core>=0.2.2 \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.29.0" \
    pydantic>=2.0.0 \
    httpx>=0.27.0

WORKDIR /app/reddit_mod_env

# PYTHONPATH=/app so that 'reddit_mod_env' is importable as a package.
# app.py adds its parent (/app) to sys.path at runtime for HF compatibility.
ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
