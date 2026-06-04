FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source
# (layer caches — changes to src/ don't invalidate pip install)
COPY pyproject.toml .
COPY README.md .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[dev]"

# NLTK data
RUN python -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt', 'stopwords', 'wordnet']]"

# Copy project source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY app/ ./app/
COPY app.py .

# Data is mounted at runtime — do not bake it into the image
VOLUME ["/app/data", "/app/outputs"]

EXPOSE 7860

# Default: launch the Gradio demo
# Override for submission generation:
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs \
#              india-runs python scripts/generate_submission.py --validate
CMD ["python", "app/demo.py", "--server-port", "7860"]
