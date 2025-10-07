# ---- Stage 0: base ----
FROM python:3.10-slim

# metadata
LABEL maintainer="D. Sajan <youremail@example.com>"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    TF_CPP_MIN_LOG_LEVEL=2

# set workdir
WORKDIR /app

# install system deps needed by transformers/torch and for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# copy dependencies file first (for layer caching)
COPY requirements.txt /app/requirements.txt

# install python deps
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# copy app source
COPY . /app

# create a non-root user for better security (optional)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Create a model cache directory and set HF cache env (optional but recommended)
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p $HF_HOME && chmod -R 777 $HF_HOME

# Expose streamlit default port
EXPOSE 8501

# default command to run the Streamlit app
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
