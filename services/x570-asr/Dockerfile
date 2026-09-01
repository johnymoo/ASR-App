FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.11.0 torchaudio==2.11.0
RUN python -m pip install --no-cache-dir \
      fastapi==0.135.1 \
      uvicorn==0.41.0 \
      python-multipart==0.0.22 \
      funasr==1.3.1 \
      modelscope==1.34.0

WORKDIR /app
COPY worker.py /app/worker.py
ENTRYPOINT ["python", "-m", "uvicorn", "worker:app"]
CMD ["--host", "0.0.0.0", "--port", "18021", "--workers", "1"]
