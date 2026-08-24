FROM python:3.11-slim

WORKDIR /app

# OpenCV needs libgl/libglib at runtime even in headless use.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gaugevision ./gaugevision
COPY models ./models

ENV GAUGEVISION_MODEL_PATH=/app/models/padim_screw.npz
EXPOSE 8000

CMD ["uvicorn", "gaugevision.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
