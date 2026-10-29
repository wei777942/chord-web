FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHORD_WEB_HOST=0.0.0.0 \
    CHORD_WEB_PORT=10000

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        poppler-utils \
        fonts-liberation \
        fonts-wqy-zenhei \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Ensure jobs dir exists and is writable
RUN mkdir -p /app/jobs && chmod 777 /app/jobs

EXPOSE 10000
CMD ["gunicorn", "-b", "0.0.0.0:10000", "-w", "1", "--timeout", "180", "--threads", "4", "app:app"]
