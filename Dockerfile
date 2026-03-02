FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ ./serving/
COPY training/ ./training/
COPY simulator/ ./simulator/

RUN mkdir -p data/raw data/processed data/reference

RUN python simulator/generate_transactions.py && \
    python training/feature_engineering.py && \
    python training/train_model.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1

CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]