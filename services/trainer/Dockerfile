FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_trainer.txt /app/requirements_trainer.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r /app/requirements_trainer.txt

COPY train_job.py /app/train_job.py
COPY scripts/ /app/scripts/


#CMD ["python", "/app/train_job.py"]
