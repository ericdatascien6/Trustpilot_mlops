FROM python:3.12-slim

WORKDIR /app

# deps système utiles (souvent nécessaires à torch / sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements_trainer.txt .
RUN pip install --no-cache-dir -r requirements_trainer.txt

COPY train.py .

CMD ["python", "train.py"]
