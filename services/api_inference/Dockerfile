FROM python:3.12-slim

WORKDIR /app

# libs système (optionnel mais utile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier le code de l'API
COPY services/api_inference/ /app/services/api_inference/

# Copier les modèles
COPY models/ /app/models/

# Installer torch CPU
RUN pip install --no-cache-dir --upgrade pip \
&& pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Installer les dépendances de l'API (sans torch dedans)
RUN pip install --no-cache-dir -r /app/services/api_inference/requirements_inference.txt

EXPOSE 8000

CMD ["uvicorn", "services.api_inference.main:app", "--host", "0.0.0.0", "--port", "8000"]
