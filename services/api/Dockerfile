FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements_inference.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir -r requirements_inference.txt

COPY . .

# ---------------------------
# Production image
# ---------------------------
FROM base AS prod
#CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------
# Test image
# ---------------------------
FROM base AS test
RUN pip install --no-cache-dir pytest==8.2.2
CMD ["pytest"]
