#!/bin/bash

set -e

echo "Initialisation de l'environnement d'inférence..."

# 1 Création du venv si absent
if [ ! -d ".venv" ]; then
    echo "Création du virtual environment..."
    python3 -m venv .venv
fi

# 2 Activation
echo "Activation du virtual environment..."
source .venv/bin/activate

# 3 Mise à jour pip
echo "Mise à jour de pip..."
pip install --upgrade pip

# 4 Installation Torch CPU uniquement (si absent)
if ! python -c "import torch" &> /dev/null; then
    echo "Installation de PyTorch (CPU uniquement)..."
    pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
else
    echo " PyTorch déjà installé."
fi

# 5 Installation des dépendances API
echo "Installation des dépendances API..."
pip install -r requirements_inference.txt

echo ""
echo " Environnement prêt."
echo "Pour lancer l'API :"
echo "   source .venv/bin/activate   (si absent)"
echo "   uvicorn main:app --host 127.0.0.1 --port 8000"
