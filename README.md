# Trustpilot MLOps – API de Topic Modeling

Ce projet implémente un microservice de **Topic Modeling** basé sur :

- Sentence-BERT pour la génération d’embeddings
- KMeans (scikit-learn) pour le clustering
- FastAPI pour l’exposition du modèle en API

L’objectif est de fournir un service d’inférence léger, reproductible et compatible CPU dans le cadre d’un projet MLOps.

---

# 1. Cloner le dépôt
git clone https://github.com/ericdatascien6/Trustpilot_mlops.git
cd Trustpilot_mlops


# 2. Tester l'API d’inférence
cd services/api_inference
./start_inference.sh
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000

L’API tourne alors sur la VM à l’adresse : http://127.0.0.1:8000
(laisser le serveur uvicorn exécuté sur cette console)


## Tester l’API depuis une machine locale (Tunnel SSH)

Si l’API tourne sur une VM distante, créer un tunnel SSH :
```bash
ssh -i "data_enginering_machine.pem" -L 9000:127.0.0.1:8000 ubuntu@IP_VM

Puis ouvrir l'interface UI de FastAPI dans le navigateur :   http://localhost:9000/docs

viter le port 8080 (souvent déjà utilisé par Docker ou autres services locaux).




# Structure simplifiée du projet

Trustpilot_mlops/
│ 
├── models/ 
│   ├── kmeans_topics.pkl 
│   ├── cluster_labels.pkl 
│ 
├── services/ 
│   ├── api_inference/ 
|       ├── main.py 
|       ├── inference.py
│       ├── schemas.py
│       ├── requirements_inference.txt
|       ├── Dockerfile
│
└── README.md
