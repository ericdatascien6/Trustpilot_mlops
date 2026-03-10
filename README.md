# Trustpilot MLOps – Topic Modeling Pipeline

Ce projet implémente une pipeline MLOps complète pour l’analyse d’avis Trustpilot et l’extraction automatique de topics.

Le système repose sur :

* Sentence-BERT pour générer des embeddings textuels
* KMeans (scikit-learn) pour le clustering
* MLflow pour le suivi des expériences et la gestion des modèles
* Airflow pour l’orchestration du pipeline
* FastAPI pour exposer le modèle via une API d’inférence
* Docker / Docker Compose pour l’isolation et la reproductibilité

L’objectif est de construire une architecture MLOps modulaire, permettant :

* l’entraînement automatisé
* la sélection du meilleur modèle
* la gestion des versions
* l’inférence via API

## Architecture du projet

**Pipeline de machine learning :**

reviews text
      ↓
Sentence-BERT embeddings
      ↓
KMeans clustering
      ↓
silhouette_score evaluation

**Pipeline MLOps :**

Airflow DAG
     ↓
train_model
     ↓
MLflow tracking
     ↓
evaluate_registry
     ↓
Model Registry (alias Production)
     ↓
FastAPI inference

**Arborescence du projet**

Trustpilot_mlops
│
├── airflow
│   ├── dags
│   │   └── trustpilot_training_pipeline.py
│   ├── airflow.db
│   └── logs
│
├── data
│   └── raw
│       └── train.csv
│
├── models
│   ├── cluster_labels.pkl
│   └── kmeans_topics.pkl
│
├── services
│   ├── api
│   │   ├── main.py
│   │   ├── inference.py
│   │   └── Dockerfile
│   │
│   └── trainer
│       ├── train_job.py
│       ├── Dockerfile
│       └── scripts
│           └── evaluate_registry.py
│
├── docker-compose.yml
├── clean_project.sh
└── README.md

1. Cloner le dépôt
git clone https://github.com/ericdatascien6/Trustpilot_mlops.git

cd Trustpilot_mlops

2. Démarrer la stack MLOps

Le projet utilise Docker Compose pour démarrer :

* MLflow
* Airflow
* API FastAPI
* Trainer container

Lancer la stack :

docker-compose up -d

Services disponibles :

Service	Port
Airflow UI	8081
MLflow UI	5000
API FastAPI	8000

3. Interface MLflow

MLflow permet de :

* suivre les runs d’entraînement
* comparer les métriques
* gérer les versions du modèle

Accéder à MLflow :

http://IP_VM:5000

Chaque run log :

* k
* silhouette_score
* training_time

Les modèles sont enregistrés dans :

MLflow Model Registry

4. Orchestration avec Airflow

Airflow orchestre le pipeline d’entraînement.

DAG principal :

trustpilot_training_pipeline

Pipeline :

train_model
      ↓
evaluate_registry

Accéder à l’interface Airflow :

http://IP_VM:8081

Depuis l’UI Airflow il est possible de :

* déclencher un entraînement
* consulter les logs
* suivre l’état des tâches

5. API d’inférence

L’API FastAPI expose le modèle via HTTP.

Le modèle chargé est :

MLflow Model Registry

alias : Production

Ce qui permet d’utiliser automatiquement la meilleure version du modèle.

* Lancer l’API manuellement
* cd services/api
* python -m venv .venv
* source .venv/bin/activate
* pip install -r requirements_inference.txt
* uvicorn main:app --host 127.0.0.1 --port 8000

## Accéder à l’API

http://127.0.0.1:8000

Interface Swagger :

http://127.0.0.1:8000/docs

Tester l’API depuis une machine locale (Tunnel SSH)

Si l’API tourne sur une VM distante :

ssh -i "data_enginering_machine.pem" -L 9000:127.0.0.1:8000 ubuntu@IP_VM

Puis ouvrir :

http://localhost:9000/docs

## Nettoyer le projet

Pour repartir d’un environnement propre :

./clean_project.sh

Ce script :

* arrête les containers
* supprime les logs Airflow
* supprime les runs MLflow
* nettoie les caches Python

## Objectif du projet

Ce projet illustre une architecture MLOps simplifiée mais réaliste intégrant :

* entraînement automatisé
* gestion de modèles
* orchestration
* service d’inférence

L’évolution future du pipeline inclura :

* recherche d’hyperparamètres (plusieurs valeurs de k)
* promotion automatique du meilleur modèle
* monitoring et détection de drift.