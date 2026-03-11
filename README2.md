# Trustpilot MLOps – Topic Modeling Pipeline

Ce projet implémente une pipeline **MLOps complète** pour l’analyse d’avis Trustpilot et l’extraction automatique de topics.

Le système repose sur :

* Sentence-BERT pour générer des embeddings textuels
* KMeans (scikit-learn) pour le clustering
* MLflow pour le suivi des expériences et la gestion des modèles
* Airflow pour l’orchestration du pipeline
* FastAPI pour exposer le modèle via une API d’inférence
* Docker / Docker Compose pour l’isolation et la reproductibilité

L’objectif est de construire une architecture MLOps modulaire permettant :

* l’entraînement automatisé
* la sélection du meilleur modèle
* la gestion des versions
* la promotion automatique du modèle en production
* l’inférence via API

---

# Architecture du projet

## Pipeline de Machine Learning
reviews text
      ↓
Sentence-BERT embeddings
      ↓
KMeans clustering
      ↓
silhouette_score evaluation

## Pipeline MLOps

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
promote_model_if_better
     ↓
Model Registry (alias Production)
     ↓
FastAPI inference


#Arborescence du projet**

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
│           └── promote_model_if_better.py
│
├── docker-compose.yml
├── clean_project.sh
└── README.md


---

# Installation

## 1. Cloner le dépôt

```bash
git clone https://github.com/ericdatascien6/Trustpilot_mlops.git

cd Trustpilot_mlops


# Démarrer la stack MLOps

Le projet utilise Docker Compose pour démarrer les services suivants :

* MLflow
* Airflow
* API
* FastAPI
* Trainer

**Démarrer tous les services :**

./start.sh

ou

docker-compose up -d

# Services disponibles

| Service     | Port |
| ----------- | ---- |
| Airflow UI  | 8081 |
| MLflow UI   | 5000 |
| FastAPI API | 8000 |


# Interface MLflow

MLflow permet de :

* suivre les runs d’entraînement
* comparer les métriques
* gérer les versions du modèle 
* promouvoir un modèle en production

**Accéder à MLflow :**

http://IP_VM:5000 

Chaque run enregistre notamment :

* k (nombre de clusters)
* silhouette_score
* training_time

Les modèles sont enregistrés dans MLflow Model Registry


# Orchestration avec Airflow

Airflow orchestre le pipeline d’entraînement.

**DAG principal :**

trustpilot_training_pipeline

Pipeline exécuté :

* train_job
        ↓
* evaluate_registry
        ↓ 
* promote_model_if_better


**Accéder à l’interface Airflow :**

http://IP_VM:8081

Depuis l’interface Airflow il est possible de  :

* déclencher un entraînement
* consulter les logs
* suivre l’état des tâches

# API d’inférence

L’API FastAPI expose le modèle via HTTP. Le modèle chargé est :

* MLflow Model Registry
* alias : Production

Cela permet d’utiliser automatiquement la meilleure version du modèle.


# Accéder à l’API

**Health check :**

curl http://127.0.0.1:8000/health

**Tester l’inférence :**

./inference.sh ou directement :

curl -X POST 
http://localhost:8000/predict \
-H "Content-Type: application/json" \
-H "x-api-key: secret123" \
-d '{"text":"This product is amazing"


# Accès à l’API depuis une machine locale (Tunnel SSH)

Si l’API tourne sur une VM distante :

ssh -i "data_enginering_machine.pem" -L 9000:127.0.0.1:8000 ubuntu@IP_VM 

Puis ouvrir :

http://localhost:9000/docs


# Réinitialiser MLflow

Pour repartir d’un environnement propre :

./reset_mlflow.sh

Ce script :

* arrête les containers
* supprime les runs MLflow
* supprime les artifacts
* redémarre la stack


# Arrêter la stack

 ./stop.sh

ou

docker-compose down


# Objectif du projet

Ce projet illustre une architecture MLOps simplifiée mais réaliste intégrant : 

* entraînement automatisé
* gestion de modèles avec MLflow
* orchestration avec Airflow
* promotion automatique du modèl
* service d’inférence via API

# Évolutions possibles

Les évolutions futures du pipeline incluront : 

* recherche d’hyperparamètres avancée
* monitoring des performances
* détection de drift
* dashboards d’observabilité (Prometheus / Grafana)
* CI/CD pour le déploiement des modèles
