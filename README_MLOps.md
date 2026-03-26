# Trustpilot MLOps – Topic Modeling Pipeline

Ce projet implémente une pipeline **MLOps complète** pour l’analyse d’avis Trustpilot et l’extraction automatique de topics.

Le système repose sur :

* Sentence-BERT pour générer des embeddings textuels
* KMeans (scikit-learn) pour le clustering
* MLflow pour le suivi des expériences et la gestion des modèles
* Airflow pour l’orchestration du pipeline
* FastAPI pour exposer le modèle via une API d’inférence
* Streamlit pour l’interface utilisateur
* Prometheus pour le monitoring des métriques
* Grafana pour la visualisation des métriques
* Docker / Docker Compose pour l’isolation et la reproductibilité

L’objectif est de construire une architecture MLOps modulaire permettant :

* l’ingestion continue de nouvelles reviews
* l’entraînement automatisé
* la sélection du meilleur modèle
* la gestion des versions
* la promotion automatique du modèle en production
* l’inférence via API
* l’observabilité du système (monitoring infra + ML)


---

## Architecture du projet

### Architecture globale du système

L’architecture du projet repose sur plusieurs microservices orchestrés via Docker Compose.

Le système sépare clairement :

* le pipeline d’entraînement
* le service d’inférence
* l’interface utilisateur
* la couche de monitoring

Diagramme d’architecture :

                        ┌─────────────────────────┐
                        │      Streamlit UI       │
                        │  Exploration & Testing  │
                        └─────────────┬───────────┘
                                      │
                                      ▼
                           ┌───────────────────┐
                           │    FastAPI API    │
                           │   /predict        │
                           │   /metrics        │
                           └─────────┬─────────┘
                                     │
                                     ▼
                          ┌────────────────────┐
                          │   MLflow Registry  │
                          │  Production Model  │
                          └────────────────────┘


   ┌───────────────────────────────────────────────────────┐
   │                    TRAINING PIPELINE                  │
   │                                                       │
   │  simulate_review_stream → update_training_dataset    │
   │                    ↓                                 │
   │             Airflow DAG orchestration                │
   │                    ↓                                 │
   │           train_k3 ... train_k8 (Trainer)            │
   │                    ↓                                 │
   │           evaluate_registry → promote_model          │
   │                    ↓                                 │
   │              MLflow Model Registry                   │
   └───────────────────────────────────────────────────────┘


                ┌─────────────────────────────┐
                │        MONITORING           │
                │                             │
                │ FastAPI /metrics endpoint   │
                │             ↓               │
                │        Prometheus           │
                │             ↓               │
                │          Grafana            │
                │     Dashboards & Alerts     │
                └─────────────────────────────┘

### Pipeline de Machine Learning

reviews text  
↓  
Sentence-BERT embeddings  
↓  
KMeans clustering  
↓  
silhouette_score evaluation

---

## Pipeline MLOps

Le pipeline est orchestré par **Airflow**.

simulate_review_stream  
↓  
update_training_dataset  
↓  
check_threshold (ShortCircuitOperator)  
↓  
train_k3
train_k4
train_k5
train_k6
train_k7
train_k8  
↓  
evaluate_registry  
↓  
promote_model_if_better  
↓  
Model Registry (alias Production)  
↓  
FastAPI inference
↓  
Streamlit user interface

---
### Entraînement avec plusieurs valeurs de K

Le pipeline entraîne plusieurs modèles KMeans avec différentes valeurs de K.

Les valeurs testées sont :

* K = 3
* K = 4
* K = 5
* K = 6
* K = 7
* K = 8

Chaque modèle est entraîné indépendamment et enregistré dans MLflow.

Pour chaque modèle sont loggés :

* valeur de K
* silhouette_score
* temps d'entraînement

Le script `evaluate_registry.py` compare ensuite les performances et identifie le meilleur modèle.

Si un modèle obtient un score supérieur au modèle actuellement en production, il est promu automatiquement dans **MLflow Model Registry** avec l’alias `Production`.


## Ingestion des données (simulation)

Le projet simule l’arrivée continue de nouvelles reviews afin de reproduire un **flux de données réel**.

Les reviews proviennent d’un dataset Amazon Reviews (Kaggle) contenant **plus de 1.5M lignes**.

Deux scripts gèrent l’ingestion :

## simulate_review_stream.py

Simule l’arrivée progressive de reviews.

Fonctionnement :

* lit le dataset source
* ajoute un batch de reviews dans une zone SAS
* utilise un offset persistant pour se souvenir de la dernière review utilisée

Configuration :

* `BATCH_SIZE = 50`

Offset stocké dans :

* `data/metadata/stream_offset.txt`

---

### update_training_dataset.py

Vérifie le nombre de reviews dans la zone SAS.

Si le seuil est atteint (**threshold = 200**), alors :

* les reviews sont ajoutées au dataset d'entraînement
* les reviews sont archivées
* la zone SAS est vidée

Le script écrit également un fichier de statut :

* `data/sas/ingestion_status.json`

Ce fichier contient :

* le nombre de reviews présentes dans la zone SAS
* le seuil d’ingestion
* un booléen `threshold_reached`

Exemple :

{
"threshold_reached": false,
"sas_review_count": 50,
"threshold": 200,
"moved_to_train": 0
}


Ce fichier est utilisé par Airflow pour décider si l'entraînement doit être lancé.

Ce script constitue le **point de vérité métier** de l’ingestion.

---

## Contrôle du déclenchement de l'entraînement

Afin d’éviter de relancer inutilement un entraînement, le pipeline utilise un mécanisme de **gating basé sur un seuil d’ingestion**.

Airflow utilise un **ShortCircuitOperator** qui lit le fichier :

* `data/sas/ingestion_status.json`

Comportement :

Si :

* `threshold_reached = false`

alors les tâches suivantes sont **skippées** :

* train_k*
* evaluate_registry
* promote_model_if_better

Si :

* `threshold_reached = true`

alors le pipeline d’entraînement est exécuté.

Ce mécanisme permet de :

* éviter des entraînements inutiles
* réduire les coûts de calcul
* simuler un comportement MLOps réaliste basé sur un volume minimal de données.

---

## Structure du répertoire data

data
│
├── dataset_source
│ └── amazon_reviews.csv
│
├── raw
│ └── train.csv
│
├── sas
│ ├── trustpilot_new_reviews.csv
│ └── ingestion_status.json
│
├── archive
│ └── integrated_reviews.csv
│
└── metadata
└── stream_offset.txt

Le dataset Amazon n'est **pas versionné dans le repository** en raison de sa taille.

---

# Monitoring et Observabilité

Le projet inclut une couche complète de **monitoring MLOps**.

**Architecture :**

Prometheus récupère les métriques en interrogeant périodiquement l’endpoint `/metrics` de l’API :

Prometheus
↓ (scrape toutes les 5 secondes)
GET /metrics
↓
FastAPI
↓
exposition des métriques
↓
stockage dans Prometheus
↓
visualisation dans Grafana

Ce mécanisme permet de transformer les métriques en **séries temporelles**, utilisées pour construire les dashboards de monitoring.


Les métriques collectées incluent :

**Monitoring API**

* nombre total de requêtes
* latence des requêtes
* nombre d’erreurs
* utilisation des endpoints

**Monitoring ML**

* nombre total de prédictions
* nombre de prédictions par cluster
* rythme de prédictions

Exemples de métriques exposées :

* trustpilot_predictions_total
* trustpilot_cluster_predictions_total{cluster_id="0"}

Ces métriques permettent de détecter :

* une dérive dans les topics prédits
* un déséquilibre dans les clusters
* une utilisation anormale de l’API

---

### Dashboards Grafana

Grafana permet de visualiser les métriques collectées par Prometheus à travers des dashboards interactifs.

Deux dashboards principaux sont fournis dans le projet :

* **Infrastructure Dashboard**  
  surveille les métriques système et API

* **Business Dashboard**  
  surveille l’activité du modèle ML et la distribution des prédictions

Les dashboards incluent notamment :

* nombre de requêtes API
* latence des prédictions
* erreurs serveur
* nombre total de prédictions
* distribution des clusters prédits

Ces dashboards permettent de détecter rapidement :

* une surcharge de l’API
* un ralentissement du modèle
* une dérive dans la distribution des topics.


---

# Génération de trafic pour les tests

Un script generate_traffic.sh permet de générer du trafic vers l’API afin de tester le monitoring.

Ce script envoie des requêtes de prédiction en continu :

./generate_traffic.sh


Cela permet de visualiser immédiatement l’activité dans :

* Prometheus
* Grafana

---

# Structure du projet


Trustpilot_mlops
│
├── airflow
│ └── dags
│ └── trustpilot_training_pipeline.py
│
├── data
│ ├── dataset_source
│ ├── raw
│ ├── sas
│ ├── archive
│ └── metadata
│
├── models
│
├── monitoring
│ ├── prometheus
│ │ └── prometheus.yml
│ │
│ └── grafana
│     ├── dashboards
│     │   ├── infrastructure_dashboard.json
│     │   └── business_dashboard.json
│     │
│     └── provisioning
│         └── datasources
│             └── prometheus.yml
│
├── services
│ ├── api
│ │ ├── main.py
│ │ ├── inference.py
│ │ ├── schemas.py
│ │ └── tests
│ │
│ ├── trainer
│ │ ├── train_job.py
│ │ └── scripts
│ │
│ └── streamlit
│ ├── Dockerfile
│ ├── requirements.txt
│ └── app
│ ├── streamlit_app.py
│ └── images
│
├── docker-compose.yml
├── generate_traffic.sh
├── reset_mlflow.sh
├── reset_pipeline.sh
├── start.sh
├── stop.sh
└── README.md


---

# Architecture des services

L’architecture repose sur plusieurs microservices conteneurisés via Docker Compose.

| Service | Rôle |
|------|------|
| Airflow | orchestration du pipeline ML |
| MLflow | tracking des expériences et registry des modèles |
| Trainer | entraînement des modèles |
| FastAPI | service d'inférence |
| Streamlit | interface utilisateur |
| Prometheus | collecte des métriques |
| Grafana | visualisation du monitoring |
| API Tests | validation automatique de l’API |

Les volumes Docker permettent de partager les données entre services.

Volumes principaux :

* `./data:/data`
* `./models:/models`
* `./mlruns:/mlruns`

Ces volumes permettent :

* à Airflow de déclencher les scripts
* au Trainer d'entraîner les modèles
* à MLflow d’enregistrer les runs
* à l’API d'accéder au modèle en production
  


---

# Services disponibles

| Service | Port |
|--------|------|
| Airflow UI | 8081 |
| MLflow UI | 5000 |
| FastAPI API | 8000 |
| FastAPI Docs | 8000/docs |
| Streamlit UI | 8501 |
| Prometheus | 9090 |
| Grafana | 3000 |


---

## Installation

### Cloner le dépôt
 
 git clone https://github.com/ericdatascien6/Trustpilot_mlops.git

cd Trustpilot_mlops


---

## Dataset

Le repository inclut une version **réduite du dataset Amazon Reviews (~50k lignes)** afin de permettre de lancer rapidement la pipeline. La pipeline est cependant conçue pour fonctionner sur des datasets beaucoup plus volumineux.

Le dataset complet (1.5M lignes) est disponible sur Kaggle :
https://www.kaggle.com/datasets/bittlingmayer/amazonreviews


---

## Démarrer la stack MLOps

Le projet utilise Docker Compose pour démarrer les services suivants :

* MLflow
* Airflow
* API FastAPI
* Trainer
* Streamlit (interface utilisateur)
* Prometheus (collecte des métriques)
* Grafana (visualisation du monitoring)

Démarrer tous les services :

./start.sh


ou


docker-compose up -d


---

## Interface MLflow

MLflow permet de :

* suivre les runs d’entraînement
* comparer les métriques
* gérer les versions du modèle
* promouvoir un modèle en production

Accéder à MLflow : http://IP_VM:5000

Chaque run enregistre notamment :

* k (nombre de clusters)
* silhouette_score
* training_time

Les modèles sont enregistrés dans **MLflow Model Registry**.

### Stratégie de promotion des modèles

Le pipeline implémente une logique simple de **model selection automatique**.

1. Les modèles candidats sont entraînés avec plusieurs valeurs de K.
2. Chaque modèle est enregistré dans MLflow.
3. Le script `evaluate_registry.py` compare le `silhouette_score` des runs.
4. Si un modèle est meilleur que le modèle actuel en production :

   * il est promu dans **MLflow Model Registry**
   * l’alias `Production` est mis à jour

L’API FastAPI charge toujours automatiquement le modèle associé à l’alias `Production`.

---

## Orchestration avec Airflow

Airflow orchestre le pipeline d’entraînement.

DAG principal :

* `trustpilot_training_pipeline`

Pipeline exécuté :

simulate_review_stream  
↓  
update_training_dataset  
↓  
check_threshold  
↓  
train_k*  
↓  
evaluate_registry  
↓  
promote_model_if_better

Accéder à l’interface Airflow : http://IP_VM:8081

Depuis l’interface Airflow il est possible de :

* déclencher un entraînement
* consulter les logs
* suivre l’état des tâches

---

## API d’inférence

L’API FastAPI expose le modèle via HTTP.

Le modèle chargé est :

* MLflow Model Registry
* alias : **Production**

L’API charge automatiquement le modèle via MLflow :

MLflow Model Registry → alias `Production`

Cela permet :

* de déployer automatiquement les nouveaux modèles
* de découpler l’inférence du pipeline d'entraînement
* d’utiliser automatiquement la meilleure version du modèle.

### Endpoint de monitoring `/metrics`

L’API expose également un endpoint spécial : GET /metrics

Cet endpoint expose les métriques internes de l’application dans un format compatible **Prometheus**.

Ces métriques sont collectées automatiquement par Prometheus afin de surveiller le comportement du système.

L’endpoint `/metrics` permet de monitorer :

* le trafic de l’API
* la latence des requêtes
* les erreurs
* l’activité du modèle ML

Prometheus interroge automatiquement cet endpoint toutes les **5 secondes** afin de construire des séries temporelles qui seront ensuite visualisées dans Grafana.

---

## Interface utilisateur (Streamlit)

Une interface utilisateur Streamlit permet d’explorer le projet et de tester le modèle en production.

Le rôle de Streamlit est de fournir :

* une interface interactive pour tester le modèle
* des exemples de prédictions sur des reviews
* des visualisations exploratoires du dataset
* un accès simplifié à l’API d’inférence

Architecture :

User  
↓  
Streamlit UI  
↓  
FastAPI inference API  
↓  
MLflow Model Registry  
↓  
Production model

Contrairement à l’ancienne version du projet Data Science, Streamlit ne charge **aucun modèle localement**.

Toutes les prédictions passent par l’API FastAPI afin de respecter une architecture MLOps réaliste.

Cela permet :

* de séparer l’interface utilisateur et le modèle
* de faciliter le déploiement
* de simuler un système d’inférence en production

Accéder à Streamlit :   http://IP_VM:8501

Fonctionnalités principales :

* exploration du dataset
* visualisation des analyses exploratoires
* prédiction d’un topic pour une review saisie par l’utilisateur
* affichage des exemples de reviews appartenant au cluster prédit
* affichage de la réponse brute de l’API pour illustrer l’architecture MLOps


---

## Accéder à l’API

Health check :


curl http://127.0.0.1:8000/health


Tester l’API d’inférence :


curl -X POST http://localhost:8000/predict

-H "Content-Type: application/json"
-H "x-api-key: secret123"
-d '{"text":"This product is amazing"}'


---

# Résumé des acces aux interfaces utilisateurs (UI)

Airflow : http://IP_VM:8081

MLflow : http://IP_VM:5000

Streamlit : http://IP_VM:8501

Prometheus : http://IP_VM:9090

Grafana : http://IP_VM:3000


---

## Réinitialiser MLflow

Pour repartir d’un environnement propre :

./reset_mlflow.sh


Ce script :

* arrête les containers
* supprime les runs MLflow
* supprime les artifacts
* redémarre la stack

---

## Réinitialiser la pipeline

Pour repartir d’un état propre :

./reset_pipeline.sh

Ce script :

* réinitialise l’offset du stream
* vide la zone SAS
* recrée un dataset d’entraînement initial

---

## Arrêter la stack


./stop.sh

ou

docker-compose down


---

# Objectif du projet

Ce projet illustre une architecture **MLOps complète et réaliste** intégrant :

* ingestion de données simulée
* entraînement automatisé
* orchestration avec Airflow
* gestion des modèles avec MLflow
* promotion automatique du modèle
* service d’inférence via API
* interface utilisateur Streamlit
* monitoring système avec Prometheus
* dashboards d’observabilité avec Grafana
* métriques ML personnalisées
* monitoring des prédictions du modèle

---

# Évolutions possibles

* hyperparameter tuning automatisé
* monitoring avancé des modèles
* détection de drift
* alerting Prometheus
* CI/CD pour le déploiement
* monitoring de la qualité des données
* versioning des datasets