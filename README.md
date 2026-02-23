# Trustpilot_mlops

## 🧭 Contexte & Objectifs

### Objectif du projet
Ce projet s’inscrit dans un cas d’usage réaliste : **Trustpilot** souhaite fournir à ses entreprises clientes un module d’analyse automatique des avis, capable de :

- classifier le **sentiment** (positif / négatif)
- extraire automatiquement les **grandes thématiques** présentes dans les retours clients
- synthétiser les insights dans un **tableau de bord métier**

---

## 📊 Données utilisées

Les avis Trustpilot n’étant pas disponibles publiquement à grande échelle, ce projet repose sur un **dataset proxy robuste** :

### Amazon Reviews Polarity (Kaggle)
- **3,6 M** avis pour l’entraînement  
- **0,4 M** avis pour le test  
- **2 classes équilibrées** (positif / négatif)  
- Données textuelles riches : livres, films, musique, jeux vidéo…  

👉 Ce dataset est particulièrement adapté pour simuler un **usage Trustpilot haute volumétrie**.

---

## 🧪 Travail réalisé dans ce dépôt

