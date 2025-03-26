## Aperçu de l'entreprise 1

![Aperçu du site web](images/DS_projet7.PNG)

## Aperçu de l'entreprise 2

![Aperçu du site web](images/DS_projet6.PNG)

## 📌 Contexte professionnel

En tant que **Data Scientist** chez **Prêt à Dépenser**, j’ai participé à deux missions stratégiques :
1. Le développement d’un **dashboard interactif de scoring crédit** destiné aux chargés de relation client pour expliquer les décisions de crédit de manière transparente.
2. Une **veille technique** sur les dernières avancées en **NLP** ou **Computer Vision**, incluant un **proof of concept** sur des données réelles.

---

## 🎯 Objectifs

### 🎯 Mission 1 : Dashboard interactif de scoring crédit

- Visualiser les scores de crédit et leur interprétabilité locale
- Comparer un client à l’ensemble de la base ou à des groupes similaires
- Intégrer l’API de prédiction pour un affichage en temps réel
- Concevoir une interface **accessible** (normes **WCAG**) pour les chargés de relation client

### 🎯 Mission 2 : Veille technique et mise en œuvre d’une méthode récente

- Réaliser un état de l’art sur une méthode récente (< 5 ans) en **NLP** ou **Computer Vision**
- Comparer cette méthode à une approche plus classique
- Tester cette méthode dans un **POC** et présenter les résultats à travers une **note méthodologique**

---

## 🧩 Démarche

### 🔹 Mission 1 : Développement du Dashboard

- **Framework** : Streamlit
- **Fonctionnalités intégrées** :
  - Visualisation du score de crédit via jauge colorée
  - Explicabilité locale (SHAP) des principales features impactantes
  - Comparaison avec la base clients via histogrammes, boxplots et graphiques bi-variés
  - Saisie de nouveaux profils pour scoring dynamique
  - Accessibilité (contraste, lisibilité, clavier, etc.)
- **Connexion à l’API** : FastAPI déployée sur le cloud pour un scoring en temps réel

### 🔹 Mission 2 : Veille technique & POC

- **Sources utilisées** : Arxiv, PapersWithCode, ConnectedPapers
- **Technique choisie** (exemple) : NLP - **DistilBERT** (modèle allégé de BERT, performant et rapide)
- **Comparatif** : BERT vs DeBerTa
- **POC réalisé** :
  - Préparation des données (corpus produit ou dataset benchmark)
  - Entraînement du modèle pré-entraîné vs baseline classique
  - Comparaison des métriques (accuracy, F1-score)
- **Note méthodologique** rédigée avec concepts clés, code, et recommandations

---

## 📂 Livrables

- ✅ Application Streamlit (dashboard interactif en ligne)
- ✅ API FastAPI connectée pour les prédictions
- ✅ Notebook de scoring et explication locale (SHAP)
- ✅ POC NLP / Vision : comparaison DistilBERT vs méthode classique
- ✅ Note méthodologique (.pdf) et présentation (.pptx)

---

## 🛠️ Outils et technologies

- **Python**, **pandas**, **scikit-learn**
- **FastAPI** pour l’API de scoring
- **Streamlit** pour l’application interactive
- **SHAP** pour l’interprétation des modèles
- **Hugging Face Transformers** (DeBerTa, BERT) pour NLP avancé
- **Evidently** (intégrable au dashboard si dérive à suivre)

---

## ✅ Résultats

- Interface intuitive & accessible pour accompagner les conseillers dans la décision
- Visualisation claire du **score de crédit et de son interprétation**
- Fonctionnalité de saisie dynamique pour simuler un scoring en temps réel
- État de l’art comparatif sur une méthode récente **NLP**, avec des **résultats supérieurs aux méthodes classiques**
- Recommandations pour **industrialiser** l’approche dans une chaîne MLOps

---

## 🔍 Aperçu

> Ce projet démontre mes compétences en **modélisation explicable**, **développement de dashboard accessible**, **mise en production de modèle**, et **veille technologique approfondie en NLP et vision par ordinateur**.

---

*Projet réalisé dans un cadre professionnel simulé, avec des responsabilités similaires à celles rencontrées dans une fintech ou une startup en data science appliquée à la relation client.*
