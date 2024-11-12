# OPC_DATA_SCIENTIST_PROJET8
Réalisez un dashboard et assurez une veille technique


# Dashboard de Credit Scoring et Veille Technique - Prêt à Dépenser

## Aperçu de l'entreprise

![Aperçu du site web](images/DS_projet7.PNG)

## Aperçu de l'entreprise

![Aperçu du site web](images/DS_projet6.PNG)


## Contexte

Je suis Data Scientist chez **"Prêt à dépenser"**, une entreprise qui propose des crédits à la consommation pour des clients ayant peu ou pas d'historique de prêt. Après avoir développé un modèle de **scoring crédit**, l’entreprise souhaite désormais rendre ce score accessible aux chargés de relation client via un **dashboard interactif** pour expliquer les décisions de crédit aux clients de façon transparente.

Le projet est structuré en deux missions :
1. **Mission 1** : Concevoir un dashboard de scoring crédit, interactif et accessible.
2. **Mission 2** : Réaliser une veille technique sur une nouvelle méthode NLP ou de vision par ordinateur (computer vision) pour explorer les dernières avancées dans le domaine.

---

## Première Mission : Concevoir un Dashboard de Credit Scoring

### Objectifs
- Développer un **dashboard interactif** permettant de visualiser le score de crédit, les informations clients, et de comparer ces informations à l'ensemble des clients ou à des groupes similaires.
- Permettre aux chargés de relation client d'expliquer les décisions d'octroi de crédit de manière compréhensible.
- Prendre en compte les critères d'accessibilité, et déployer le dashboard sur une plateforme cloud.

### Étape 1 : Élaboration du Dashboard
- **Objectif** : Créer un prototype fonctionnel de dashboard intégrant toutes les fonctionnalités spécifiées, et le déployer sur une plateforme accessible.
- **Détails** :
  - **Visualisation du score de crédit** : Inclure une jauge colorée affichant la probabilité et le score de crédit, ainsi que la **feature importance locale** qui explique les principales contributions au score.
  - **Affichage des caractéristiques clients** : Permettre la comparaison des informations du client avec l’ensemble des clients ou un groupe similaire via des graphiques interactifs.
  - **Analyse bi-variée** : Ajouter des graphiques permettant de visualiser la relation entre deux variables choisies par l’utilisateur.
  - **Accessibilité** : Suivre les critères WCAG pour s’assurer que le dashboard soit compréhensible par tous les utilisateurs, y compris les personnes en situation de handicap.
  - **API de prédiction** : Utiliser l’API déjà créée pour calculer et afficher les résultats du modèle en temps réel.
  - **Optionnel** : Ajouter la fonctionnalité permettant de modifier les informations d’un client et rafraîchir le score, ou de saisir un nouveau client pour obtenir une prédiction.
- **Livrable** : Un **dashboard déployé sur le cloud** (via **Streamlit**, **Dash** ou **Bokeh**), permettant de visualiser les scores de crédit et les informations clients de manière interactive.

### Points de Vigilance
- S'assurer que l'interface est accessible et facilement compréhensible pour les utilisateurs non techniques.
- Optimiser la lisibilité des graphiques et des informations pour une présentation claire et efficace.

### Ressources
- **Streamlit Documentation** : Guide pour la création d'une application Streamlit.
- **WCAG Guidelines** : Critères d'accessibilité pour les interfaces.

---

## Deuxième Mission : Veille Technique sur une Nouvelle Méthode NLP ou Computer Vision

### Objectifs
- Réaliser un état de l’art sur une **nouvelle technique de modélisation NLP** ou **computer vision** datant de moins de 5 ans, et la comparer à une approche plus classique.
- Tester cette technique sur des données existantes ou similaires, et en réaliser une **preuve de concept (POC)**.
- Présenter la technique et ses résultats dans une **note méthodologique** et lors d'une présentation orale.

### Étape 1 : Préparation de la Veille Technique
- **Objectif** : Sélectionner une méthode récente de NLP ou de computer vision et réaliser un Proof of Concept (POC) pour tester son efficacité.
- **Détails** :
  - **Choisir une méthode** : Utiliser des ressources telles qu’**Arxiv**, **Paperswithcode**, ou **ConnectedPapers** pour choisir une technique récente (comme les **Transformers**, **BERT**, ou des architectures avancées de vision par ordinateur).
  - **Réaliser un POC** : Tester cette nouvelle méthode sur un dataset NLP ou image déjà utilisé dans un projet précédent ou sur un nouveau dataset similaire. Comparer les performances avec une approche classique.
  - **Écrire une note méthodologique** : Présenter la nouvelle technique, expliquer ses concepts clés, et comparer les résultats obtenus avec les méthodes précédemment implémentées.
- **Livrable** : Une **note méthodologique** détaillant la nouvelle technique et ses résultats, ainsi qu’un **notebook de POC** démontrant son efficacité.

### Étape 2 : Vérification et Préparation de la Soutenance
- **Objectif** : Préparer une présentation claire et détaillée de la technique étudiée et de ses résultats pour un public métier et technique.
- **Détails** :
  - Structurer la présentation autour de l'analyse de la nouvelle technique, des résultats obtenus, et des différences avec les méthodes traditionnelles.
  - Réviser tous les livrables pour assurer la clarté, la précision et l’alignement avec les attentes métier et techniques.
- **Livrable** : Une présentation de **30 slides maximum** résumant l'état de l’art, la mise en œuvre de la technique et les résultats obtenus lors du POC.

### Points de Vigilance
- Choisir une méthode récente et pertinente pour le domaine (moins de 5 ans).
- Veiller à bien expliquer les concepts sous-jacents de la nouvelle technique à un public non technique.
- S’assurer de la validité des sources bibliographiques (articles avec citations et références solides).

### Ressources
- **Arxiv** : Répertoire de publications scientifiques.
- **ConnectedPapers** : Visualisation de papiers de recherche pour explorer les relations entre différents travaux.
- **Paperswithcode** : Liste de papiers de recherche accompagnés de code pour tester les résultats.

---

## Détails Techniques

- **Fichiers** :
  - **Dataset Prêt à Dépenser** : Contient les données des clients et des institutions financières utilisées pour le modèle de scoring.
  - **Notebook de Modélisation** : Inclut l’entraînement du modèle, le scoring, et les API pour la prédiction en temps réel.
  - **Dashboard** : Application Streamlit/Bokeh/Dash pour la visualisation des scores de crédit et des informations clients.
  - **Note Méthodologique** : Explication des techniques récentes (NLP/Computer Vision) et comparaison avec des approches plus classiques.

- **Outils Utilisés** :
  - **Python** (pandas, scikit-learn) pour le traitement des données et la modélisation.
  - **Streamlit**, **Dash**, ou **Bokeh** pour la création du dashboard interactif.
  - **API FastAPI** pour l'intégration des prédictions en temps réel.
  - **Transformers/BERT** ou architectures avancées de **vision par ordinateur** pour la veille technique.

---

## Résumé

Le projet comporte deux missions distinctes. La première consiste à développer un **dashboard interactif** permettant de visualiser et d'expliquer les décisions de crédit à l'aide d'un modèle de scoring. La seconde mission consiste en une **veille technique** pour explorer et tester une technique récente de **NLP** ou **computer vision**, avec une preuve de concept comparant cette nouvelle méthode à une approche plus classique.
