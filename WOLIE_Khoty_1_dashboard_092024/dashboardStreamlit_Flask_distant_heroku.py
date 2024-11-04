# Importation des bibliothèques nécessaires pour la manipulation de données et les calculs
import numpy as np
import pandas as pd
import random
import warnings
import re
import json
from faker import Faker  # Importation de Faker pour générer des prénoms et noms aléatoires

# Importation des bibliothèques pour la création de graphiques et la visualisation des données
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from PIL import Image

# Importation de la bibliothèque Streamlit pour créer une interface utilisateur interactive
import streamlit as st

# Importation de joblib pour charger et enregistrer des modèles de machine learning
import joblib

# Importation de la bibliothèque requests pour envoyer des requêtes HTTP, utilisée pour communiquer avec l'API
import requests

# Importation de SHAP pour l'explication des prédictions des modèles
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import shap
shap.initjs()

# Ignorer les avertissements de dépréciation spécifiques à Numba
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Configuration de la page Streamlit avec un titre spécifique pour l'onglet du navigateur
st.set_page_config(page_title='Scoring Nouveau Client')

# Fonction pour charger les données à partir d'un fichier CSV
@st.cache_data
def charger_donnees(url):
    df = pd.read_csv(url)
    return df

# Essayer de charger les données depuis un chemin local pour les tests
try:
    AnciennesDonnees = charger_donnees(
        r"C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 8\Application\Donnees_generees\Application_projet7_Streamlit.csv")
except:
    AnciennesDonnees = charger_donnees("Application//Donnees_generees//Application_projet7_Streamlit.csv")

# Création d'une liste contenant les noms de toutes les colonnes du dataframe chargé
ListeVariables = list(AnciennesDonnees.columns)

# Création d'une barre latérale dans Streamlit permettant de naviguer entre différentes sections (onglets)
option = st.sidebar.selectbox(
    "Sommaire :",
    ("Page d'accueil", "Informations Clients"))

# Ajout de contenu à gauche sur la page d'accueil
if option == "Page d'accueil":
    with st.sidebar:
        st.markdown("### À propos de l'application")
        st.markdown("""
        Cette application a été développée pour permettre une transparence totale dans l'analyse financière, 
        en aidant à comprendre les scores de crédit et les décisions financières de manière claire et accessible.
        """)

        st.markdown("### Conseils Financiers")
        st.markdown("""
        - **Épargnez régulièrement** : Mettez de côté une partie de vos revenus chaque mois.
        - **Minimisez vos dettes** : Évitez de contracter des dettes à haut intérêt.
        - **Investissez intelligemment** : Diversifiez vos investissements pour réduire les risques.
        """)

        st.markdown("### Support & Contact")
        st.markdown("""
        Si vous avez des questions ou avez besoin d'assistance, veuillez contacter notre équipe de support.
        - **Email** : support@finapp.com
        - **Téléphone** : +33 1 23 45 67 89
        - **Chat en ligne** : Disponible 24/7 sur notre site web.
        """)

# Fonction pour ouvrir et afficher une image à partir d'un fichier donné
def ouvrir_image(url, taille):
    image = Image.open(url)
    return st.image(image, width=taille)

# Fonction pour charger la liste des nouveaux clients à partir d'un fichier CSV
def liste_nouveaux_clients(fichier_csv):
    df_nouveaux_clients = pd.read_csv(fichier_csv)
    df_nouveaux_clients.reset_index(inplace=True)
    liste_clients = list(df_nouveaux_clients['SK_ID_CURR'])
    liste_clients.insert(0, ' ')
    return liste_clients, df_nouveaux_clients

# Fonction pour afficher un graphique de type jauge, indiquant le niveau de risque ou score du client
def afficher_jauge_client(score):
    couleur_arriere_plan = "#def"
    couleurs_quadrants = [couleur_arriere_plan, "#2bad4e", "#90EE90", "#f2a529", "#f25829"]
    textes_quadrants = ["", "<b>Bon</b>", "<b>Faible</b>", "<b>Très faible</b>", "<b>Ultra faible</b>"]
    nombre_quadrants = len(couleurs_quadrants) - 1

    valeur_actuelle = score
    valeur_minimale = 0
    valeur_maximale = 100
    longueur_aiguille = np.sqrt(2) / 4
    angle_aiguille = np.pi * (1 - (max(valeur_minimale, min(valeur_maximale, valeur_actuelle)) - valeur_minimale) / (valeur_maximale - valeur_minimale))

    fig = go.Figure(
        data=[go.Pie(values=[0.5] + (np.ones(nombre_quadrants) / 2 / nombre_quadrants).tolist(),
                rotation=90, hole=0.5, marker_colors=couleurs_quadrants, text=textes_quadrants, textinfo="text", hoverinfo="skip")],
        layout=go.Layout(showlegend=False, margin=dict(b=0, t=50, l=5, r=5), width=350, height=350, paper_bgcolor=couleur_arriere_plan,
            annotations=[go.layout.Annotation(text=f"<b>Niveau de remboursement :</b><br>{valeur_actuelle}%", x=0.5, xanchor="center", xref="paper", y=0.2, yanchor="bottom", yref="paper", showarrow=False)],
            shapes=[go.layout.Shape(type="circle", x0=0.48, x1=0.52, y0=0.48, y1=0.52, fillcolor="#333", line_color="#333"),
                    go.layout.Shape(type="line", x0=0.5, x1=0.5 + longueur_aiguille * np.cos(angle_aiguille), y0=0.5, y1=0.5 + longueur_aiguille * np.sin(angle_aiguille), line=dict(color="#333", width=4))])
    )
    return st.plotly_chart(fig)

# Fonction pour générer une explication locale (SHAP) pour un client donné
def expliquer_shap_local(modele_charge, anciennes_donnees):
    explainer = shap.TreeExplainer(modele_charge, anciennes_donnees)
    valeurs_shap = explainer(DataClient, check_additivity=False)
    return valeurs_shap[0]

# Fonction pour extraire les variables les plus importantes pour une prédiction donnée
def recuperer_variables_importantes(valeurs_shap, anciennes_donnees):
    colonnes = DataClient.columns
    variables_importantes = pd.DataFrame(zip(expliquer_shap_local(modele_charge, anciennes_donnees).values, colonnes))
    variables_importantes[0] = abs(variables_importantes[0]).round(2)
    variables_importantes = variables_importantes.sort_values(0, ascending=False)
    variables_principales = list(variables_importantes.iloc[:15][1])
    variables_principales.insert(0, ' ')
    return variables_principales

# Fonction pour tracer des graphiques comparant deux variables sélectionnées par l'utilisateur
def tracer_graphiques_importants(var1, var2, cible, anciennes_donnees, data_client, liste_resultats):
    # Création de légendes personnalisées pour le graphique de dispersion
    legendes_personnalisees = [Line2D([0], [0], marker='o', color='w', label='Classe 1', markerfacecolor='g'),
                               Line2D([0], [0], marker='o', color='w', label='Classe 0', markerfacecolor='r')]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax = fig.subplot_mosaic("""
                            AB
                            CC
                            """)

    # Vérification si les variables sont numériques
    if np.issubdtype(anciennes_donnees[var1].dtype, np.number) and np.issubdtype(anciennes_donnees[var2].dtype, np.number):
        # Extraction des données unidimensionnelles pour var1, var2 et cible
        var1_data = anciennes_donnees[var1]
        var2_data = anciennes_donnees[var2]
        cible_data = cible

        # Combinaison des données dans un DataFrame
        data_combined = pd.DataFrame({var1: var1_data, var2: var2_data, 'cible': cible_data})

        # Suppression des lignes contenant des valeurs manquantes (NaN) de manière coordonnée
        data_combined_clean = data_combined.dropna()

        # Ajustement de `liste_resultats` pour correspondre aux données nettoyées
        liste_resultats_clean = liste_resultats[:data_combined_clean.shape[0]]

        # Vérification que les dimensions correspondent avant de tracer les courbes
        if data_combined_clean.shape[0] > 0:
            # Tracé des courbes de densité pour la première variable sélectionnée
            sns.kdeplot(data=data_combined_clean, x=var1, hue='cible', multiple="stack", ax=ax['A'])
            ax["A"].axvline(data_client[var1].values[0], linewidth=2, color='r')

            # Tracé des courbes de densité pour la deuxième variable sélectionnée
            sns.kdeplot(data=data_combined_clean, x=var2, hue='cible', multiple="stack", ax=ax['B'])
            ax["B"].axvline(data_client[var2].values[0], linewidth=2, color='r')

            # Tracé d'un graphique de dispersion pour visualiser la relation entre les deux variables
            sns.scatterplot(x=data_combined_clean[var1], y=data_combined_clean[var2], hue=liste_resultats_clean, palette="blend:red,green", s=100, ax=ax['C'])

            # Encapsuler les valeurs scalaires pour Seaborn
            sns.scatterplot(x=[data_client[var1].values[0]], y=[data_client[var2].values[0]], s=400, color='blue', marker='*', ax=ax['C'])
            ax['C'].legend(handles=legendes_personnalisees, loc='upper right')
        else:
            st.write("Pas de données suffisantes après suppression des valeurs manquantes.")
    else:
        st.write(f"Les variables {var1} et/ou {var2} ne sont pas numériques et ne peuvent pas être tracées avec kdeplot.")
        # Utilisation d'un autre type de graphique pour les variables catégorielles, comme un countplot.
        sns.countplot(x=var1, hue=cible, data=anciennes_donnees, ax=ax['A'])
        sns.scatterplot(x=var2, y=var1, hue=liste_resultats, data=anciennes_donnees, ax=ax['C'])

    # Affichage du graphique dans Streamlit
    st.pyplot(fig)

# Configuration de la première page "Page d'accueil" de l'application
if option == "Page d'accueil":
    st.markdown("<h1 style='text-align: center; color: red;'>Bonjour, Bienvenue sur votre portail de transparence financière :</h1>", unsafe_allow_html=True)
    st.markdown('')
    st.markdown('')

    col1, col2, col3, col4 = st.columns(4)
    with col2:
        try:
            ouvrir_image(r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 8\Application\Images\Logo_Entreprise_pres_a_depenser.png', 300)
        except:
            ouvrir_image('Application/Images/Logo_Entreprise_pres_a_depenser.png', 300)

    st.markdown('')
    st.markdown('')

    st.markdown("<h2 style='text-align: center; color: grey;'>Faciliter l’analyse et la compréhension des résultats pour tous les utilisateurs.</h2>", unsafe_allow_html=True)
    st.markdown('')

    st.markdown("<h3 style='font-weight:bold;'>Objectif de l’application:</h3>", unsafe_allow_html=True)
    st.markdown('')
    st.markdown("""
    <p style='text-align: center; font-size:22px;'>
    Aider les clients et conseillers à naviguer facilement dans les données et à prendre des décisions éclairées.
    </p>
    """, unsafe_allow_html=True)

# Configuration de la deuxième page "Informations Clients" de l'application
if option == "Informations Clients":
    st.markdown("<h2 style='text-align: center; color: green;'>Informations Clients :</h2>", unsafe_allow_html=True)

    try:
        liste_clients = liste_nouveaux_clients(
            r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 8\Application\Donnees_generees\ListeNouveauxClients.csv')[0]
        df_nouveaux_clients = liste_nouveaux_clients(
            r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 8\Application\Donnees_generees\ListeNouveauxClients.csv')[1]
    except:
        liste_clients = liste_nouveaux_clients('Application/Donnees_generees/ListeNouveauxClients.csv')[0]
        df_nouveaux_clients = liste_nouveaux_clients('Application/Donnees_generees/ListeNouveauxClients.csv')[1]

    def selectionner_client(liste_clients):
        client = st.selectbox('Veuillez choisir le numéro de votre client : ', liste_clients)
        return client

    client = selectionner_client(liste_clients)

    if client == ' ':
        st.markdown("### Veuillez sélectionner un client pour afficher ses informations détaillées.")
        st.markdown("L'application vous permet de visualiser les scores de crédit, l'historique des clients et d'autres informations financières importantes.")
    else:
        index_client = list(df_nouveaux_clients[df_nouveaux_clients['SK_ID_CURR'] == client]['index'].values)
        for i in index_client:
            index_client = i
        index_autres = list(df_nouveaux_clients['index'])
        index_autres.remove(index_client + 1)
        index_autres.remove(0)

        try:
            DataClient = pd.read_csv(
                r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 8\Application\Donnees_generees\NouveauxClientsReduits.csv',
                skiprows=index_autres, nrows=1)
        except:
            DataClient = pd.read_csv('Application/Donnees_generees/NouveauxClientsReduits.csv', skiprows=index_autres, nrows=1)

        DataClient = DataClient.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        numero_client = DataClient['SK_ID_CURR'].values
        for i in numero_client:
            numero_client = i
        st.write('**N° Client :** ', numero_client)
        DataClient = DataClient[ListeVariables]
        DataClient = DataClient.drop(columns='TARGET')

        fake = Faker('fr_FR')
        prenom_selectionne = fake.first_name()
        nom_selectionne = fake.last_name()

        st.write('**Prénom :** ', prenom_selectionne)
        st.write('**Nom :** ', nom_selectionne)

        st.markdown("<h2 style='text-align: center; color: green;'>  Résultats du prêt :</h2>", unsafe_allow_html=True)

        # Envoi des données du client à une API Flask pour obtenir une prédiction du score
        # URL de l'API Heroku
        url = 'https://predictions-app-projet7-f3b0b3d90518.herokuapp.com/api/'

        # Conversion des données du client en format JSON pour les envoyer à l'API
        ListeVariables.remove('TARGET')
        data = DataClient.values.tolist()
        j_data = json.dumps(data)
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        r = requests.post(url, data=j_data, headers=headers)

        # Affichage de la réponse de l'API pour débogage
        st.write("Réponse brute de l'API : ", r.text)

        # Extraction du score de la réponse de l'API et arrondi pour l'affichage
        try:
            # La réponse semble être une liste de listes, donc nous devons traiter cela correctement
            prediction = json.loads(r.text)
            score_0 = prediction[0][0]  # Classe 0: risque faible (accepté)
            score_1 = prediction[0][1]  # Classe 1: risque élevé (refusé)
            score = int(round(score_0 * 100, 2))  # Nous utilisons maintenant le score de la classe 0
        except Exception as e:
            st.error(f"Erreur : La réponse de l'API n'est pas formatée comme prévu. Détails : {e}")
            score = 50  # Valeur par défaut si l'API ne fonctionne pas correctement

        # Si l'API n'est pas utilisée, charger le modèle directement depuis un fichier
        try:
            modele_charge = joblib.load(open(
                r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 8\Notebook\mlflow_runs\290555362347125930\1e46374402274ffe9572106d93203ef9\artifacts\model\model.pkl',
                'rb'))
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle : {e}")

        # Affichage du score du client sous forme de jauge
        col1, col2 = st.columns(2)
        with col2:
            afficher_jauge_client(score)

        # Affichage d'une image conditionnée au score (rouge pour un score faible, vert pour un bon score)
        with col1:
            if score >= 59:  # Classe 0 signifie que le crédit est accepté
                try:
                    ouvrir_image(r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 8\Application\Images\pngtree_vert.png', 300)
                except:
                    ouvrir_image('Application/Images/pngtree_vert.png', 300)
            if score < 59:  # Classe 1 signifie que le crédit est refusé
                try:
                    ouvrir_image(r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 8\Application\Images\pngtree_rouge.png', 300)
                except:
                    ouvrir_image('Application/Images/pngtree_rouge.png', 300)

        st.markdown("<h2 style='text-align: center; color: green;'>Importance des caractéristiques globales :</h2>", unsafe_allow_html=True)
        try:
            ouvrir_image(r'C:\Users\Infogene\Documents\Khoty_Privé\DOSSIER FORMATION DATA SCIENTIST\PROJET 8\Application\Images\SHAP_Image_Globale.png', 600)
        except:
            ouvrir_image('Application/Images/SHAP_Image_Globale.png', 700)

        st.markdown("<h2 style='text-align: center; color: green;'>Importance des caractéristiques locales :</h2>", unsafe_allow_html=True)
        valeurs_shap = expliquer_shap_local(modele_charge, AnciennesDonnees)

        shap.waterfall_plot(valeurs_shap, max_display=10)
        fig = plt.gcf()
        st.pyplot(fig)

        cible = AnciennesDonnees['TARGET']
        AnciennesDonnees = AnciennesDonnees.drop(columns='TARGET')
        resultats = modele_charge.predict_proba(AnciennesDonnees)
        resultats = pd.DataFrame(resultats)
        resultats = 100 * resultats
        resultats = resultats.astype(int)
        liste_resultats = list(resultats[1])

        st.markdown("<h2 style='text-align: center; color: green;'>Analyse des variables importantes :</h2>", unsafe_allow_html=True)

        variable1 = st.selectbox('Veuillez choisir la variable N°1 : ', recuperer_variables_importantes(valeurs_shap, AnciennesDonnees))
        variable2 = st.selectbox('Veuillez choisir la variable N°2 : ', recuperer_variables_importantes(valeurs_shap, AnciennesDonnees))

        if variable1 == ' ' or variable2 == ' ':
            st.write('')

        else:
            tracer_graphiques_importants(variable1, variable2, cible, AnciennesDonnees, DataClient, liste_resultats)