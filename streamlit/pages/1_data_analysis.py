import sys
import joblib
import streamlit as st
import numpy as np

from utils import (
    plot_disc_variables,
    plot_disc_vs_target,
    plot_cont_vs_cont,
    plot_cont_vs_target,
    plot_clusters_scatterplots,
)


st.set_page_config(page_title="Analyse des données", layout="wide")

st.title("Analyse des données")

# Récupération des données
if "df" not in st.session_state:
    st.error(
        "Impossible d'accéder aux données, veuillez retournez sur la page principale pour les charger."
    )
    sys.exit()

df = st.session_state["df"]

# Intro
st.write(
    """
    - L'idée de présenter les principaux résultats d'analyse s'inscrit dans une logique de transparence vis-à-vis des différentes parties prenantes (métier, tech, client, audit, etc.)
    - Les uns et les autres pourront donc avoir de meilleures pistes pour comprendre les prédictions, critiquer les méthodes utilisées, apporter des améliorations, etc.
    - Le notebook `notebooks/eda.ipynb` contient davantage d'informations et de précisions (je suis un apprenant, aidez-moi à m'améliorer si vous passez par là 😉).
"""
)

# Séparation
st.divider()


# ---------------- Vision d'ensemble ----------------
st.subheader("Vision d'ensemble")
st.dataframe(df.head())

# Commentaire général
st.write(
    """
    - On constate une répartition inégale de la variable cible : 16% des 6011 clients résilient leur contrat.
    - Les clients sont divisés en 2 grands groupes selon qu'ils aient ou non souscrit à un service Internet.
    - On obtient de meilleures performances avec 2 modèles spécialisés plutôt qu'avec un modèle unique.
    - Les lignes ayant des valeurs manquantes représentent un faible pourcentage des données (2.64%).
"""
)

# Suppression des valeurs manquantes
df.dropna(inplace=True)

# Catégorisation des variables
disc = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]
cont = ["Monthly Charges", "Total Charges", "Tenure Months", "CLTV"]
target = "Churn Value"

# Séparation Avec / Sans Internet
df_internet = df[df["Internet Service"] != "No"].copy()
df_no_internet = df[df["Internet Service"] == "No"].copy()

# Autres groupes de variables
disc_no_internet = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Multiple Lines",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]
internet_services = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]

# Séparation
st.divider()


# ---------------- Variables discrètes ---------------
st.subheader("1. Variables discrètes")

# Création de 2 colonnes pour chaque segment
internet, no_internet = st.columns(2)

with internet:
    st.markdown("#### Avec Internet")
    segment = "Avec Internet"

    # Distribution
    with st.expander("📊 Distribution"):
        plot_disc_variables(df=df_internet, cols=disc, segment=segment)

    # Taux de churn
    with st.expander("🚨 Taux de churn"):
        plot_disc_vs_target(df=df_internet, cols=disc_no_internet, segment=segment)

    # Taux de churn (services Internet)
    with st.expander("🛜 Taux de churn (services Internet)"):
        plot_disc_vs_target(
            df=df_internet,
            cols=internet_services,
            n_rows=1,
            n_cols=6,
            segment=segment,
            height=300,
            width=1200,
            fig_title=f"Taux de churn selon les services Internet, Segment = {segment}",
            subplot_titles=["<br>".join(col.split()) for col in internet_services],
        )


with no_internet:
    st.markdown("#### Sans Internet")
    segment = "Sans Internet"

    # Distribution
    with st.expander("📊 Distribution"):
        plot_disc_variables(
            df=df_no_internet,
            cols=disc_no_internet,
            n_rows=2,
            n_cols=4,
            segment=segment,
        )

    # Taux de churn
    with st.expander("🚨 Taux de churn"):
        plot_disc_vs_target(df=df_no_internet, cols=disc_no_internet, segment=segment)

    # Taux de churn (services Internet)
    with st.expander("🛜 Taux de churn (services Internet) `Impertinent`"):
        plot_disc_vs_target(
            df=df_no_internet,
            cols=internet_services,
            n_rows=1,
            n_cols=6,
            segment=segment,
            height=300,
            width=1200,
            fig_title=f"Taux de churn selon les services Internet, Segment = {segment}",
            subplot_titles=["<br>".join(col.split()) for col in internet_services],
            br_xtickslabels=True,
        )

with st.expander("💬 Commentaire"):
    st.write(
        """
        - Le taux de churn est globalement plus élevé chez les clients ayant souscrit à des services Internet (20% vs. 4%).
        - Pour l'ensemble des clients, on remarque que le taux de résiliation est plus élevé lorsque le client :
            - n'a pas de partenaire
            - n'a pas de personnes à charge
            - renouvelle mensuellement son contrat
            - ne possède qu'une seule ligne de téléphone
        - Pour les cliens `Avec Internet`, le taux de churn est plus élevé lorsque le client :
            - a plus de 65 ans
            - opte pour une facturation sans papier
            - règle ses factures par chèque électronique
        - Pour les clients `Sans Internet`, le churn est élevé lorsque le client :
            - règle ses factures via un chèque envoyé par courrier
        - Ceci nous aide à identifier ces variables comme potentiellement importantes pour notre modèle.
        - Concernant les différents services Internet, les clients qui s'y abonnent sont plus fidèles.
        - Cette tendance est moins marquée pour les services de streaming TV et films (peut-être en raison de leur gratuité).
        - On observe également que plus un client souscrit à des options, plus il y'a des chances de le garder.
        - L'introduction dans notre modèle final d'une variable `Service Count` pour dénombrer les services Internet par client s'est avérée pertinente.
    """
    )

# Séparation
st.divider()


# ---------------- Variables continues ---------------
st.subheader("2. Variables continues")

# Création de 2 colonnes pour chaque segment
internet, no_internet = st.columns(2)


with internet:
    st.markdown("#### Avec Internet")
    segment = "Avec Internet"

    # Distribution
    with st.expander("📈 Distribution"):
        plot_cont_vs_target(df=df_internet, cols=cont, target_value=0, segment=segment)
        plot_cont_vs_target(df=df_internet, cols=cont, target_value=1, segment=segment)

    # Relations
    with st.expander("🔗 Relations"):
        plot_cont_vs_cont(df=df_internet, cols=cont, segment=segment)


with no_internet:
    st.markdown("#### Sans Internet")
    segment = "Sans Internet"

    # Distribution
    with st.expander("📈 Distribution"):
        plot_cont_vs_target(
            df=df_no_internet, cols=cont, target_value=0, segment=segment
        )
        plot_cont_vs_target(
            df=df_no_internet, cols=cont, target_value=1, segment=segment
        )

    # Relations
    with st.expander("🔗 Relations"):
        plot_cont_vs_cont(df=df_no_internet, cols=cont, segment=segment)

with st.expander("💬 Commentaire"):
    st.write(
        """
        - Un nombre important de clients qui partent ont de faibles charges totales.
        - Ceux qui résilient le font majoritairement dans les premiers mois, et au bout de 70 mois d'ancienneté, il devient très difficile de perdre un client.
        - Les clients ayant une valeur vie > 6 000 ont une faible probabilité de résiliation.
        - Selon leur ancienneté, il est possible de regrouper les clients en 2 catégories (nuage de points entre la valeur client et l'ancienneté).
        - La variable `Is New Client`, créé pour distinguer les clients ayant moins de 2 ans d'ancienneté a eu un impact positif sur le segment `Avec Internet`.
        - On observe des groupes de charges mensuelles au travers des agglomérations verticales de points sur la 1ère colonne des nuages de points.
        - En utilisant un algorithme de clustering, on a respectivement identifié 20 et 2 classes de `Monthly Charges` pour les clients avec et sans Internet.
        - Au sein de chaque cluster, il y'a une très forte corrélation entre l'ancienneté et le montant des charges totales.
        - L'intégration des clusters de `Monthly Charges` améliore les performances du modèle sur le segment `Avec Internet` uniquement.
        - L'exclusion de `Total Charges`, justifiée par sa corrélation avec `Tenure Months` due aux clusters, s'est révélée utile chez les clients `Avec Internet`.
        - Rien de particulier ne se dégage de l'analyse des coordonnées géographiques, mais les variables `Lat` et `Long` n'ont pas été écartées.
        - Le modèle final des clients `Sans Internet` ne prend en entrée que 7 variables sur 16.
    """
    )

with st.expander("Corrélations entre `Total Charges` et `Tenure Months`"):
    from scipy.stats import pearsonr

    c1, c2 = "Total Charges", "Tenure Months"

    kmeans_internet = joblib.load("artifacts/internet/kmeans_internet.pkl")
    clusters_mapping_internet = joblib.load(
        "artifacts/internet/clusters_mapping_internet.pkl"
    )

    df_internet["Monthly Charges Group"] = kmeans_internet.predict(
        df_internet[["Monthly Charges"]]
    )
    df_internet["Monthly Charges Group"] = df_internet["Monthly Charges Group"].map(
        clusters_mapping_internet
    )

    corr_all = pearsonr(df_internet[c1], df_internet[c2]).statistic

    corr_clusters = []
    for i in range(20):
        tmp_df = df_internet.query("`Monthly Charges Group` == @i")
        corr_clusters.append(pearsonr(tmp_df[c1], tmp_df[c2]).statistic)

    st.write(
        f"""
        - Corrélation totale : {corr_all:.4f}
        - Moyenne des corrélations intra-groupes : {np.mean(corr_clusters):.4f}
    """
    )

    plot_clusters_scatterplots(
        df_internet, corr_all=corr_all, corr_clusters=corr_clusters
    )
