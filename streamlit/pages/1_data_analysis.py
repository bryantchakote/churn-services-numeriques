import streamlit as st
from utils import (
    plot_disc_variables,
    plot_disc_vs_target,
    plot_cont_vs_cont,
    plot_cont_vs_target,
)


st.set_page_config(page_title="Analyse des données", layout="wide")

st.title("Analyse des données")

# Récupération des données
if "df" in st.session_state:
    df = st.session_state["df"]
else:
    st.error("Impossible d'accéder aux données, veuillez retournez sur la page principale pour les charger.")


# Intro
st.write("- L'idée de présenter les principaux résultats d'analyse s'inscrit dans une logique de transparence vis-à-vis des différentes parties prenantes (métier, tech, client, audit, etc.)")
st.write("- Les uns et les autres pourront donc avoir de meilleures pistes pour comprendre les prédictions, critiquer les méthodes utilisées, apporter des améliorations, etc.")

# Séparation
st.divider()





# ---------------- Vision d'ensemble ----------------
st.subheader("Vision d'ensemble")
st.dataframe(df.head())

# Commentaire général
st.write("""
- On constate une répartition inégale de la variable cible : 16% des clients résilient leur contrat.
- Les clients sont divisés en 2 grands groupes selon qu'ils aient ou non souscrit à un service Internet.
- Les lignes ayant des valeurs manquantes représentent un faible pourcentage des données (2.64%).
""")

# Extraction de la latitude et de la longitude
df["Lat"] = df["Lat Long"].str.split(", ").str[0].astype(float)
df["Long"] = df["Lat Long"].str.split(", ").str[1].astype(float)

# Suppression des valeurs manquantes
df.dropna(inplace=True)

# Catégorisation des variables
coords = ["Lat", "Long"]
disc = ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", "Multiple Lines", "Internet Service", "Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies", "Contract", "Paperless Billing", "Payment Method"]
cont = ["Monthly Charges", "Total Charges", "Tenure Months", "CLTV"]
target = "Churn Value"

# Séparation Avec / Sans Internet
df_internet = df[df["Internet Service"] != "No"].copy()
df_no_internet = df[df["Internet Service"] == "No"].copy()

# Autres groupes de variables
disc_no_internet = ["Gender", "Senior Citizen", "Partner", "Dependents", "Multiple Lines", "Contract", "Paperless Billing", "Payment Method"]
internet_services = ["Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"]
    
# Séparation
st.divider()

# Création de 2 colonnes pour chaque segment
internet, no_internet = st.columns(2)





# ---------------- Internet ----------------
with internet:
    st.subheader("Internet")
    segment = "Avec Internet"

    # Analyse des variables discrètes
    st.markdown("#### 1. Variables discrètes")

    # Distribution
    with st.expander("📊 Distribution"):
        plot_disc_variables(df=df_internet, cols=disc, segment=segment)

    # Taux de churn
    with st.expander("🚨 Taux de churn"):
        plot_disc_vs_target(df=df_internet, cols=disc_no_internet, segment=segment)
    
    # Taux de churn (services Internet)
    with st.expander("🛜 Taux de churn (services Internet)"):        
        plot_disc_vs_target(df=df_internet, cols=internet_services, n_rows=1, n_cols=6, segment=segment, height=300, width=1200, fig_title=f"Taux de churn selon les services Internet, Segment = {segment}", subplot_titles=["<br>".join(col.split()) for col in internet_services])
        
    # Séparation
    st.divider()
    
    # Analyse des variables continues
    st.markdown("#### 2. Variables continues")

    # Distribution
    with st.expander("📈 Distribution"):
        plot_cont_vs_target(df=df_internet, cols=cont, target_value=0, segment=segment)
        plot_cont_vs_target(df=df_internet, cols=cont, target_value=1, segment=segment)
    
    # Relations
    with st.expander("🔗 Relations"):
        plot_cont_vs_cont(df=df_internet, cols=cont, segment=segment)





# ---------------- No Internet ----------------
with no_internet:
    st.subheader("No Internet")
    
    segment = "Sans Internet"

    # Analyse des variables discrètes
    st.markdown("#### 1. Variables discrètes")

    # Distribution
    with st.expander("📊 Distribution"):
        plot_disc_variables(df=df_no_internet, cols=disc_no_internet, n_rows=2, n_cols=4, segment=segment)

    # Taux de churn
    with st.expander("🚨 Taux de churn"):
        plot_disc_vs_target(df=df_no_internet, cols=disc_no_internet, segment=segment)
    
    # Taux de churn (services Internet)
    with st.expander("🛜 Taux de churn (services Internet) : Impertinent, juste pour la symétrie"):
        plot_disc_vs_target(df=df_no_internet, cols=internet_services, n_rows=1, n_cols=6, segment=segment, height=300, width=1200, fig_title=f"Taux de churn selon les services Internet, Segment = {segment}", subplot_titles=["<br>".join(col.split()) for col in internet_services], br_xtickslabels=True)

    # Séparation
    st.divider()
    
    # Analyse des variables continues
    st.markdown("#### 2. Variables continues")

    # Distribution
    with st.expander("📈 Distribution"):
        plot_cont_vs_target(df=df_no_internet, cols=cont, target_value=0, segment=segment)
        plot_cont_vs_target(df=df_no_internet, cols=cont, target_value=1, segment=segment)
    
    # Relations
    with st.expander("🔗 Relations"):
        plot_cont_vs_cont(df=df_no_internet, cols=cont, segment=segment)
