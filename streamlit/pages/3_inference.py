import streamlit as st
import pandas as pd
import joblib
from utils import (
    prepare_data,
    inference_prediction,
    INTERNET_SERVICES,
    COLS_SPACE,
)

st.set_page_config(page_title="Inférence", layout="wide")

st.title("Inférence")

# Intro
st.write(
    """
    - Les preprocesseurs et modèles de benchmark sauvegardés dans le dossier `artifacts` peuvent être utilisés ici pour les prédictions.
"""
)

# Séparation
st.divider()

# Configuration utilisée
use_separated_models = st.toggle("Modèles séparés", True)
multiple_predictions = st.toggle("Prédictions multiples", True)

# Artefacts (Avec Internet)
if "preprocessor_internet" not in st.session_state:
    st.session_state["preprocessor_internet"] = joblib.load(
        "artifacts/internet/preprocessor_internet.pkl"
    )

if "model_internet" not in st.session_state:
    st.session_state["model_internet"] = joblib.load(
        "artifacts/internet/model_internet.pkl"
    )

# Artefacts (Sans Internet)
if "preprocessor_no_internet" not in st.session_state:
    st.session_state["preprocessor_no_internet"] = joblib.load(
        "artifacts/no_internet/preprocessor_no_internet.pkl"
    )

if "model_no_internet" not in st.session_state:
    st.session_state["model_no_internet"] = joblib.load(
        "artifacts/no_internet/model_no_internet.pkl"
    )

# Artefacts (All)
if "preprocessor_all" not in st.session_state:
    st.session_state["preprocessor_all"] = joblib.load(
        "artifacts/all/preprocessor_all.pkl"
    )

if "model_all" not in st.session_state:
    st.session_state["model_all"] = joblib.load("artifacts/all/model_all.pkl")

# Séparation
st.divider()

# Prédictions sur données CSV
if multiple_predictions:
    # Chargement du fichier
    uploaded_file = st.file_uploader(
        "Veuillez uploader un fichier CSV pour effectuer un ensemble de prédictions.",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Préparation des données
            df_internet, df_no_internet = prepare_data(
                df, extract_coords=True, identify_new_clients=True, split_internet=True
            )

            df = pd.concat([df_internet, df_no_internet]).sort_index()

        except Exception as e:
            st.error(
                f"Erreur lors du chargement ou de la préparation des données : {e}"
            )

        else:
            # Modèles séparés
            if use_separated_models:
                # Avec Internet
                df_internet = inference_prediction(
                    df_internet,
                    preprocessor=st.session_state["preprocessor_internet"],
                    model=st.session_state["model_internet"],
                )

                # Sans Internet
                df_no_internet = inference_prediction(
                    df_no_internet,
                    preprocessor=st.session_state["preprocessor_no_internet"],
                    model=st.session_state["model_no_internet"],
                )

                # Concat
                df = pd.concat([df_internet, df_no_internet]).sort_index()

            # Modèle unique
            else:
                df = inference_prediction(
                    df,
                    preprocessor=st.session_state["preprocessor_all"],
                    model=st.session_state["model_all"],
                )

            # Résultats
            st.write("Prédictions :")
            st.dataframe(df)

# Prédiction unique
else:
    # Variables discrètes
    disc = {}
    for i in range(4):
        disc_cols = st.columns(4)
        for j, col in enumerate(disc_cols):
            with col:
                col_name = list(COLS_SPACE["disc"].keys())[i * 4 + j]
                col_values = COLS_SPACE["disc"][col_name]

                if col_name == "Multiple Lines":
                    col_values = (
                        col_values[0]
                        if disc["Phone Service"] == "Yes"
                        else col_values[1]
                    )

                if col_name in INTERNET_SERVICES:
                    col_values = (
                        col_values[0]
                        if disc["Internet Service"] != "No"
                        else col_values[1]
                    )

                disc[col_name] = st.selectbox(col_name, col_values)
        st.write("\n")

    # Variables continues
    cont = {}
    cont_cols = iter(st.columns(5))

    # 1. Coordonnées géographiques
    zip_codes_coords = pd.read_csv("data/ZipCodesAndCoords.csv")
    with next(cont_cols):
        zip_code = st.selectbox("Zip Code", COLS_SPACE["coords"]["Zip Code"])
        coords_row = zip_codes_coords.query("`Zip Code` == @zip_code").iloc[0]
        coords = {"Lat": coords_row["Lat"], "Long": coords_row["Long"]}

    # 2. Autres variables continues
    for col_name, (min_value, max_value, value, step) in COLS_SPACE["cont"].items():
        with next(cont_cols):
            cont[col_name] = st.number_input(
                label=col_name,
                min_value=min_value,
                max_value=max_value,
                value=value,
                step=step,
            )

    # Séparation
    st.divider()

    # Réunion
    input_data = pd.DataFrame([disc | cont | coords])

    # Variables calculées manuellement
    df_internet, df_no_internet = prepare_data(
        input_data, extract_coords=False, identify_new_clients=True, split_internet=True
    )

    df = pd.concat([df_internet, df_no_internet]).sort_index()

    # Modèles séparés
    if use_separated_models:
        # Avec Internet
        df_internet = inference_prediction(
            df_internet,
            preprocessor=st.session_state["preprocessor_internet"],
            model=st.session_state["model_internet"],
        )

        # Sans Internet
        df_no_internet = inference_prediction(
            df_no_internet,
            preprocessor=st.session_state["preprocessor_no_internet"],
            model=st.session_state["model_no_internet"],
        )

        # Concat
        df = pd.concat([df_internet, df_no_internet]).sort_index()

    # Modèle unique
    else:
        df = inference_prediction(
            df,
            preprocessor=st.session_state["preprocessor_all"],
            model=st.session_state["model_all"],
        )

    # Affichage des résultats
    churn_score = df.loc[0, "Pred Proba"]
    signal = (
        "🟢"
        if churn_score < 0.3
        else "🟡" if churn_score < 0.5 else "🟠" if churn_score < 0.7 else "🔴"
    )
    st.write(f"{signal} Probabilité de résiliation : {churn_score:.0%}.")
