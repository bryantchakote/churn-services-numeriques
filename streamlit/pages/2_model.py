import sys
import joblib
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from utils import (
    DISC_INTERNET,
    DISC_INTERNET_DEFAULT,
    CONT_INTERNET,
    CONT_INTERNET_DEFAULT,
    DISC_NO_INTERNET,
    DISC_NO_INTERNET_DEFAULT,
    CONT_NO_INTERNET,
    CONT_NO_INTERNET_DEFAULT,
    DISC_UNIQUE_MODEL,
    DISC_UNIQUE_MODEL_DEFAULT,
    CONT_UNIQUE_MODEL,
    CONT_UNIQUE_MODEL_DEFAULT,
    preprocess_data,
    train_model,
    RESULTS_TWO_MODELS_DF,
    RESULTS_UNIQUE_MODEL_DF,
    COLS_TO_SHOW_TWO_MODELS,
    COLS_TO_SHOW_UNIQUE_MODEL,
)


st.set_page_config(page_title="Modèle", layout="wide")

st.title("Modèle")

# Récupération des données
if "df" not in st.session_state:
    st.error(
        "Impossible d'accéder aux données, veuillez retournez sur la page principale pour les charger."
    )
    sys.exit()

df = st.session_state["df"]

# Suppression des valeurs manquantes
df.dropna(inplace=True)

# Extraction de la latitude et de la longitude
df["Lat"] = df["Lat Long"].str.split(", ").str[0].astype(float)
df["Long"] = df["Lat Long"].str.split(", ").str[1].astype(float)

# Identification des nouveaux clients
df["Is New Client"] = (df["Tenure Months"] <= 48).astype(int)

# Affichage des résultats
if "RESULTS_TWO_MODELS_DF" not in st.session_state:
    st.session_state["RESULTS_TWO_MODELS_DF"] = RESULTS_TWO_MODELS_DF

if "RESULTS_UNIQUE_MODEL_DF" not in st.session_state:
    st.session_state["RESULTS_UNIQUE_MODEL_DF"] = RESULTS_UNIQUE_MODEL_DF

# Préparation des données
st.write(
    """
    - Cette interface vous permet d'entraîner votre propre modèle de prédiction du churn.
    - Vous pouvez choisir entre deux approches : entraîner des modèles séparés pour les clients avec et sans Internet, ou un modèle unique pour l'ensemble des clients.
    - Le split train/test a déjà été effectué
    - Vous pouvez également choisir les variables à inclure dans les différents modèles.
    - Le pipeline de prétraitement comprend un `OneHotEncoder` pour les variables discrètes et un `MinMaxScaler` pour les variables continues.
    - Un `XGBClassifier` sera utilisé pour la prédiction du churn.
    - Il est possible de modifier certains hyperparamètres du modèle : `learning_rate`, `scale_pos_weight` et `subsample`.
"""
)


# Séparation avec/sans Internet
df_internet = df.query("`Internet Service` != 'No'").copy()
df_no_internet = df.drop(df_internet.index).copy()


# ---------------- Avec Internet ----------------
# Ajout des groupes de `Monthly Charges`
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

# Ajout de la colonne `Services Count` pour compter le nombre de services auxquels le client a souscrit
internet_services = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]
df_internet["Services Count"] = (
    df_internet[internet_services]
    .replace({"No": "0", "Yes": "1"})
    .astype(int)
    .sum(axis=1)
)

# Split train/test
X_internet = df_internet.drop(columns="Churn Value")
y_internet = df_internet["Churn Value"]
X_train_internet, X_test_internet, y_train_internet, y_test_internet = train_test_split(
    X_internet, y_internet, test_size=0.2, random_state=42, stratify=y_internet
)


# ---------------- Sans Internet ----------------
# Ajout des groupes de `Monthly Charges`
kmeans_no_internet = joblib.load("artifacts/no_internet/kmeans_no_internet.pkl")
clusters_mapping_no_internet = joblib.load(
    "artifacts/no_internet/clusters_mapping_no_internet.pkl"
)

df_no_internet["Monthly Charges Group"] = kmeans_no_internet.predict(
    df_no_internet[["Monthly Charges"]]
)
df_no_internet["Monthly Charges Group"] = df_no_internet["Monthly Charges Group"].map(
    clusters_mapping_no_internet
)

# Split train/test
X_no_internet = df_no_internet.drop(columns="Churn Value")
y_no_internet = df_no_internet["Churn Value"]
X_train_no_internet, X_test_no_internet, y_train_no_internet, y_test_no_internet = (
    train_test_split(
        X_no_internet,
        y_no_internet,
        test_size=0.2,
        random_state=42,
        stratify=y_no_internet,
    )
)


# ---------------- All ----------------
X_train = pd.concat([X_train_internet, X_train_no_internet])
X_test = pd.concat([X_test_internet, X_test_no_internet])
y_train = pd.concat([y_train_internet, y_train_no_internet])
y_test = pd.concat([y_test_internet, y_test_no_internet])


# Séparation
st.divider()


train_separated_models = st.toggle("Modèles séparés", True)


if train_separated_models:
    internet, no_internet = st.columns(2, gap="large")

    with internet:
        st.subheader("Clients avec Internet")

        with st.expander("Ajustement des hyperparamètres"):
            param1_internet, param2_internet, param3_internet = st.columns(
                3, gap="large"
            )
            learning_rate_internet = param1_internet.slider(
                "learning_rate",
                min_value=0.01,
                max_value=0.05,
                value=0.02,
                step=0.01,
                key="learning_rate_internet",
            )
            scale_pos_weight_internet = param2_internet.slider(
                "scale_pos_weight",
                min_value=3,
                max_value=7,
                value=3,
                step=1,
                key="scale_pos_weight_internet",
            )
            subsample_internet = param3_internet.slider(
                "Subsample",
                min_value=0.8,
                max_value=1.0,
                value=0.8,
                step=0.1,
                key="Subsample_internet",
            )
            params_internet = {
                "learning_rate": learning_rate_internet,
                "scale_pos_weight": scale_pos_weight_internet,
                "subsample": subsample_internet,
            }

        with st.expander("Sélection des variables", expanded=True):
            disc_internet = st.multiselect(
                "Variables discrètes",
                DISC_INTERNET,
                default=DISC_INTERNET_DEFAULT,
            )
            disc_internet_added = list({*disc_internet}.difference({*DISC_INTERNET}))
            disc_internet_removed = list({*DISC_INTERNET}.difference({*disc_internet}))

            cont_internet = st.multiselect(
                "Variables continues",
                CONT_INTERNET,
                default=CONT_INTERNET_DEFAULT,
            )
            cont_internet_added = list({*cont_internet}.difference({*CONT_INTERNET}))
            cont_internet_removed = list({*CONT_INTERNET}.difference({*cont_internet}))

            # Preprocessing
            X_train_internet_preprocessed, X_test_internet_preprocessed = (
                preprocess_data(
                    X_train_internet, X_test_internet, disc_internet, cont_internet
                )
            )

    with no_internet:
        st.subheader("Clients sans Internet")

        with st.expander("Ajustement des hyperparamètres"):
            param1_no_internet, param2_no_internet, param3_no_internet = st.columns(
                3, gap="large"
            )
            learning_rate_no_internet = param1_no_internet.slider(
                "learning_rate",
                min_value=0.01,
                max_value=0.05,
                value=0.04,
                step=0.01,
                key="learning_rate_no_internet",
            )
            scale_pos_weight_no_internet = param2_no_internet.slider(
                "scale_pos_weight",
                min_value=3,
                max_value=7,
                value=6,
                step=1,
                key="scale_pos_weight_no_internet",
            )
            subsample_no_internet = param3_no_internet.slider(
                "Subsample",
                min_value=0.8,
                max_value=1.0,
                value=0.8,
                step=0.1,
                key="Subsample_no_internet",
            )
            params_no_internet = {
                "learning_rate": learning_rate_no_internet,
                "scale_pos_weight": scale_pos_weight_no_internet,
                "subsample": subsample_no_internet,
            }

        with st.expander("Sélection des variables", expanded=True):
            disc_no_internet = st.multiselect(
                "Variables discrètes",
                DISC_NO_INTERNET,
                default=DISC_NO_INTERNET_DEFAULT,
            )
            disc_no_internet_added = list(
                {*disc_no_internet}.difference({*DISC_NO_INTERNET})
            )
            disc_no_internet_removed = list(
                {*DISC_NO_INTERNET}.difference({*disc_no_internet})
            )

            cont_no_internet = st.multiselect(
                "Variables continues",
                CONT_NO_INTERNET,
                default=CONT_NO_INTERNET_DEFAULT,
            )
            cont_no_internet_added = list(
                {*cont_no_internet}.difference({*CONT_NO_INTERNET})
            )
            cont_no_internet_removed = list(
                {*CONT_NO_INTERNET}.difference({*cont_no_internet})
            )

            # Preprocessing
            X_train_no_internet_preprocessed, X_test_no_internet_preprocessed = (
                preprocess_data(
                    X_train_no_internet,
                    X_test_no_internet,
                    disc_no_internet,
                    cont_no_internet,
                )
            )

else:
    st.subheader("Tous les clients (modèle unique)")

    with st.expander("Ajustement des hyperparamètres"):
        param1_unique_model, param2_unique_model, param3_unique_model = st.columns(
            3, gap="large"
        )
        learning_rate_unique_model = param1_unique_model.slider(
            "learning_rate",
            min_value=0.01,
            max_value=0.05,
            value=0.02,
            step=0.01,
            key="learning_rate_unique_model",
        )
        scale_pos_weight_unique_model = param2_unique_model.slider(
            "scale_pos_weight",
            min_value=3,
            max_value=7,
            value=4,
            step=1,
            key="scale_pos_weight_unique_model",
        )
        subsample_unique_model = param3_unique_model.slider(
            "Subsample",
            min_value=0.8,
            max_value=1.0,
            value=0.8,
            step=0.1,
            key="Subsample_unique_model",
        )
        params_unique_model = {
            "learning_rate": learning_rate_unique_model,
            "scale_pos_weight": scale_pos_weight_unique_model,
            "subsample": subsample_unique_model,
        }

    with st.expander("Sélection des variables", expanded=True):
        disc_unique_model = st.multiselect(
            "Variables discrètes",
            DISC_UNIQUE_MODEL,
            default=DISC_UNIQUE_MODEL_DEFAULT,
        )
        disc_unique_model_added = list(
            {*disc_unique_model}.difference({*DISC_UNIQUE_MODEL_DEFAULT})
        )
        disc_unique_model_removed = list(
            {*DISC_UNIQUE_MODEL_DEFAULT}.difference({*disc_unique_model})
        )

        cont_unique_model = st.multiselect(
            "Variables continues",
            CONT_UNIQUE_MODEL,
            default=CONT_UNIQUE_MODEL_DEFAULT,
        )
        cont_unique_model_added = list(
            {*cont_unique_model}.difference({*CONT_UNIQUE_MODEL_DEFAULT})
        )
        cont_unique_model_removed = list(
            {*CONT_UNIQUE_MODEL_DEFAULT}.difference({*cont_unique_model})
        )

        # Preprocessing
        X_train_preprocessed, X_test_preprocessed = preprocess_data(
            X_train, X_test, disc_unique_model, cont_unique_model
        )

run_train = st.button("Entraîner le modèle")

if run_train:
    if train_separated_models:
        metrics_internet, y_pred_test_internet, y_pred_train_internet = train_model(
            X_train_internet_preprocessed,
            y_train_internet,
            X_test_internet_preprocessed,
            y_test_internet,
            params_internet,
            disc_internet,
            cont_internet,
            "INTERNET",
        )

        metrics_no_internet, y_pred_test_no_internet, y_pred_train_no_internet = (
            train_model(
                X_train_no_internet_preprocessed,
                y_train_no_internet,
                X_test_no_internet_preprocessed,
                y_test_no_internet,
                params_no_internet,
                disc_no_internet,
                cont_no_internet,
                "NO_INTERNET",
            )
        )

        y_pred_test = np.concat([y_pred_test_internet, y_pred_test_no_internet])
        y_pred_train = np.concat([y_pred_train_internet, y_pred_train_no_internet])

        metrics_two_models = dict()
        model_types = ["internet", "no_internet"]
        for metrics, model_type in zip(
            [metrics_internet, metrics_no_internet], model_types
        ):
            metrics_two_models.update(
                {k: v for k, v in metrics.items() if k.endswith(model_type)}
            )

        # Calcul des performances globales
        for split in ["test", "train"]:
            for metric in ["accuracy", "precision", "recall", "f1"]:
                metrics_two_models[f"{metric}_{split}"] = globals()[f"{metric}_score"](
                    y_true=globals()[f"y_{split}"], y_pred=globals()[f"y_pred_{split}"]
                )

        # Append des hyperparamètres
        metrics_two_models.update(
            {
                k: [metrics_internet[k]] + [metrics_no_internet[k]]
                for k in list(params_internet)
            }
        )

        # Append des variables
        metrics_two_models.update(
            {
                k: metrics_internet[k] + ["AND"] + metrics_no_internet[k]
                for k in ["disc", "cont"]
            }
        )

        metrics_two_models = {k: [v] for k, v in metrics_two_models.items()}

        st.session_state["RESULTS_TWO_MODELS_DF"] = pd.concat(
            [
                st.session_state["RESULTS_TWO_MODELS_DF"],
                pd.DataFrame(metrics_two_models),
            ],
            ignore_index=True,
        )
    else:
        metrics_unique_model, _, _ = train_model(
            X_train_preprocessed,
            y_train,
            X_test_preprocessed,
            y_test,
            params_unique_model,
            disc_unique_model,
            cont_unique_model,
            "UNIQUE_MODEL",
        )

        metrics_unique_model = {k: [v] for k, v in metrics_unique_model.items()}

        st.session_state["RESULTS_UNIQUE_MODEL_DF"] = pd.concat(
            [
                st.session_state["RESULTS_UNIQUE_MODEL_DF"],
                pd.DataFrame(metrics_unique_model),
            ],
            ignore_index=True,
        )


st.divider()

two_models, unique_model = st.columns(2, gap="large")

with two_models:
    st.subheader("Résultats (modèles séparés)")
    st.dataframe(
        st.session_state["RESULTS_TWO_MODELS_DF"], column_order=COLS_TO_SHOW_TWO_MODELS
    )

with unique_model:
    st.subheader("Résultats (modèle unique)")
    st.dataframe(
        st.session_state["RESULTS_UNIQUE_MODEL_DF"],
        column_order=COLS_TO_SHOW_UNIQUE_MODEL,
    )
