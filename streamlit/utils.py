import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier

from typing import List, Tuple, Dict, Literal, Union


# Fonctions pour afficher les graphiques
def plot_disc_variables(
    df,
    cols: List[str],
    n_rows: int = 4,
    n_cols: int = 4,
    segment: Literal["Avec Internet", "Sans Internet", "All"] = "All",
    **kwargs,
) -> None:
    """Représente la distribution des variables discrètes."""
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols)

    for i, col in enumerate(cols):
        row, col_pos = divmod(i, n_cols)
        row += 1

        value_counts = df[col].value_counts(normalize=True).sort_index()

        bar_trace = go.Bar(
            x=value_counts.index,
            y=value_counts.values,
            text=[f"{v:.0%}" for v in value_counts.values],
            marker_color=px.colors.qualitative.Plotly[0],
            opacity=0.8,
        )

        fig.add_trace(bar_trace, row=row, col=col_pos + 1)

    fig.update_layout(
        height=kwargs.get("height", 800),
        width=kwargs.get("width", 1200),
        title_text=kwargs.get(
            "fig_title", f"Distribution des variables discrètes, Segment = {segment}"
        ),
        showlegend=False,
    )

    fig.update_yaxes(showticklabels=False)

    st.plotly_chart(fig)


def plot_disc_vs_target(
    df,
    cols: List[str],
    n_rows: int = 2,
    n_cols: int = 4,
    segment: Literal["All", "Avec Internet", "Sans Internet"] = "All",
    target: str = "Churn Value",
    **kwargs,
) -> None:
    """Représente le taux de churn selon les différentes modalités des variables discrètes."""

    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=kwargs.get("subplot_titles", cols)
    )

    colors = px.colors.qualitative.Plotly[1:3][::-1]

    for i, col in enumerate(cols):
        row, col_pos = divmod(i, n_cols)
        row += 1

        prop_df = (
            df.groupby([col, target])
            .size()
            .div(df.groupby(col).size() / 100)
            .reset_index(name="Percentage")
            .round()
        )

        for j in [0, 1]:
            sub_prop_df = prop_df[prop_df[target] == j]

            bar_trace = go.Bar(
                x=(
                    ["<br>".join(val.split()) for val in sub_prop_df[col]]
                    if kwargs.get("br_xtickslabels")
                    else sub_prop_df[col]
                ),
                y=sub_prop_df["Percentage"],
                marker_color=colors[j],
                name="Stayed" if (j == 0) else "Exited",
                text=(
                    sub_prop_df["Percentage"].astype(str).str.split(".").str[0] + "%"
                    if j == 0
                    else ""
                ),
                showlegend=(i == 0),
                opacity=0.8,
            )

            fig.add_trace(bar_trace, row=row, col=col_pos + 1)

    fig.update_layout(
        height=kwargs.get("height", 400),
        width=kwargs.get("width", 1200),
        title_text=kwargs.get(
            "fig_title",
            f"Taux de churn selon les variables discrètes, Segment = {segment}",
        ),
        barmode="stack",
    )

    fig.update_yaxes(showticklabels=False)

    st.plotly_chart(fig)


def plot_cont_vs_cont(
    df,
    cols: List[str],
    segment: Literal["All", "Avec Internet", "Sans Internet"] = "All",
    target: str = "Churn Value",
    **kwargs,
) -> None:
    """Représente les relations entre plusieurs variables continues."""

    colors = {
        val: color for val, color in zip([1, 0], px.colors.qualitative.Plotly[1:3])
    }

    fig = go.Figure(
        data=go.Splom(
            dimensions=[
                {"label": col, "values": df[col].tolist()} for col in df[cols].columns
            ],
            marker_color=df[target].map(colors),
            marker_size=2,
            opacity=0.8,
            diagonal_visible=False,
            showupperhalf=False,
            showlegend=False,
        )
    )

    fig.update_layout(
        title_text=f"Relations entre les variables continues, Segment = {segment}",
        height=kwargs.get("height", 600),
        width=kwargs.get("width", 600),
    )

    # Ajouter un plot vide pour avoir la légende
    for val, color in colors.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker_color=color,
                name="Stayed" if (val == 0) else "Exited",
            )
        )

    st.plotly_chart(fig)


def plot_cont_vs_target(
    df,
    cols: List[str],
    target_value: Literal[0, 1],
    segment: Literal["All", "Avec Internet", "Sans Internet"] = "All",
    target: str = "Churn Value",
    **kwargs,
) -> None:
    """Représente la distribution des variables continues selon une valeur fixée de la variable cible."""

    color_id = 1 if target_value == 1 else 2
    color = px.colors.qualitative.Plotly[color_id]

    fig = make_subplots(
        rows=1,
        cols=len(cols),
        subplot_titles=["<br>".join(col.split()) for col in cols],
    )

    for i, col in enumerate(cols):
        fig.add_trace(
            go.Histogram(
                x=df.loc[df[target] == target_value, col],
                nbinsx=100,
                marker_color=color,
                opacity=0.8,
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        title_text=f"Distribution des variables continues, Segment = {segment}, Churn Value = {target_value}",
        showlegend=False,
        height=kwargs.get("height", 300),
        width=kwargs.get("width", 1200),
    )

    st.plotly_chart(fig)


def plot_clusters_scatterplots(
    df,
    x: str = "Total Charges",
    y: str = "Tenure Months",
    clusters: List[int] = [0, 5, 10, 15, 19],
    segment: Literal["Avec Internet", "Sans Internet"] = "Avec Internet",
    **kwargs,
) -> None:
    # Exemples graphiques
    fig = make_subplots(
        rows=1,
        cols=len(clusters) + 1,
        shared_yaxes=True,
        subplot_titles=["Cluster = All"] + [f"Cluster = {i}" for i in clusters],
    )

    # Scatterplot pour toutes les données
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="markers",
            marker_size=2,
            marker_color=px.colors.qualitative.Plotly[0],
            opacity=0.8,
        ),
        row=1,
        col=1,
    )

    corr_all = kwargs.get("corr_all")
    fig.add_annotation(
        x=max(df[x]),
        y=5,
        text=f"Corr = {corr_all:.4f}" if corr_all else "",
        showarrow=False,
        font_weight="bold",
        xanchor="right",
        row=1,
        col=1,
    )

    # Scatterplots pour chaque cluster spécifique
    for idx, i in enumerate(clusters):
        tmp_df = df.query("`Monthly Charges Group` == @i")
        fig.add_trace(
            go.Scatter(
                x=tmp_df[x],
                y=tmp_df[y],
                mode="markers",
                marker_size=2,
                marker_color=px.colors.qualitative.Plotly[0],
                opacity=0.8,
            ),
            row=1,
            col=idx + 2,
        )

        corr_clusters = kwargs.get("corr_clusters")

        fig.add_annotation(
            x=max(tmp_df[x]),
            y=5,
            text=f"Corr = {corr_clusters[i]:.4f}" if corr_clusters else "",
            showarrow=False,
            font_weight="bold",
            xanchor="right",
            row=1,
            col=idx + 2,
        )

    fig.update_xaxes(title_text=x, row=1, col=len(clusters) // 2 + 1)

    fig.update_layout(
        title=f"Relations entre les variables `{x}` et `{y}` pour différents clusters, Segment = {segment}",
        height=400,
        width=1200,
        yaxis_title=y,
        showlegend=False,
    )

    st.plotly_chart(fig)


# Préparation des données
def prepare_data(
    df,
    extract_coords: bool = True,
    identify_new_clients: bool = True,
    split_internet: bool = True,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    # Extraction de la latitude et de la longitude
    if extract_coords:
        df["Lat"] = df["Lat Long"].str.split(", ").str[0].astype(float)
        df["Long"] = df["Lat Long"].str.split(", ").str[1].astype(float)

    # Identification des nouveaux clients
    if identify_new_clients:
        df["Is New Client"] = (df["Tenure Months"] <= 48).astype(int)

    # Séparation avec/sans Internet
    if split_internet:
        df_internet = df.query("`Internet Service` != 'No'").copy()
        df_no_internet = df.drop(df_internet.index).copy()

        # Ajout des groupes de `Monthly Charges`
        if "kmeans_internet" not in st.session_state:
            st.session_state["kmeans_internet"] = joblib.load(
                "artifacts/internet/kmeans_internet.pkl"
            )

        if "clusters_mapping_internet" not in st.session_state:
            st.session_state["clusters_mapping_internet"] = joblib.load(
                "artifacts/internet/clusters_mapping_internet.pkl"
            )

        if "kmeans_no_internet" not in st.session_state:
            st.session_state["kmeans_no_internet"] = joblib.load(
                "artifacts/no_internet/kmeans_no_internet.pkl"
            )

        if "clusters_mapping_no_internet" not in st.session_state:
            st.session_state["clusters_mapping_no_internet"] = joblib.load(
                "artifacts/no_internet/clusters_mapping_no_internet.pkl"
            )

        if len(df_internet) > 0:
            df_internet["Monthly Charges Group"] = st.session_state[
                "kmeans_internet"
            ].predict(df_internet[["Monthly Charges"]])
            df_internet["Monthly Charges Group"] = df_internet[
                "Monthly Charges Group"
            ].map(st.session_state["clusters_mapping_internet"])

        if len(df_no_internet) > 0:
            df_no_internet["Monthly Charges Group"] = st.session_state[
                "kmeans_no_internet"
            ].predict(df_no_internet[["Monthly Charges"]])
            df_no_internet["Monthly Charges Group"] = df_no_internet[
                "Monthly Charges Group"
            ].map(st.session_state["clusters_mapping_no_internet"])

        # Ajout de la colonne `Services Count`
        df_internet["Services Count"] = (
            df_internet[INTERNET_SERVICES]
            .replace({"No": "0", "Yes": "1"})
            .astype(int)
            .sum(axis=1)
        )

        return df_internet, df_no_internet

    return df


# Variables pour le modèle (discrètes, continues, par défaut, par segment)
DISC_INTERNET = [
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
    "Is New Client",
]

DISC_INTERNET_DEFAULT = [
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Contract",
    "Paperless Billing",
    "Is New Client",
]

CONT_INTERNET = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
    "Lat",
    "Long",
    "Monthly Charges Group",
    "Services Count",
]

CONT_INTERNET_DEFAULT = [
    "Monthly Charges",
    "Total Charges",
    "CLTV",
    "Lat",
    "Long",
    "Monthly Charges Group",
    "Services Count",
]

DISC_NO_INTERNET = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Multiple Lines",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Is New Client",
]

DISC_NO_INTERNET_DEFAULT = [
    "Dependents",
    "Contract",
    "Paperless Billing",
]

CONT_NO_INTERNET = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
    "Lat",
    "Long",
    "Monthly Charges Group",
]

CONT_NO_INTERNET_DEFAULT = [
    "Tenure Months",
    "CLTV",
    "Lat",
    "Long",
]

DISC_UNIQUE_MODEL = [
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
    "Is New Client",
]

DISC_UNIQUE_MODEL_DEFAULT = [
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Paperless Billing",
]

CONT_UNIQUE_MODEL = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
    "Lat",
    "Long",
]

CONT_UNIQUE_MODEL_DEFAULT = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
    "Lat",
    "Long",
]

# Valeurs par défaut des hyperparamètres
PARAMS_INTERNET_DEFAULT = {
    "learning_rate": 0.02,
    "scale_pos_weight": 3,
    "subsample": 0.8,
}

PARAMS_NO_INTERNET_DEFAULT = {
    "learning_rate": 0.04,
    "scale_pos_weight": 6,
    "subsample": 0.8,
}

PARAMS_UNIQUE_MODEL_DEFAULT = {
    "learning_rate": 0.02,
    "scale_pos_weight": 4,
    "subsample": 0.8,
}


# Fonctions de modélisation
def preprocess_data(
    X_train, X_test, disc: List[str], cont: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    preprocessor = ColumnTransformer(
        transformers=[
            ("disc", OneHotEncoder(drop="first", handle_unknown="ignore"), disc),
            ("cont", MinMaxScaler(), cont),
        ],
        remainder="drop",
    )

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed


def train_model(
    X_train_preprocessed,
    y_train,
    X_test_preprocessed,
    y_test,
    params: Dict[str, Union[int, float]],
    disc: List[str],
    cont: List[str],
    model_type: Literal["INTERNET", "NO INTERNET", "UNIQUE MODEL"],
) -> Tuple[Dict[str, Union[int, float, List[str]]], np.ndarray, np.ndarray]:
    if X_train_preprocessed.shape[1] == 0:
        st.error("Aucune colonne sélectionnée")
        sys.exit()

    with st.spinner("Entraînement du modèle..."):
        # Entraînement
        clf = XGBClassifier(random_state=42, **params)
        clf.fit(X_train_preprocessed, y_train)

        # Prédictions
        y_pred_train = clf.predict(X_train_preprocessed)
        y_pred_test = clf.predict(X_test_preprocessed)

        # Métriques
        metrics = {
            # Test
            "accuracy_test": accuracy_score(y_test, y_pred_test),
            "precision_test": precision_score(y_test, y_pred_test),
            "recall_test": recall_score(y_test, y_pred_test),
            "f1_test": f1_score(y_test, y_pred_test),
            # Train
            "accuracy_train": accuracy_score(y_train, y_pred_train),
            "precision_train": precision_score(y_train, y_pred_train),
            "recall_train": recall_score(y_train, y_pred_train),
            "f1_train": f1_score(y_train, y_pred_train),
        }

        suffix = "" if model_type == "UNIQUE_MODEL" else f"_{model_type.lower()}"
        metrics = {f"{key}{suffix}": value for key, value in metrics.items()}

        # Sauvegarde des paramètres
        metrics.update(params)

        # Sauvegarde des variables suivant le modèle :
        # VARIABLES_PAR_DEFAUT + variables rajoutées - variables supprimées
        DISC_MODEL_TYPE_DEFAULT = globals()[f"DISC_{model_type}_DEFAULT"]
        disc_added = list({*disc}.difference({*DISC_MODEL_TYPE_DEFAULT}))
        disc_added = ["+"] + disc_added if len(disc_added) > 0 else []
        disc_removed = list({*DISC_MODEL_TYPE_DEFAULT}.difference({*disc}))
        disc_removed = ["-"] + disc_removed if len(disc_removed) > 0 else []

        CONT_MODEL_TYPE_DEFAULT = globals()[f"CONT_{model_type}_DEFAULT"]
        cont_added = list({*cont}.difference({*CONT_MODEL_TYPE_DEFAULT}))
        cont_added = ["+"] + cont_added if len(cont_added) > 0 else []
        cont_removed = list({*CONT_MODEL_TYPE_DEFAULT}.difference({*cont}))
        cont_removed = ["-"] + cont_removed if len(cont_removed) > 0 else []

        metrics["disc"] = [f"DISC_{model_type}_DEFAULT"] + disc_added + disc_removed
        metrics["cont"] = [f"CONT_{model_type}_DEFAULT"] + cont_added + cont_removed

        st.toast("Modèle entraîné avec succès !")

        return metrics, y_pred_test, y_pred_train


# Résultats des modèles de benchmark
RESULTS_TWO_MODELS_DATA = {
    # Performances - All - Test
    "accuracy_test": [0.8462177888611804],
    "precision_test": [0.5189393939393939],
    "recall_test": [0.7025641025641025],
    "f1_test": [0.5969498910675382],
    # Performances - All - Train
    "accuracy_train": [0.8754159733777038],
    "precision_train": [0.5807314897413024],
    "recall_train": [0.8346153846153846],
    "f1_train": [0.6849026827985271],
    # Performances - Avec Internet - Test
    "accuracy_test_internet": [0.8140043763676149],
    "precision_test_internet": [0.5238095238095238],
    "recall_test_internet": [0.7252747252747253],
    "f1_test_internet": [0.6082949308755761],
    # Performances - Avec Internet - Train
    "accuracy_train_internet": [0.839671682626539],
    "precision_train_internet": [0.5676442762535477],
    "recall_train_internet": [0.823045267489712],
    "f1_train_internet": [0.671892497200448],
    # Performances - Sans Internet - Test
    "accuracy_test_no_internet": [0.9480968858131488],
    "precision_test_no_internet": [0.4166666666666667],
    "recall_test_no_internet": [0.38461538461538464],
    "f1_test_no_internet": [0.4],
    # Performances - Sans Internet - Train
    "accuracy_train_no_internet": [0.9887250650477016],
    "precision_train_no_internet": [0.796875],
    "recall_train_no_internet": [1.0],
    "f1_train_no_internet": [0.8869565217391304],
    # Hyperparamètres
    "learning_rate": [
        [
            PARAMS_INTERNET_DEFAULT["learning_rate"],
            PARAMS_NO_INTERNET_DEFAULT["learning_rate"],
        ]
    ],
    "scale_pos_weight": [
        [
            PARAMS_INTERNET_DEFAULT["scale_pos_weight"],
            PARAMS_NO_INTERNET_DEFAULT["scale_pos_weight"],
        ]
    ],
    "subsample": [
        [PARAMS_INTERNET_DEFAULT["subsample"], PARAMS_NO_INTERNET_DEFAULT["subsample"]]
    ],
    # Variables
    "disc": [["DISC_INTERNET_DEFAULT", "AND", "DISC_NO_INTERNET_DEFAULT"]],
    "cont": [["CONT_INTERNET_DEFAULT", "AND", "CONT_NO_INTERNET_DEFAULT"]],
}

RESULTS_TWO_MODELS_DF = pd.DataFrame(RESULTS_TWO_MODELS_DATA)

COLS_TO_SHOW_TWO_MODELS = [
    "f1_test",
    "f1_train",
    "f1_test_internet",
    "f1_train_internet",
    "f1_test_no_internet",
    "f1_train_no_internet",
    "learning_rate",
    "scale_pos_weight",
    "subsample",
    "disc",
    "cont",
]

RESULTS_UNIQUE_MODEL_DATA = {
    # Performances - Test
    "accuracy_test": [0.8196176226101413],
    "precision_test": [0.46381578947368424],
    "recall_test": [0.7230769230769231],
    "f1_test": [0.5651302605210421],
    # Performances - Train
    "accuracy_train": [0.8514975041597338],
    "precision_train": [0.5270491803278688],
    "recall_train": [0.8243589743589743],
    "f1_train": [0.643],
    # Hyperparamètres
    "learning_rate": [PARAMS_UNIQUE_MODEL_DEFAULT["learning_rate"]],
    "scale_pos_weight": [PARAMS_UNIQUE_MODEL_DEFAULT["scale_pos_weight"]],
    "subsample": [PARAMS_UNIQUE_MODEL_DEFAULT["subsample"]],
    # Variables
    "disc": [["DISC_UNIQUE_MODEL_DEFAULT"]],
    "cont": [["CONT_UNIQUE_MODEL_DEFAULT"]],
}

RESULTS_UNIQUE_MODEL_DF = pd.DataFrame(RESULTS_UNIQUE_MODEL_DATA)

COLS_TO_SHOW_UNIQUE_MODEL = [
    "f1_test",
    "f1_train",
    "learning_rate",
    "scale_pos_weight",
    "subsample",
    "disc",
    "cont",
]

# Ensemble des valeurs prises par les variables
COLS_SPACE = {
    "disc": {
        "Gender": ["Male", "Female"],
        "Senior Citizen": ["No", "Yes"],
        "Partner": ["No", "Yes"],
        "Dependents": ["No", "Yes"],
        "Phone Service": ["No", "Yes"],
        "Multiple Lines": (["No", "Yes"], ["No phone service"]),
        "Internet Service": ["DSL", "Fiber optic", "No"],
        "Online Security": (["No", "Yes"], ["No internet service"]),
        "Online Backup": (["No", "Yes"], ["No internet service"]),
        "Device Protection": (["No", "Yes"], ["No internet service"]),
        "Tech Support": (["No", "Yes"], ["No internet service"]),
        "Streaming TV": (["No", "Yes"], ["No internet service"]),
        "Streaming Movies": (["No", "Yes"], ["No internet service"]),
        "Contract": ["Month-to-month", "Two year", "One year"],
        "Paperless Billing": ["No", "Yes"],
        "Payment Method": [
            "Credit card (automatic)",
            "Mailed check",
            "Electronic check",
            "Bank transfer (automatic)",
        ],
    },
    "cont": {
        "Tenure Months": [0, 72, 30, 1],
        "Monthly Charges": [15.0, 120.0, 70.0, 0.05],
        "Total Charges": [15.0, 9000.0, 1500.0, 0.05],
        "CLTV": [2000, 7000, 4500, 1],
    },
    "coords": {
        "Zip Code": pd.read_csv("data/ZipCodesAndCoords.csv")["Zip Code"],
    },
}

# Liste des services Internet
INTERNET_SERVICES = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]


# Prédictions en mode inférence
def inference_prediction(df, preprocessor, model) -> pd.DataFrame:
    if len(df) > 0:
        processed_df = preprocessor.transform(df)
        y_pred = model.predict(processed_df)
        y_pred_proba = model.predict_proba(processed_df)
        df["Pred"] = y_pred
        df["Pred Proba"] = y_pred_proba[:, 1]

    return df
