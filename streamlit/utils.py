import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.subplots as sp
# from plotly.graph_objects import Figure

plt.style.use("seaborn-v0_8")


def plot_disc_variables(df, cols, n_rows=2, n_cols=8, segment="All", **kwargs):
    """Représente la distribution des variables discrètes."""
    
    _, axes = plt.subplots(n_rows, n_cols, figsize=kwargs.get("figsize", (24, 6)))
    
    if len(cols) > 1:
        axes = axes.flatten()
    
    for i, col in enumerate(cols):
        ax = axes[i] if len(cols) > 1 else axes

        sns.countplot(data=df, x=col, order=df[col].drop_duplicates().sort_values(), ax=ax)
        
        # Ajout des pourcentages sur les barres
        n = df[col].count()
        for p in ax.patches:
            ax.text(
                p.get_x() + p.get_width() / 2,
                p.get_y() + p.get_height() / 2,
                f"{p.get_height() / n:.0%}",
                ha="center", va="center",
                c="w", weight="bold",
            )
            
            ax.set_xlabel(col, weight="bold")
            ax.set_xticks(ax.get_xticks())
            
            xticklabels = ["\n".join(label.get_text().split()) for label in ax.get_xticklabels()]
            ax.set_xticklabels(xticklabels, rotation=0)
            
            ax.set_yticks([])
            ax.set_ylabel("")

    title = kwargs.get("fig_title", f"Distribution des variables discrètes, Segment = {segment}")
    size = kwargs.get("fig_title_font_size", 16)
    
    plt.suptitle(title, size=size, weight="bold")
    plt.tight_layout()
    
    st.pyplot(plt)


def plot_disc_vs_target(df, cols, n_rows=1, n_cols=8, segment="All", target="Churn Value", **kwargs):
    """Représente le taux de churn selon les différentes modalités des variables discrètes."""
    
    _, axes = plt.subplots(n_rows, n_cols, figsize=kwargs.get("figsize", (28, 4)))
    
    if len(cols) > 1:
        axes = axes.flatten()
        
    for i, col in enumerate(cols):
        prop_df = pd.crosstab(df[col], df[target])
        prop_df = prop_df.div(prop_df.sum(axis=1), axis=0)
        
        ax = axes[i] if len(cols) > 1 else axes
        
        # Couleurs des classes 0 (vert) et 1 (rouge)
        colors_01 = kwargs.get("colors_01", sns.color_palette()[1:3])
        
        bar = prop_df.plot(kind="bar", stacked=True, legend=False, color=colors_01, alpha=0.8, ax=ax)
        
        # Ajout des pourcentages sur les barres
        for i, p in enumerate(bar.patches):
            if i < len(bar.patches) / 2:
                bar.text(
                    p.get_x() + p.get_width() / 2,
                    p.get_y() + p.get_height() / 2,
                    f"{p.get_height():.0%}",
                    ha="center", va="center",
                    c="w", weight="bold",
                )
                
        bar.set_xlabel(col, weight="bold")
        bar.set_xticks(bar.get_xticks())
        
        xticklabels = ["\n".join(label.get_text().split()) for label in ax.get_xticklabels()]
        bar.set_xticklabels(xticklabels, rotation=0)
        
        bar.set_yticks([])
        bar.set_ylabel("")

    ax = axes[0] if len(cols) > 1 else axes
    ax.legend(title=target, bbox_to_anchor=(0, 1))
    
    size = kwargs.get("fig_title_font_size", 16)
    
    plt.suptitle(f"Taux de churn selon les variables discrètes, Segment = {segment}", size=size, weight="bold")
    plt.tight_layout()
    
    st.pyplot(plt)


def plot_cont_vs_cont(df, cols, segment="All", target="Churn Value", **kwargs):
    """Représente les relations entre plusieurs variables continues."""
    
    # Couleurs des classes 0 (vert) et 1 (rouge)
    colors_01 = kwargs.get("colors_01", sns.color_palette()[1:3])
    
    g = sns.PairGrid(df, hue=target, vars=cols, palette=colors_01, corner=True, diag_sharey=False)
    g.map_diag(sns.histplot, bins=100)
    g.map_lower(sns.scatterplot, s=10, alpha=0.8)
    g.add_legend(bbox_to_anchor=(0.92, 0.95))
    
    plt.suptitle(f"Relations entre les variables continues, Segment = {segment}", y=1.01, size=14, weight="bold")
    
    st.pyplot(plt)

    
def plot_cont_vs_target(df, cols, target_value, segment="All", target="Churn Value", **kwargs):
    """Représente la distribution des variables continues selon une valeur fixée de la variable cible."""
    
    _, axes = plt.subplots(1, len(cols), figsize=kwargs.get("figsize", (18, 3)))

    # Couleurs des classes 0 (vert : green) et 1 (rouge : reg)
    color_g, color_r = kwargs.get("colors_01", sns.color_palette()[1:3])
    color = color_g if (target_value == 0) else color_r
    
    for i, col in enumerate(cols):
        axes[i].hist(df.loc[df[target] == target_value, col], bins=100, color=color, alpha=0.8, label=target_value)
        axes[i].set_title(col)

    plt.suptitle(f"Distribution des variables continues\nSegment = {segment}, Churn Value = {target_value}", y=0.95, size=12, weight="bold")
    plt.tight_layout()
    
    st.pyplot(plt)
