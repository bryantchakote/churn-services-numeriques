import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_disc_variables(df, cols, n_rows=4, n_cols=4, segment="All", **kwargs):
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
        title_text=kwargs.get("fig_title", f"Distribution des variables discrètes, Segment = {segment}"),
        showlegend=False,
    )
    
    fig.update_yaxes(showticklabels=False)
    
    st.plotly_chart(fig)


def plot_disc_vs_target(df, cols, n_rows=2, n_cols=4, segment="All", target="Churn Value", **kwargs):
    """Représente le taux de churn selon les différentes modalités des variables discrètes."""
    
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=kwargs.get("subplot_titles", cols))
    
    colors = px.colors.qualitative.Plotly[1:3][::-1]
    
    for i, col in enumerate(cols):
        row, col_pos = divmod(i, n_cols)
        row += 1
        
        prop_df = df.groupby([col, target]).size().div(df.groupby(col).size() / 100).reset_index(name="Percentage").round()
        
        for j in [0, 1]:
            sub_prop_df = prop_df[prop_df[target] == j]
                        
            bar_trace = go.Bar(
                x=["<br>".join(val.split()) for val in sub_prop_df[col]] if kwargs.get("br_xtickslabels") else sub_prop_df[col],
                y=sub_prop_df["Percentage"],
                marker_color=colors[j],
                name="Stayed" if (j == 0) else "Exited",
                text=sub_prop_df['Percentage'].astype(str).str.split(".").str[0] + "%" if j==0 else "",
                showlegend=(i == 0),
                opacity=0.8,
            )
            
            fig.add_trace(bar_trace, row=row, col=col_pos + 1)
    
    fig.update_layout(
        height=kwargs.get("height", 400), 
        width=kwargs.get("width", 1200), 
        title_text=kwargs.get("fig_title", f"Taux de churn selon les variables discrètes, Segment = {segment}"),
        barmode="stack",
    )
    
    fig.update_yaxes(showticklabels=False)
    
    st.plotly_chart(fig)


def plot_cont_vs_cont(df, cols, segment="All", target="Churn Value", **kwargs):
    """Représente les relations entre plusieurs variables continues."""
    
    colors = {val: color for val, color in zip([1, 0], px.colors.qualitative.Plotly[1:3])}

    fig = go.Figure(
        data=go.Splom(
            dimensions=[{"label": col, "values": df[col].tolist()} for col in df[cols].columns],
            marker=dict(
                color=df[target].map(colors),
                size=2,
            ),
            opacity=0.8,
            diagonal_visible=False,
            showupperhalf=False,
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Relations entre les variables continues, Segment = {segment}",
        ),
        height=kwargs.get("height", 600),
        width=kwargs.get("width", 600),
    )

    # Ajouter un plot vide pour avoir la légende
    for val, color in colors.items():
        print(val, color)
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                color=color,
            ),
            name="Stayed" if (val == 0) else "Exited",
        ))
    
    st.plotly_chart(fig)


def plot_cont_vs_target(df, cols, target_value, segment="All", target="Churn Value", **kwargs):
    """Représente la distribution des variables continues selon une valeur fixée de la variable cible."""
    
    color_id = 1 if target_value == 1 else 2
    color = px.colors.qualitative.Plotly[color_id]
    
    fig = make_subplots(rows=1, cols=len(cols), subplot_titles=["<br>".join(col.split()) for col in cols])
    
    for i, col in enumerate(cols):
        fig.add_trace(
            go.Histogram(
                x=df.loc[df[target] == target_value, col],
                nbinsx=100,
                marker_color=color,
                opacity=0.8,
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title_text=f"Distribution des variables continues, Segment = {segment}, Churn Value = {target_value}",
        showlegend=False,
        height=kwargs.get("height", 300),
        width=kwargs.get("width", 1200),
    )
    
    st.plotly_chart(fig)
