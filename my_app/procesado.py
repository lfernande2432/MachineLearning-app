import streamlit as st
import pandas as pd
import altair as alt

def mejor_modelo(df_metrics):

# Columnas métricas que quieres graficar
    metric_cols = [
        "Balanced_accuracy", "Precision_macro", "Precision_micro", "Precision_weighted",
        "Recall_macro", "Recall_micro", "Recall_weighted", "F1_weighted", "F1_macro",
        "Precision_0", "Recall_0", "Precision_1", "Recall_1", "Roc_auc", "PR_auc", "Roc_auc_byFold"
    ]

    # Transformar a formato largo para Altair
    df_long = df_metrics.melt(
        id_vars=["Tipo", "Seed", "ModelName"],
        value_vars=metric_cols,
        var_name="Métrica",
        value_name="Valor"
    )

    st.title("Comparación de modelos por métricas")

    # Selector para la métrica
    metrica_sel = st.selectbox("Selecciona la métrica a visualizar", metric_cols)

    # Filtramos el df por la métrica seleccionada
    df_plot = df_long[df_long["Métrica"] == metrica_sel]

    # Gráfico Altair: comparar distribución de la métrica por modelo
    chart = alt.Chart(df_plot).mark_boxplot().encode(
        x=alt.X("ModelName:N", sort='-y', title="Modelo"),
        y=alt.Y("Valor:Q", title=metrica_sel),
        color="ModelName:N",
        tooltip=["ModelName", "Valor"]
    ).properties(
        width=700,
        height=400,
        title=f"Distribución de {metrica_sel} por modelo"
    ).interactive()

    st.altair_chart(chart, use_container_width=True)
def procesar(df_feature_importance, df_metrics, df_test_pred, df_feature_importance_folds, df_leaderboard_testset):
    a=1
    st.subheader("📁 Estructura de los datos")