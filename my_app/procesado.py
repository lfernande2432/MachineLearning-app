import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def mejor_modelo(df_metrics):
    metric_cols = [
        "Balanced_accuracy","Precision_macro", "Precision_weighted",
        "Recall_macro", "Recall_weighted", "F1_macro", "F1_weighted","Roc_auc", "PR_auc"
    ]
    st.title("Comparación de modelos por métricas")
    metrica_sel = st.selectbox("Selecciona la métrica a visualizar", metric_cols)

    # Agrupar por modelo y calcular la media de la métrica seleccionada
    df_grouped = df_metrics.groupby("ModelBase")[metrica_sel].mean().sort_values(ascending=False)

    # Crear gráfico de barras con matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    df_grouped.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_ylabel(metrica_sel)
    ax.set_xlabel("Modelo")
    ax.set_title(f"Precisión promedio por modelo ({metrica_sel})")
    plt.xticks(rotation=45, ha='right')
    # Ajustar la escala del eje Y para resaltar diferencias
    ymin = max(0, df_grouped.min() - 0.02)
    ymax = min(1, df_grouped.max() + 0.02)
    ax.set_ylim(ymin, ymax)
    plt.tight_layout()

    st.pyplot(fig)
def procesar(df_feature_importance, df_metrics, df_test_pred, df_feature_importance_folds, df_leaderboard_testset):
    mejor_modelo(df_metrics)
    return "Procesamiento completado"