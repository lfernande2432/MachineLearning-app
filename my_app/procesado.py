import streamlit as st
import altair as alt
import pandas as pd
import ast

def S01(df_metrics):
    metric_cols = [
        "Balanced_accuracy", "Precision_macro", "Precision_micro", "Precision_weighted",
        "Recall_macro", "Recall_micro","Recall_weighted", "F1_macro", "F1_weighted",
        "Roc_auc", "PR_auc"
    ]

    metrica_seleccionada = st.selectbox(
        "Selecciona una métrica:",
        options=metric_cols, 
        key='metric_cols'
    )
    chart = alt.Chart(df_metrics).mark_boxplot(extent='min-max').encode(
        x='ModelBase:O',
        y=alt.Y(metrica_seleccionada + ':Q', scale=alt.Scale(zero=False))
    )
    st.altair_chart(chart, use_container_width=True)
    mejores_modelos = [
        'RandomForest', 'RandomForestEntr', 'RandomForestGini',
        'CatBoost', 'LightGBMLarge'
    ]
    st.markdown(
        "Los modelos con mayor rendimiento y consistencia son:\n\n" +
        "\n".join([f"- {modelo}" for modelo in mejores_modelos])
    )
    return mejores_modelos
    
def SO2(df_metrics):
    semillas = sorted(df_metrics["Seed"].unique())
    st.markdown(f"Las semillas utilizadas para los diferentes modelos son: {', '.join(map(str, semillas))}")
    metric_cols = ["F1_weighted","Roc_auc", "PR_auc"]
    
    modelo_sel = st.selectbox("Selecciona un modelo:", df_metrics["ModelBase"].unique())
    metrica_sel = st.selectbox("Selecciona una métrica:", metric_cols, index=metric_cols.index("Roc_auc"))

    # Filtrar datos del modelo
    df_modelo = df_metrics[df_metrics["ModelBase"] == modelo_sel].copy()

    # Calcular la media global
    media_global = df_modelo[metrica_sel].mean()
    df_media = pd.DataFrame({metrica_sel: [media_global]})

    # Boxplot por semilla con escala personalizada en el eje y
    boxplot = (
        alt.Chart(df_modelo)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("Seed:N", title="Semilla"),
            y=alt.Y(
                f"{metrica_sel}:Q",
                title=metrica_sel,
                scale=alt.Scale(domain=[df_modelo[metrica_sel].min() * 0.98, df_modelo[metrica_sel].max() * 1.02])
            ),
            color=alt.Color("Seed:N", legend=None)
        )
    )

    # Línea roja del promedio global
    linea_media = (
        alt.Chart(df_media)
        .mark_rule(color="red", size=3)
        .encode(y=f"{metrica_sel}:Q")
    )

    st.altair_chart(
        (boxplot + linea_media).properties(
            width=700,
            height=400,
            title=f"{modelo_sel} - Distribución de {metrica_sel} por semilla (línea roja = promedio global)"
        ),
        use_container_width=True
    )


    

def SO3(df_leaderboard_testset):

    modelos = df_leaderboard_testset["model"].unique()
    modelo_sel = st.selectbox("Selecciona un modelo:", modelos)
    df_modelo = df_leaderboard_testset[df_leaderboard_testset["model"] == modelo_sel]
    # Valores distintos de hyperparameters
    st.markdown(f"**Número de configuraciones distintas de hyperparameters:** {df_modelo['hyperparameters'].nunique()}")
    # Mostrar todas las columnas
    pd.set_option('display.max_columns', None)
    # Parsear hyperparameters
    def parse_hyperparams(hp):
        if isinstance(hp, dict):
            parsed = hp
        else:
            try:
                parsed = ast.literal_eval(hp)
            except Exception:
                parsed = {}
        # Prefijar las claves con 'hyperparameter_'
        return {f"hyperparameter_{k}": v for k, v in parsed.items()}

    # Parsear hyperparameters y expandir a columnas
    hiperparams_df = df_modelo["hyperparameters"].apply(parse_hyperparams).apply(pd.Series)
    # Eliminar columna 'hyperparameters' y unir las columnas parseadas
    df_modelo = pd.concat([df_modelo.reset_index(drop=True), hiperparams_df.reset_index(drop=True)], axis=1)

    # Imprimir dataframes
    st.dataframe(df_modelo)

    # Número de valores distintos de los hiperparámetros
    st.markdown("**Número de valores distintos por hiperparámetro:**")
    hyperparam_cols = [col for col in df_modelo.columns if col.startswith("hyperparameter_")]
    distinct_counts = df_modelo[hyperparam_cols].apply(lambda col: col.nunique())
    st.dataframe(distinct_counts.rename("Valores distintos"))

    # Gráfica: métrica en eje y y columna hyperparameters en eje x (sin parsear)
    metricas = ["score_test", "balanced_accuracy", "f1", "f1_macro",
                "f1_micro", "roc_auc", "average_precision", "precision", "recall",
                "log_loss"]
    metrica_sel = st.selectbox("Selecciona una métrica para graficar:", metricas, key="so3_metricas")
    if metrica_sel in df_modelo.columns and "hyperparameters" in df_modelo.columns:
        ranking = alt.Chart(df_modelo).mark_bar().encode(
            y=alt.Y(f"mean({metrica_sel}):Q", title=f"Media de {metrica_sel}"),
            x=alt.X("hyperparameters:N", sort="-x", title="Hyperparameters"),
            tooltip=["hyperparameters", f"mean({metrica_sel}):Q", f"stddev({metrica_sel}):Q"]
        ).properties(
            width=700,
            height=400,
            title=f"Ranking de configuraciones según {metrica_sel}"
        )

        st.altair_chart(ranking, use_container_width=True)





def procesar(df_feature_importance, df_metrics, df_test_pred, df_feature_importance_folds, df_leaderboard_testset):

    # --- Objetivo 1: Modelos más robustos ---
    with st.expander("**3.1. Selección de los mejores modelos (SO1)**", expanded=False):
        st.markdown("""
        El análisis de los diagramas de cajas y bigotes nos permitirá analizar los mejores modelos en términos de rendimiento y consistencia.
        """)
        mejores_modelos=S01(df_metrics)


    # --- Objetivo 2: Mejores semillas ---
    with st.expander("**3.2. Selección de las mejores semillas (SO2)**", expanded=False):
        df_metrics_mejores = df_metrics[df_metrics["ModelBase"].isin(mejores_modelos)]
        SO2(df_metrics_mejores)

    # --- Objetivo 3: Mejores configuraciones ---
    with st.expander("**3.3. Selección de las mejores configuraciones (SO3)**", expanded=False):
        SO3(df_leaderboard_testset)
    
    
    # --- Objetivo 4: Variables más relevantes ---

    with st.expander("**3.4. Selección de las variables más relevantes (SO4)**", expanded=False):
        st.markdown("""
        El análisis de importancia de variables nos permite identificar qué características tienen el mayor impacto en las predicciones del modelo. Al centrarnos en las variables más relevantes, podemos simplificar el modelo, mejorar su interpretabilidad y potencialmente aumentar su rendimiento al reducir el ruido.
        """)
    
    # --- Objetivo 5: Análisis detallado de la métrica ROC AUC por fold  ---
    with st.expander("3.5. Análisis detallado de la métrica ROC AUC por fold (SO5)", expanded=False):
        st.markdown("""
        El análisis de la métrica ROC AUC por fold nos permite evaluar la estabilidad y consistencia del rendimiento del modelo a través de diferentes particiones de los datos. Al examinar cómo varía esta métrica en cada fold, podemos identificar posibles problemas de sobreajuste y asegurar que el modelo generalice bien a datos no vistos.
        """)
    # --- Objetivo 6: Análisis de errores y casos difíciles  ---
    with st.expander("3.6. Análisis de errores y casos difíciles (SO6)", expanded=False):
        st.markdown("""
        El análisis de errores y casos difíciles nos ayuda a comprender mejor las limitaciones del modelo y las áreas donde puede necesitar mejoras. Al identificar patrones en los errores, podemos ajustar el modelo o los datos para abordar estos desafíos específicos, mejorando así la precisión y la confiabilidad del modelo en situaciones del mundo real.
        """)
    # st.subheader("3.3. Selección de las mejores semillas (SO1)")
    #...

