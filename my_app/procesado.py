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
    st.markdown(f"Número de configuraciones distintas de hyperparameters: {df_modelo['hyperparameters'].nunique()}")

    # Parsear hyperparameters
    def parse_hyperparams(hp):
        if isinstance(hp, dict):
            parsed = hp
        else:
            try:
                parsed = ast.literal_eval(hp)
            except Exception:
                parsed = {}
        return {f"hyperparameter_{k}": v if v is not None else "None" for k, v in parsed.items()}

    hiperparams_df = df_modelo["hyperparameters"].apply(parse_hyperparams).apply(pd.Series)
    df_modelo = pd.concat([df_modelo.reset_index(drop=True), hiperparams_df.reset_index(drop=True)], axis=1)

    # Valores distintos por hiperparámetro
    st.markdown("*Valores distintos por hiperparámetro:*")
    hyperparam_cols = [col for col in df_modelo.columns if col.startswith("hyperparameter_")]
    distinct_summary = (
        df_modelo[hyperparam_cols]
        .fillna("None")
        .agg([lambda x: x.nunique(), lambda x: list(x.unique())])
        .T
        .reset_index()
    )
    distinct_summary.columns = ["Hiperparámetro", "Nº valores distintos", "Valores distintos"]

    # Convertir la columna de listas a strings
    distinct_summary["Valores distintos"] = distinct_summary["Valores distintos"].apply(lambda x: ", ".join(map(str, x)))

    st.dataframe(distinct_summary)

    # Selección de métrica
    metricas = ["score_test", "balanced_accuracy", "f1", "f1_macro",
                "f1_micro", "roc_auc", "average_precision", "precision", "recall",
                "log_loss"]
    metrica_sel = st.selectbox("Selecciona una métrica para graficar:", metricas, key="so3_metricas")

    if metrica_sel in df_modelo.columns and "hyperparameters" in df_modelo.columns:
        # Agrupar por configuraciones únicas de hyperparameters
        df_unique = df_modelo.groupby("hyperparameters").agg(
            mean_metric=(metrica_sel, "mean"),
            std_metric=(metrica_sel, "std")
        ).reset_index()

        # Crear IDs cortos para la gráfica
        df_unique["config_id"] = [f"Config_{i+1}" for i in range(len(df_unique))]

        # Gráfica
        ranking = alt.Chart(df_unique).mark_bar().encode(
            y=alt.Y("mean_metric:Q", title=f"Media de {metrica_sel}"),
            x=alt.X("config_id:N", sort="-y", title="Configuración"),
            tooltip=["config_id", "hyperparameters", "mean_metric:Q", "std_metric:Q"]
        ).properties(
            width=700,
            height=400,
            title=f"Ranking de configuraciones según {metrica_sel}"
        )

        st.altair_chart(ranking, use_container_width=True)

        # Tabla de correspondencia
        st.markdown("*Correspondencia entre identificadores y configuraciones de hyperparameters:*")
        st.dataframe(df_unique[["config_id", "hyperparameters"]])

def SO4(df_feature_importance_mejores):

    # Filtrar filas
    df_nv8 = df_feature_importance_mejores[df_feature_importance_mejores["nV"] == 8]
    df_nv19 = df_feature_importance_mejores[df_feature_importance_mejores["nV"] == 19]

    # Features únicas
    features_nv8 = df_nv8["feature"].unique()
    features_nv19 = df_nv19["feature"].unique()

    # Todas las features únicas de ambos conjuntos
    all_features = sorted(set(features_nv8).union(features_nv19))

    # Crear DataFrame indicando presencia/ausencia
    df_comparacion = pd.DataFrame({
        "feature": all_features,
        "en_nv8": [f in features_nv8 for f in all_features],
        "en_nv19": [f in features_nv19 for f in all_features]
    })

    # Mostrar tabla en Streamlit
    st.markdown("*Tabla de presencia de features en nV=8 y nV=19*")
    st.dataframe(df_comparacion)

    modelos = df_feature_importance_mejores["model"].unique()
    modelo_sel = st.selectbox("Selecciona un modelo para ver su importancia de variables:", modelos, key="so4_modelos")
    df_modelo = df_feature_importance_mejores[df_feature_importance_mejores["model"] == modelo_sel]
    # Checkbox para seleccionar mostrar solo nv8 o todas
    solo_nv8 = st.checkbox("Mostrar solo features de nV=8", value=False)

    # Filtrar según el checkbox
    if solo_nv8:
        df_grafico = df_modelo[df_modelo["nV"] == 8]
    else:
        df_grafico = df_modelo.copy()
    # Gráfica de barras horizontales
    barra_importancia = (
        alt.Chart(df_grafico)
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Importancia"),
            y=alt.Y("feature:N", sort='-x', title="Variable"),
            color=alt.Color("importance:Q", scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=["feature", "importance"]
        )
        .properties(
            width=700,
            height=700,
            title=f"Importancia de variables para el modelo {modelo_sel}"
        )
    )

    st.altair_chart(barra_importancia, use_container_width=True)

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
        df_feature_importance_mejores=df_feature_importance[df_feature_importance["model"].isin(mejores_modelos)]
        SO4(df_feature_importance_mejores)
    
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

