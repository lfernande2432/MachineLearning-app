import streamlit as st
import altair as alt
import pandas as pd
import ast
import numpy as np
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
    df_leaderboard_testset['model_simplified'] = df_leaderboard_testset['model'].str.split('_').str[0]
    modelos = sorted(df_leaderboard_testset['model_simplified'].unique())
    modelo_sel = st.selectbox("Selecciona un modelo:", modelos)
    df_modelo = df_leaderboard_testset[df_leaderboard_testset['model_simplified'] == modelo_sel].copy()

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
    metricas = ["roc_auc", "average_precision", "f1_macro", "balanced_accuracy", "log_loss", "precision", "recall"]
    metrica_sel = st.selectbox("Selecciona una métrica para graficar:", metricas, key="so3_metricas")

    if metrica_sel in df_modelo.columns and "hyperparameters" in df_modelo.columns:
        # Agrupar por configuraciones únicas de hyperparameters
        df_unique = df_modelo.groupby("hyperparameters").agg(
            mean_metric=(metrica_sel, "mean"),
            std_metric=(metrica_sel, "std")
        ).reset_index()

        # Crear IDs cortos para la gráfica
        df_unique["config_id"] = [f"Config_{i+1}" for i in range(len(df_unique))]
        # Encontrar la mejor configuración (max para la mayoría de métricas, min para log_loss)
        if metrica_sel == "log_loss":
            best_config_id = df_unique.loc[df_unique["mean_metric"].idxmin()]["config_id"]
        else:
            best_config_id = df_unique.loc[df_unique["mean_metric"].idxmax()]["config_id"]

        # Añadir un punto o tick para la mejor configuración
        best_config_marker = alt.Chart(df_unique[df_unique["config_id"] == best_config_id]).mark_text(
            text='✓',       # símbolo de check
            color='green',
            fontSize=30,    # tamaño del check
            dy=-10          # desplazamiento vertical hacia arriba de la barra
        ).encode(
            x=alt.X("config_id:N", sort="-y"),
            y=alt.Y("mean_metric:Q", scale=alt.Scale(zero=False)),
            tooltip=["config_id", "mean_metric:Q"]
        )
        # Gráfica
        bars = alt.Chart(df_unique).mark_bar().encode(
            y=alt.Y("mean_metric:Q", title=f"Media de {metrica_sel}",scale=alt.Scale(zero=False)),
            x=alt.X("config_id:N", sort="-y", title="Configuración"),
            tooltip=["config_id", "hyperparameters", "mean_metric:Q", "std_metric:Q"],
            color=alt.condition(
                alt.datum.config_id == best_config_id,
                alt.value("green"),  # Color especial para la mejor configuración
                alt.value("steelblue")  # Color estándar para las demás
            )
        )

        # Combina los dos gráficos usando alt.layer()
        ranking = alt.layer(bars, best_config_marker).properties(
            width=700,
            height=400,
            title=f"Ranking de configuraciones según {metrica_sel}"
        ).interactive()

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
        .mark_tick()
        .encode(
            x=alt.X("importance:Q", title="Importancia"),
            y=alt.Y("feature:N", sort=alt.EncodingSortField(field="importance", op="max", order="descending"), title="Variable"),
            color=alt.Color(
                "importance:Q",
                scale=alt.Scale(scheme='redblue', domainMid=0),
                legend=None
            ),
            tooltip=["feature", "importance"]
        )
        .properties(
            width=700,
            height=700,
            title=f"Importancia de variables para el modelo {modelo_sel}"
        )
    )

    st.altair_chart(barra_importancia, use_container_width=True)


def SO5(df_leaderboard_testset, df_metrics):

    # Selección del modelo
    modelos = sorted(df_leaderboard_testset["model"].unique())
    modelo_sel = st.selectbox("Selecciona un modelo", modelos)
    df = df_leaderboard_testset[df_leaderboard_testset["model"] == modelo_sel].copy()
    
    # Obtener y ordenar los valores únicos de la semilla como números
    semillas = sorted(df["seed"].unique())

    semillas_sel = st.multiselect(
        "Selecciona las semillas",
        semillas,
        default=[0, 7]
    )

    # Filtrar el DataFrame
    df = df[df["seed"].isin(semillas_sel)]
    df['seed'] = df['seed'].astype(str)
    if df.empty:
        st.warning("No hay datos para mostrar con las semillas seleccionadas.")
        return

    # ---- Preparar df_metrics ----
    df_metrics['model'] = np.where(
    df_metrics['Replica'] == 0,
    df_metrics['ModelBase'],
    df_metrics['ModelBase'] + '_r' + df_metrics['Replica'].astype(str)
)
    df_line = df_metrics[df_metrics["model"] == modelo_sel].copy()

    # ---- Crear una gráfica por cada nF ----
    charts = []
    for nF in sorted(df["nF"].unique()):
        df_nF = df[df["nF"] == nF]
        df_line_nF = df_line[df_line["nF"] == nF]

        base = alt.Chart(df_nF).mark_point(size=50).encode(
            x=alt.X('fold:N', title='Fold'),
            y=alt.Y('roc_auc', title='ROC AUC (fold)'),
            color=alt.Color('seed', legend=alt.Legend(title="Semilla")),
            tooltip=['fold', 'roc_auc', 'seed']
        ).transform_filter(
            alt.datum.fold <= alt.datum.nF
        ).properties(
            width=200,
            height=200
        )
        roc_mean = df_line_nF["Roc_auc"].mean()  # media de todos los valores
        line = alt.Chart(pd.DataFrame({"Roc_auc": [roc_mean]})).mark_rule(
            color="black",
            strokeDash=[4, 2]   # línea discontinua
        ).encode(
            y="Roc_auc:Q",
            tooltip=["Roc_auc:Q"]
        )

        charts.append(
            alt.layer(base, line).properties(title=f"nF = {nF}")
        )

    # Concatenar las gráficas verticalmente
    final_chart = alt.vconcat(*charts).properties(
        title=f"Rendimiento por nF para el modelo: {modelo_sel}"
    )

    st.altair_chart(final_chart, use_container_width=True)
def convertir_a_predicciones(df, threshold=0.5):
    # Buscar columnas de probabilidades
    prob_cols = [c for c in df.columns if c.startswith("testPredProba_")]
    
    # Crear columnas de predicciones binarias
    for col in prob_cols:
        pred_col = col.replace("testPredProba_", "pred_")
        df[pred_col] = (df[col] >= threshold).astype(int)
    
    return df

def graficar_predicciones_por_instancia(df_pred):
    # Columnas de predicción
    pred_cols = [c for c in df_pred.columns if c.startswith("pred_")]
    
    # Transformar a formato largo
    df_melt = df_pred.melt(
        id_vars=["etiq-id", "ED_2Clases"],
        value_vars=pred_cols,
        var_name="Modelo",
        value_name="Prediccion"
    )
    df_melt["Modelo"] = df_melt["Modelo"].str.replace("pred_", "")
    
    # Columna de error
    df_melt["Error"] = (df_melt["Prediccion"] != df_melt["ED_2Clases"]).astype(int)
    
    # Selección interactiva de instancia
    instancia_seleccionada = st.selectbox("Selecciona la instancia (etiq-id):", df_melt["etiq-id"].unique())
    df_filtrado = df_melt[df_melt["etiq-id"] == instancia_seleccionada]
    
    # Contar cuántas veces cada modelo predice cada etiqueta
    counts = df_filtrado.groupby(["Modelo", "Prediccion"]).size().reset_index(name="Freq")
    df_filtrado = df_filtrado.merge(counts, on=["Modelo", "Prediccion"], how="left")
    
    # Círculos: predicciones de los modelos con degradado por frecuencia
    pred_chart = alt.Chart(df_filtrado).mark_circle(size=150).encode(
        x=alt.X('Modelo:N', title='Modelo'),
        y=alt.Y('Prediccion:O', title='Predicción (0/1)', sort=[0,1]),
        color=alt.Color('Error:N', title='Predicciones', scale=alt.Scale(domain=[0,1], range=['green','red'])),
        opacity=alt.Opacity('Freq:Q', scale=alt.Scale(range=[0.2,1]), legend=None),  # sin leyenda
        tooltip=['etiq-id', 'Modelo', 'Prediccion', 'ED_2Clases', 'Error', 'Freq']
    )
    
    # Solo el gráfico de predicciones (sin línea)
    chart = pred_chart.properties(
        width=700,
        height=300,
        title=f'Predicciones para la instancia {instancia_seleccionada}'
    ).interactive()
    
    return chart


def SO6(df_test_pred, mejores_modelos, threshold=0.5):
    # 1. Filtrar columnas de probabilidades de los mejores modelos
    pattern = "|".join([f"testPredProba_{m}(_.*)?$" for m in mejores_modelos])
    prob_cols = df_test_pred.filter(regex=pattern)

    # 2. Convertir probabilidades a predicciones binarias
    df_pred = prob_cols.apply(lambda col: (col >= threshold).astype(int))
    
    # 3. Renombrar columnas: testPredProba_ -> pred_
    df_pred.columns = [c.replace("testPredProba_", "pred_") for c in df_pred.columns]

    # 4. Añadir las columnas 'etiq-id' y 'ED_2Clases'
    df_pred = pd.concat([df_pred, df_test_pred[["etiq-id", "ED_2Clases"]]], axis=1)

    # Interacción para seleccionar el umbral
        # Columnas de predicción
    pred_cols = [c for c in df_pred.columns if c.startswith("pred_")]

    # Slider interactivo: mínimo de modelos que deben fallar
    min_modelos_fallidos = st.slider(
        "Número mínimo de modelos con predicción distinta a la etiqueta",
        min_value=1,
        max_value=len(pred_cols),
        value=1
    )

    # Contar por fila cuántos modelos difieren de la etiqueta
    errores_por_fila = (df_pred[pred_cols] != df_pred["ED_2Clases"].values[:, None]).sum(axis=1)

    # Filtrar filas donde al menos 'min_modelos_fallidos' modelos fallan
    df_filtrado = df_pred[errores_por_fila >= min_modelos_fallidos]
    st.altair_chart(graficar_predicciones_por_instancia(df_filtrado), use_container_width=True)


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
        df_leaderboard_testset_mejores = df_leaderboard_testset[df_leaderboard_testset["model"].str.split('_').str[0].isin(mejores_modelos)]
        SO3(df_leaderboard_testset_mejores)
    
    
    # --- Objetivo 4: Variables más relevantes ---

    with st.expander("**3.4. Selección de las variables más relevantes (SO4)**", expanded=False):
        df_feature_importance_mejores=df_feature_importance[df_feature_importance["model"].isin(mejores_modelos)]
        SO4(df_feature_importance_mejores)
    
    # --- Objetivo 5: Análisis detallado de la métrica ROC AUC por fold  ---
    with st.expander("**3.5. Análisis detallado de la métrica ROC AUC por fold (SO5)**", expanded=False):
        SO5( df_leaderboard_testset,df_metrics)
    # --- Objetivo 6: Análisis de errores y casos difíciles  ---
    with st.expander("**3.6. Análisis de errores y casos difíciles (SO6)**", expanded=False):
        st.markdown("""
        El análisis de errores y casos difíciles nos ayuda a comprender mejor las limitaciones del modelo y las áreas donde puede necesitar mejoras. Al identificar patrones en los errores, podemos ajustar el modelo o los datos para abordar estos desafíos específicos, mejorando así la precisión y la confiabilidad del modelo en situaciones del mundo real.
        """)
        SO6(df_test_pred,mejores_modelos)
    return mejores_modelos


