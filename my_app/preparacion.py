import streamlit as st
import pandas as pd
import altair as alt
def preparar_datos(df_feature_importance, df_metrics, df_test_pred, df_feature_importance_folds, df_leaderboard_testset):

    st.markdown("""
    Este trabajo parte de numerosos ficheros CSV organizados en carpetas que codifican la configuración de cada experimento (por ejemplo: número de variables, folds y semilla).  
    Antes del análisis, se ha realizado una fase de **preparación, unificación y limpieza** de los datos para facilitar su análisis posterior.
    """)

    st.subheader("🗄️ Fuente de los datos")
    st.markdown("""
    Los datos están organizados en dos carpetas (según el número de variables: 8 o 19) y contienen cuatro tipos principales de archivos:
    - **Metrics**: Métricas promedio de evaluación por modelo.
    - **FeatureImportance**: Importancia de las variables para cada modelo.
    - **TestPredCV**: Predicciones de validación cruzada (out-of-fold).
    - **Leaderboard_testset**: Métricas por fold y parámetros de los modelos.
    
    Algunos metadatos como `semilla`, `fold`, `número de variables`, etc., no están en el contenido del archivo sino en el nombre de la carpeta, por lo que han sido extraídos de ahí.
    """)

    st.subheader("📁 Estructura de los datos")
    st.markdown("""
    Los archivos procesados se han agrupado en los siguientes `DataFrames`:
    - **`df_metrics`**: Contiene medidas como precisión, recall, F1, AUC y más, por modelo.
    - **`df_feature_importance`**: Importancia de características usadas en los modelos.
    - **`df_test_pred`**: Predicciones sobre los conjuntos de prueba (por modelo y réplica).
    - **`df_feature_importance_folds`**: Importancia de características para cada fold.
    - **`df_leaderboard_testset`**: Métricas detalladas por fold y parámetros del modelo.
    """)

    st.subheader("🧹 Limpieza y normalización")
    st.markdown("""
    Durante la carga de datos se ha realizado:
    - Eliminar registros/variables con demasiados valores nulos.
    """)

    with st.expander("Número de valores nulos y porcentaje"):
        for df_name, df in [("df_feature_importance", df_feature_importance),
                        ("df_metrics", df_metrics),
                        ("df_test_pred", df_test_pred),
                        ("df_feature_importance_folds", df_feature_importance_folds),
                        ("df_leaderboard_testset", df_leaderboard_testset)]:
            st.subheader(f"{df_name}")
            nulos_count = df.isnull().sum()
            nulos_pct = df.isnull().mean()
            resumen_nulos = pd.concat([nulos_count, nulos_pct], axis=1)
            resumen_nulos.columns = ["Nulos", "% Nulos"]
            resumen_nulos_con_nulos = resumen_nulos[resumen_nulos["Nulos"] > 0]
            if resumen_nulos_con_nulos.empty:
                st.write("No hay valores nulos.")
            else:
                st.dataframe(resumen_nulos_con_nulos)

    # Eliminar las columnas que contienen muchos valores nulos para df_leaderboard_testset
    columnas_a_eliminar = ['compile_time', 'child_model_type','child_hyperparameters', 'child_hyperparameters_fit', 'child_ag_args_fit']
    df_leaderboard_testset = df_leaderboard_testset.drop(columns=columnas_a_eliminar)

    with st.expander("Estudio de los registros con valores nulos para df_test_pred"):

    #Estudio de los registros con valores nulos para df_test_pred

    # Detectar columnas con valores nulos
        cols_con_nulos = df_test_pred.columns[df_test_pred.isnull().any()]

        # Filtrar filas que tengan al menos un nulo en esas columnas
        filas_con_nulos = df_test_pred[df_test_pred[cols_con_nulos].isnull().any(axis=1)]

        # Mostrar esas filas para que puedas analizarlas
        st.dataframe(filas_con_nulos)
        st.write(f"Total de filas con nulos en las columnas seleccionadas: {filas_con_nulos.shape[0]}")
    # Eliminar filas con cualquier valor nulo
    df_test_pred = df_test_pred.dropna()
    st.markdown("""
    - Eliminar columnas no relevantes, repetitas o con valor único.
    """)
    with st.expander("Número de valores distintos por variable"):
        st.markdown("###  df_feature_importance")
        unique_values = df_feature_importance.nunique().sort_values(ascending=False)
        st.dataframe(unique_values.reset_index().rename(columns={"index": "Variable", 0: "Valores distintos"}))

        st.markdown("###  df_metrics")
        unique_values = df_metrics.nunique().sort_values(ascending=False)
        st.dataframe(unique_values.reset_index().rename(columns={"index": "Variable", 0: "Valores distintos"}))

        st.markdown("###  df_test_pred")
        unique_values = df_test_pred.nunique().sort_values(ascending=False)
        st.dataframe(unique_values.reset_index().rename(columns={"index": "Variable", 0: "Valores distintos"}))

        st.markdown("###  df_feature_importance_folds")
        unique_values = df_feature_importance_folds.nunique().sort_values(ascending=False)
        st.dataframe(unique_values.reset_index().rename(columns={"index": "Variable", 0: "Valores distintos"}))

        st.markdown("###  df_leaderboard_testset")
        unique_values = df_leaderboard_testset.nunique().sort_values(ascending=False)
        st.dataframe(unique_values.reset_index().rename(columns={"index": "Variable", 0: "Valores distintos"}))
    # Eliminar columnas constantes (que solo tienen un valor único) de cada DataFrame
    df_feature_importance = df_feature_importance.loc[:, df_feature_importance.nunique() > 1]
    df_metrics = df_metrics.loc[:, df_metrics.nunique() > 1]
    df_test_pred = df_test_pred.loc[:, df_test_pred.nunique() > 1]
    df_feature_importance_folds = df_feature_importance_folds.loc[:, df_feature_importance_folds.nunique() > 1]
    df_leaderboard_testset = df_leaderboard_testset.loc[:, df_leaderboard_testset.nunique() > 1]
    # Eliminar columnas específicas de df_feature_importance_folds
    df_feature_importance_folds = df_feature_importance_folds.drop(columns=["fold", "runID"], errors="ignore")
    # Eliminar columnas específicas de df_leaderboard_testset
    df_leaderboard_testset = df_leaderboard_testset.drop(columns=["fold", "runID"], errors="ignore")
    # Eliminar columnas específicas de df_test_pred
    df_test_pred = df_test_pred.drop(columns=["etiq-id"], errors="ignore")
    st.markdown("""
    - Modificar tipo de datos.
    """)
    convertir_a_categoria(df_feature_importance, ["feature", "model", "seed", "fold", "nV"])
    convertir_a_categoria(df_metrics, ["Replica", "ModelBase", "Seed", "nF", "nV"])

    cols_test_pred = [col for col in df_test_pred.columns if col.startswith('testNumFold_')] + \
                 ['Seed', 'nF', 'nV', 'ED_2Clases', 'testPredProba_KNeighborsUnif']
    convertir_a_categoria(df_test_pred, cols_test_pred)

    convertir_a_categoria(df_feature_importance_folds, ['feature', 'seed', 'nF', 'nV'])

    convertir_a_categoria(df_leaderboard_testset, [
        'seed', 'model_type', 'ag_args_fit', 'nF', 'stopping_metric', 'nV', 'stack_level'])

    with st.expander("Tipos de datos"):
            for df_name, df in {
                "df_feature_importance": df_feature_importance,
                "df_metrics": df_metrics,
                "df_test_pred": df_test_pred,
                "df_feature_importance_folds": df_feature_importance_folds,
                "df_leaderboard_testset": df_leaderboard_testset,
            }.items():
                st.markdown(f"### Tipos de datos en {df_name}")
                st.write(df.dtypes)
    


    dataframes = {
        "Feature Importance": df_feature_importance,
        "Metrics": df_metrics,
        "Test Predictions": df_test_pred,
        "Feature Importance Folds": df_feature_importance_folds,
        "Leaderboard Testset": df_leaderboard_testset,
    }

    st.title("Diagrama de Caja y Bigotes con Altair")

    df_name = st.selectbox("Selecciona un DataFrame", list(dataframes.keys()))
    df = dataframes[df_name]

    st.write(f"Dataframe: {df_name} — filas: {df.shape[0]}, columnas: {df.shape[1]}")

    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    var_seleccionada = st.selectbox("Selecciona una variable numérica", num_cols)

    if var_seleccionada:
        df_sample = df.head(1000)
        data = df_sample[[var_seleccionada]].dropna()
        if data.empty:
            st.warning("No hay datos disponibles para esta variable.")
        else:
            chart = alt.Chart(data).mark_boxplot().encode(
                y=alt.Y(f'{var_seleccionada}:Q', title=var_seleccionada),
                x=alt.value(0),
            ).properties(width=300, height=400)
            st.altair_chart(chart, use_container_width=True)



    st.subheader("🔍 Vista previa de los datos cargados")

    with st.expander("📄 df_feature_importance"):   
        st.dataframe(df_feature_importance.head())

    with st.expander("📄 df_metrics"):
        st.dataframe(df_metrics.head())

    with st.expander("📄 df_test_pred"):
        st.dataframe(df_test_pred.head())

    with st.expander("📄 df_feature_importance_folds"):
        st.dataframe(df_feature_importance_folds.head())

    with st.expander("📄 df_leaderboard_testset"):
        st.dataframe(df_leaderboard_testset.head())

def convertir_a_categoria(df, cols):
    cols_existentes = [c for c in cols if c in df.columns]
    for c in cols_existentes:
        df[c] = df[c].astype('category')