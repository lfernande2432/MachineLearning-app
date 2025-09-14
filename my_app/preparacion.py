import streamlit as st
import pandas as pd
@st.cache_data
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
        #Normalizar valores de las variables modelo y ModelBase
      # Eliminar el sufijo a partir de "_" en los nombres de modelos
    df_feature_importance['model'] = df_feature_importance['model'].str.split('_').str[0]
    df_feature_importance_folds['model'] = df_feature_importance_folds['model'].str.split('_').str[0]


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
    df_feature_importance_folds = df_feature_importance_folds.drop(columns=["runID"], errors="ignore")
    # Eliminar columnas específicas de df_leaderboard_testset
    df_leaderboard_testset = df_leaderboard_testset.drop(columns=[ "runID"], errors="ignore")


   
  
    st.subheader("🔍 Vista previa de los datos cargados")

    with st.expander("📄 df_feature_importance"):   
        st.dataframe(df_feature_importance.head())

    with st.expander("📄 df_metrics"):
        st.dataframe(df_metrics.head())

    with st.expander("📄 df_test_pred"):
        st.dataframe(df_test_pred.head())

    with st.expander("📄 df_feature_importance_folds"):
        st.dataframe(df_feature_importance.head())

    with st.expander("📄 df_leaderboard_testset"):
        st.dataframe(df_leaderboard_testset.head())

    # Dataframe df_metrics balanceado por clase ModelBase
    st.subheader("⚖️ Balanceo de df_metrics por clase ModelBase")
    st.markdown("""
    El conjunto de datos muestra un desbalance significativo entre las clases de la variable `ModelBase`.
    Este desbalance puede afectar el análisis de robustez y la selección de modelos, ya que algunas clases están subrepresentadas.
    """)
    st.markdown("### Distribución original de `ModelBase`")
    conteo_original = df_metrics['ModelBase'].value_counts()
    st.bar_chart(conteo_original)
    return df_feature_importance, df_metrics, df_test_pred, df_feature_importance_folds, df_leaderboard_testset



