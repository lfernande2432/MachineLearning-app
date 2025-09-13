import streamlit as st
import pandas as pd

from my_app.resumen import mostrar_resumen
from my_app.objetivos import mostrar_objetivos
from my_app.leer_datos import cargar_datos
from my_app.preparacion import preparar_datos
from my_app.procesado import procesar

base_path = "dataset"


df_feature_importance, df_metrics, df_test_pred, df_feature_importance_folds, df_leaderboard_testset=cargar_datos(base_path)

# Título principal
st.markdown("<h1 style='text-align: center;'>Análisis de modelos de Machine Learning</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: right'>
    Laura Fernández Asensio  <br>
    Visualización de Datos  <br>
    Curso 2024/2025
    </div>
    """, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
# Crear pestañas
tabs = st.tabs([
    "Resumen",
    "1. Planteamiento y objetivos",
    "2. Preparación de los datos",
    "3. Procesado y análisis",
    "4. Visualización",
    "5. Conclusiones"
])


# Espaciado entre título y resumen
st.markdown("<br>", unsafe_allow_html=True)



# Contenido de cada pestaña
with tabs[0]:
     mostrar_resumen()

with tabs[1]:
     st.header("1. Planteamiento y objetivos")
     mostrar_objetivos()

with tabs[2]:
    st.header("2. Preparación de los datos")
    df_feature_importance, df_metrics, df_test_pred, df_feature_importance_folds, df_leaderboard_testset=preparar_datos(df_feature_importance, df_metrics, df_test_pred, df_feature_importance_folds, df_leaderboard_testset)

with tabs[3]:
    st.header("3. Procesado y análisis")
    procesar(df_feature_importance, df_metrics, df_test_pred, df_feature_importance_folds, df_leaderboard_testset)


with tabs[4]:
    st.header("4. Visualización")

with tabs[5]:
  st.header("5. Conclusiones")
