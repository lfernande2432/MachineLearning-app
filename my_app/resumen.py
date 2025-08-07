
import streamlit as st

def mostrar_resumen():
    st.markdown("""
    <div style='text-align: justify'>
    Esta herramienta ha sido desarrollada con el objetivo de facilitar el análisis de múltiples experimentos de aprendizaje automático realizados con distintas configuraciones, modelos, semillas y métricas.  
    <br><br>
    Normalmente, entrenar modelos de Machine Learning implica probar muchas combinaciones que requieren un alto coste computacional. Con esta aplicación, busco ayudarte a identificar qué configuraciones han dado mejores resultados, reduciendo así la necesidad de repetir entrenamientos innecesarios.  
    <br><br>
    A través de visualizaciones interactivas, filtros y comparaciones, podrás explorar de manera intuitiva los resultados de tus experimentos, detectar patrones y tomar decisiones más informadas para etapas futuras del desarrollo de modelos.
    </div>
    """, unsafe_allow_html=True)
