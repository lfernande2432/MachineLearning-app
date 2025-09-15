import streamlit as st

def conclusion():

    st.write("""
        La visualización de datos ha facilitado la interpretación y comparación de modelos, réplicas y métricas como ROC AUC por fold. 
        Los gráficos interactivos y codificados por colores permiten identificar errores y casos difíciles rápidamente, enfocándose en lo más relevante sin saturar la visualización. 
        Durante el desarrollo surgieron dificultades por la falta de estandarización y preprocesamiento de los datos, lo que resalta la importancia de preparar correctamente los datos antes del análisis. 
        Futuras mejoras podrían incluir estudiar el tiempo de ejecución de los modelos en función del número de folds y su relación con el rendimiento. 
        En conjunto, la visualización complementa el análisis cuantitativo, mejora la comprensión de los resultados y ayuda a identificar patrones y oportunidades de mejora.
    """)