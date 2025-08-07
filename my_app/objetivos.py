import streamlit as st

def mostrar_objetivos():
    st.markdown("""
    <div style='text-align: justify'>
    El objetivo de esta herramienta es ayudarte a analizar y entender mejor los resultados obtenidos al entrenar múltiples modelos de Machine Learning con distintas configuraciones, sin necesidad de repetir todos los experimentos.

    Para ello, he incorporado funcionalidades que permiten:

    - **Seleccionar las mejores semillas:** Identificar aquellas semillas que han generado modelos representativos del rendimiento medio, lo que permite reducir la cantidad de experimentos necesarios.
    
    - **Detectar los modelos más robustos:** Saber qué modelos ofrecen un buen rendimiento de forma consistente, para centrar el esfuerzo en los algoritmos que realmente aportan valor.

    - **Elegir las mejores configuraciones de hiperparámetros:** Entre todas las variantes probadas para un mismo modelo, ver cuáles ofrecen el mejor equilibrio entre rendimiento y estabilidad.

    - **Identificar las variables más relevantes:** Analizar qué variables influyen más en el rendimiento de los modelos y cuáles podrían eliminarse para simplificar los modelos sin perder calidad.

    - **Analizar la métrica ROC AUC por fold:** Comparar el comportamiento del modelo en cada fold de validación cruzada para detectar posibles inconsistencias o inestabilidades.

    - **Revisar los errores y casos difíciles:** Estudiar los ejemplos que los modelos han clasificado mal para entender por qué fallan y si presentan patrones o características especiales.

    - **Explorar dimensiones adicionales:** Analizar el impacto de aspectos como el número de variables, la cantidad de folds o la evolución de resultados a lo largo de distintas fases del experimento.
    </div>
    """, unsafe_allow_html=True)