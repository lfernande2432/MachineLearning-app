import streamlit as st

def conclusion():

    st.markdown("""
El desarrollo de este trabajo ha permitido apreciar de manera práctica los beneficios de la visualización de datos en el análisis de modelos de aprendizaje automático. Los gráficos interactivos y codificados por colores facilitaron la interpretación de métricas como ROC AUC por fold, la comparación de modelos y réplicas, y la identificación de errores y casos difíciles. La posibilidad de seleccionar modelos, réplicas, semillas o instancias concretas permitió enfocar el análisis en los aspectos más relevantes, evitando saturar la visualización y resaltando patrones que serían difíciles de detectar únicamente mediante tablas o métricas agregadas.

Durante el proceso también surgieron dificultades derivadas de la preparación de los datos. La falta de estandarización en los nombres de las variables y la ausencia de un preprocesamiento inicial completo generaron problemas a la hora de escribir el código, evidenciando la importancia de dedicar tiempo a limpiar y estructurar correctamente los datos antes de realizar cualquier análisis visual o computacional.

Asimismo, el trabajo deja líneas de investigación futuras interesantes, como estudiar el tiempo de ejecución de los modelos en función del número de folds y analizar si existe alguna relación con el rendimiento. Este tipo de análisis podría aportar información adicional sobre la eficiencia de los modelos y ayudar a optimizar la selección de parámetros y recursos computacionales.

En conjunto, el trabajo demuestra que la visualización no solo complementa el análisis cuantitativo, sino que también potencia la comprensión, la comunicación de los resultados y la identificación de problemas y oportunidades de mejora en los modelos analizados.
""")