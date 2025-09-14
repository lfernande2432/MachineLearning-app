import streamlit as st
import pandas as pd
import altair as alt
import ast
def parse_hyperparams(hp):
    if isinstance(hp, dict):
        return hp
    try:
        return ast.literal_eval(hp)
    except Exception:
        return {}

def visualizar(df_metrics, mejores_modelos, df_leaderboard_testset,df_feature_importance):
    filtrar_mejores_modelos = st.checkbox("Mejores modelos", value=True)

    df_metrics_filtrado = df_metrics.copy()
    
    st.write("### Visualización de las métricas por modelo")
    if filtrar_mejores_modelos:
        df_metrics_filtrado = df_metrics_filtrado[df_metrics_filtrado['ModelBase'].isin(mejores_modelos)]

    metric_cols = [
        "Balanced_accuracy", "Precision_macro",
        "Recall_macro",  "F1_macro",
        "Roc_auc"
    ]
    # Crear copia con columnas renombradas
    df_leaderboard_testset = df_leaderboard_testset.copy()
    df_leaderboard_testset['Balanced_accuracy'] = df_leaderboard_testset['balanced_accuracy']
    df_leaderboard_testset['Precision_macro'] = df_leaderboard_testset['precision']
    df_leaderboard_testset['Recall_macro'] = df_leaderboard_testset['recall']
    df_leaderboard_testset['F1_macro'] = df_leaderboard_testset['f1_macro']
    df_leaderboard_testset['Roc_auc'] = df_leaderboard_testset['roc_auc']
    df_leaderboard_testset['ModelBase'] = df_leaderboard_testset['model'].str.split('_').str[0]
    df_leaderboard_testset.drop(columns=['balanced_accuracy', 'precision', 'recall', 'f1_macro', 'roc_auc'], inplace=True)

    metrica_seleccionada = st.selectbox(
        "Selecciona una métrica:",
        options=metric_cols, 
        key='metric_cols2'
    )

    # Calcular la media global
    media_global = df_metrics_filtrado[metrica_seleccionada].mean()
    df_media = pd.DataFrame({metrica_seleccionada: [media_global]})

    # Boxplot general
    chart = alt.Chart(df_metrics_filtrado).mark_boxplot(extent='min-max').encode(
        x='ModelBase:O',
        y=alt.Y(metrica_seleccionada + ':Q', scale=alt.Scale(zero=False))
    )
    st.altair_chart(chart, use_container_width=True)

    # Pestañas dinámicas para cada modelo
    tabs = st.tabs(mejores_modelos)
    for i, modelo in enumerate(mejores_modelos):
        with tabs[i]:
            st.markdown(f"### Métricas por semilla para el modelo: **{modelo}**")
            df_modelo = df_metrics_filtrado[df_metrics_filtrado['ModelBase'] == modelo]

            # Boxplot por semilla
            chart_modelo = alt.Chart(df_modelo).mark_boxplot(extent='min-max').encode(
                x='Seed:O', 
                y=alt.Y(metrica_seleccionada + ':Q', scale=alt.Scale(zero=False)),
                tooltip=['Seed', metrica_seleccionada]
            ).properties(width=600)

            # Línea de media global
            linea_media = alt.Chart(df_media).mark_rule(color="red", size=2).encode(
                y=f"{metrica_seleccionada}:Q"
            )
            st.altair_chart(chart_modelo + linea_media, use_container_width=True)


            # --- Configuraciones ---
            st.markdown(f"### Hiperparámetros del modelo: **{modelo}**")
            df_modelo_testset = df_leaderboard_testset[df_leaderboard_testset['ModelBase'] == modelo]

            # Parsear configs
            configs = [parse_hyperparams(hp) for hp in df_modelo_testset['hyperparameters'].unique()]
            df_configs = pd.DataFrame(configs)
            df_configs.index = [f"conf{i+1}" for i in range(len(df_configs))]
            df_configs.reset_index(inplace=True)
            df_configs.rename(columns={"index": "config_id"}, inplace=True)

            df_configs = df_configs.astype(str)
            st.dataframe(df_configs)
'''
            # Ranking de configuraciones
            df_unique = df_modelo_testset.groupby("hyperparameters").agg(
                mean_metric=(metrica_seleccionada, "mean"),
                std_metric=(metrica_seleccionada, "std")
            ).reset_index()
            df_unique["config_id"] = [f"conf{i+1}" for i in range(len(df_unique))]

            bars = alt.Chart(df_unique).mark_bar().encode(
                x=alt.X("config_id:N", title="Configuración"),
                y=alt.Y("mean_metric:Q", title=f"Media de {metrica_seleccionada}", scale=alt.Scale(zero=False)),
                tooltip=["config_id", "mean_metric:Q", "std_metric:Q"]
            )

            st.altair_chart(bars, use_container_width=True)
            # Mejor configuración
            best_config = df_unique.loc[df_unique['mean_metric'].idxmax()]

            st.write(f"##### Mejor configuración según {metrica_seleccionada}")
            st.write("ID de configuración:", best_config['config_id'])
            st.write("Hiperparámetros:", best_config['hyperparameters'])
            st.markdown(f"### Importancia de las variables para el: **{modelo}**")

            # Filtrar filas
            df_modelo = df_feature_importance[df_feature_importance["model"] == modelo]
            #  Seleccionar más relevantes
            # Calcular ranking global de importancia para el modelo
            df_ranking = df_modelo.groupby("feature")["importance"].mean().reset_index()
            df_ranking = df_ranking.sort_values("importance", ascending=False)

            # Slider para seleccionar cuántas features mostrar
            num_features = st.slider(
                "Número de features más relevantes a mostrar según el ranking global:",
                min_value=1,
                max_value=len(df_ranking),
                value=min(8, len(df_ranking)),
                key=f"top_features_{modelo}"
            )

            # Tomar las top N features según ranking
            top_features = df_ranking.head(num_features)["feature"].tolist()
            df_grafico = df_modelo[df_modelo["feature"].isin(top_features)]

            # Gráfica de barras horizontales
            barra_importancia = (
                alt.Chart(df_grafico)
                .mark_tick()
                .encode(
                    x=alt.X("importance:Q", title="Importancia"),
                    y=alt.Y(
                        "feature:N",
                        sort=alt.EncodingSortField(field="importance", op="max", order="descending"),
                        title="Variable"
                    ),
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
                    title=f"Importancia de variables para el modelo {modelo}"
                )
            )

            st.altair_chart(barra_importancia, use_container_width=True)

            st.markdown(f"### Análisis del Roc_auc por Fold para el: **{modelo}**")
'''