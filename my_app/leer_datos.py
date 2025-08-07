import os
import pandas as pd
import streamlit as st

def cargar_pickle_dividido(filepath_base):
    """
    Carga un archivo pickle o varios archivos divididos con sufijo _partX.pkl
    y concatena todo en un DataFrame.
    """
    if os.path.exists(filepath_base):
        return pd.read_pickle(filepath_base)

    dfs = []
    base_noext = filepath_base.replace(".pkl", "")
    i = 1
    while True:
        part_path = f"{base_noext}_part{i}.pkl"
        if not os.path.exists(part_path):
            break
        dfs.append(pd.read_pickle(part_path))
        i += 1

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        raise FileNotFoundError(f"No se encontró {filepath_base} ni partes {base_noext}_partX.pkl")

@st.cache_data(show_spinner="Cargando datos...")
def cargar_datos(pkl_dir):
    pkl_paths = {
        "feature_importance": os.path.join(pkl_dir, "feature_importance.pkl"),
        "metrics": os.path.join(pkl_dir, "metrics.pkl"),
        "test_pred": os.path.join(pkl_dir, "test_pred.pkl"),
        "feature_importance_folds": os.path.join(pkl_dir, "feature_importance_folds.pkl"),
        "leaderboard_testset": os.path.join(pkl_dir, "leaderboard_testset.pkl"),
    }

    # Validar que exista al menos el archivo o alguna de las partes para cada uno
    for key, path in pkl_paths.items():
        base_noext = path.replace(".pkl", "")
        # Verificamos existencia archivo o al menos una parte
        if not os.path.exists(path) and not any(
            os.path.exists(f"{base_noext}_part{i}.pkl") for i in range(1, 20)
        ):
            st.error(f"❌ No se encontró el archivo {path} ni sus partes.")
            st.stop()

    # Cargar todos los pickles con la función que maneja partes
    return (
        cargar_pickle_dividido(pkl_paths["feature_importance"]),
        cargar_pickle_dividido(pkl_paths["metrics"]),
        cargar_pickle_dividido(pkl_paths["test_pred"]),
        cargar_pickle_dividido(pkl_paths["feature_importance_folds"]),
        cargar_pickle_dividido(pkl_paths["leaderboard_testset"]),
    )