import pickle
import streamlit as st
from pathlib import Path


@st.cache_resource
def load_model(model_path: str = None):
    """
    Загружает сериализованную модель из файла.

    Args:
        model_path (str): Путь к файлу модели (pkl).

    Returns:
        object: Загруженная модель.
    """
    if model_path is None:
        model_path = (
            Path("D:/diplom/spark-3.5.0-bin-hadoop3/Maxim_maize/")
            / "Maxim_maize_2/Maxim_maize/workdir/workdir/model.pkl"
        )

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model
