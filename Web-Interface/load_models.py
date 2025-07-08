import pickle
import streamlit as st
# Загрузка модели (десериализация)
@st.cache_resource
def load_model():
    with open("D:\diplom\spark-3.5.0-bin-hadoop3\Maxim_maize\Maxim_maize_2\Maxim_maize\workdir\workdir\model.pkl", "rb") as f:
        return pickle.load(f)