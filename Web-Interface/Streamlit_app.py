
# ---------- STREAMLIT ПРИЛОЖЕНИЕ ----------
import streamlit as st
from PIL import Image
import os
from load_models import load_model
from Classification.Inference import make_pipeline_predict
from Classification.config import PREDICT_PATH
model = load_model()
def Web_Interface(model):
    '''
    Запускает веб-итерфейс локальный
    :param model:
    :return:
    '''
    # Интерфейс
    st.title("\U0001F33D Предсказание всхожести кукурузы по фото")
    st.write("Укажите путь к изображению с БПЛА и (опционально) путь к маске")

    image_path = st.text_input("\U0001F4F7 Путь до изображения поля (например, ./images/field.jpg)")
    mask_path = st.text_input("\U0001F3A8 Путь до маски кукурузы (по желанию, например, ./masks/mask.jpg)")

    if image_path and os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption="Загруженное изображение", use_column_width=True)

        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path)
            st.image(mask, caption="Маска", use_column_width=True)

        prediction = make_pipeline_predict(model, image, mask, PREDICT_PATH)

        st.success(f"\U0001F4C8 Предсказанное количество всходов: {int(prediction[0])}")
    else:
        if image_path:
            st.error("Файл изображения не найден. Проверьте путь.")


if __name__ == "__main__":
    Web_Interface(model)
