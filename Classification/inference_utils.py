import numpy as np
import pandas as pd


def prepare_features_for_inference(df: pd.DataFrame, best_features: list):
    """
    Готовит признаки и делает предсказания модели.

    Args:
        df (pd.DataFrame): Таблица признаков.
        best_features (List[str]): Список используемых признаков.

    Returns:
        pd.DataFrame: Массив признаков X.
    """
    X = df[best_features].fillna(0)
    return X


def predict_model(model, X: pd.DataFrame) -> np.ndarray:
    """
    Выполняет предсказание модели и округляет результат.

    Args:
        model: Обученная модель.
        X (pd.DataFrame): Таблица признаков.

    Returns:
        np.ndarray: Массив целевых меток (0/1).
    """
    preds = model.predict(X)
    preds = np.uint8(np.round(preds))
    return preds