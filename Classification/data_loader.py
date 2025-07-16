import pandas as pd

from config import best_features
from prepare import WORK_DIR


def load_data(filename: str) -> pd.DataFrame:
    """Загружает данные из CSV и добавляет признак max-box-side."""
    df = pd.read_csv(f"{WORK_DIR}/{filename}")
    df = df.fillna(0)
    df['max-box-side'] = df[['width', 'height']].max(axis=1)
    return df


def update_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Обновляет колонку max-box-side на основе width и height."""
    df['max-box-side'] = df[['width', 'height']].max(axis=1)
    return df


def prepare_features_targets(df: pd.DataFrame):
    """Выделяет признаки и целевую переменную."""
    X = df[best_features]
    y = df['target']
    return X, y
