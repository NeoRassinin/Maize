import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_shap_values(model, X, model_type='tree'):
    """
    Вычисляет SHAP значения для обученной модели.

    Args:
        model: Обученная модель (XGBoost, LightGBM, CatBoost, RandomForest и т.д.).
        X (pd.DataFrame): Признаки.
        model_type (str): Тип модели — 'tree', 'catboost', 'linear' и т.д.

    Returns:
        shap_values: SHAP значения.
        explainer: SHAP Explainer.
    """
    if model_type == 'catboost':
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    else:
        explainer = shap.Explainer(model, X)

    shap_values = explainer(X)
    return shap_values, explainer


def plot_shap_summary(shap_values, X, max_display=20, save_path=None):
    """
    Строит SHAP summary plot.

    Args:
        shap_values: SHAP значения.
        X (pd.DataFrame): Признаки.
        max_display (int): Сколько признаков отобразить.
        save_path (str): Путь для сохранения графика (если нужно).
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, max_display=max_display, show=save_path is None)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


def get_best_features(shap_values, X, top_n=10):
    """
    Возвращает список наиболее важных признаков по SHAP.

    Args:
        shap_values: SHAP значения.
        X (pd.DataFrame): Признаки.
        top_n (int): Количество признаков в списке.

    Returns:
        List[str]: Список названий признаков.
    """
    importance = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values(by='importance', ascending=False)

    return feature_importance['feature'].head(top_n).tolist()
