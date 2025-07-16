import numpy as np

def estimate_accuracy(model_preds: np.ndarray, true_ids: list[int]) -> float:
    """
    Оценивает точность модели на основе количества предсказаний и истинных ID.

    Args:
        model_preds (np.ndarray): Бинарные предсказания модели.
        true_ids (list[int]): Список индексов целевых объектов.

    Returns:
        float: Значение точности в интервале [0, 1].
    """
    if len(true_ids) == 0 or model_preds.sum() == 0:
        return 0.0
    return min(model_preds.sum() / len(true_ids), len(true_ids) / model_preds.sum())