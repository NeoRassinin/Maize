import cv2
import numpy as np
from skimage import morphology


def clear_mask(mask: np.ndarray, min_length: float = 70.0) -> np.ndarray:
    """
    Удаляет маленькие контуры из бинарной маски по порогу длины.

    Args:
        mask (np.ndarray): Входная бинарная маска.
        min_length (float): Минимальная длина контура, чтобы он остался.

    Returns:
        np.ndarray: Очищенная бинарная маска.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    new_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > min_length]
    cv2.drawContours(new_mask, new_contours, -1, 1, thickness=-1)
    return new_mask


def make_skeleton(mask: np.ndarray, method: str = 'lee') -> np.ndarray:
    """
    Применяет скелетизацию к бинарной маске.

    Args:
        mask (np.ndarray): Входная бинарная маска.
        method (str): Метод скелетизации ('lee', 'zhang', и др.).

    Returns:
        np.ndarray: Скелетизированное изображение.
    """
    skeleton = morphology.skeletonize(mask, method=method)
    return skeleton.astype(np.uint8)


def sort_function(tup: tuple[int, int]) -> tuple[int, int]:
    """
    Сортирует кортеж сначала по y, затем по x.

    Args:
        tup (tuple): Кортеж координат (x, y).

    Returns:
        tuple: Отсортированный кортеж (y, x).
    """
    x, y = tup
    return y, x


def make_bool_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """
    Преобразует скелет в булеву маску: 1 — скелет, 0 — фон.

    Args:
        skeleton (np.ndarray): Входной скелет.

    Returns:
        np.ndarray: Булева маска скелета.
    """
    skeleton = (skeleton > 0).astype(np.uint8)
    return skeleton


