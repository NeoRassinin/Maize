import cv2
import numpy as np
from scipy.stats import mode


def hough_transform_with_angles(mask_path: str) -> float | None:
    """
    Применяет преобразование Хафа к бинарной маске для обнаружения прямых линий.
    Вычисляет преобладающий угол наклона (в градусах) рядов кукурузы.

    Args:
        mask_path (str): Путь к бинарной маске (черно-белое изображение).

    Returns:
        float | None: Угол в градусах (если линии найдены), иначе None.
    """
    # Загрузка бинарной маски в оттенках серого
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Не удалось загрузить маску: {mask_path}")

    # Убедимся, что маска содержит только значения 0 и 255
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Применяем преобразование Хафа
    lines = cv2.HoughLinesP(
        binary_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=430,
        minLineLength=100,
        maxLineGap=60
    )

    # Хранилище для углов наклона найденных линий
    angles = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                angle = 90.0  # Вертикальная линия
            else:
                angle = np.degrees(np.arctan2(dy, dx))

            angles.append(angle)

    if angles:
        rounded_angles = np.round(angles, 2)
        dominant_angle = mode(rounded_angles, keepdims=False).mode
        return float(dominant_angle)

    return None
