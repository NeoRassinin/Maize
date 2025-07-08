def hough_transform_with_angles(mask_path):
    """
    Применяет преобразование Хафа к бинарной маске для обнаружения прямых линий.
    Вычисляет средний, максимальный и минимальный углы наклона линий.
    Визуализирует результат.
    """
    # Загрузка бинарной маски
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Не удалось загрузить маску: {mask_path}")
    mask = (mask * 255).astype(np.uint8)

    # Убедимся, что маска содержит значения [0, 255]
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(
        mask,
        rho=1,  # Разрешение по расстоянию (в пикселях)
        theta=np.pi / 180,  # Разрешение по углу (в радианах)
        threshold=430,  # Минимальное количество голосов для обнаружения линии
        minLineLength=100,  # Минимальная длина линии
        maxLineGap=60  # Максимальный разрыв между частями одной линии
    )

    # Создаем цветное изображение для визуализации
    color_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Список для хранения углов наклона
    angles = []

    # Если линии найдены, рисуем их и вычисляем углы
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # Координаты начала и конца линии
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Угол наклона в градусах
            angles.append(angle)

            # Рисуем линию на изображении
            cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Зелёная линия

    if angles:
        angles_rounded = np.round(angles, 2)

        # Находим моду
        mode_result = mode(angles_rounded, keepdims=False)
        avg_angle = mode_result.mode
    else:
        avg_angle = None

    return avg_angle