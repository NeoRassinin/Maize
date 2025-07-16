import os


def validate_file_exists(file_path: str, file_description: str = "Файл") -> None:
    """
    Проверяет существование файла.

    Args:
        file_path (str): Путь к файлу.
        file_description (str): Описание файла для сообщения об ошибке.

    Raises:
        FileNotFoundError: Если файл не найден.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_description} не найден: {file_path}")


def validate_prediction_inputs(contours, skeletons, model_output) -> None:
    """
    Проверяет, что данные для предсказания не пусты.

    Args:
        contours (list): Список контуров.
        skeletons (list): Список скелетов.
        model_output (np.ndarray): Предсказания модели.

    Raises:
        ValueError: Если один из элементов пуст.
    """
    if not contours:
        raise ValueError("Контуры не найдены.")
    if not skeletons:
        raise ValueError("Скелеты не найдены.")
    if model_output is None or len(model_output) == 0:
        raise ValueError("Модель не вернула предсказаний.")