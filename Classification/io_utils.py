import os
import pickle
import cv2


def safe_mkdir(path: str) -> None:
    """
    Безопасно создаёт директорию по указанному пути, если она ещё не существует.

    Args:
        path (str): Путь к директории, которую нужно создать.
    """
    os.makedirs(path, exist_ok=True)


def save_models(models: dict, dir_path="models"):
    """
    Сохраняет несколько моделей в указанный каталог.

    Parameters:
    - models (dict): словарь вида {'имя_модели': model_object}
    - dir_path (str): путь к папке для сохранения
    """
    os.makedirs(dir_path, exist_ok=True)  # создаёт папку, если её нет

    for name, model in models.items():
        path = os.path.join(dir_path, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)

def save_prediction_image(image, image_path, output_path, method='', model_name=''):
    """
    Сохраняет изображение с предсказаниями модели.

    Args:
        image (np.ndarray): RGB изображение.
        image_path (str): Исходный путь к изображению.
        output_path (str): Папка для сохранения.
        method (str): Метод скелетизации или имя этапа.
        model_name (str): Название модели (для суффикса).
    """
    os.makedirs(output_path, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    file_name = f"{method}_{image_name}_{model_name}.jpg"
    save_path = os.path.join(output_path, file_name)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image_bgr)
