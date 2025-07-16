import cv2
import matplotlib.pyplot as plt


def draw_predictions(image_path, contours, ids, top_left_points_skel, model_predict):
    """
    Рисует предсказанные значения на изображении с контурами.

    Args:
        image_path (str): Путь к исходному изображению.
        contours (List[np.ndarray]): Контуры объектов.
        ids (List[int]): ID контуров, для которых есть предсказания.
        top_left_points_skel (List[Tuple[int, int]]): Координаты подписи.
        model_predict (np.ndarray): Предсказанные значения.

    Returns:
        np.ndarray: Изображение с нарисованными предсказаниями.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    index_draw = 0
    for j, contour in enumerate(contours):
        if j in ids:
            x, y = top_left_points_skel[j]
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 1)
            cv2.putText(
                image,
                str(model_predict[index_draw]),
                (x + 10, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                6
            )
            index_draw += 1

    return image


def show_image_with_title(image, title="Изображение с предсказаниями"):
    """
    Показывает изображение с помощью matplotlib.

    Args:
        image (np.ndarray): RGB изображение.
        title (str): Заголовок графика.
    """
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()
