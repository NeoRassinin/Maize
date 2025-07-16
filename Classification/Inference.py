from feature_engineeing import get_features,make_table
from metrics import estimate_accuracy
from io_utils import save_prediction_image
from visualization import draw_predictions, show_image_with_title
from validation import validate_prediction_inputs,validate_file_exists
from data_loader import update_cols
from config import best_features
from inference_utils import prepare_features_for_inference, predict_model




def make_pipeline_predict(
    model,
    image_path,
    mask_path,
    output_path,
    method='lee',
    model_name='rf'
):
    """Полный конвейер инференса: от признаков до визуализации и метрики."""
    # Проверка существования файлов
    validate_file_exists(image_path, "Изображение")
    validate_file_exists(mask_path, "Маска")

    # Получение признаков
    image, mask, skeleton, contours, boxes_features, target, \
        top_left_points_skel = get_features(
            image_path, mask_path, method=method, predict=True
        )

    # Создание таблицы признаков
    df = make_table(skeleton, boxes_features, target)
    df = df.toPandas()
    update_cols(df)

    # Предсказание
    X = prepare_features_for_inference(df, best_features)
    model_predict = predict_model(model, X)

    # Извлечение ID скелетов
    ids = df['skeleton-id'].tolist()

    # Проверка на пустые данные
    validate_prediction_inputs(contours, top_left_points_skel, model_predict)

    # Рисование результатов
    image = draw_predictions(
        image_path, contours, ids, top_left_points_skel, model_predict
    )
    show_image_with_title(image)

    # Сохранение результата
    save_prediction_image(
        image, image_path, output_path, method, model_name
    )

    # Вычисление точности
    accuracy = estimate_accuracy(model_predict, ids)

    return accuracy



