def make_pipeline_predict(model, image_path, mask_path, output_path, method='None', model_name=''):
    # Проверка существования файлов
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Маска не найдена: {mask_path}")

    print(f"Обрабатываю изображение: {image_path}")
    print(f"Обрабатываю маску: {mask_path}")

    # Получение признаков
    image, mask, skeleton, contours, boxes_features, target, top_left_points_skel, hough = get_features(
        image_path, mask_path, method=method, predict=True
    )

    # Создание таблицы признаков
    df = make_table(skeleton, boxes_features, target, hough)
    df = df.toPandas()
    df['max-box-side'] = df[['width', 'height']].max(axis=1)

    # Подготовка данных для модели
    X = df[best_features].fillna(0)
    model_predict = model.predict(X)
    model_predict = np.uint8(np.round(model_predict, 0))

    # Извлечение ID скелетов
    ids = df['skeleton-id'].tolist()

    # Проверка на пустые данные
    if len(contours) == 0:
        raise ValueError("Контуры не найдены.")
    if len(top_left_points_skel) == 0:
        raise ValueError("Скелеты не найдены.")
    if len(model_predict) == 0:
        raise ValueError("Модель не вернула предсказаний.")

    # Рисование результатов
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    index_draw = 0
    for j, contour in enumerate(contours):
        if j in ids:  # Только если контур соответствует ID
            x, y = top_left_points_skel[j][0], top_left_points_skel[j][1]
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 1)
            cv2.putText(image, str(model_predict[index_draw]), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 6)
            index_draw += 1
    plt.imshow(image)
    plt.title("Изображение с таргетными метками")
    plt.axis("off")
    plt.show()

    # Сохранение результата
    os.makedirs(output_path, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_path, f'{method}_{image_name}_{model_name}.jpg'), image)

    # Вычисление точности
    if len(ids) == 0 or model_predict.sum() == 0:
        accuracy = 0
    else:
        accuracy = np.min([model_predict.sum() / len(ids), len(ids) / model_predict.sum()])

    print(f"{image_name}: {model_predict.sum()}")
    return accuracy