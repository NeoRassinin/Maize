def map_skeleton_contours(contours_skeletons, contours_objects, top_left_points_skel, top_left_points_contour):
    ''' Предположим, что у нас есть два списка контуров: contours_objects и contours_skeletons
    Если расстояние между центрами и разница между площадями достаточно малы, то считаем, что контуры соответствуют друг другу
    '''
    len_i = len(top_left_points_skel)
    len_j = len(top_left_points_contour)
    matches = []
    best_points = []
    boxes_features = []
    for i in range(len_i):
        x_skel, y_skel, w_skel, h_skel = cv2.boundingRect(contours_skeletons[i])
        best_match = None
        best_point = None
        best_index = i
        min_dist = np.inf
        min_area_diff = np.inf
        for j in range(len_j):
            x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(contours_objects[j])
            area = cv2.contourArea(contours_objects[j])
            rect = cv2.minAreaRect(contours_objects[j])
            center, size, angle = rect
            width, height = size
            box_ratio = np.min(np.array([width/height, height/width]))
            area_box_ratio = area/(width*height)
            length = cv2.arcLength(contours_objects[j], True)
            ratio_skel_contour = cv2.arcLength(contours_skeletons[i], True) / cv2.arcLength(contours_objects[j], True)

            dist = np.linalg.norm(np.array(top_left_points_skel[i]) - np.array(top_left_points_contour[j]))
            if dist < min_dist:
                min_dist = dist
                box_features = [i, x_cnt, y_cnt, width, height, area, length, ratio_skel_contour, area_box_ratio, float(box_ratio)]
                best_index = j
                best_match = [contours_skeletons[i], contours_objects[best_index]]
                best_point = [top_left_points_skel[i], top_left_points_contour[best_index]]
        matches.append(best_match.copy())
        best_points.append(best_point.copy())
        boxes_features.append(box_features.copy())
    boxes_features = sorted(boxes_features, key=lambda x: x[0])
    # Теперь в списке matches содержатся сопоставленные контуры объектов и контуры скелетов
    contours_skeletons = [contour_skeleton[0] for contour_skeleton in matches]
    contours_objects = [contour_object[1] for contour_object in matches]
    top_left_points_skel = [top_left_point_skel[0] for top_left_point_skel in best_points]
    top_left_points_contour = [top_left_point_contour[1] for top_left_point_contour in best_points]
    return contours_skeletons, contours_objects, top_left_points_skel, top_left_points_contour, boxes_features


def get_features(image_path, mask_path, method=None, predict=False):
    '''
    Функция для формирования геометрических признаков
    :param image_path:
    :param mask_path:
    :param method:
    :param predict:
    :return: image, mask, skeleton, contours, boxes_features, target, top_left_points_skel, hough
    '''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target = []
    indexes = []
    boxes_features = []
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.imread(mask_path, 0)
    mask = clear_mask(mask, min_length=70)
    print(f"Mask shape: {mask.shape if mask is not None else 'None'}")

    print("Минимальное значение:", np.min(mask))
    print("Максимальное значение:", np.max(mask))
    print("Количество ненулевых пикселей:", np.count_nonzero(mask))

    hough = hough_transform_with_angles(mask_path)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    skeleton = morphology.skeletonize(mask, method=method)
    skeleton = make_bool_skeleton(skeleton)
    contours_skel, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(len(contours), len(contours_skel))
    while (len(contours) != len(contours_skel)):
        mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        skeleton = morphology.skeletonize(mask, method=method)
        skeleton = make_bool_skeleton(skeleton)
        contours_skel, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours), len(contours_skel))

    top_left_points_contour = []
    for contour in contours:
        min_x = np.min(contour[:, 0, 0])
        min_y = np.min(contour[:, 0, 1])
        top_left_points_contour.append((min_x, min_y))
    contours_points = [(point, contour) for point, contour in zip(top_left_points_contour, contours)]
    contours_points = sorted(contours_points, key=lambda x: (x[0][1], x[0][0]))
    contours = [contour for _, contour in contours_points]
    top_left_points_contour = sorted(top_left_points_contour, key=sort_function)

    top_left_points_skel = []
    for contour_skel in contours_skel:
        min_x = np.min(contour_skel[:, 0, 0])
        min_y = np.min(contour_skel[:, 0, 1])
        top_left_points_skel.append((min_x, min_y))
    contours_skel_points = [(point, contour) for point, contour in zip(top_left_points_skel, contours_skel)]
    contours_skel_points = sorted(contours_skel_points, key=lambda x: (x[0][1], x[0][0]))
    contours_skel = [contour for _, contour in contours_skel_points]
    top_left_points_skel = sorted(top_left_points_skel, key=sort_function)
    contours_skel, contours, top_left_points_skel, top_left_points_contour, boxes_features = map_skeleton_contours(
        contours_skel, contours, top_left_points_skel, top_left_points_contour)
    if not predict:
        for j in range(len(top_left_points_contour)):
            cv2.drawContours(image, [contours[j]], -1, (255, 0, 0), 1)
            # cv2.putText(image, str(j), (top_left_points_contour[j][0] + 10, top_left_points_contour[j][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        for i in range(len(top_left_points_skel)):
            cv2.putText(image, str(i), (top_left_points_skel[i][0] + 15, top_left_points_skel[i][1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    for index in range(len(top_left_points_skel)):
        target.append(1)

    # Отрисовка bounding box'ов для контуров маски
    for contour in contours:
        # Нахождение минимального ограничивающего прямоугольника
        rect = cv2.minAreaRect(contour)
        center, size, angle = rect
        width, height = size
        area = (width * height)
        box_ratios = np.min(np.array([width / height, height / width]))

        # Получение вершин прямоугольника
        box = cv2.boxPoints(rect)
        box = np.int0(box)  # Преобразование координат в целые числа

        # Рисование прямоугольника
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    # Преобразуем изображение обратно в BGR для сохранения через OpenCV
    image_bgrs = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Сохраняем изображение
    output_image_path = "output_image_with_boxes.jpg"
    cv2.imwrite(output_image_path, image_bgrs)


    return image, mask, skeleton, contours, boxes_features, target, top_left_points_skel, hough


def delete_spurs(df):
    """
        Фильтрует короткие боковые ветви (spurs) на скелетах объектов по пороговому расстоянию.

        Функция анализирует расстояния боковых ветвей (`branch-distance`) в скелетах и удаляет
        те из них, которые считаются "короткими" по сравнению с локальным и глобальным порогом.
        Локальный порог — 25-й перцентиль расстояний внутри одного скелета, глобальный — 5-й перцентиль
        по всем скелетам. Ветвь сохраняется, если она относится к основному типу (`branch-type != 1`)
        или её длина превышает минимум из двух порогов.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Входной DataFrame с колонками:
            - 'skeleton-id': идентификатор скелета
            - 'branch-type': тип ветви (1 — боковая ветвь)
            - 'branch-distance': длина ветви

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame без коротких боковых ветвей, с сохранёнными основными ветвями
            и длинными боковыми ветвями. Отсортирован по 'skeleton-id'.

        Notes
        -----
        - Ветки типа `branch-type = 1` фильтруются по порогам.
        - Ветки других типов сохраняются без условий.
        """
    grouped_data = df.groupBy(['skeleton-id', 'branch-type']).agg(percentile_approx('branch-distance', 0.25).alias('25th_percentile_local'))
    quantile_5 = df.filter(df['branch-type']==1).stat.approxQuantile('branch-distance', [0.2], relativeError=0.05)
    grouped_data = grouped_data.withColumn("5th_percentile_global", lit(quantile_5[0]))
    grouped_data = grouped_data.withColumn('min-threshold', least(grouped_data['25th_percentile_local'], grouped_data['5th_percentile_global']))
    grouped_data = grouped_data.select(['skeleton-id', 'branch-type', 'min-threshold'])
    #grouped_data.show()
    result = df.join(grouped_data, on=['skeleton-id', 'branch-type'], how='left')
    result = result.filter((result['branch-type']!=1) | (result['branch-distance'] >= result['min-threshold'])).orderBy('skeleton-id')
    return result


def make_stats(df, group_col, agg_col):
    """
        Строит агрегированные статистики по длинам ветвей для разных типов ветвлений в скелетах.

        Функция группирует данные по `group_col` (например, 'skeleton-id') и типу ветви (`branch-type`),
        затем считает суммарные и описательные статистики по колонке `agg_col` (например, 'branch-distance') —
        отдельно для типов ветвей 0, 1, 2, а также объединённо для типов 1 и 2.

        Возвращает объединённую таблицу с результатами по всем типам ветвей (0, 1, 2 и 1+2) с медианой,
        средним, минимумом, максимумом и суммой.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Входной DataFrame с колонками:
            - `group_col` (например, 'skeleton-id') — ID скелета или другой группы
            - `branch-type` — тип ветви (0: основная, 1: боковая, 2: шумовая)
            - `agg_col` (например, 'branch-distance') — метрика, по которой агрегируются значения

        group_col : str
            Название колонки, по которой группируются данные (обычно 'skeleton-id').

        agg_col : str
            Название числовой колонки, по которой считаются статистики (например, 'branch-distance').

        Returns
        -------
        pyspark.sql.DataFrame
            Таблица со следующими агрегатами для каждого `skeleton-id`:
            - Сумма, среднее, минимум, максимум и медиана по `agg_col`
              отдельно для ветвей типа 0 (`_sum`, `_avg`, ...), типа 1 (`_sum_1`, ...) и типа 2 (`_sum_2`, ...)
            - Агрегаты по типам ветвей 1 и 2 вместе (`_sum`, `_avg`, ...) — совмещены с типом 0
            - Все пропущенные значения заменены на 0
            - Отсортировано по `skeleton-id`

        Notes
        -----
        Используется `percentile(agg_col, 0.5)` для оценки медианы, а также `na.fill(0)` для пустых значений.
        """
    approx_quantile_value = [0.5]  # Медиана
    approx_quantile_accuracy = 0.01  # Точность

    aggregated_df_0 = df.groupBy(group_col, "branch-type").agg(
        sum(agg_col).alias(agg_col + "_sum"),
        avg(agg_col).alias(agg_col + "_avg"),
        min(agg_col).alias(agg_col + "_min"),
        max(agg_col).alias(agg_col + "_max"),
        percentile(agg_col, 0.5).alias(agg_col + "_median")
    ).filter(col("branch-type") == 0)
    aggregated_df_0 = aggregated_df_0.drop('branch-type')

    aggregated_df_1 = df.groupBy(group_col, "branch-type").agg(
        sum(agg_col).alias(agg_col + "_sum_1"),
        avg(agg_col).alias(agg_col + "_avg_1"),
        min(agg_col).alias(agg_col + "_min_1"),
        max(agg_col).alias(agg_col + "_max_1"),
        percentile(agg_col, 0.5).alias(agg_col + "_median_1")
    ).filter(col("branch-type") == 1)
    aggregated_df_1 = aggregated_df_1.drop('branch-type')

    aggregated_df_2 = df.groupBy(group_col, "branch-type").agg(
        sum(agg_col).alias(agg_col + "_sum_2"),
        avg(agg_col).alias(agg_col + "_avg_2"),
        min(agg_col).alias(agg_col + "_min_2"),
        max(agg_col).alias(agg_col + "_max_2"),
        percentile(agg_col, 0.5).alias(agg_col + "_median_2")
    ).filter(col("branch-type") == 2)
    aggregated_df_2 = aggregated_df_2.drop('branch-type')

    aggregated_df_1_2 = df.filter((col("branch-type") == 1) | (col("branch-type") == 2)).groupBy(group_col).agg(
        sum(agg_col).alias(agg_col + "_sum"),
        avg(agg_col).alias(agg_col + "_avg"),
        min(agg_col).alias(agg_col + "_min"),
        max(agg_col).alias(agg_col + "_max"),
        percentile(agg_col, 0.5).alias(agg_col + "_median")
    )
    aggregated_df_1_2 = aggregated_df_1_2.union(aggregated_df_0).orderBy('skeleton-id')
    aggregated_df = aggregated_df_1_2
    aggregated_df = aggregated_df.join(aggregated_df_1, on='skeleton-id', how='left')
    aggregated_df = aggregated_df.join(aggregated_df_2, on='skeleton-id', how='left')
    aggregated_df = aggregated_df.orderBy('skeleton-id').na.fill(0)
    return aggregated_df


def make_table(skeleton_img, features, target):
    """
       Формирует итоговую таблицу признаков для анализа и предсказания всхожести по скелетизированным изображениям.

       Функция объединяет структурные признаки (геометрию объектов), скелетные признаки (ветвление скелета),
       и целевую переменную. Включает предварительную фильтрацию шумов и коротких ветвей, расчёт статистик
       по различным типам ветвей, агрегации и генерацию новых признаков на основе геометрии и скелетной структуры.

       Parameters
       ----------
       skeleton_img : np.ndarray
           Бинарное изображение с уже извлечёнными скелетами объектов (маска), используется для получения ветвлений.

       features : List[Tuple]
           Список признаков объектов (ширина, длина, площадь и др.) до скелетизации. Каждый элемент — кортеж
           с признаками одного объекта. Преобразуется в Spark DataFrame.

       target : List[int]
           Список целевых значений (например, количество всходов на каждый скелет), используется для обучения модели.

       Returns
       -------
       pyspark.sql.DataFrame
           Финальный DataFrame с объединёнными признаками:
           - Метрики геометрии объекта (area, box ratio, length, etc.)
           - Скелетные статистики (branch count, длины ветвей по типам, отношения длин)
           - Целевая переменная (target)
           - Все значения округлены до 3 знаков после запятой
           - Группировка по 'skeleton-id'

       Notes
       -----
       - Используются функции `delete_spurs`, `summarize`, `make_stats`
       - Типы ветвей (branch-type):
           0 — endpoint-to-endpoint,
           1 — junction-to-endpoint,
           2 — junction-to-junction
       - Отбрасываются короткие ветви и шумы на основе относительного расстояния
       - Формируются новые признаки: отношения длин ветвей разных типов
       """
    # add features
    column_names = ['skeleton-id', 'x', 'y', 'width', 'height', 'area', 'length', 'ratio-skel-contour',
                    'area-box-ratio', 'box-ratio']
    rdd_features = spark.sparkContext.parallelize(features)
    df_features = rdd_features.toDF(column_names)
    df_features = df_features.withColumn('skeleton-id', row_number().over(Window.orderBy(lit(1).asc())) - 1)

    # add target
    df_target = spark.createDataFrame(target, IntegerType())
    window_spec = Window.partitionBy().orderBy(lit(1).asc())
    df_target = df_target.withColumn('skeleton-id', row_number().over(window_spec) - 1)

    # features + target
    df_features = df_features.join(df_target, on='skeleton-id').withColumnRenamed("value", "target")
    df_features = df_features.drop('x', 'y')

    # make branch data
    branch_df = summarize(Skeleton(skeleton_img))
    df = spark.createDataFrame(data=branch_df)

    # selecting the required columns
    df = df.select(['skeleton-id', 'branch-type', 'branch-distance', 'euclidean-distance'])

    # filtering of noise segments
    df = df.withColumn("relative-distance",
                       (abs(col('branch-distance') - col('euclidean-distance')) / col('branch-distance')))
    df = df.join(df_features, on='skeleton-id', how='inner')
    df = df.filter(((df['relative-distance'] >= 0.2) | (df['branch-type'] != 0) | (df['ratio-skel-contour'] <= 0.45)))
    df = delete_spurs(df)
    # base dataframe for statistics
    df = df.select(['skeleton-id', 'branch-type', 'branch-distance', 'euclidean-distance'])
    # df.createOrReplaceTempView('skeleton_edges')
    # df.groupBy('skeleton-id').agg(count(col('euclidean-distance'))).join(df_features, on='skeleton-id', how='inner').orderBy('skeleton-id').show(144)
    group_col, pivot_col = "skeleton-id", "branch-type"  # группировка по скелету, новые колонки по типу ребра
    agg_col_dst = 'branch-distance'
    agg_col_euc = 'euclidean-distance'
    df_dst = make_stats(df, group_col, agg_col_dst)
    df_euc = make_stats(df, group_col, agg_col_euc)
    df_pivot_count = df.groupBy(group_col).pivot(pivot_col).agg(count(col(pivot_col))).na.fill(0)

    df_pivot_count = df_pivot_count.withColumnsRenamed({
        "0": "endpoint-to-endpoint", \
        "1": "junction-to-endpoint", \
        "2": "junction-to-junction", \
        "3": "isolated cycle"})

    df_pivot_count = df_pivot_count.select(['skeleton-id', 'junction-to-endpoint', 'junction-to-junction']) \
        .withColumns({'sum-edges': col('junction-to-junction') + col('junction-to-endpoint')})
    df_pivot = df_pivot_count.join(df_features, on='skeleton-id')
    df_final = df_dst.join(df_euc, on='skeleton-id', how='left')
    df_final = df_final.withColumns(
        {'ratio-branch-distance-jte': col('branch-distance_sum_1') / col('branch-distance_sum'),
         'ratio-branch-distance-jtj': col('branch-distance_sum_2') / col('branch-distance_sum')})
    df_final = df_final.join(df_pivot, on='skeleton-id', how='left').orderBy('skeleton-id').na.fill(0)
    df_final = df_final.select([rnd(col_, 3).alias(col_name) for col_, col_name in zip(df_final.columns, df_final.columns)])
    return df_final








