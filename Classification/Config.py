"""
Конфигурационный файл проекта: настройки Spark, пути, параметры, имена признаков.
"""

# Параметры памяти Spark
MAX_MEMORY = "15G"

# Конфигурация Spark
SPARK_CONF = {
    "spark.executor.heartbeatInterval": "10000",
    "spark.network.timeout": "10000",
    "spark.core.connection.ack.wait.timeout": "3600",
    "spark.executor.memory": MAX_MEMORY,
    "spark.driver.memory": MAX_MEMORY,
    "spark.sql.execution.arrow.pyspark.enabled": "true",
}

# Путь к JDK
JAVA_HOME_PATH = r"D:\idea\OpenJDK11U-jdk_x64_windows_hotspot_11.0.26_4\jdk-11.0.26+4"


# Список признаков
cols = [
    "skeleton-id",
    "branch-distance_sum",
    "branch-distance_avg",
    "branch-distance_min",
    "branch-distance_max",
    "branch-distance_median",
    "branch-distance_sum_1",
    "branch-distance_avg_1",
    "branch-distance_min_1",
    "branch-distance_max_1",
    "branch-distance_median_1",
    "branch-distance_sum_2",
    "branch-distance_avg_2",
    "branch-distance_min_2",
    "branch-distance_max_2",
    "branch-distance_median_2",
    "euclidean-distance_sum",
    "euclidean-distance_avg",
    "euclidean-distance_min",
    "euclidean-distance_max",
    "euclidean-distance_median",
    "euclidean-distance_sum_1",
    "euclidean-distance_avg_1",
    "euclidean-distance_min_1",
    "euclidean-distance_max_1",
    "euclidean-distance_median_1",
    "euclidean-distance_sum_2",
    "euclidean-distance_avg_2",
    "euclidean-distance_min_2",
    "euclidean-distance_max_2",
    "euclidean-distance_median_2",
    "ratio-branch-distance-jte",
    "ratio-branch-distance-jtj",
    "junction-to-endpoint",
    "junction-to-junction",
    "sum-edges",
    "width",
    "height",
    "area",
    "length",
    "ratio-skel-contour",
    "area-box-ratio",
    "box-ratio",
    "target",
]

# список лучших признаков на основе shap-анализа
best_features = ['branch-distance_sum_2',
                 'branch-distance_sum',
                 'euclidean-distance_sum_2',
                 'euclidean-distance_sum',
                 'max-box-side',
                 'area-box-ratio',
                 'length',
                 'euclidean-distance_sum_1',
                 'euclidean-distance_max_1'
]


