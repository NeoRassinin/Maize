import os

import findspark
from pyspark import SparkConf
from pyspark.sql import SparkSession

from config import SPARK_CONF, JAVA_HOME_PATH


def init_spark(app_name: str = "PySpark", master: str = "local[*]") -> SparkSession:
    """
    Инициализация Spark-сессии с кастомной конфигурацией.

    Args:
        app_name (str): Название Spark-приложения.
        master (str): Адрес мастера (локально или кластер).

    Returns:
        SparkSession: Инициализированная Spark-сессия.
    """
    os.environ["JAVA_HOME"] = JAVA_HOME_PATH
    findspark.init()

    conf = SparkConf().setMaster(master)
    for key, value in SPARK_CONF.items():
        conf.set(key, value)

    spark = (
        SparkSession.builder
        .appName(app_name)
        .config(conf=conf)
        .getOrCreate()
    )

    return spark


if __name__ == "__main__":
    spark = init_spark()
