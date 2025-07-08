import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
import pyspark
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions
from pyspark.sql import Window
from pyspark.sql.functions import row_number
from pyspark.sql.functions import lit, desc, col, size, array_contains\
, isnan, udf, hour, array_min, array_max, countDistinct, count, abs\
, percentile_approx, least, greatest, min, max, sum, avg, expr, percentile
from pyspark.sql.functions import round as rnd
from pyspark.sql import DataFrameStatFunctions as stat
from pyspark.sql.types import *
from pyspark.sql.functions import monotonically_increasing_id
import skan
import cv2
import numpy as np
import pandas as pd
from glob import glob
from skimage import measure, morphology, filters, feature
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from skan import Skeleton, summarize
pd.DataFrame.iteritems = pd.DataFrame.items
from scipy.stats import mode
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

import optuna
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold

os.environ["JAVA_HOME"] = r"D:\idea\OpenJDK11U-jdk_x64_windows_hotspot_11.0.26_4\jdk-11.0.26+4"

import findspark
findspark.init()

MAX_MEMORY = '15G'
# Initialize a spark session.
conf = pyspark.SparkConf().setMaster("local[*]") \
        .set('spark.executor.heartbeatInterval', 10000) \
        .set('spark.network.timeout', 10000) \
        .set("spark.core.connection.ack.wait.timeout", "3600") \
        .set("spark.executor.memory", MAX_MEMORY) \
        .set("spark.driver.memory", MAX_MEMORY) \
        .set("spark.sql.execution.arrow.pyspark.enabled", "true")
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Pyspark_") \
        .config(conf=conf) \
        .getOrCreate()
    return spark

spark = init_spark()

import logging
logging.getLogger('pyspark').setLevel(logging.ERROR)
log4j_logger = spark._jvm.org.apache.log4j
log4j_logger.LogManager.getLogger("org.apache.spark.sql.execution.window.WindowExec").setLevel(log4j_logger.Level.ERROR)

optuna.logging.set_verbosity(optuna.logging.WARNING)

WORK_DIR = os.getcwd() # рабочий каталог
DATASET_PATH = WORK_DIR + '/Dataset'
os.makedirs(DATASET_PATH, exist_ok = True)

PREDICT_PATH = WORK_DIR + '/PREDICTS_'
os.makedirs(PREDICT_PATH, exist_ok=True)

dirs = sorted([
    os.path.join(DATASET_PATH, d)
    for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])
ALE_PATH, ORTHO_CROP_PATH, POS_PATH = dirs
ALE_PATH_MASKS, ALE_PATH_IMAGES = sorted([os.path.join(ALE_PATH, d) for d in os.listdir(ALE_PATH) if os.path.isdir(os.path.join(ALE_PATH, d))])
ORTHO_PATH_MASKS, ORTHO_PATH_IMAGES = sorted([os.path.join(ORTHO_PATH, d) for d in os.listdir(ORTHO_PATH) if os.path.isdir(os.path.join(ORTHO_PATH, d))])
POS_PATH_MASKS, POS_PATH_IMAGES = sorted([os.path.join(POS_PATH, d) for d in os.listdir(POS_PATH) if os.path.isdir(os.path.join(POS_PATH, d))])

ale_image_path_list = sorted(glob(ALE_PATH_IMAGES + '/*'))
ale_mask_path_list = sorted(glob(ALE_PATH_MASKS + '/*'))
ortho_image_path_list = sorted(glob(ORTHO_PATH_IMAGES + '/*'))
ortho_mask_path_list = sorted(glob(ORTHO_PATH_MASKS + '/*'))
pos_image_path_list = sorted(glob(POS_PATH_IMAGES + '/*'))
pos_mask_path_list = sorted(glob(POS_PATH_MASKS + '/*'))

pos_predict_image_path = WORK_DIR + '/POS_PREDICT_FULL'
ale_predict_image_path = WORK_DIR + '/ALE_PREDICT_FULL'
pos_predict_image_path_list = sorted(glob(POS_PATH_MASKS + '/*'))
ale_predict_image_path_list = sorted(glob(ALE_PATH_IMAGES + '/*'))

pos_images_path = WORK_DIR + '/POS_PREDICT_FULL'
pos_masks_path = WORK_DIR + '/POS_PREDICT_U-Net-512_TRUE'
ale_images_path = WORK_DIR + '/ALE_PREDICT_FULL'
ale_masks_path = WORK_DIR + '/ALE_PREDICT_U-Net-512_TRUE'


list_image_path = sorted(glob(ale_images_path + '/*'))[5:] + sorted(glob(pos_images_path + '/*'))
list_mask_path = sorted(glob(ale_masks_path + '/*'))[5:] + sorted(glob(pos_masks_path + '/*'))

ortho_path = WORK_DIR + '/PREDICT_ORTHO_U-Net'
list_mask_path = os.listdir(ortho_path)
list_mask_ortho_path = sorted(glob(ortho_path + '/*'))

cols = ['skeleton-id',
 'branch-distance_sum',
 'branch-distance_avg',
 'branch-distance_min',
 'branch-distance_max',
 'branch-distance_median',
 'branch-distance_sum_1',
 'branch-distance_avg_1',
 'branch-distance_min_1',
 'branch-distance_max_1',
 'branch-distance_median_1',
 'branch-distance_sum_2',
 'branch-distance_avg_2',
 'branch-distance_min_2',
 'branch-distance_max_2',
 'branch-distance_median_2',
 'euclidean-distance_sum',
 'euclidean-distance_avg',
 'euclidean-distance_min',
 'euclidean-distance_max',
 'euclidean-distance_median',
 'euclidean-distance_sum_1',
 'euclidean-distance_avg_1',
 'euclidean-distance_min_1',
 'euclidean-distance_max_1',
 'euclidean-distance_median_1',
 'euclidean-distance_sum_2',
 'euclidean-distance_avg_2',
 'euclidean-distance_min_2',
 'euclidean-distance_max_2',
 'euclidean-distance_median_2',
 'ratio-branch-distance-jte',
 'ratio-branch-distance-jtj',
 'junction-to-endpoint',
 'junction-to-junction',
 'sum-edges',
 'width',
 'height',
 'area',
 'length',
 'ratio-skel-contour',
 'area-box-ratio',
 'box-ratio',
 'target']

ale_label_path = WORK_DIR + '/ALE_LABEL'
pos_label_path = WORK_DIR + '/POS_LABEL'
ortho_label_path = WORK_DIR + '/ORTHO_LABEL'
os.makedirs(pos_label_path, exist_ok=True)
os.makedirs(ale_label_path, exist_ok=True)
os.makedirs(ortho_label_path, exist_ok=True)

os.makedirs("counting", exist_ok=True)

image_path_list = pos_image_path_list + ortho_image_path_list + ale_image_path_list
mask_path_list = pos_mask_path_list + ortho_mask_path_list + ale_mask_path_list


