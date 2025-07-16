import os
import logging
import warnings
from glob import glob
from spark_session import init_spark
import optuna
from io_utils import safe_mkdir

spark = init_spark()


# Логгеры
logging.getLogger("pyspark").setLevel(logging.ERROR)

log4j_logger = spark._jvm.org.apache.log4j
log4j_logger.LogManager.getLogger(
    "org.apache.spark.sql.execution.window.WindowExec"
).setLevel(log4j_logger.Level.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
warnings.filterwarnings("ignore")


optuna.logging.set_verbosity(optuna.logging.WARNING)

# Рабочие директории
WORK_DIR = os.getcwd()

DATASET_PATH = os.path.join(WORK_DIR, "Dataset")
PREDICT_PATH = os.path.join(WORK_DIR, "PREDICTS_")
safe_mkdir(DATASET_PATH)
safe_mkdir(PREDICT_PATH)

# Вложенные директории датасета
dirs = sorted([
    os.path.join(DATASET_PATH, d)
    for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

ALE_PATH, ORTHO_CROP_PATH, POS_PATH = dirs

ALE_PATH_MASKS, ALE_PATH_IMAGES = sorted([
    os.path.join(ALE_PATH, d)
    for d in os.listdir(ALE_PATH)
    if os.path.isdir(os.path.join(ALE_PATH, d))
])

ORTHO_PATH_MASKS, ORTHO_PATH_IMAGES = sorted([
    os.path.join(ORTHO_CROP_PATH, d)
    for d in os.listdir(ORTHO_CROP_PATH)
    if os.path.isdir(os.path.join(ORTHO_CROP_PATH, d))
])

POS_PATH_MASKS, POS_PATH_IMAGES = sorted([
    os.path.join(POS_PATH, d)
    for d in os.listdir(POS_PATH)
    if os.path.isdir(os.path.join(POS_PATH, d))
])

# Пути к изображениям и маскам
ale_image_path_list = sorted(glob(os.path.join(ALE_PATH_IMAGES, "*")))
ale_mask_path_list = sorted(glob(os.path.join(ALE_PATH_MASKS, "*")))

pos_image_path_list = sorted(glob(os.path.join(POS_PATH_IMAGES, "*")))
pos_mask_path_list = sorted(glob(os.path.join(POS_PATH_MASKS, "*")))

# Папки с предсказаниями
pos_predict_image_path = os.path.join(WORK_DIR, "POS_PREDICT_FULL")
ale_predict_image_path = os.path.join(WORK_DIR, "ALE_PREDICT_FULL")
safe_mkdir(pos_predict_image_path)
safe_mkdir(ale_predict_image_path)

pos_predict_image_path_list = sorted(glob(os.path.join(POS_PATH_MASKS, "*")))
ale_predict_image_path_list = sorted(glob(os.path.join(ALE_PATH_IMAGES, "*")))

# Используемые изображения и маски
pos_images_path = pos_predict_image_path
pos_masks_path = os.path.join(WORK_DIR, "POS_PREDICT_U-Net-512_TRUE")

ale_images_path = ale_predict_image_path
ale_masks_path = os.path.join(WORK_DIR, "ALE_PREDICT_U-Net-512_TRUE")

list_image_path = (
    sorted(glob(os.path.join(ale_images_path, "*")))[:] +
    sorted(glob(os.path.join(pos_images_path, "*")))
)

list_mask_path = (
    sorted(glob(os.path.join(ale_masks_path, "*")))[:] +
    sorted(glob(os.path.join(pos_masks_path, "*")))
)

# LABEL директории
ale_label_path = os.path.join(WORK_DIR, "ALE_LABEL")
pos_label_path = os.path.join(WORK_DIR, "POS_LABEL")
ortho_label_path = os.path.join(WORK_DIR, "ORTHO_LABEL")

safe_mkdir(ale_label_path)
safe_mkdir(pos_label_path)
safe_mkdir(ortho_label_path)

# Счётчики
safe_mkdir(os.path.join(WORK_DIR, "counting"))

# Полный список изображений и масок
image_path_list = (
    pos_image_path_list + ale_image_path_list
)
mask_path_list = (
    pos_mask_path_list + ale_mask_path_list
)

print("Длина list_image_path:", len(list_image_path))
print("Длина list_mask_path:", len(list_mask_path))
