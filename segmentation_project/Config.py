import os
import torch

# Общая рабочая директория
WORK_DIR = os.getcwd()

# Пути к директориям с данными
DATASET_PATH = os.path.join(WORK_DIR, 'Dataset')
TRAIN_IMAGES_PATH = os.path.join(WORK_DIR, 'train/images')
TRAIN_MASKS_PATH = os.path.join(WORK_DIR, 'train/masks')
VAL_IMAGES_PATH = os.path.join(WORK_DIR, 'val/images')
VAL_MASKS_PATH = os.path.join(WORK_DIR, 'val/masks')
TEST_IMAGES_PATH = os.path.join(WORK_DIR, 'test/images')
TEST_MASKS_PATH = os.path.join(WORK_DIR, 'test/masks')

# Параметры аугментации и модели
CROP_SIZE = 256
BATCH_SIZE = 8
MIN_CONTOUR_AREA = 70

# Настройки модели
ENCODER = 'efficientnet-b7'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
LR = 1e-5

# Устройства
DEVICE1 = "cuda:1" if torch.cuda.is_available() else "cpu"
DEVICE2 = "cuda:2" if torch.cuda.is_available() else "cpu"
DEVICE3 = "cuda:3" if torch.cuda.is_available() else "cpu"
