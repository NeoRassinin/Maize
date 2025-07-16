from config import (
    DATASET_PATH, TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH,
    VAL_IMAGES_PATH, VAL_MASKS_PATH, TEST_IMAGES_PATH, TEST_MASKS_PATH
)
import os
import shutil
from glob import glob
from PIL import Image

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A


def prepare_directories():
    """Создаёт необходимые директории для train/val/test выборок."""
    for path in [
        TRAIN_IMAGES_PATH,
        TRAIN_MASKS_PATH,
        VAL_IMAGES_PATH,
        VAL_MASKS_PATH,
        TEST_IMAGES_PATH,
        TEST_MASKS_PATH
    ]:
        os.makedirs(path, exist_ok=True)


def collect_dataset_paths():
    """Собирает пути к изображениям и маскам из исходных папок."""
    subdirs = sorted([
        os.path.join(DATASET_PATH, d)
        for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ])

    image_dirs = []
    mask_dirs = []
    for sd in subdirs:
        img, mask = sorted([
            os.path.join(sd, d)
            for d in os.listdir(sd)
            if os.path.isdir(os.path.join(sd, d))
        ], reverse=True)
        image_dirs.append(img)
        mask_dirs.append(mask)

    target_img_dir = os.path.join(DATASET_PATH, '..', 'images')
    target_mask_dir = os.path.join(DATASET_PATH, '..', 'masks')
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_mask_dir, exist_ok=True)

    for img_dir, mask_dir in zip(image_dirs, mask_dirs):
        img_files = sorted(glob(os.path.join(img_dir, '*.jpg')))
        mask_files = sorted(glob(os.path.join(mask_dir, '*.png')))
        for img, mask in zip(img_files, mask_files):
            shutil.copy(img, target_img_dir)
            shutil.copy(mask, target_mask_dir)

    image_paths = sorted(glob(os.path.join(target_img_dir, '*')))
    mask_paths = sorted(glob(os.path.join(target_mask_dir, '*')))
    return image_paths, mask_paths


def split_dataset(image_paths, mask_paths):
    """Разбивает данные на train, val и test."""
    df = pd.DataFrame({'images_path': image_paths, 'masks_path': mask_paths})
    train_df, valid_df = train_test_split(df, test_size=0.12, random_state=42)
    train_df, test_df = train_test_split(train_df, test_size=0.10, random_state=42)
    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


def cnt_num(number: int, max_zeros: int) -> str:
    """Генерирует строку с ведущими нулями."""
    zeros = '0' * max_zeros
    while number > 0:
        number //= 10
        zeros = zeros[:-1]
    return zeros


def make_subimages_by_cropping(
    images_list, masks_list, crop_size,
    output_images, output_masks
):
    """
    Делает подизображения фиксированного размера из исходных масок/изображений.

    Args:
        images_list (list): Пути к изображениям.
        masks_list (list): Пути к маскам.
        crop_size (int): Размер кадра.
        output_images (str): Папка сохранения изображений.
        output_masks (str): Папка сохранения масок.
    """
    for idx in range(len(masks_list)):
        image = np.array(Image.open(images_list[idx]))
        mask = np.array(Image.open(masks_list[idx]).convert("L"))
        h, w = mask.shape
        grid_w = w // crop_size
        grid_h = h // crop_size

        for i in range(grid_w - 1):
            for j in range(grid_h - 1):
                transform = A.Compose([
                    A.Crop(
                        x_min=i * crop_size,
                        y_min=j * crop_size,
                        x_max=(i + 2) * crop_size,
                        y_max=(j + 2) * crop_size,
                        p=1
                    )
                ])
                result = transform(image=image, mask=mask)
                number = grid_w * grid_h * idx + grid_h * i + j
                filename = f'{cnt_num(number, 5)}{number}'
                Image.fromarray(result['image']).save(
                    f'{output_images}/{filename}.jpg'
                )
                Image.fromarray(result['mask']).save(
                    f'{output_masks}/{filename}.png'
                )


def filter_small_contours(masks_dir, min_area=70):
    """Удаляет мелкие контуры из масок по площади."""
    mask_files = glob(os.path.join(masks_dir, '*.png'))
    for mask_file in mask_files:
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        new_mask = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(new_mask, [contour], -1, 1, -1)
        cv2.imwrite(mask_file, new_mask)

