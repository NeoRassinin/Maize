import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A

def get_train_augs():
    """Аугментации для обучения."""
    train_transform = [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.3, border_mode=0),
        A.GaussNoise(p=0.05),
        A.OneOf([
            A.CLAHE(p=0.6),
            A.RandomBrightnessContrast(p=0.6),
            A.RandomGamma(p=0.6),
        ], p=0.6),
        A.OneOf([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.6, 1.0)),
            A.Blur(blur_limit=3, p=0.1),
            A.MotionBlur(blur_limit=3, p=0.1),
        ], p=0.6),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.2), p=0.6),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.6),
        ], p=0.6),
    ]
    return A.Compose(train_transform)

def get_val_augs():
    """Аугментации для валидации и теста (только ресайз)."""
    return A.Compose([A.Resize(256, 256)], is_check_shapes=False)

class SegmentationDataset(Dataset):
    """Пользовательский датасет для сегментации изображений."""
    def __init__(self, df, augs=None):
        self.df = df
        self.augs = augs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        image = cv2.imread(sample.images_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample.masks_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)

        if self.augs:
            augmented = self.augs(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        image = torch.Tensor(image) / 255.0
        mask = torch.round(torch.Tensor(mask))

        return image, mask
