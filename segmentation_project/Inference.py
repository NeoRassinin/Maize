import torch
import numpy as np


def make_inference(model, dataset, device, index):
    """
    Делает инференс на одном элементе датасета.

    Args:
        model: обученная модель сегментации
        dataset: датасет
        device: CUDA или CPU
        index: индекс изображения

    Returns:
        image: исходное изображение
        mask: ground truth маска
        pred: предсказанная маска
    """
    model.eval()
    image, mask = dataset[index]
    image = image.to(device).unsqueeze(0)
    mask = mask.to(device)

    with torch.no_grad():
        pred = model(image)
        pred = (pred > 0.5).float()

    return image.squeeze().cpu(), mask.squeeze().cpu(), pred.squeeze().cpu()
