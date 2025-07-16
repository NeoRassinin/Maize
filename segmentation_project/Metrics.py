import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torch


def compute_metrics(model, dataloader, device):
    """
    Вычисляет метрики качества сегментации: Accuracy, IoU, F1, Precision, Recall.

    Args:
        model: обученная модель
        dataloader: DataLoader с тестовыми или валидационными данными
        device: CUDA или CPU
    """
    model.eval()
    all_true = []
    all_pred = []
    confusion = np.zeros((2, 2))

    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            pred_masks = (outputs > 0.5).float()

            true_np = masks.cpu().numpy().ravel()
            pred_np = pred_masks.cpu().numpy().ravel()
            confusion += confusion_matrix(true_np, pred_np, labels=[0, 1])

    TN, FP, FN, TP = confusion.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    iou = TP / (TP + FP + FN + 1e-7)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"IoU      : {iou:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
