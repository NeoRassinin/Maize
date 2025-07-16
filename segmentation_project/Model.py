import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss


def get_models(encoder, encoder_weights, activation):
    """Создает 3 модели: Unet, DeepLabV3, DeepLabV3Plus."""
    model_1 = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=activation
    )
    model_2 = smp.DeepLabV3(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=activation
    )
    model_3 = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=activation
    )
    return model_1, model_2, model_3


class SegmentationModel(nn.Module):
    """Обёртка над моделью сегментации с двумя функциями потерь."""
    def __init__(self, model):
        super(SegmentationModel, self).__init__()
        self.model = model

    def forward(self, images, masks=None):
        logits = self.model(images)

        if masks is not None:
            loss1 = DiceLoss(mode='binary')(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss1 + loss2

        return logits
