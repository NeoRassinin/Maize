import cv2
import matplotlib.pyplot as plt
import numpy as np

def generate_sample_images(image_paths):
    """Отображает до 2 изображений в ряду."""
    _, ax = plt.subplots(1, 2, figsize=(15, 6))
    for i, path in enumerate(image_paths):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax[i].imshow(image)
        ax[i].set_title("IMAGE")
        ax[i].axis("off")
    plt.tight_layout()
    plt.show()

def generate_sample_masks(mask_paths):
    """Отображает до 2 бинарных масок в ряду."""
    _, ax = plt.subplots(1, 2, figsize=(15, 6))
    for i, path in enumerate(mask_paths):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255.0
        ax[i].imshow(mask, cmap='gray')
        ax[i].set_title("GROUND TRUTH")
        ax[i].axis("off")
    plt.tight_layout()
    plt.show()

def visualize(**images):
    """Отображает любое количество изображений в одной строке."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name)
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.tight_layout()
    plt.show()

def processed_image(dataset, idx):
    """Отображает одно изображение и его маску из датасета."""
    image, mask = dataset[idx]
    image_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.squeeze().numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.axis('off')
    plt.title("IMAGE")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.axis('off')
    plt.title("GROUND TRUTH")
    plt.tight_layout()
    plt.show()
