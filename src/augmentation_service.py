# src/augmentation_service.py
# Handles image augmentations and comparative Grad-CAM analysis.

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

class AugmentationService:
    """
    Applies deterministic augmentations to images and supports Grad-CAM comparison.
    """

    def __init__(self, image_size: int = 224, seed: int = 42):
        random.seed(seed)
        torch.manual_seed(seed)

        resize = T.Resize((image_size, image_size))

        self.augmentations = {
            "original": T.Compose([resize]),
            "horizontal_flip": T.Compose([T.RandomHorizontalFlip(p=1.0), resize]),
            "rotation_15": T.Compose([T.RandomRotation(15), resize]),
            "color_jitter": T.Compose([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                resize
            ]),
            "gaussian_blur": T.Compose([T.GaussianBlur(kernel_size=5), resize]),
            "grayscale": T.Compose([T.RandomGrayscale(p=1.0), resize]),
        }

    def apply(self, image: Image.Image):
        """
        Apply all augmentations to an image and return as dict[name] = PIL.Image.
        """
        aug_dict = {}
        for name, transform in self.augmentations.items():
            aug_img = transform(image)
            aug_dict[name] = aug_img
        return aug_dict

    def visualize_comparison(self, overlays_dict):
        """
        Display original + augmented Grad-CAM overlays side-by-side for visual comparison.
        """
        num_imgs = len(overlays_dict)
        fig, axes = plt.subplots(1, num_imgs, figsize=(4*num_imgs, 4))
        if num_imgs == 1:
            axes = [axes]
        for ax, (name, img) in zip(axes, overlays_dict.items()):
            ax.imshow(img)
            ax.set_title(name)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def to_numpy(self, image: Image.Image) -> np.ndarray:
        """
        Utility to convert PIL image to numpy array (RGB).
        """
        return np.array(image)
