# src/gradcam_service.py
# Unified Grad-CAM service with visualization + MLflow-ready outputs

# Importing necessary libraries
import torch # for tensor operations
import numpy as np # for numerical operations
import cv2 # for image processing
import matplotlib.pyplot as plt # for plotting

class GradCAMService:
    """
    Generates Grad-CAM heatmaps and overlays for selected CNN layers.
    Combines the visual fidelity of the original implementation
    with MLflow-friendly outputs (NumPy arrays of the overlays).
    """

    def __init__(self):
        pass

    def _compute_gradcam(self, activation: torch.Tensor, gradient: torch.Tensor):
        """
        Compute Grad-CAM heatmap from given activation & gradient tensors.
        """
        grad = gradient.mean(dim=(2, 3), keepdim=True)         # [1, C, 1, 1]
        cam = (grad * activation).sum(dim=1, keepdim=True)     # [1, 1, H, W]
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # normalize to [0, 1]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam

    def _overlay(self, orig_img, heatmap, alpha=0.4):
        """
        Overlay heatmap on top of the original RGB image.
        """
        if not isinstance(orig_img, np.ndarray):
            orig_img = np.array(orig_img)

        if orig_img.ndim == 2:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)

        heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR),
                                  1 - alpha, heatmap_color, alpha, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        return overlay

    def visualize_gradcam(self, orig_img, activation, gradient, layer_name=None):
        """
        Visualize a single layer's Grad-CAM result and return the matplotlib figure.
        """
        heatmap = self._compute_gradcam(activation, gradient)
        overlay = self._overlay(orig_img, heatmap)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(orig_img)
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(overlay)
        ax[1].set_title(f"Grad-CAM: {layer_name or ''}")
        ax[1].axis("off")
        plt.tight_layout()
        return fig, heatmap, overlay

    def generate_stagewise_outputs(self, orig_img, activations, gradients):
        """
        Generate and return per-layer Grad-CAM heatmaps + overlays as numpy arrays.
        """
        heatmaps = {}
        overlays = {}

        for layer_name, act in activations.items():
            grad = gradients.get(layer_name)
            if act is None or grad is None:
                print(f"[Warning] Skipping {layer_name}: missing activation/gradient.")
                continue

            with torch.no_grad():
                fig, heatmap, overlay = self.visualize_gradcam(orig_img, act, grad, layer_name)
                # convert matplotlib fig to numpy array for MLflow logging
                fig.canvas.draw()
                img_array = np.array(fig.canvas.renderer.buffer_rgba())
                plt.close(fig)

            heatmaps[layer_name] = (heatmap * 255).astype(np.uint8)
            overlays[layer_name] = img_array

        print(f"✅ Generated {len(heatmaps)} stagewise Grad-CAM maps.")
        return heatmaps, overlays
