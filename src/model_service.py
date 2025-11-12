# src/model_service.py
# Loads VGG16, sets up hooks, and runs forward/backward passes for Grad-CAM.

# Import necessary libraries
import torch # for tensor operations
import torch.nn as nn # for neural network modules
from torchvision import models # for VGG16 model

# HookManager class definition
class HookManager:
    """
    Manages forward and backward hooks to extract activations and gradients.
    Clones/detaches tensors inside hooks to avoid autograd view + inplace issues.
    """
    # Constructor
    def __init__(self):
        self.activations = {}
        self.gradients = {}
        self._handles = []

    # Internal hook functions

    def _forward_hook(self, name):
        def hook(module, input, output):
            # quick debug:
            # print(f"🔥 Forward hook fired for {name} with output shape {output.shape if isinstance(output, torch.Tensor) else 'n/a'}")
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().clone()
            else:
                self.activations[name] = None
        return hook

    def _backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            grad = grad_output[0] if isinstance(grad_output, (list, tuple)) else grad_output
            if isinstance(grad, torch.Tensor):
                self.gradients[name] = grad.detach().clone()
            else:
                self.gradients[name] = None
        return hook

    # Clear all hooks and stored data
    def clear(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self.activations.clear()
        self.gradients.clear()

    def register(self, model: nn.Module, layer_indices: list[int]):
        """
        Registers hooks on given layer indices of model.features.
        Example: layer_indices=[14, 20, 30]
        """
        self.clear()
        print(f"Registering hooks on layers: {layer_indices}")
        for idx in layer_indices:
            layer = model.features[idx]
            layer_name = f"layer_{idx}_{layer.__class__.__name__}"

            fh = layer.register_forward_hook(self._forward_hook(layer_name))
            bh = layer.register_full_backward_hook(self._backward_hook(layer_name))

            self._handles.extend([fh, bh])
            print(f"✅ Hooked: {layer_name} ({idx})")

# ModelService class definition
class ModelService:
    """
    Loads VGG16, registers hooks, and provides forward/backward methods.
    """
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # use the new weights enum API; falls back if older torchvision installed
        try:
            self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(self.device)
        except Exception:
            # fallback for older torchvision
            self.model = models.vgg16(pretrained=True).to(self.device)
        
        # Disable inplace ReLU to avoid issues during backward
        self._disable_inplace_relu(self.model)
        
        self.model.eval()
        self.hook_manager = HookManager()
        self.output = None  # will hold last forward output
    
    def _disable_inplace_relu(self, module: nn.Module):
        """
        Recursively set inplace=False for all nn.ReLU instances.
        This avoids autograd view + inplace conflicts when using backward hooks.
        """
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                # replace with an equivalent non-inplace ReLU
                setattr(module, name, nn.ReLU(inplace=False))
            else:
                # recurse
                self._disable_inplace_relu(child)

    # Register hooks method
    def register_hooks(self, layer_indices: list[int]):
        """
        Registers hooks on the model using the HookManager.
        """
        self.hook_manager.register(self.model, layer_indices)

    def forward(self, input_tensor):
        """
        Runs a forward pass and returns logits and predicted class index.
        Also stores the output tensor on self.output (needed for backward).
        :param input_tensor: preprocessed input [1, 3, 224, 224]
        """
        input_tensor = input_tensor.to(self.device)
        # ensure gradients will be computed for backward pass (do NOT wrap with no_grad)
        self.model.zero_grad()
        output = self.model(input_tensor)  # shape [1, 1000]
        self.output = output  # store for backward reference
        pred_class = output.argmax(dim=1).item()
        return output, pred_class

    def backward(self, class_index, retain_graph=False):
        """
        Performs backward pass for Grad-CAM using the stored self.output.
        :param class_index: integer index of target class
        :param retain_graph: whether to retain graph (default False)
        """
        if self.output is None:
            raise RuntimeError("No forward output stored. Call forward() before backward().")
        self.model.zero_grad()
        # If you want to explain a particular class, use its logit (pre-softmax)
        class_score = self.output[0, class_index]
        # Backpropagate from this single scalar
        class_score.backward(retain_graph=retain_graph)

    def run(self, input_tensor, target_class=None):
        """
        Executes forward + backward passes, storing activations/gradients.
        If target_class is None, uses the predicted top-1 class.
        Returns: predicted_class, activations, gradients
        """
        self.hook_manager.activations.clear()
        self.hook_manager.gradients.clear()
        
        # Forward pass (fires forward hooks)
        output, pred_class = self.forward(input_tensor)

        # default to predicted class if user did not specify
        class_to_explain = pred_class if target_class is None else int(target_class)

        # backward to compute gradients w.r.t activations
        self.backward(class_to_explain)

        return class_to_explain, self.hook_manager.activations, self.hook_manager.gradients

