"""
Phase 0 — Fine-tune ResNet50 on IQ-OTH/NCCD lung cancer CT dataset.

Usage:
    python src/train.py --data-root data --epochs 25 --batch-size 32
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# ── constants ─────────────────────────────────────────────────────────────────

CLASS_DIRS  = {0: "Normal cases", 1: "Bengin cases", 2: "Malignant cases"}
CLASS_NAMES = {0: "Normal", 1: "Benign", 2: "Malignant"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── dataset ───────────────────────────────────────────────────────────────────

class IQOTHNCCDTorchDataset(Dataset):
    """
    PyTorch Dataset wrapping the IQ-OTH/NCCD directory structure.
    Applies ImageNet-standard preprocessing (resize 224, normalize).
    Supports train/val/test splits via a pre-shuffled index list.
    """

    def __init__(self, data_root: str, samples: list[tuple[str, int]], augment: bool = False):
        """
        samples: list of (image_path, label) — produced by split_dataset()
        augment: if True, applies random horizontal flip + rotation during training
        """
        self.samples = samples
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)
        return tensor, label


def split_dataset(
    data_root: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Discovers all images, shuffles deterministically, splits into
    train/val/test lists of (image_path, label) tuples.
    Returns: (train_samples, val_samples, test_samples)
    """
    all_samples = []
    for label, dir_name in CLASS_DIRS.items():
        class_dir = os.path.join(data_root, dir_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        for f in files:
            all_samples.append((os.path.join(class_dir, f), label))

    rng = random.Random(seed)
    rng.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_samples = all_samples[:n_train]
    val_samples   = all_samples[n_train:n_train + n_val]
    test_samples  = all_samples[n_train + n_val:]

    print(f"Dataset split — train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}")
    return train_samples, val_samples, test_samples


# ── model ─────────────────────────────────────────────────────────────────────

def build_resnet50(num_classes: int = 3, freeze_backbone: bool = True) -> nn.Module:
    """
    Loads ImageNet-pretrained ResNet50.
    Replaces the fc head with a num_classes linear layer.
    If freeze_backbone=True, freezes all layers except the new fc head.
    Returns the modified model.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    # replace head — in_features is 2048 for ResNet50
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ── training utilities ────────────────────────────────────────────────────────

def compute_class_weights(samples: list[tuple[str, int]], num_classes: int = 3) -> torch.Tensor:
    """
    Computes inverse-frequency class weights to handle dataset imbalance.
    Returns a float tensor of shape [num_classes] for nn.CrossEntropyLoss(weight=...).
    """
    counts = [0] * num_classes
    for _, label in samples:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (num_classes * c) if c > 0 else 1.0 for c in counts]
    print(f"Class counts: {counts}, weights: {[f'{w:.3f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


def save_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_path: str = "results/figures/training/training_curves_resnet50.png",
):
    """
    Saves a 2-panel figure (loss | accuracy) across epochs to save_path.
    Called at the end of train() regardless of whether training converged.
    Also logs the figure to MLflow as an artifact.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, val_losses,   "r-o", label="Val Loss",   markersize=4)
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, val_accs,   "r-o", label="Val Acc",   markersize=4)
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {save_path}")

    # log to MLflow if a run is active
    try:
        import mlflow
        if mlflow.active_run() is not None:
            mlflow.log_artifact(save_path, artifact_path="figures/training")
    except Exception as e:
        print(f"[Warning] Could not log training curves to MLflow: {e}")


# ── main training loop ────────────────────────────────────────────────────────

def train(
    data_root: str = "data",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    freeze_backbone: bool = True,
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
    seed: int = 42,
    experiment_name: str = "ResNet50_IQ_OTH_NCCD_Finetune",
):
    """
    Full training loop. Logs train/val loss and accuracy to MLflow each epoch.
    Saves best checkpoint to checkpoints/resnet50_iqothnc.pth.
    Saves training curves to results/figures/training/training_curves_resnet50.png.

    The saved checkpoint dict contains:
        {
            "epoch": int,
            "model_state_dict": ...,
            "val_acc": float,
            "val_loss": float,
            "class_names": {0: "Normal", 1: "Benign", 2: "Malignant"},
            "arch": "resnet50",
            "num_classes": 3,
        }
    """
    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    curves_dir = os.path.join(results_dir, "figures", "training")
    os.makedirs(curves_dir, exist_ok=True)

    # data
    train_samples, val_samples, _ = split_dataset(data_root, seed=seed)

    train_dataset = IQOTHNCCDTorchDataset(data_root, train_samples, augment=True)
    val_dataset   = IQOTHNCCDTorchDataset(data_root, val_samples,   augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    # model
    model = build_resnet50(num_classes=3, freeze_backbone=freeze_backbone)
    model = model.to(device)

    # optimizer — only train fc if backbone is frozen
    if freeze_backbone:
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam([
            {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": lr * 0.1},
            {"params": model.fc.parameters(), "lr": lr},
        ])

    # weighted loss
    class_weights = compute_class_weights(train_samples).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # MLflow setup
    try:
        import mlflow
        mlflow.set_experiment(experiment_name)
        mlflow_run = mlflow.start_run()
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "freeze_backbone": freeze_backbone,
            "seed": seed,
            "arch": "resnet50",
            "num_classes": 3,
        })
        use_mlflow = True
    except Exception as e:
        print(f"[Warning] MLflow not available: {e}")
        use_mlflow = False

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc  = 0.0
    best_ckpt_path = os.path.join(checkpoint_dir, "resnet50_iqothnc.pth")

    try:
        for epoch in range(1, epochs + 1):
            # ── train ──
            model.train()
            running_loss = 0.0
            correct = 0
            total   = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += inputs.size(0)

            train_loss = running_loss / total
            train_acc  = correct / total

            # ── validate ──
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total   = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total   += inputs.size(0)

            val_loss = val_running_loss / val_total
            val_acc  = val_correct / val_total

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(
                f"Epoch [{epoch:3d}/{epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if use_mlflow:
                try:
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "train_acc":  train_acc,
                        "val_loss":   val_loss,
                        "val_acc":    val_acc,
                    }, step=epoch)
                except Exception as e:
                    print(f"[Warning] MLflow metrics log failed: {e}")

            # save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          val_acc,
                    "val_loss":         val_loss,
                    "class_names":      CLASS_NAMES,
                    "arch":             "resnet50",
                    "num_classes":      3,
                }
                torch.save(checkpoint, best_ckpt_path)
                print(f"  -> Saved best checkpoint (val_acc={val_acc:.4f}) to {best_ckpt_path}")

    finally:
        # always save curves
        curves_path = os.path.join(curves_dir, "training_curves_resnet50.png")
        save_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)

        if use_mlflow:
            try:
                mlflow.log_metrics({"best_val_acc": best_val_acc})
                mlflow.end_run()
            except Exception as e:
                print(f"[Warning] MLflow end_run failed: {e}")

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
    print(f"Checkpoint: {best_ckpt_path}")
    return best_ckpt_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fine-tune ResNet50 on IQ-OTH/NCCD")
    p.add_argument("--data-root",      default="data")
    p.add_argument("--epochs",         default=25, type=int)
    p.add_argument("--batch-size",     default=32, type=int)
    p.add_argument("--lr",             default=1e-3, type=float)
    p.add_argument("--no-freeze",      action="store_true")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--results-dir",    default="results")
    p.add_argument("--seed",           default=42, type=int)
    args = p.parse_args()

    train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_backbone=not args.no_freeze,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        seed=args.seed,
    )
