# src/pipeline.py
r"""
All-in-one pipeline orchestrator for:
 - DataService (loading / preprocessing / augmentations)
 - ModelService (VGG16 + hooks)
 - GradCAMService (stagewise heatmaps + overlays)
 - AugmentationService (multiple augmentations)
 - MLflowService (logging everything into one run)
Usage (example):
    python src/pipeline.py --image "C:\path\to\img.jpg" --layers 14 20 30 --use-postgres
"""
# Import necessary libraries
import argparse # for argument parsing
import os # for file operations
import pprint # for pretty-printing
import numpy as np # for numerical operations

# Importing service modules
from data_service import DataService # for data handling
from model_service import ModelService # for model and hooks
from gradcam_service import GradCAMService # for Grad-CAM computations
from augmentation_service import AugmentationService # for image augmentations
from mlflow_service import MLflowService # for MLflow logging

# Main pipeline function
def run_pipeline(
        image_path: str,
        layers: list[int],
        use_postgres: bool = False,
        do_augmentations: bool = True,
        run_name: str | None = None,
):
    # --- instantiate services
    ds = DataService()
    ms = ModelService()
    gradcam = GradCAMService()
    aug = AugmentationService()
    mlf = MLflowService()

    # --- load & preprocess image
    print(f"\n[1] Loading image: {image_path}")
    input_tensor, orig_pil = ds.get_image_tensor(image_path, augment=False)
    orig_np = np.array(orig_pil)

    # --- register hooks once for desired layers
    print(f"\n[2] Registering hooks on layers: {layers}")
    ms.register_hooks(layers)

    # --- run forward+backward on base image and compute Grad-CAMs
    print("\n[3] Running model and computing stagewise Grad-CAM for base image...")
    class_idx, activations, gradients = ms.run(input_tensor)

    # generate stagewise heatmaps + overlays using GradCAMService
    print("\n[4] Generating stagewise outputs (heatmaps + overlays)")
    stage_heatmaps, stage_overlays = gradcam.generate_stagewise_outputs(orig_np, activations, gradients)

    # --- Start MLflow run and log base results
    run_params = {
        "image_path": os.path.abspath(image_path),
        "layers": ", ".join(map(str, layers)),
        "explained_class_index": int(class_idx),
        "do_augmentations": bool(do_augmentations),
    }
    mlf.start_run(run_name=run_name, params=run_params)

    print("\n[5] Logging base stagewise heatmaps and overlays to MLflow")
    mlf.log_stagewise_heatmaps(stage_heatmaps)
    mlf.log_augmented_overlays(stage_overlays)  # "augmented" naming is generic; these are overlays

    # --- Augmentation loop: compute and log per-augmentation
    if do_augmentations:
        print("\n[6] Applying augmentations and logging results per augmentation")
        augmented_imgs = aug.apply(orig_pil)  # dict: name -> PIL.Image
        for aug_name, aug_img in augmented_imgs.items():
            print(f"  -> Augmentation: {aug_name}")
            # preprocess augmented image (returns tensor and PIL) -- use preprocess directly
            tensor = ds.preprocess(aug_img).unsqueeze(0) if isinstance(ds.preprocess(aug_img), np.ndarray) else ds.preprocess(aug_img)
            # Note: DataService.preprocess returns a tensor already with batch dim in your earlier impl (it did .unsqueeze(0))
            # To be safe, do:
            aug_tensor = ds.preprocess(aug_img)  # returns [1,3,H,W]
            # run model on augmented image
            class_idx_aug, activations_aug, gradients_aug = ms.run(aug_tensor)
            # generate heatmaps + overlays for augmented image
            aug_heatmaps, aug_overlays = gradcam.generate_stagewise_outputs(np.array(aug_img), activations_aug, gradients_aug)
            # log under augmentation subfolder in the same MLflow run
            mlf.log_augmented_results(aug_name, aug_heatmaps, aug_overlays)

    # --- optional metrics
    mlf.log_scalar("explained_class_index", int(class_idx))
    mlf.log_scalar("num_stage_maps", len(stage_heatmaps))

    # --- finish run
    mlf.end_run()
    print("\n[7] Pipeline finished. All artifacts logged to MLflow.")

def parse_args():
    p = argparse.ArgumentParser(description="Run full Grad-CAM pipeline and log to MLflow")
    p.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    p.add_argument("--layers", "-l", type=int, nargs="+", default=[14, 20, 30], help="Layer indices to hook")
    p.add_argument("--no-augment", dest="augment", action="store_false", help="Disable augmentations")
    p.add_argument("--use-postgres", action="store_true", help="Use PostgreSQL MLflow backend (requires configured mlflow_service)")
    p.add_argument("--run-name", type=str, default=None, help="Optional MLflow run name")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Pipeline arguments:")
    pprint.pprint(vars(args))

    run_pipeline(
        image_path=args.image,
        layers=args.layers,
        use_postgres=args.use_postgres,
        do_augmentations=args.augment,
        run_name=args.run_name,
    )