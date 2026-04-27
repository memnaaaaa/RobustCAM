# RobustCAM — Faithful Grad-CAM Explanations for Lung CT Classification

RobustCAM fine-tunes ResNet50 on the IQ-OTH/NCCD 3-class lung CT dataset, fuses
Grad-CAM heatmaps across augmented views (Robust-CAM), quantifies explanation quality
using a unified 9-metric suite, and cross-validates saliency maps with LIME and SHAP.

## Research context

| Item | Detail |
|---|---|
| Primary paper | Panboonyuen 2026 — "Seeing Isn't Always Believing" |
| Secondary paper | Akgündoğdu & Çelikbaş 2025 — LIME+Grad-CAM+SHAP voting mask fusion |
| Published baseline | ResNet50 on IQ-OTH/NCCD: accuracy=0.85, sensitivity=0.82 (Panboonyuen Table I) |
| This project result | ResNet50 val_acc=0.8813 at epoch 18 ✓ |

## Dataset — IQ-OTH/NCCD Lung Cancer CT

| Class | Folder | Images | Train | Val | Test |
|---|---|---|---|---|---|
| 0 Normal | `Normal cases` | 416 | ~250 | ~83 | ~83 |
| 1 Benign | `Bengin cases` | 120 | ~72 | ~24 | ~24 |
| 2 Malignant | `Malignant cases` | 561 | ~337 | ~112 | ~112 |
| **Total** | | **1097** | **~659** | **~219** | **~219** |

Note: "Bengin cases" is the actual folder name (intentional misspelling in the dataset).

## Setup

```cmd
conda create -n robustcam python=3.12 -y
conda activate robustcam
pip install -r requirements.txt

# Additional XAI dependencies
conda activate robustcam && pip install lime
conda activate robustcam && pip install shap
```

## Phase progress

| Phase | Status | Description |
|---|---|---|
| Phase 0 | ✓ Complete | ResNet50 fine-tuning — val_acc=0.8813 (epoch 18) |
| Phase 1 | ✓ Complete | IQ-OTH/NCCD dataset loader with 60/20/20 split |
| Phase 2 | ✓ Complete | 9-metric faithfulness suite (`faithfulness_metrics.py`) |
| Phase 3 | ✓ Complete | LIME integration (`lime_service.py`) |
| Phase 4 | In progress | SHAP integration + cross-method voting mask |
| Phase 5 | In progress | Batch evaluation pipeline + results/ export |
| Phase 6 | Pending | Multi-arch support + augmentation extensions |

## Run training (Phase 0)

```cmd
conda activate robustcam
python src/train.py --data-root data --epochs 25 --batch-size 32
```

Checkpoint saved to `checkpoints/resnet50_iqothnc.pth`.
Training curves saved to `results/figures/training/training_curves_resnet50.png`.

## Run batch evaluation (Phase 5, when complete)

```cmd
# Grad-CAM only, 5 images (smoke test)
python src/eval_pipeline.py --split test --max-images 5 --no-lime --no-shap

# Full evaluation with LIME + SHAP
python src/eval_pipeline.py --split test --max-images 10 --run-name full_eval
```

## Project layout

```
src/
├── train.py                ← Phase 0: ResNet50 fine-tuning
├── iq_othncc_dataset.py    ← Phase 1: IQ-OTH/NCCD dataset loader
├── faithfulness_metrics.py ← Phase 2: 9-metric evaluation suite
├── lime_service.py         ← Phase 3: LIME explanation wrapper
├── shap_service.py         ← Phase 4: SHAP wrapper (pending)
├── xai_fusion.py           ← Phase 4: cross-method voting mask (pending)
├── eval_pipeline.py        ← Phase 5: batch evaluation + results/ export (pending)
├── model_service.py        ← model loading, hook manager, multi-arch support
├── data_service.py         ← image loading and preprocessing
├── gradcam_service.py      ← Grad-CAM computation
├── augmentation_service.py ← augmentation definitions
├── robust_cam.py           ← augmentation fusion + uncertainty
├── mlflow_service.py       ← MLflow experiment logging
├── pipeline.py             ← legacy VGG16 single-image CLI (preserved)
└── vgg_structure.py        ← VGG16 layer index helper (legacy)

results/
├── figures/
│   ├── training/           ← loss/accuracy curves from Phase 0
│   ├── qualitative/        ← per-image XAI panels (Phase 5)
│   └── quantitative/       ← metric bar charts (Phase 5)
└── tables/                 ← CSV metric tables (Phase 5)

```

## 9-Metric evaluation suite

| Group | Metric | Symbol | Source |
|---|---|---|---|
| A | Perturbation faithfulness | Faith↑ | Panboonyuen 2026 Eq. 5 |
| A | Localization accuracy | LocAcc↑ | Panboonyuen 2026 Eq. 4 |
| A | Explanation consistency (IoU) | Consist↑ | Panboonyuen 2026 Eq. 6 |
| B | Fidelity | Fid↑ | Akgündoğdu 2025 Eq. 11-15 |
| B | Stability | Stab↑ | Akgündoğdu 2025 Eq. 16-17 |
| B | Consistency (Pearson) | Cons↑ | Akgündoğdu 2025 Eq. 18 |
| C | Mean per-pixel variance | Var↓ | RobustCAM codebase |
| C | Mean top-k IoU | IoU_k↑ | RobustCAM codebase |
| C | Mean Spearman ρ | Spear↑ | RobustCAM codebase |

Group C metrics (Var, IoU_k, Spear) apply to Robust-CAM only (require multiple augmented views). Reported as NaN for single-view methods.

## MLflow tracking

```cmd
mlflow ui --backend-store-uri ./mlruns --port 5000
# Open http://127.0.0.1:5000
```

## Legacy VGG16 pipeline (preserved for reference)

```cmd
conda activate robustcam
python src/pipeline.py --image "data\Normal cases\Normal case (1).jpg" --layers 14 20 30 --num-aug 6 --fusion-method mean
```

Note: the legacy pipeline predicts ImageNet classes, not lung CT classes. Use `eval_pipeline.py` for all research-quality runs.

## Results

Final results tables and figures will be written to `results/` by Phase 5.
