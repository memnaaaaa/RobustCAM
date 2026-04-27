# CLAUDE.md — RobustCAM Project Guide

This file is the authoritative guide for all development on this project.
Read it fully before touching any code. Every command, path, and phase
instruction here is specific to the actual repo layout and dataset on disk.

---

## Project summary

RobustCAM enhances Grad-CAM faithfulness for lung CT classification by:
1. Fine-tuning ResNet50 on the IQ-OTH/NCCD 3-class lung CT task to produce a
   clinically meaningful classifier before any XAI analysis is run.
2. Fusing Grad-CAM heatmaps across multiple augmented views of each image.
3. Quantitatively validating explanations using a unified 9-metric suite drawn
   from both source papers and the existing RobustCAM codebase.
4. Cross-validating Robust-CAM regions against LIME and SHAP outputs.

**Primary backbone: ResNet50** — fine-tuned on IQ-OTH/NCCD (3 classes).
This directly matches Panboonyuen 2026, which tests ResNet50 on the same dataset
and reports 0.85 accuracy, making results directly comparable to the published baseline.
VGG16 remains in the codebase as a legacy fallback but is not the primary model.

**Primary research papers:**
- Panboonyuen 2026 — "Seeing Isn't Always Believing" (Grad-CAM faithfulness analysis
  on IQ-OTH/NCCD using ResNet50, ResNet101, DenseNet161, EfficientNet-B0, ViT-Base)
- Akgündoğdu & Çelikbaş 2025 — "Explainable deep learning for brain tumor detection"
  (LIME + Grad-CAM + SHAP voting mask fusion with two-stage training)

**Dataset:** IQ-OTH/NCCD lung cancer CT — 3 classes, JPEG images.
- `data/Normal cases/` — 416 images  (label 0)
- `data/Bengin cases/` — 120 images  (label 1, folder name intentionally misspelled)
- `data/Malignant cases/` — 561 images  (label 2)

---

## Environment — mandatory

All code MUST run inside the `robustcam` conda environment. Never run
scripts with a different Python or global pip.

```cmd
conda activate robustcam
```

Verify before running anything:
```cmd
conda activate robustcam && python --version
```
Expected: Python 3.12.x

Install new packages only with:
```cmd
conda activate robustcam && pip install <package>
```

Never use `pip install --break-system-packages` or modify the base environment.

---

## Repo layout

```
RobustCAM/
├── CLAUDE.md                   ← this file
├── README.md
├── requirements.txt            ← includes lime==0.2.0.1, shap==0.51.0
├── .gitignore
├── checkpoints/
│   └── resnet50_iqothnc.pth   ← fine-tuned checkpoint: epoch=18, val_acc=0.8813
├── data/
│   ├── Bengin cases/           ← 120 CT JPEGs (class 1, folder name intentionally misspelled)
│   ├── Malignant cases/        ← 561 CT JPEGs (class 2)
│   ├── Normal cases/           ← 416 CT JPEGs (class 0)
│   └── IQ-OTH_NCCD lung cancer dataset.txt
├── report/
│   ├── midterm_report.md       ← primary report; contains Section 4.4 XAI figures
│   ├── midterm_report.tex
│   └── midterm_report.pdf
├── results/                    ← all report-ready outputs; see results/README.md
│   ├── README.md
│   ├── figures/
│   │   ├── qualitative/        ← *_xai_comparison.png + individual overlays (Phase 4 ✓)
│   │   ├── quantitative/       ← metric bar charts (Phase 5, pending)
│   │   └── training/           ← training_curves_resnet50.png (Phase 0 ✓)
│   └── tables/                 ← metrics_table_test.csv (Phase 5, pending)
└── src/
    ├── pipeline.py             ← VGG16 legacy CLI; do NOT modify
    ├── gradcam_service.py      ← Grad-CAM computation, architecture-agnostic
    ├── augmentation_service.py ← 6 augmentation types + meta dict for warp
    ├── robust_cam.py           ← fuse_mean/median, warp_heatmap_back, global_stability_metrics
    ├── model_service.py        ← ModelService + HookManager; arch/checkpoint params added ✓
    ├── data_service.py         ← image loading + ImageNet preprocessing
    ├── mlflow_service.py       ← MLflow logging (PostgreSQL + local fallback)
    ├── vgg_structure.py        ← VGG16 layer index helper (legacy only)
    │
    │   — BUILT —
    ├── train.py                ← Phase 0 ✓ ResNet50 fine-tuning on IQ-OTH/NCCD
    ├── iq_othncc_dataset.py    ← Phase 1 ✓ dataset loader (658/219/220 split)
    ├── faithfulness_metrics.py ← Phase 2 ✓ 9-metric suite, compute_all_metrics()
    ├── lime_service.py         ← Phase 3 ✓ LIME explanation wrapper
    ├── shap_service.py         ← Phase 4 ✓ SHAP via GradientExplainer
    ├── xai_fusion.py           ← Phase 4 ✓ pixel-wise voting mask + colormap
    ├── visualize_xai.py        ← Phase 4 ✓ qualitative figure generator (report output)
    │
    │   — BUILT —
    └── eval_pipeline.py        ← Phase 5 ✓ batch evaluation + metrics_table_test.csv
```

---

## Metric system — all 9 metrics, their sources, and what they measure

The project computes a unified 9-metric suite. All 9 are computed for every explanation
method (Grad-CAM baseline, Robust-CAM, LIME, SHAP, voting mask) and logged to MLflow.
They are also exported as a CSV table to `results/tables/metrics_table_test.csv` for
direct use in the report.

### Group A — Panboonyuen 2026 faithfulness metrics (3 metrics)
Defined in `src/faithfulness_metrics.py` (Phase 2). Measure whether the explanation
reflects the model's actual decision-making.

| Metric | Symbol | Source | What it measures |
|---|---|---|---|
| Perturbation faithfulness | Faith | Panboonyuen Eq. 5 | Softmax confidence drop when top-k salient pixels are zeroed out. Higher = more faithful. |
| Localization accuracy | LocAcc | Panboonyuen Eq. 4 | IoU between explanation mask and ground-truth tumor region. Returns NaN for IQ-OTH/NCCD (no pixel masks). |
| Explanation consistency | Consist | Panboonyuen Eq. 6 | Mean IoU of heatmaps across random seeds / augmentation runs. Higher = more stable. |

### Group B — Akgündoğdu 2025 explainability quality metrics (3 metrics)
Defined in `src/faithfulness_metrics.py` (Phase 2). Measure XAI method quality
independent of the specific model or dataset.

| Metric | Symbol | Source | What it measures |
|---|---|---|---|
| Fidelity | Fid | Akgündoğdu Eq. 11-15 | Confidence change when salient mask is applied vs removed. Positive = explanation highlights causally important regions. |
| Stability | Stab | Akgündoğdu Eq. 16-17 | Pearson correlation between explanations for an image and its noise-perturbed version. Higher = more robust to input noise. |
| Consistency (XAI) | Cons | Akgündoğdu Eq. 18 | Average pairwise Pearson correlation across n repeated explanation runs. Higher = more reproducible. |

Note: Akgündoğdu's Consistency (Cons) and Panboonyuen's Explanation Consistency (Consist)
measure the same concept via different formulas (Pearson vs IoU). Report both — the
difference in values is itself a finding about metric sensitivity.

### Group C — RobustCAM augmentation stability metrics (3 metrics)
Computed by the existing `robust_cam.global_stability_metrics()` function. Measure how
stable the fused heatmap is across the augmentation sweep.

| Metric | Symbol | Source | What it measures |
|---|---|---|---|
| Mean per-pixel variance | Var | `robust_cam.py` | Average variance of heatmap values across all augmented views. Lower = more stable fusion. |
| Mean top-k IoU | IoU_k | `robust_cam.py` | Average IoU between each augmented heatmap's top-10% region and the fused map's top-10% region. Higher = augmentations agree on salient regions. |
| Mean Spearman ρ | Spear | `robust_cam.py` | Average rank correlation between each augmented heatmap and the fused map. Higher = augmentations preserve spatial ranking. |

Group C metrics only apply to Robust-CAM (since they require multiple augmented views).
They are reported as NaN for single-view methods (Grad-CAM baseline, LIME, SHAP).

### Report comparison table structure
The final table in the report compares all methods across all applicable metrics:

| Method | Faith↑ | Fid↑ | Stab↑ | Cons↑ | Consist↑ | Var↓ | IoU_k↑ | Spear↑ | LocAcc↑ |
|---|---|---|---|---|---|---|---|---|---|
| Grad-CAM (baseline) | | | | | | NaN | NaN | NaN | NaN |
| Robust-CAM (mean) | | | | | | | | | NaN |
| Robust-CAM (median) | | | | | | | | | NaN |
| LIME | | | | | | NaN | NaN | NaN | NaN |
| SHAP | | | | | | NaN | NaN | NaN | NaN |
| Voting mask | | | | | | NaN | NaN | NaN | NaN |

↑ = higher is better, ↓ = lower is better, NaN = not applicable for this method.

---

## Existing code — what works and what needs fixing

### `src/data_service.py`
**Status:** Functional for single-image CLI use. Needs extension for dataset-level use.
- `get_image_tensor()` works correctly and uses standard ImageNet normalization
  (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) — keep this unchanged.
  ResNet50 pretrained weights expect exactly these normalization values.
- Missing: no batch loading, no class label handling, no train/val/test split.
- Do NOT modify the existing methods — extend only.

### `src/model_service.py`
**Status:** Extended (Phase 6a minimal complete). Supports ResNet50 + checkpoint loading.
- `ModelService(arch, checkpoint_path)` — `arch` defaults to `"vgg16"` for backward compat.
- `register_hooks_by_name(layer_names)` added — architecture-agnostic via `named_modules()`.
- `_disable_inplace_relu()` still present — applies to ResNet too; do not remove.
- `run()` remains the canonical forward+backward entry point.
- Full multi-arch (densenet161, efficientnet_b0, vit_b_16) is Phase 6a-full, still pending.

### `src/gradcam_service.py`
**Status:** Fully functional and architecture-agnostic.
- `compute_raw_heatmap()` returns float32 [0,1] heatmaps — use for all metric computation.
- `generate_stagewise_raw()` is preferred over `generate_stagewise_outputs()` for pipelines.
- `_overlay()` produces RGB numpy arrays suitable for MLflow artifact logging.
- No changes needed here — works identically for ResNet50 and VGG16.

### `src/augmentation_service.py`
**Status:** Functional but limited augmentation set.
- Only 6 augmentation types; rotation fixed at 15°.
- `meta` dict is used by `robust_cam.warp_heatmap_back()` — do not change its schema.
- When adding new augmentations, always populate `meta` with `{"type": "none"}` for
  non-geometric transforms so the warp logic degrades gracefully.

### `src/robust_cam.py`
**Status:** Fully functional. Provides Group C metrics via `global_stability_metrics()`.
- `warp_heatmap_back()` correctly inverts flips and rotations.
- `fuse_mean()`, `fuse_median()`, `compute_uncertainty()` are production-ready.
- `global_stability_metrics()` returns mean_variance, mean_iou_topk, mean_spearman
  (Group C). These are called by `eval_pipeline.py` for Robust-CAM runs only.
- **Gap:** No Group A/B faithfulness metrics — built in Phase 2.

### `src/pipeline.py`
**Status:** Functional for single-image CLI runs using VGG16/ImageNet (legacy).
- Preserved as-is. The new ResNet50-based evaluation uses `eval_pipeline.py` instead.
- Do NOT modify `pipeline.py`.

### `src/mlflow_service.py`
**Status:** Fully functional.
- PostgreSQL backend with local `mlruns/` fallback.
- Uses `.env` for `MLFLOW_TRACKING_URI_POSTGRES`.

### `src/vgg_structure.py`
**Status:** Legacy utility script only.
- Run as: `conda activate robustcam && python src/vgg_structure.py`
- Only relevant for the legacy `pipeline.py` VGG16 flow.

---

## Development phases

Work through phases in order. Each phase has a definition of done —
do not proceed to the next phase until it is met.

---

### Phase 0 — Fine-tune ResNet50 on IQ-OTH/NCCD ✓ COMPLETE

**File:** `src/train.py` | **Checkpoint:** `checkpoints/resnet50_iqothnc.pth`

- ResNet50 + ImageNet pretrained weights; frozen backbone; 3-class fc head
- Adam lr=1e-3, 25 epochs, batch_size=32, seed=42, num_workers=0
- Weighted cross-entropy: `w_c = N_total / (num_classes × N_c)` — Benign ~4.6× weight
- Best checkpoint: **epoch=18, val_acc=0.8813** (published baseline: 0.85)
- Training curves saved to `results/figures/training/training_curves_resnet50.png`

**Checkpoint format:**
```python
{
    "epoch": 18,
    "model_state_dict": ...,
    "val_acc": 0.8813,
    "val_loss": float,
    "class_names": {0: "Normal", 1: "Benign", 2: "Malignant"},
    "arch": "resnet50",
    "num_classes": 3,
}
```

**Verify:**
```cmd
D:/miniconda3/envs/robustcam/python.exe -c "
import torch, os
ckpt = torch.load('checkpoints/resnet50_iqothnc.pth', map_location='cpu')
print(f'epoch={ckpt[\"epoch\"]}, val_acc={ckpt[\"val_acc\"]:.4f}')
"
```

---

### Phase 1 — Dataset loader for IQ-OTH/NCCD ✓ COMPLETE

**File:** `src/iq_othncc_dataset.py`

- `IQOTHNCCDDataset(data_root, split, seed=42)` — returns `(path, label, class_name)` tuples
- Deterministic 60/20/20 split matching `train.py` seed=42: **658 train / 219 val / 220 test**
- `split="all"` returns full dataset; `get_all_samples()` returns the active split
- `"Bengin cases"` folder spelling preserved exactly as on disk — do not rename

---

### Phase 2 — Unified 9-metric evaluation suite ✓ COMPLETE

**File:** `src/faithfulness_metrics.py`

`compute_all_metrics()` returns a flat dict with all 9 keys, ready for MLflow and CSV:

| Key | Group | Formula source |
|---|---|---|
| `faith` | A | Panboonyuen Eq. 5 — confidence drop when top-k pixels zeroed |
| `loc_acc` | A | Panboonyuen Eq. 4 — IoU vs GT mask; always NaN (no pixel labels) |
| `consist_iou` | A | Panboonyuen Eq. 6 — mean IoU across repeated runs |
| `fidelity` | B | Akgündoğdu Eq. 11-15 — Fid+ − Fid− |
| `stability` | B | Akgündoğdu Eq. 16-17 — Pearson ρ vs noise-perturbed input |
| `consist_pearson` | B | Akgündoğdu Eq. 18 — mean pairwise Pearson across runs |
| `mean_variance` | C | `robust_cam.global_stability_metrics()` — NaN if no aug_heatmaps |
| `mean_iou_topk` | C | `robust_cam.global_stability_metrics()` — NaN if no aug_heatmaps |
| `mean_spearman` | C | `robust_cam.global_stability_metrics()` — NaN if no aug_heatmaps |

Key constraint: all Group B metrics use `model_service.forward()` directly — NOT `run()`.

---

### Phase 3 — LIME integration ✓ COMPLETE

**File:** `src/lime_service.py` | **Dependency:** `lime==0.2.0.1`

- `LIMEService(num_samples, random_state)` wraps `lime.lime_image.LimeImageExplainer`
- `explain(pil_img, predict_fn, target_class)` → float32 [224,224] normalized to [0,1]
- `build_predict_fn(model_service, data_service)` → `[N, 3]` softmax array; calls
  `model_service.model` inside `torch.no_grad()` — does NOT call `run()`
- Quick test: `num_samples=200`; production: `num_samples=500`

---

### Phase 4 — SHAP + voting mask + qualitative visualizations ✓ COMPLETE

**Files:** `src/shap_service.py`, `src/xai_fusion.py`, `src/visualize_xai.py`
**Dependency:** `shap==0.51.0`

#### SHAP (`src/shap_service.py`)
- `SHAPService.explain(ms, input_tensor, background_tensor, target_class)` →
  float32 [224,224]: `shap.GradientExplainer` → abs channel-sum → normalize
- `build_background_tensor(ds, train_paths, n_background=10)` → [N, 3, 224, 224]
- Background must be training images only; ≥5 samples; model in `eval()` mode

#### Voting mask (`src/xai_fusion.py`)
- `compute_voting_mask(gradcam, lime, shap)` → int [224,224] ∈ {0,1,2,3} (Akgündoğdu Eq. 7-9)
- `voting_mask_to_colormap()` → uint8 RGB: 3→red, 2→purple, 1→pink, 0→blue
- `compute_high_confidence_mask(mask, min_votes=2)` → binary agreement mask

#### Qualitative visualizations (`src/visualize_xai.py`)
Standalone report-output script. Run once per class set to regenerate figures.

- Runs Grad-CAM (single view), Robust-CAM (6 augmented views → `fuse_mean`), and LIME
- SHAP is computed in the codebase but **excluded from report figures**
- Comparison panel: **Original CT | Grad-CAM | Robust-CAM | LIME** (4 columns, 16×4 in, 150 dpi)
- Outputs to `results/figures/qualitative/`: individual `*_overlay.png` + `*_xai_comparison.png`
- Already generated for all 3 classes: Normal (conf=0.854), Benign (conf=0.612), Malignant (conf=0.925)
- Added to `report/midterm_report.md` Section 4.4

**Run:**
```cmd
D:/miniconda3/envs/robustcam/python.exe src/visualize_xai.py --images-per-class 1 --lime-samples 200
```

**Args:** `--images-per-class N` (default 1), `--lime-samples` (default 500), `--seed` (default 42)

---

### Phase 5 — Batch evaluation pipeline + results/ export

**Goal:** Wire everything together into `eval_pipeline.py`. This file is responsible for
two outputs simultaneously: MLflow (for experiment tracking) and `results/` (for the
report). Every figure and table that goes into the report is written here.

**File to create:** `src/eval_pipeline.py`

**What to build:**

```python
def run_eval_pipeline(
    data_root: str,
    checkpoint_path: str = "checkpoints/resnet50_iqothnc.pth",
    arch: str = "resnet50",
    layers: list[str] = ["layer3", "layer4"],
    split: str = "test",
    num_aug: int = 6,
    fusion_method: str = "mean",
    run_lime: bool = True,
    run_shap: bool = True,
    n_shap_background: int = 10,
    lime_num_samples: int = 500,
    max_images: int = None,
    results_dir: str = "results",
    experiment_name: str = "RobustCAM_ResNet50_IQ_OTH_NCCD",
    run_name: str = None,
):
    """
    Per-image loop:
    1.  Load image. Run fine-tuned ResNet50 forward+backward for Grad-CAM.
    2.  Compute baseline Grad-CAM heatmap (layer4, single-image, no augmentation).
    3.  Run Robust-CAM augmentation sweep → fused heatmap + aug_heatmaps list.
    4.  (if run_lime) Run LIME → lime_heatmap.
    5.  (if run_shap) Run SHAP → shap_heatmap.
    6.  (if both) Compute voting mask.
    7.  Compute all 9 metrics for each available heatmap via compute_all_metrics().
        - Grad-CAM baseline: Group A+B only; Group C = NaN.
        - Robust-CAM: all 9 metrics (pass aug_heatmaps + fused_heatmap for Group C).
        - LIME: Group A+B only; Group C = NaN.
        - SHAP: Group A+B only; Group C = NaN.
        - Voting mask: Group A+B only; Group C = NaN.
    8.  Save qualitative panel figure to results/figures/qualitative/.
    9.  Log all artifacts + metrics to MLflow.

    After the loop:
    10. Aggregate metrics across all images → mean per method per metric.
    11. Export results/tables/metrics_table_<split>.csv (rows=methods, cols=metrics).
    12. Save bar-chart figures to results/figures/quantitative/.
    13. Save classification report CSV to results/tables/classification_report.csv.
    14. End MLflow run.

    IMPORTANT: Instantiate ModelService, LIMEService, SHAPService ONCE before the
    loop. Do NOT reinstantiate them per image.
    """
    ...
```

**MLflow metrics logged per image (prefix = `<class_name>_<image_stem>`):**
```
<prefix>_gradcam_faith       <prefix>_gradcam_fidelity      <prefix>_gradcam_stability
<prefix>_gradcam_consist_iou <prefix>_gradcam_consist_pearson
<prefix>_robustcam_faith     <prefix>_robustcam_fidelity     <prefix>_robustcam_stability
<prefix>_robustcam_consist_iou <prefix>_robustcam_consist_pearson
<prefix>_robustcam_mean_variance <prefix>_robustcam_mean_iou_topk <prefix>_robustcam_mean_spearman
<prefix>_lime_faith          <prefix>_lime_fidelity          <prefix>_lime_stability
<prefix>_shap_faith          <prefix>_shap_fidelity          <prefix>_shap_stability
```

**MLflow aggregate metrics (logged once at run end):**
```
mean_gradcam_faith           mean_gradcam_fidelity         mean_gradcam_stability
mean_robustcam_faith         mean_robustcam_fidelity        mean_robustcam_stability
mean_robustcam_mean_variance mean_robustcam_mean_iou_topk  mean_robustcam_mean_spearman
mean_lime_faith              mean_lime_fidelity             mean_lime_stability
mean_shap_faith              mean_shap_fidelity             mean_shap_stability
val_acc_from_checkpoint
```

**`results/` outputs written by this function:**

```
results/
├── figures/
│   ├── qualitative/
│   │   ├── normal_<stem>_xai_panel.png      ← 1 per Normal image sampled
│   │   ├── benign_<stem>_xai_panel.png      ← 1 per Benign image sampled
│   │   └── malignant_<stem>_xai_panel.png   ← 1 per Malignant image sampled
│   └── quantitative/
│       ├── metric_comparison_faith.png      ← bar chart: Faith per method
│       ├── metric_comparison_fidelity.png   ← bar chart: Fidelity per method
│       ├── metric_comparison_stability.png  ← bar chart: Stability per method
│       ├── metric_comparison_consistency.png← bar chart: both Consist metrics
│       └── robustcam_stability_metrics.png  ← bar chart: Var / IoU_k / Spear
└── tables/
    ├── metrics_table_<split>.csv            ← full 9-metric table, rows=methods
    └── classification_report.csv           ← per-class precision/recall/F1
```

**Qualitative panel figure format:**
Each `*_xai_panel.png` is a single matplotlib figure with 2 rows × 4 columns:
- Row 1: Original CT | Grad-CAM overlay | Robust-CAM overlay | Uncertainty map
- Row 2: LIME overlay | SHAP overlay | Voting colormap | Confidence bar (3 classes)

This is the primary visualization for the report. Aim for at least one panel per class.

**CSV table format for `metrics_table_<split>.csv`:**
```
Method,faith,loc_acc,consist_iou,fidelity,stability,consist_pearson,mean_variance,mean_iou_topk,mean_spearman
Grad-CAM,0.xxx,NaN,0.xxx,0.xxx,0.xxx,0.xxx,NaN,NaN,NaN
Robust-CAM (mean),0.xxx,NaN,0.xxx,0.xxx,0.xxx,0.xxx,0.xxx,0.xxx,0.xxx
Robust-CAM (median),0.xxx,...
LIME,0.xxx,...
SHAP,0.xxx,...
Voting mask,0.xxx,...
```

**CLI entry point:**
```python
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",   default="data")
    p.add_argument("--checkpoint",  default="checkpoints/resnet50_iqothnc.pth")
    p.add_argument("--arch",        default="resnet50")
    p.add_argument("--layers",      default=["layer3","layer4"], nargs="+")
    p.add_argument("--split",       default="test", choices=["train","val","test","all"])
    p.add_argument("--num-aug",     default=6, type=int)
    p.add_argument("--fusion",      default="mean", choices=["mean","median"])
    p.add_argument("--max-images",  default=None, type=int)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--no-lime",     action="store_true")
    p.add_argument("--no-shap",     action="store_true")
    p.add_argument("--run-name",    default=None)
    p.add_argument("--experiment",  default="RobustCAM_ResNet50_IQ_OTH_NCCD")
    args = p.parse_args()
    run_eval_pipeline(**vars(args))
```

**Smoke test (Grad-CAM only, 5 images):**
```cmd
conda activate robustcam
cd D:\RobustCAM
python src/eval_pipeline.py --split test --max-images 5 --no-lime --no-shap --run-name phase5_smoke
```

**Full test (all methods, 10 images):**
```cmd
conda activate robustcam
cd D:\RobustCAM
python src/eval_pipeline.py --split test --max-images 10 --run-name phase5_full
```

**Definition of done:**
```cmd
conda activate robustcam
python -c "
import os
assert os.path.exists('results/tables/metrics_table_test.csv'), 'CSV not written'
import pandas as pd
df = pd.read_csv('results/tables/metrics_table_test.csv')
assert 'Method' in df.columns and 'faith' in df.columns and 'fidelity' in df.columns
assert 'mean_variance' in df.columns
print(df.to_string(index=False))
print('Phase 5 PASSED')
"
```

---

### Phase 6 — Improvements to existing files

#### Phase 6a (minimal) ✓ COMPLETE — `src/model_service.py`

The subset needed for Phases 2–4 is already implemented:
- `ModelService(arch="vgg16", checkpoint_path=None)` — `arch` defaults for backward compat
- Checkpoint loading: ImageNet weights first, then overwritten by `model_state_dict`
- `HookManager.register_by_name(model, layer_names)` — via `named_modules()`
- `ModelService.register_hooks_by_name(layer_names)` — call before `run()`
- Supported arches so far: `vgg16` (legacy), `resnet50`, `resnet101`

#### Phase 6a (full multi-arch) — PENDING

Extend `model_service.py` to all 6 architectures. Add ViT degraded-heatmap warning.

**Target architectures:**

| Architecture | Hook layers | Notes |
|---|---|---|
| resnet50 | `layer3`, `layer4` | Primary — all main experiments |
| resnet101 | `layer3`, `layer4` | Same structure as resnet50 |
| densenet161 | `features.denseblock3`, `features.denseblock4` | Last two dense blocks |
| efficientnet_b0 | `features.6`, `features.7`, `features.8` | Last three MBConv stages |
| vit_b_16 | `encoder.layers.encoder_layer_10`, `encoder.layers.encoder_layer_11` | Expect degraded output |
| vgg16 | integer `register_hooks([14, 20, 30])` | Legacy only |

Inspect names: `python -c "from torchvision import models; m = models.densenet161(weights='DEFAULT'); [print(n) for n,_ in m.named_modules() if n]"`

#### Phase 6b — PENDING — minor extensions to existing files

- **`src/augmentation_service.py`:** add `rotation_neg15`, `rotation_10`, `center_crop`, `contrast_high` — always populate `meta` with `{"type": "none"}` for non-geometric transforms
- **`src/robust_cam.py`:** add `fuse_weighted(heatmaps, weights)` — weighted average, weights must sum to 1.0
- **`src/gradcam_service.py`:** add `resize_heatmap_to_image(heatmap, target_h, target_w)` — bilinear resize, clipped to [0,1]
- **`src/mlflow_service.py`:** add `log_voting_mask_artifacts(voting_mask, colormap, image_stem)` — logs grayscale + RGB colormap under `images/<stem>/`

---

### Phase 7 — Optional: two-stage training with XAI masks

Implement only if Phases 0-6 are complete and time permits.
Create `src/two_stage_trainer.py`. Do not modify any other existing file.

---

## Running the legacy single-image pipeline (VGG16)

```cmd
conda activate robustcam
cd D:\RobustCAM
python src/pipeline.py --image "data\Normal cases\Normal case (1).jpg" --layers 14 20 30 --num-aug 6 --fusion-method mean
```

This explains arbitrary ImageNet class predictions, not lung CT classes.
Use `eval_pipeline.py` for all research-quality runs.

---

## Installing new dependencies

```cmd
conda activate robustcam && pip install lime
conda activate robustcam && pip install shap
conda activate robustcam && pip install scipy
```

Update `requirements.txt` after installing. Freeze after Phase 0:
```cmd
conda activate robustcam && pip freeze > requirements_freeze.txt
```

---

## Coding conventions

- **Python 3.12.** Use `list[str]`, `dict[str, int]`, `X | None` type hints directly.
- **Docstrings.** Every public function needs one. Multi-line for non-obvious behavior.
- **No bare `except`.** Catch `Exception as e` at minimum.
- **No `print()` in library functions** except progress indicators; use `[Warning]` prefix.
- **Imports:** stdlib → third-party → local. All at top of file.
- **No global state.** Instantiate every service; no module-level mutable globals.
- **Path handling.** Always `os.path.join()`. Never hardcode absolute paths.
- **Windows DataLoader.** Always `num_workers=0`. `num_workers > 0` without a spawn
  guard crashes Windows with a recursion error.
- **results/ writes.** Always call `os.makedirs(path, exist_ok=True)` before writing
  to any subdirectory of `results/` — directories may not exist yet.

---

## Common failure modes and fixes

**`ModuleNotFoundError: No module named 'src'`**
- Run from `D:\RobustCAM\`. Or prepend `sys.path.insert(0, 'src')`.

**`RuntimeError: Expected all tensors to be on the same device`**
- Call `.to(ms.device)` on all tensors before passing to the model.

**`RuntimeError: No forward output stored`**
- Called `backward()` before `forward()`. Always use `model_service.run()`, or call
  `forward()` then `backward()` in sequence.

**`KeyError` or `RuntimeError` when loading checkpoint**
- Print `ckpt.keys()` to inspect. Expected format documented in `train.py`.

**Predicted class > 2 (e.g., 207, 388)**
- Checkpoint was not loaded. The model outputs 1000-class ImageNet logits.
  Verify `checkpoint_path` is correct and the file exists.

**`compute_all_metrics()` returns all NaN for Group A/B**
- The model is likely not in `eval()` mode, causing batch norm to behave differently,
  or the input tensor is on a different device than the model.

**`LIME taking forever`**
- Use `num_samples=200` for quick tests. `--max-images 5` for smoke tests.

**`shap.GradientExplainer` giving NaN values**
- Background tensor needs at least 5 samples. Model must be in `eval()` mode.

**Training loss stuck / val_acc < 0.70**
- Print class weights tensor before training — verify they are not all 1.0.
- Try `--no-freeze` for full backbone fine-tuning.
- Verify `seed=42` is used consistently in `train.py`.

**`ValueError: Layer 'X' not found in model`**
- Print valid names with `dict(model.named_modules()).keys()`.

**Grad-CAM heatmap is a flat gray blob**
- Hook did not fire. Check `register_hooks_by_name()` was called before `run()`.
- For ViT: flat heatmaps are expected — documented in Panboonyuen 2026.

**`results/` figure not written**
- Missing `os.makedirs(path, exist_ok=True)` before writing. The directories exist
  but subdirectories created at runtime must be made explicitly.

**MLflow run stuck active**
- Call `mlf.end_run()` in a `finally:` block. Restart Python if stuck.

---

## Key numbers for the IQ-OTH/NCCD dataset

| Class | Folder | Count | Train (60%) | Val (20%) | Test (20%) |
|---|---|---|---|---|---|
| 0 Normal | `Normal cases` | 416 | ~250 | ~83 | ~83 |
| 1 Benign | `Bengin cases` | 120 | ~72 | ~24 | ~24 |
| 2 Malignant | `Malignant cases` | 561 | ~337 | ~112 | ~112 |
| **Total** | | **1097** | **~659** | **~219** | **~219** |

Always use weighted cross-entropy during training. Always report per-class metrics.

**Published baseline (Panboonyuen 2026, Table I — same dataset):**

| Metric | ResNet50 | ResNet101 | DenseNet161 |
|---|---|---|---|
| Sensitivity | 0.82 | 0.85 | 0.87 |
| Specificity | 0.90 | 0.92 | 0.89 |
| Accuracy | 0.85 | 0.87 | 0.88 |
| Inference (ms) | 12 | 15 | 18 |

---

## Runtime note (2026-03-18)
`conda run` is broken in this shell. Use `D:/miniconda3/envs/robustcam/python.exe` directly.

---

## Deliverables checklist

- [x] Phase 0: ResNet50 fine-tuned — epoch=18, val_acc=0.8813 ✓
      `checkpoints/resnet50_iqothnc.pth` + `results/figures/training/training_curves_resnet50.png`
- [x] Phase 1: `src/iq_othncc_dataset.py` — 658 train / 219 val / 220 test, seed=42 ✓
- [x] Phase 6a (minimal): `model_service.py` — `arch` param, `checkpoint_path`, `register_hooks_by_name` ✓
- [x] Phase 2: `src/faithfulness_metrics.py` — all 9 metrics, `compute_all_metrics()` ✓
      Group C returns NaN when `aug_heatmaps=None`; all Group B use `forward()` not `run()`
- [x] Phase 3: `src/lime_service.py` — 3-class predict_fn, heatmap shape (224,224) ✓
- [x] Phase 4: `src/shap_service.py` + `src/xai_fusion.py` + `src/visualize_xai.py` ✓
      Qualitative figures (Original | Grad-CAM | Robust-CAM | LIME) in `results/figures/qualitative/`
      3 comparison panels in `report/midterm_report.md` Section 4.4
      Normal conf=0.854, Benign conf=0.612, Malignant conf=0.925

- [x] Phase 5: `src/eval_pipeline.py` ✓ COMPLETE (2026-03-19)
      → `results/tables/metrics_table_test.csv` (9 metrics × all methods)
      → `results/figures/quantitative/` bar charts (faith, fidelity, stability, consistency, robustcam)
      → `results/tables/classification_report.csv`
      → `results/figures/qualitative/*_xai_panel.png` (2-row × 4-col panels per image)
      → all metrics logged to MLflow per image + aggregated
      Smoke test: 5 images, Grad-CAM + Robust-CAM only — PASSED
      Note: Run with --no-lime --no-shap for fast eval; full run adds LIME + SHAP + voting mask
      Note: Use PYTHONIOENCODING=utf-8 or Python -X utf8 if emoji print errors occur (handled in code)
- [x] Phase 6a (full): `model_service.py` — densenet161, efficientnet_b0, vit_b_16 support ✓
      Added `vit_b_16` head rebuild (`model.heads.head`) in checkpoint loading section
- [x] Phase 6b: augmentation extensions, `fuse_weighted`, `resize_heatmap_to_image`, voting mask MLflow logging ✓
      `augmentation_service.py`: added rotation_neg15, rotation_10, center_crop, contrast_high
      `robust_cam.py`: added fuse_weighted(heatmaps, weights) — normalized weighted average
      `gradcam_service.py`: added resize_heatmap_to_image(heatmap, h, w) — bilinear, clipped [0,1]
      `mlflow_service.py`: added log_voting_mask_artifacts(voting_mask, colormap, image_stem)
- [ ] Final report:
      (a) Classification results vs Panboonyuen 2026 Table I baseline (ResNet50 row)
      (b) 9-metric comparison table from `results/tables/metrics_table_test.csv`
      (c) Qualitative XAI panels but this time with SHAP included (Original | Grad-CAM | Robust-CAM | LIME | SHAP)
      (d) Discussion: does Robust-CAM improve Faith/Fidelity/Stability vs single-image
          Grad-CAM? What do the Group C metrics reveal about augmentation agreement?
