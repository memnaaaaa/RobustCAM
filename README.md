# RobustCAM: Stagewise Grad-CAM with Augmentation Fusion and Uncertainty Estimation

A compact research pipeline created as part of an Intro to Deep Learning project. RobustCAM helps inspect how convolutional neural networks attend to input images at multiple intermediate stages, measure how attention shifts under common image augmentations, and produce fused heatmaps and uncertainty estimates for reproducible analysis.

This README is intentionally concise and focused on reproducing experiments on Windows (cmd.exe) and understanding the core outputs.

## Highlights

- Stagewise Grad-CAM generation (intermediate layers).
- Augmentation sweeps and RobustCAM-style fusion across augmentations.
- Per-pixel uncertainty maps and numeric stability metrics logged to MLflow.
- Modular code organized under `src/` for easy extension.

## Quick setup (Windows / cmd.exe)

1. Clone the repo and enter the project directory:

```cmd
git clone https://github.com/memnaaaaa/RobustCAM.git
cd "C:\path\to\RobustCAM"
```

2. Create & activate a virtual environment (recommended). If using Conda:

```cmd
conda create -n robustcam python=3.12 -y
conda activate robustcam
```

> Note: This project was developed and tested with Python 3.12. Adjust accordingly if using a different version.

3. Install dependencies:

```cmd
pip install -r requirements.txt
```

4. (Optional) Configure MLflow remote tracking by adding a local `.env` file:

```text
# .env (DO NOT COMMIT)
MLFLOW_TRACKING_URI_POSTGRES=postgresql://<user>:<password>@<host>:<port>/<database>
```

If a remote tracking URI is configured and reachable, runs will be recorded there; otherwise the project falls back to the local `mlruns/` directory.

## Run examples

- List available VGG layers (useful to choose stage indices):

```cmd
python src\vgg_structure.py
```

- Run pipeline on a single image (example):

```cmd
python pipeline.py --image "C:\path\to\image.jpg" --layers 14 20 30 --num-aug 16 --fusion-method mean
```

Common CLI flags:
- `--image` — input image path
- `--layers` — space-separated layer indices (see `src/vgg_structure.py`)
- `--num-aug` — number of augmentations for fusion (default: project default)
- `--fusion-method` — `mean` or `median`
    - `mean` — average heatmaps across augmentations
    - `median` — median heatmaps across augmentations (more robust to outliers)
- `--use-postgres` — prefer configured Postgres MLflow backend

> Note: `--help` shows all available options.

## Outputs & where to find them

- Artifacts (heatmaps, overlays, augmentations, uncertainty) are uploaded to the active MLflow run's artifact directory.
- With local fallback you will find artifacts under `mlruns/<experiment_id>/<run_id>/artifacts/`.
- The pipeline also uses temporary folders during runs: `temp_aug/`, `temp_heatmaps/`, `temp_overlays/`, `temp_uncert/` (these are ignored by git).

## Important files (quick map)

- `src/pipeline.py` — CLI entrypoint for running experiments!
- `src/vgg_structure.py` — model hooks & layer indexing helper.
- `src/gradcam_service.py` — computes stagewise Grad-CAMs.
- `src/augmentation_service.py` — augmentation definitions.
- `src/robust_cam.py` — augmentation fusion and uncertainty logic.
- `src/mlflow_service.py` — helper for MLflow logging and configuration.
- `src/model_service.py` — model loading and preprocessing.
- `src/data_service.py` — image I/O utilities.

## MLflow UI (local)

To inspect runs locally (when using `mlruns/`):

```cmd
mlflow ui --backend-store-uri ./mlruns --port 5000
# Open http://127.0.0.1:5000
```

## Research notes & best practices

- Use `src/vgg_structure.py` to determine stable layer indices for your model before running large sweeps.
- Start with a modest `--num-aug` (e.g., 8–16) on limited hardware and scale up as needed.
- Log environment details (Python, PyTorch versions) with each MLflow run for reproducibility.

## Contributing and attribution

This code was developed for academic use. If you reuse or adapt components, please cite the project and consider opening a PR with tests or documentation for changes.

For questions or collaboration, contact: memnaaaaa (repo owner)

---

Thank you for trying RobustCAM — happy visualizing!

