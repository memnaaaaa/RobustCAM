# vizCNN — Stagewise Grad-CAM + Augmentation Visualizer

vizCNN is a compact, modular pipeline for exploring how convolutional neural networks (CNNs) attend to input images across intermediate stages. It:

- 🧠 Captures activations and gradients at multiple intermediate stages (convolution, ReLU, pooling).
- 🔥 Produces per-stage Grad-CAM heatmaps and overlay visualizations (heatmaps blended with the input image).
- 🔄 Re-runs the pipeline on common augmentations and logs how attention shifts under transformations (rotation, flips, color jitter, etc.).
- 📊 Logs experiments, parameters, and image artifacts to MLflow. MLflow can write to a remote PostgreSQL-backed tracking store (recommended for centralized teams) or a local `mlruns/` directory (default fallback).

This README explains why the project exists, how to run the code end-to-end (Windows/cmd example), where outputs are written, and safety notes about secret handling.

---

## 💡 Why this repository

When exploring model explanations, it is often useful to inspect not just the final Grad-CAM for the last layer, but how attention evolves through earlier layers. This helps:

- 🖼️ Understand which layers detect low-level edges vs. higher-level concepts.
- 🔍 Compare how augmentations affect attention (robustness / dataset biases).
- 🖌️ Produce publication-quality stagewise visualizations and store experiment artifacts reproducibly using MLflow.

vizCNN is intentionally modular so you can:

- 🛠️ Inspect the hook wiring in `src/vgg_structure.py`.
- 🚀 Run the end-to-end pipeline from `pipeline.py` with simple CLI flags.
- 🔧 Swap in different augmentation sets or models.

---

## ⚡ Quickstart (Windows / cmd.exe)

The steps below assume you're on Windows and will use `cmd.exe`. Adjust for macOS / Linux as needed.

### 1️⃣ Clone the repository

```cmd
git clone https://github.com/memnaaaaa/vizCNN
cd vizCNN
```

### 2️⃣ Create and activate a virtual environment (recommended). If using Conda:

```cmd
conda create -n *env_name* python==3.12 -y
conda activate *env_name*
```

### 3️⃣ Install dependencies

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Configure MLflow tracking (optional)

- By default, the project will fall back to local directories (`mlruns/`, `temp_aug/`, `temp_heatmaps/`, `temp_mlflow/`, `temp_overlays/`) if no remote MLflow tracking URI is provided.
- To use a PostgreSQL-backed MLflow tracking store, create a `.env` file in the repository root with the following variable (this file is ignored by git):

```env
# .env (DO NOT COMMIT)
MLFLOW_TRACKING_URI_POSTGRES=postgresql://<user>:<password>@<host>:<port>/<database>
```

### 5️⃣ Inspect the hook wiring (recommended)

Before running the full pipeline, the helper `src/vgg_structure.py` shows how hooks are attached to the VGG model to capture intermediate activations/gradients. Run it to print available layer indices/names and to confirm your environment can import PyTorch / torchvision.

```cmd
python src\vgg_structure.py
```

### 6️⃣ Run the pipeline

The main entrypoint is `pipeline.py`. Example usage (Windows path example):

```cmd
python pipeline.py --image "C:\path\to\image.jpg" --layers 14 20 30 --use-postgres
```

- `--image` : Path to a single input image.
- `--layers` : Space-separated list of layer indices (integers) to generate stagewise Grad-CAMs for. These correspond to the layers exposed by `vgg_structure.py`.
- `--use-postgres` : Optional flag indicating the run should attempt to use the PostgreSQL MLflow URI from the environment. If the remote tracking server is unreachable, the code will log a warning and fall back to local storage.

Run `python pipeline.py --help` for all available CLI flags.

---

## 📂 What the pipeline produces

- 🎨 Per-layer Grad-CAM heatmaps (standalone PNGs).
- 🖼️ Per-layer overlay images (original input blended with heatmap).
- 📁 Augmentation-specific folders that contain heatmaps and overlays for each augmentation.
- 📊 MLflow experiment entries (if tracking configured): parameters, metrics, and image artifacts organized under artifact paths such as `heatmaps/`, `overlays/`, or `augmentations/<augment_name>/`.

### Local output layout (when using local fallback)

- `mlruns/` — MLflow local store (experiments, runs, metadata, artifacts)
- `temp_heatmaps/` — temporary files written during logging (these are cleaned up by the code but exist during runtime)
- `temp_overlays/`, `temp_aug/` — temporary overlays and augmentation artifacts

> **Note:** The repository `.gitignore` already ignores these directories so they are not committed.

---

## 🔒 Security & secrets

- 🚫 Never check `.env` into source control. It is already listed in `.gitignore`.
- 🔑 Use the environment variable `MLFLOW_TRACKING_URI_POSTGRES` (in `.env`) to store your full MLflow/Postgres URI. Example format:

```env
MLFLOW_TRACKING_URI_POSTGRES=postgresql://<user>:<password>@<host>:<port>/<database>
```

- The code will read this value via `python-dotenv` (already added to `requirements.txt`) and attempt to use it. If you prefer to provide the URI at runtime, the `MLflowService` accepts a `tracking_uri_postgres` parameter when instantiated.

---

## 🖥️ Running the MLflow UI locally (when using local fallback)

If the pipeline uses the local `mlruns/` directory, you can inspect runs with the MLflow UI:

```cmd
mlflow ui --backend-store-uri ./mlruns --port 5000

# Then open http://127.0.0.1:5000 in your browser
```

---

## 🛠️ Troubleshooting

- ⚠️ If you see a warning about the Postgres connection failing, the library will print a message and continue using the local `mlruns/` store. This is expected behavior when you haven't provided a valid `MLFLOW_TRACKING_URI_POSTGRES`.
- 🛑 If `cv2` fails to save images, ensure `opencv-python` is installed (it's in `requirements.txt`) and the process has write permissions to the working directory.
- 🔄 If you accidentally committed secrets, rotate credentials and consider using `git filter-repo` / BFG to purge secrets from history.

---

## 📝 Example workflow summary

1. 🛠️ Clone repository, create venv, install requirements.
2. 🔒 Create `.env` with `MLFLOW_TRACKING_URI_POSTGRES` or skip to use local `mlruns/`.
3. 🔍 Run `python src\vgg_structure.py` to inspect available layer indices/names.
4. 🚀 Run `python pipeline.py --image "C:\path\to\image.jpg" --layers 14 20 30 --use-postgres` to produce stagewise and augmentation visualizations and log them to MLflow.

---

## 📜 License & contribution

This project is provided as-is for research and teaching. If you make improvements, please open a PR and include tests or documentation for the change.

---