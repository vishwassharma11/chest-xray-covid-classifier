# Chest X-ray Classification (Research Starter)

This project trains a binary image classifier on the included chest X-ray dataset:

- `archive-3/COVID/` — COVID images
- `archive-3/non-COVID/` — non-COVID images

Dataset source: [SARS-CoV-2 CT-scan Dataset on Kaggle](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset). Please consult the source page for its license, attribution requirements, and terms before redistributing or using the data.

> **Important:** This is an educational/research project only.

## Quick start

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train a baseline model:

```bash
python train.py --data-dir archive-3 --epochs 10
```

Run the local web app after training:

```bash
python app.py
```

Then open `http://127.0.0.1:5000` and upload a PNG or JPG image.

## Deploy online

This repo includes the trained model in `artifacts/best_model.keras`, so it can be deployed as a small Flask web app. On Render, create a new Web Service from this GitHub repo and use:

```bash
pip install -r requirements.txt
```

Start command:

```bash
gunicorn app:app
```

Results are written to `artifacts/`:

- `best_model.keras` — best validation model
- `metrics.json` — test-set metrics and configuration
- `confusion_matrix.png` — test-set confusion matrix
- `class_names.json` — label mapping

## How it works

The training script uses a frozen ImageNet-pretrained EfficientNetB0 backbone and a small classification head. It performs a stratified 70/15/15 train/validation/test split, applies augmentation only to training data, and evaluates the final selected checkpoint on the held-out test set.

For any serious research, audit image provenance, patient-level leakage, demographics, acquisition devices, and external validation before interpreting results.

## Project layout

```
archive-3/       Dataset supplied with this project
app.py           Local upload-and-predict web app
train.py         Reproducible training and evaluation entry point
requirements.txt Python dependencies
artifacts/       Generated after training (ignored by Git)
templates/       Web page template for the app
static/          Web app styling
```
