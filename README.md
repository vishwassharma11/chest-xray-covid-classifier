# Chest X-ray Classification (Research Starter)

This project trains a binary image classifier on the included chest X-ray dataset:

- `archive-3/COVID/` — COVID images
- `archive-3/non-COVID/` — non-COVID images

> **Important:** This is an educational/research project only. It is not a medical device and must not be used to diagnose, rule out, or make treatment decisions for COVID-19. Model performance on this dataset does not establish clinical safety or generalization.

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
train.py         Reproducible training and evaluation entry point
requirements.txt Python dependencies
artifacts/       Generated after training (ignored by Git)
```
