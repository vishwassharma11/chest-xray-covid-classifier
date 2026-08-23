"""Local web app for research-only COVID image classification."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "artifacts" / "class_names.json"
IMAGE_SIZE = 224
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

app = Flask(__name__)
model: tf.keras.Model | None = None
class_names: list[str] = []


def load_assets() -> tuple[tf.keras.Model, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Train first with: python train.py --data-dir archive-3 --epochs 10"
        )
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError("Class names not found. Train the model first.")

    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    loaded_classes = json.loads(CLASS_NAMES_PATH.read_text())
    return loaded_model, loaded_classes


def prepare_image(file_bytes: bytes) -> tuple[np.ndarray, str]:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    preview = image.copy()
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    buffer = io.BytesIO()
    preview.thumbnail((900, 900))
    preview.save(buffer, format="PNG")
    preview_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return array, preview_base64


def predict(file_bytes: bytes) -> dict[str, object]:
    global model, class_names
    if model is None or not class_names:
        model, class_names = load_assets()

    image_array, preview_base64 = prepare_image(file_bytes)
    raw_probability = float(model.predict(image_array, verbose=0).ravel()[0])
    predicted_index = int(raw_probability >= 0.5)
    predicted_label = class_names[predicted_index]

    scores = {
        class_names[0]: 1.0 - raw_probability,
        class_names[1]: raw_probability,
    }

    return {
        "label": predicted_label,
        "confidence": scores[predicted_label],
        "scores": scores,
        "preview": preview_base64,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        if not uploaded_file or not uploaded_file.filename:
            error = "Please choose a PNG or JPG image."
        elif Path(uploaded_file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            error = "Only PNG, JPG, and JPEG files are supported."
        else:
            try:
                result = predict(uploaded_file.read())
            except (UnidentifiedImageError, OSError):
                error = "That file does not look like a valid image."
            except Exception as exc:
                error = str(exc)

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    model, class_names = load_assets()
    app.run(host="127.0.0.1", port=5000, debug=False)
