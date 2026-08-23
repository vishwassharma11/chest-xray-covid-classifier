"""Train and evaluate a research-only chest X-ray image classifier."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("archive-3"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_files(data_dir: Path) -> tuple[list[Path], list[str]]:
    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if len(class_dirs) != 2:
        raise ValueError(f"Expected exactly two class folders in {data_dir}")

    paths, labels = [], []
    for class_dir in class_dirs:
        images = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"No supported images found in {class_dir}")
        paths.extend(images)
        labels.extend([class_dir.name] * len(images))
    return paths, labels


def make_dataset(paths, labels, image_size, batch_size, training=False):
    ds = tf.data.Dataset.from_tensor_slices((list(map(str, paths)), labels))
    if training:
        ds = ds.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)

    def load_image(path, label):
        image = tf.io.decode_png(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, (image_size, image_size))
        image = tf.cast(image, tf.float32)
        return image, label

    return ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {args.data_dir}")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    tf.keras.utils.set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths, labels = collect_files(args.data_dir)
    class_names = sorted(set(labels))
    label_to_id = {name: index for index, name in enumerate(class_names)}
    encoded = np.array([label_to_id[label] for label in labels])

    train_paths, temp_paths, train_y, temp_y = train_test_split(
        paths, encoded, test_size=0.30, stratify=encoded, random_state=args.seed
    )
    val_paths, test_paths, val_y, test_y = train_test_split(
        temp_paths, temp_y, test_size=0.50, stratify=temp_y, random_state=args.seed
    )

    train_ds = make_dataset(train_paths, train_y, args.image_size, args.batch_size, training=True)
    val_ds = make_dataset(val_paths, val_y, args.image_size, args.batch_size)
    test_ds = make_dataset(test_paths, test_y, args.image_size, args.batch_size)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ])
    backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(args.image_size, args.image_size, 3))
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(args.image_size, args.image_size, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

    model_path = args.output_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(test_ds, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    report = classification_report(test_y, predictions, target_names=class_names, output_dict=True, zero_division=0)
    test_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
    payload = {"classes": class_names, "test_metrics": {key: float(value) for key, value in test_metrics.items()}, "classification_report": report, "split_sizes": {"train": len(train_paths), "validation": len(val_paths), "test": len(test_paths)}, "seed": args.seed}
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    (args.output_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))

    matrix = confusion_matrix(test_y, predictions)
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    print(json.dumps({"test_metrics": payload["test_metrics"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
