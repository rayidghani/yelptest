#!/usr/bin/env python3
"""Train a binary latte-art quality model (good vs bad) with TensorFlow/Keras.

Expected dataset structure:

<dataset_dir>/
  good/
    img1.jpg
    ...
  bad/
    img2.jpg
    ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a latte-art quality classifier.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path with good/ and bad/ subfolders")
    parser.add_argument("--output-model", type=Path, default=Path("models/latte_art_model.keras"))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--fine-tune", action="store_true", help="Unfreeze base model after warmup")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for training. Install requirements-tensorflow.txt first."
        ) from exc

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir does not exist: {args.dataset_dir}")

    good_dir = args.dataset_dir / "good"
    bad_dir = args.dataset_dir / "bad"
    if not good_dir.exists() or not bad_dir.exists():
        raise RuntimeError(
            f"Dataset must include '{good_dir}' and '{bad_dir}' directories."
        )

    image_size = (args.img_size, args.img_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.dataset_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=args.validation_split,
        subset="training",
        seed=args.seed,
        image_size=image_size,
        batch_size=args.batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.dataset_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=args.validation_split,
        subset="validation",
        seed=args.seed,
        image_size=image_size,
        batch_size=args.batch_size,
    )

    class_names = train_ds.class_names
    if sorted(class_names) != ["bad", "good"]:
        print(
            "Warning: class names are expected to be ['bad', 'good']. "
            f"Detected: {class_names}."
        )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(args.img_size, args.img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(args.img_size, args.img_size, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="latte_art_score")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=3, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    warmup_epochs = min(args.warmup_epochs, args.epochs)
    history = model.fit(train_ds, validation_data=val_ds, epochs=warmup_epochs, callbacks=callbacks)

    if args.fine_tune and args.epochs > warmup_epochs:
        base_model.trainable = True
        fine_tune_at = int(len(base_model.layers) * 0.8)
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate * 0.1),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=history.epoch[-1] + 1,
            epochs=args.epochs,
            callbacks=callbacks,
        )
        history.history.update({f"fine_tune_{k}": v for k, v in fine_tune_history.history.items()})

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output_model)

    metadata = {
        "image_size": args.img_size,
        "class_names": class_names,
        "output_model": str(args.output_model),
    }
    metadata_path = args.output_model.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved model: {args.output_model}")
    print(f"Saved metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
