"""
Training script for CK+ facial expression recognition.

This script mirrors and extends the `assets/facial/fer.ipynb` notebook by:
- Handling local CK+48 images stored under `assets/facial/CK+48`
- Optionally creating train / test splits if they do not already exist
- Providing a configurable CLI for hyperparameters
- Adding richer callbacks (checkpointing, TensorBoard, LR scheduling, early stopping)

Usage (from project root):
    python -m emotion_game_ai.vision.ck_train \
        --ck48-dir assets/facial/CK+48 \
        --output-dir models/ck_cnn \
        --epochs 40 \
        --batch-size 64
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import pathlib
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def _ensure_split_dirs(
    ck48_dir: pathlib.Path,
    train_subdir: str = "CK+48_train",
    test_subdir: str = "CK+48_test",
    train_ratio: float = 0.8,
) -> Tuple[pathlib.Path, pathlib.Path]:
    """
    Ensure we have train and test directories with the standard Keras
    directory structure: one subdirectory per class label.

    If they already exist and are non-empty, they are reused. Otherwise they
    are created by splitting images in `ck48_dir` according to `train_ratio`.
    """
    train_dir = ck48_dir.parent / train_subdir
    test_dir = ck48_dir.parent / test_subdir

    def _has_images(root: pathlib.Path) -> bool:
        if not root.exists():
            return False
        for sub in root.rglob("*"):
            if sub.is_file() and sub.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                return True
        return False

    if _has_images(train_dir) and _has_images(test_dir):
        return train_dir, test_dir

    # Fresh split from ck48_dir
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=42)

    for emotion_dir in sorted(ck48_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue

        images = [
            p for p in emotion_dir.iterdir() if p.is_file()
            and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        ]
        if not images:
            continue

        rng.shuffle(images)
        train_size = int(len(images) * train_ratio)
        train_images = images[:train_size]
        test_images = images[train_size:]

        train_emotion_dir = train_dir / emotion_dir.name
        test_emotion_dir = test_dir / emotion_dir.name
        train_emotion_dir.mkdir(parents=True, exist_ok=True)
        test_emotion_dir.mkdir(parents=True, exist_ok=True)

        for src in train_images:
            dst = train_emotion_dir / src.name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
        for src in test_images:
            dst = test_emotion_dir / src.name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())

    return train_dir, test_dir


def build_generators(
    train_dir: pathlib.Path,
    test_dir: pathlib.Path,
    image_size: Tuple[int, int] = (48, 48),
    batch_size: int = 64,
) -> Tuple[tf.keras.utils.Sequence, tf.keras.utils.Sequence]:
    """Create training and validation generators with augmentation."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=image_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )

    val_gen = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=image_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )

    return train_gen, val_gen


def build_model(
    input_shape: Tuple[int, int, int] = (48, 48, 1),
    num_classes: int = 7,
) -> tf.keras.Model:
    """
    CNN architecture inspired by the notebook, kept reasonably compact so it
    trains quickly on CK+.
    """
    model = models.Sequential(name="ck_emotion_cnn")

    model.add(
        layers.Conv2D(
            64,
            (5, 5),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            input_shape=input_shape,
            name="conv2d_1",
        )
    )
    model.add(layers.BatchNormalization(name="bn_1"))
    model.add(
        layers.Conv2D(
            64,
            (5, 5),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_2",
        )
    )
    model.add(layers.BatchNormalization(name="bn_2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool_1"))
    model.add(layers.Dropout(0.25, name="dropout_1"))

    model.add(
        layers.Conv2D(
            128,
            (3, 3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_3",
        )
    )
    model.add(layers.BatchNormalization(name="bn_3"))
    model.add(
        layers.Conv2D(
            128,
            (3, 3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_4",
        )
    )
    model.add(layers.BatchNormalization(name="bn_4"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool_2"))
    model.add(layers.Dropout(0.25, name="dropout_2"))

    model.add(
        layers.Conv2D(
            256,
            (3, 3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_5",
        )
    )
    model.add(layers.BatchNormalization(name="bn_5"))
    model.add(
        layers.Conv2D(
            256,
            (3, 3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_6",
        )
    )
    model.add(layers.BatchNormalization(name="bn_6"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool_3"))
    model.add(layers.Dropout(0.25, name="dropout_3"))

    model.add(layers.Flatten(name="flatten"))
    model.add(
        layers.Dense(
            256,
            activation="elu",
            kernel_initializer="he_normal",
            name="fc_1",
        )
    )
    model.add(layers.BatchNormalization(name="bn_7"))
    model.add(layers.Dropout(0.5, name="dropout_4"))
    model.add(
        layers.Dense(
            num_classes,
            activation="softmax",
            name="output",
        )
    )

    opt = optimizers.Nadam(learning_rate=1e-3)
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_callbacks(output_dir: pathlib.Path) -> list[callbacks.Callback]:
    """Create a set of training callbacks for better convergence and monitoring."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "best_model.keras"
    log_dir = output_dir / "logs" / _dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    early_stopping = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        min_delta=5e-4,
        restore_best_weights=True,
        verbose=1,
    )

    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    )

    model_ckpt = callbacks.ModelCheckpoint(
        filepath=str(ckpt_path),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    tensorboard_cb = callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1,
    )

    return [early_stopping, lr_scheduler, model_ckpt, tensorboard_cb]


def train(
    ck48_dir: pathlib.Path,
    output_dir: pathlib.Path,
    epochs: int = 40,
    batch_size: int = 64,
) -> tf.keras.Model:
    """End-to-end training pipeline."""
    train_dir, test_dir = _ensure_split_dirs(ck48_dir)

    train_gen, val_gen = build_generators(
        train_dir=train_dir,
        test_dir=test_dir,
        image_size=(48, 48),
        batch_size=batch_size,
    )

    num_classes = train_gen.num_classes
    model = build_model(input_shape=(48, 48, 1), num_classes=num_classes)

    cbs = build_callbacks(output_dir)

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=cbs,
    )

    # Final evaluation
    val_loss, val_acc = model.evaluate(val_gen, verbose=1)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc * 100:.2f}%")

    # Save final model (in addition to best checkpoint)
    final_path = output_dir / "final_model.keras"
    model.save(str(final_path))
    print(f"Saved final model to: {final_path}")

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN on CK+48 facial expression images."
    )
    parser.add_argument(
        "--ck48-dir",
        type=str,
        default=os.path.join("assets", "facial", "CK+48"),
        help="Path to CK+48 directory with one subdirectory per emotion.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("models", "ck_emotion"),
        help="Directory to write trained model and logs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = pathlib.Path(__file__).resolve().parents[2]
    ck48_dir = (project_root / args.ck48_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    if not ck48_dir.exists():
        raise FileNotFoundError(f"CK+48 directory not found: {ck48_dir}")

    print(f"Using CK+48 images from: {ck48_dir}")
    print(f"Outputs will be saved under: {output_dir}")

    # Set a reproducible seed for TensorFlow / NumPy where possible
    tf.random.set_seed(42)
    np.random.seed(42)

    train(
        ck48_dir=ck48_dir,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

