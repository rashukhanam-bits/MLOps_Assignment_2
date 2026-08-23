"""
Trains the baseline CNN on data/processed/{train,val,test} and logs
params, metrics, and artifacts (confusion matrix, loss curve, model) to MLflow.

Usage:
    python src/models/train.py --epochs 10 --batch-size 32 --lr 0.001
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import keras

from src.models.model import build_model

PROCESSED_DIR = Path("data/processed")
MODEL_OUT = Path("models/cats_dogs_cnn.keras")
REPORTS_DIR = Path("reports")


def load_datasets(image_size: int, batch_size: int):
    train_ds = keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "train",
        label_mode="binary",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "val",
        label_mode="binary",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "test",
        label_mode="binary",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )
    class_names = train_ds.class_names  # ['cats', 'dogs']

    autotune = keras.utils.experimental
    train_ds = train_ds.prefetch(buffer_size=1)
    val_ds = val_ds.prefetch(buffer_size=1)
    test_ds = test_ds.prefetch(buffer_size=1)
    return train_ds, val_ds, test_ds, class_names


def plot_loss_curve(history, out_path: Path):
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(out_path)
    plt.close()
    return cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    image_size = params["preprocess"]["image_size"]
    epochs = args.epochs or params["train"]["epochs"]
    batch_size = args.batch_size or params["train"]["batch_size"]
    lr = args.lr or params["train"]["learning_rate"]

    REPORTS_DIR.mkdir(exist_ok=True)
    MODEL_OUT.parent.mkdir(exist_ok=True)

    mlflow.set_experiment("cats-vs-dogs")
    with mlflow.start_run():
        mlflow.log_params(
            {"epochs": epochs, "batch_size": batch_size, "learning_rate": lr, "image_size": image_size}
        )

        train_ds, val_ds, test_ds, class_names = load_datasets(image_size, batch_size)

        model = build_model(image_size)
        model.optimizer.learning_rate.assign(lr)

        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

        test_loss, test_acc = model.evaluate(test_ds)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        for epoch, (tl, ta, vl, va) in enumerate(
            zip(history.history["loss"], history.history["accuracy"],
                history.history["val_loss"], history.history["val_accuracy"])
        ):
            mlflow.log_metrics(
                {"train_loss": tl, "train_accuracy": ta, "val_loss": vl, "val_accuracy": va},
                step=epoch,
            )

        # Confusion matrix on test set
        y_true, y_pred = [], []
        for images, labels in test_ds:
            preds = (model.predict(images, verbose=0) > 0.5).astype(int).flatten()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().astype(int).flatten().tolist())

        cm_path = REPORTS_DIR / "confusion_matrix.png"
        loss_path = REPORTS_DIR / "loss_curve.png"
        plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
        plot_loss_curve(history, loss_path)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(loss_path))

        model.save(MODEL_OUT)
        mlflow.log_artifact(str(MODEL_OUT))
        mlflow.keras.log_model(model, "model")

        metrics_summary = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "final_train_accuracy": float(history.history["accuracy"][-1]),
            "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        }
        with open(REPORTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics_summary, f, indent=2)

        print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
        print(f"Model saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
