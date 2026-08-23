"""
Model-loading and prediction utilities, kept separate from FastAPI routing
so they're easy to unit test in isolation (see tests/test_inference.py).
"""
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/cats_dogs_cnn_weights.pkl"))
CLASS_NAMES = ["cats", "dogs"]  # index 0 -> cats, index 1 -> dogs
IMAGE_SIZE = 224

_model = None  # lazy-loaded singleton

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train it first with src/models/train.py"
            )
        import pickle
        from src.models.model import build_model

        _model = build_model()
        with open(MODEL_PATH, "rb") as f:
            weights = pickle.load(f)
        _model.set_weights(weights)
    return _model


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes -> (1, 224, 224, 3) float array ready for the model."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)  # model has its own Rescaling layer


def predict(image_bytes: bytes) -> dict:
    """Runs inference and returns label + probabilities. Pure function, easy to unit test."""
    model = get_model()
    x = preprocess_image_bytes(image_bytes)
    prob_dog = float(model.predict(x, verbose=0)[0][0])
    prob_cat = 1.0 - prob_dog

    label = "dogs" if prob_dog > 0.5 else "cats"
    confidence = prob_dog if label == "dogs" else prob_cat

    return {
        "label": label,
        "probability": confidence,
        "class_probabilities": {"cats": prob_cat, "dogs": prob_dog},
    }
