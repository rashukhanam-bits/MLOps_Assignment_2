"""
Unit tests for the model inference utility.

These test the image-preprocessing and prediction contract WITHOUT requiring
a real trained model file, by monkeypatching get_model() with a stub Keras-like
object. This keeps CI fast and independent of the (large, git/DVC-tracked) model
artifact.
"""
import io

import numpy as np
import pytest
from PIL import Image

from src.api import inference


def make_image_bytes(size=(300, 200), color=(0, 255, 0)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_preprocess_image_bytes_shape_and_dtype():
    img_bytes = make_image_bytes(size=(500, 400))

    arr = inference.preprocess_image_bytes(img_bytes)

    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.float32


class _StubModel:
    """Mimics model.predict(x) -> shape (1, 1) sigmoid output."""

    def __init__(self, dog_probability: float):
        self.dog_probability = dog_probability

    def predict(self, x, verbose=0):
        return np.array([[self.dog_probability]], dtype=np.float32)


def test_predict_labels_as_dog_when_probability_high(monkeypatch):
    monkeypatch.setattr(inference, "get_model", lambda: _StubModel(0.9))
    img_bytes = make_image_bytes()

    result = inference.predict(img_bytes)

    assert result["label"] == "dogs"
    assert result["probability"] == pytest.approx(0.9)
    assert result["class_probabilities"]["dogs"] == pytest.approx(0.9)
    assert result["class_probabilities"]["cats"] == pytest.approx(0.1)


def test_predict_labels_as_cat_when_probability_low(monkeypatch):
    monkeypatch.setattr(inference, "get_model", lambda: _StubModel(0.2))
    img_bytes = make_image_bytes()

    result = inference.predict(img_bytes)

    assert result["label"] == "cats"
    assert result["probability"] == pytest.approx(0.8)


def test_predict_raises_file_not_found_when_model_missing(monkeypatch):
    def raise_missing():
        raise FileNotFoundError("no model")

    monkeypatch.setattr(inference, "get_model", raise_missing)

    with pytest.raises(FileNotFoundError):
        inference.predict(make_image_bytes())
