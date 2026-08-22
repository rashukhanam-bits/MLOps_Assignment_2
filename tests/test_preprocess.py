"""Unit tests for the data preprocessing utilities (no dataset download needed)."""
import io

import numpy as np
import pytest
from PIL import Image

from src.data.preprocess import resize_and_save, split_files


def make_dummy_image(path, size=(300, 200), color=(255, 0, 0)):
    img = Image.new("RGB", size, color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def test_resize_and_save_produces_correct_shape(tmp_path):
    src = tmp_path / "src.jpg"
    dst = tmp_path / "out" / "dst.jpg"
    make_dummy_image(src, size=(500, 350))

    resize_and_save(src, dst, size=224)

    assert dst.exists()
    with Image.open(dst) as out_img:
        assert out_img.size == (224, 224)
        assert out_img.mode == "RGB"


def test_resize_and_save_converts_non_rgb(tmp_path):
    src = tmp_path / "gray.jpg"
    Image.new("L", (100, 100), 128).save(src)  # grayscale
    dst = tmp_path / "out.jpg"

    resize_and_save(src, dst, size=224)

    with Image.open(dst) as out_img:
        assert out_img.mode == "RGB"


def test_split_files_respects_ratios_and_is_exhaustive():
    files = [f"img_{i}.jpg" for i in range(100)]

    splits = split_files(files, train_split=0.8, val_split=0.1, seed=42)

    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10
    # every file accounted for exactly once
    all_out = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_out) == sorted(files)


def test_split_files_is_deterministic_given_seed():
    files = [f"img_{i}.jpg" for i in range(50)]
    splits_a = split_files(files, 0.8, 0.1, seed=7)
    splits_b = split_files(files, 0.8, 0.1, seed=7)
    assert splits_a["train"] == splits_b["train"]
