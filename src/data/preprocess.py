"""
Preprocesses raw Cats vs Dogs images:
  - Resizes to 224x224 RGB
  - Splits into train/val/test (80/10/10 by default, from params.yaml)
  - Copies into data/processed/{train,val,test}/{cats,dogs}/

Data augmentation (random flip/rotation/zoom) is applied at TRAIN TIME inside
src/models/train.py via a Keras augmentation layer, not baked into the files
on disk, so validation/test data stays clean.

Usage:
    python src/data/preprocess.py
"""
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["preprocess"]


def resize_and_save(src_path: Path, dst_path: Path, size: int):
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize((size, size), Image.BILINEAR)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, quality=95)


def split_files(files, train_split, val_split, seed):
    files = list(files)
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def preprocess_class(class_name: str, params: dict):
    src_dir = RAW_DIR / class_name
    files = sorted(src_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(
            f"No images found in {src_dir}. Run src/data/download_data.py first, "
            f"or place images manually."
        )

    splits = split_files(
        files, params["train_split"], params["val_split"], params["seed"]
    )

    for split_name, split_files_list in splits.items():
        out_dir = PROCESSED_DIR / split_name / class_name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        for f in split_files_list:
            resize_and_save(f, out_dir / f.name, params["image_size"])

    print(
        f"{class_name}: train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])}"
    )


def main():
    params = load_params()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for class_name in ["cats", "dogs"]:
        preprocess_class(class_name, params)
    print(f"Preprocessing complete. Output: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
