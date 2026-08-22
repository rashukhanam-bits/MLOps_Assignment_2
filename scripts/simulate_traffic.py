"""
M5: Model Performance Tracking (Post-Deployment)

Sends a batch of real test-set images with known true labels to the deployed
/predict endpoint, records predictions, and reports accuracy on live traffic --
a simple stand-in for production drift monitoring.

Usage:
    python scripts/simulate_traffic.py --n 50 --endpoint http://localhost:8000
"""
import argparse
import csv
import random
from pathlib import Path

import requests

TEST_DIR = Path("data/processed/test")
REPORT_PATH = Path("reports/post_deploy_eval.csv")


def collect_sample(n: int):
    samples = []
    for class_name in ["cats", "dogs"]:
        for f in (TEST_DIR / class_name).glob("*.jpg"):
            samples.append((f, class_name))
    random.shuffle(samples)
    return samples[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    samples = collect_sample(args.n)
    if not samples:
        print("No test images found under data/processed/test/. Run preprocessing first.")
        return

    REPORT_PATH.parent.mkdir(exist_ok=True)
    correct = 0
    rows = []

    for path, true_label in samples:
        with open(path, "rb") as f:
            resp = requests.post(f"{args.endpoint}/predict", files={"file": f})
        resp.raise_for_status()
        pred = resp.json()
        is_correct = pred["label"] == true_label
        correct += int(is_correct)
        rows.append(
            {
                "file": path.name,
                "true_label": true_label,
                "predicted_label": pred["label"],
                "probability": pred["probability"],
                "correct": is_correct,
            }
        )

    with open(REPORT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    accuracy = correct / len(samples)
    print(f"Sent {len(samples)} requests. Live accuracy: {accuracy:.2%}")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
