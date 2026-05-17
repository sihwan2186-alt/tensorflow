from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ZIP_PATH = DATA_DIR / "uci_student_dropout.zip"
RAW_DIR = DATA_DIR / "uci_student_dropout_raw"
OUTPUT_PATH = DATA_DIR / "student_dropout_real.csv"
SOURCE_PATH = DATA_DIR / "student_dropout_real_source.json"

DATASET_URL = "https://cdn.uci-ics-mlr-prod.aws.uci.edu/697/predict%2Bstudents%2Bdropout%2Band%2Bacademic%2Bsuccess.zip"

SOURCE_INFO = {
    "name": "Predict Students' Dropout and Academic Success",
    "repository": "UCI Machine Learning Repository",
    "dataset_id": 697,
    "official_page": "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success",
    "download_url": DATASET_URL,
    "doi": "10.24432/C5MC89",
    "license": "CC BY 4.0",
    "instances": 4424,
    "features": 36,
    "task": "classification",
}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"downloading: {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(RAW_DIR)

    raw_csv = RAW_DIR / "data.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"expected file not found: {raw_csv}")

    OUTPUT_PATH.write_bytes(raw_csv.read_bytes())
    SOURCE_PATH.write_text(json.dumps(SOURCE_INFO, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved data: {OUTPUT_PATH}")
    print(f"saved source metadata: {SOURCE_PATH}")


if __name__ == "__main__":
    main()
