import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd

BASE_URL = "http://127.0.0.1:5000"
CSV_PATH = Path("data/transactions_test.csv")


def clean_row(row_dict):
    row_dict = dict(row_dict)
    row_dict.pop("is_fraud", None)
    cleaned = {}
    for k, v in row_dict.items():
        if pd.isna(v):
            cleaned[k] = None
        elif hasattr(v, "item"):
            cleaned[k] = v.item()
        else:
            cleaned[k] = v
    return cleaned


def post_json(url, payload_dict):
    req = Request(
        url,
        data=json.dumps(payload_dict).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=30) as res:
        return json.loads(res.read().decode("utf-8"))


def main():
    print("1. Checking /api/health ...")
    try:
        with urlopen(f"{BASE_URL}/api/health", timeout=10) as res:
            health = json.loads(res.read().decode("utf-8"))
            print(json.dumps(health, indent=2))
    except URLError as exc:
        print("Cannot connect to Flask on port 5000:", exc)
        sys.exit(1)

    if not CSV_PATH.exists():
        print(f"ERROR: Dataset not found: {CSV_PATH.resolve()}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    genuine_row = df[df["is_fraud"] == 0].iloc[0].to_dict()
    fraud_row = df[df["is_fraud"] == 1].iloc[0].to_dict()

    print("\n2. Testing GENUINE Transaction (Expected: Approved)...")
    res_gen = post_json(f"{BASE_URL}/predict", clean_row(genuine_row))
    print(json.dumps(res_gen, indent=2, ensure_ascii=False))

    print("\n3. Testing FRAUD Transaction (Expected: Hold & Verify)...")
    res_fraud = post_json(f"{BASE_URL}/predict", clean_row(fraud_row))
    print(json.dumps(res_fraud, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()