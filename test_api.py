
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd

API_URL = "http://127.0.0.1:5000/predict"
CSV_PATH = Path("data/transactions_test.csv")


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: Dataset not found: {CSV_PATH.resolve()}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    if df.empty:
        print("ERROR: Test dataset is empty.")
        sys.exit(1)

    print("Dataset shape:", df.shape)
    print("Dataset columns:", list(df.columns))

    # Use one actual transaction from the test dataset.
    actual_label = None
    if "is_fraud" in df.columns:
        actual_label = int(df.iloc[0]["is_fraud"])

    data = df.iloc[0].to_dict()

    # Ground-truth label is for evaluation only, never prediction input.
    data.pop("is_fraud", None)

    # Convert pandas missing values and NumPy values to JSON-safe values.
    clean_data = {}
    for key, value in data.items():
        if pd.isna(value):
            clean_data[key] = None
        elif hasattr(value, "item"):
            clean_data[key] = value.item()
        else:
            clean_data[key] = value

    payload = json.dumps(clean_data).encode("utf-8")

    req = Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print("\nSending REAL dataset row to API...")
    print("Input feature columns:", len(clean_data))
    print("Actual dataset label (evaluation only):", actual_label)

    try:
        with urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))

            print("\n========== PREDICTION RESULT ==========")
            print(json.dumps(result, indent=4, ensure_ascii=False))
            print("=======================================")

    except HTTPError as error:
        print(f"\nAPI returned HTTP {error.code}")
        print(error.read().decode("utf-8", errors="replace"))
        sys.exit(1)

    except URLError as error:
        print("\nCannot connect to Flask.")
        print("Check that app.py is running on port 5000.")
        print(error.reason)
        sys.exit(1)


if __name__ == "__main__":
    main()