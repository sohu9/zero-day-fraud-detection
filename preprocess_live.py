import joblib
import numpy as np
import pandas as pd
from data_pipeline import add_time_features, ID_COLUMNS, TARGET_COLUMN


def load_live_preprocessor(artifact_path="preprocessor.joblib"):
    """Saved training preprocessor artifact load karta hai (19 features)."""
    artifact = joblib.load(artifact_path)
    preprocessor = artifact["preprocessor"]
    expected_columns = (
        list(artifact["numerical_columns"]) + list(artifact["categorical_columns"])
    )
    feature_names = list(artifact.get("feature_names", expected_columns))
    return preprocessor, expected_columns, feature_names


def preprocess_transaction(data, preprocessor, expected_columns):
    """Single live transaction dictionary ko 19 scaled features mein convert karta hai."""
    frame = pd.DataFrame([data])

    if TARGET_COLUMN in frame.columns:
        frame = frame.drop(columns=[TARGET_COLUMN])

    frame = frame.drop(
        columns=[col for col in ID_COLUMNS if col in frame.columns],
        errors="ignore"
    )

    frame = add_time_features(frame)
    frame = frame[expected_columns].copy()

    scaled = np.asarray(preprocessor.transform(frame), dtype=np.float32)
    return scaled