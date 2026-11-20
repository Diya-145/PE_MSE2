import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from mlops_monitor.data import FEATURES, TARGET_COLUMN


def calculate_metrics(model: Pipeline, frame: pd.DataFrame) -> dict[str, float]:
    truth = frame[TARGET_COLUMN]
    prediction = model.predict(frame[FEATURES])
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
    }
