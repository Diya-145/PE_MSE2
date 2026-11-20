from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_monitor.data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_validated_data,
)


def build_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_candidate(
    training_data: pd.DataFrame, adaptation_data: pd.DataFrame
) -> tuple[Pipeline, int]:
    combined_training_data = pd.concat(
        [training_data, adaptation_data], ignore_index=True
    )
    candidate = build_pipeline()
    candidate.fit(
        combined_training_data[FEATURES], combined_training_data[TARGET_COLUMN]
    )
    return candidate, len(combined_training_data)


def bootstrap_champion(train_data_path: Path, champion_path: Path) -> None:
    training_data = load_validated_data(train_data_path, "Original training data")
    champion = build_pipeline()
    champion.fit(training_data[FEATURES], training_data[TARGET_COLUMN])
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, champion_path)
    print(f"Initial champion saved to {champion_path}")
