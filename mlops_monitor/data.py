from pathlib import Path

import pandas as pd


ID_COLUMN = "customer_id"
TARGET_COLUMN = "churn"
NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL_FEATURES = ["contract_type", "payment_method", "paperless_billing"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [ID_COLUMN, *FEATURES, TARGET_COLUMN]


def load_validated_data(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{dataset_name} data file does not exist: {path}")

    frame = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{dataset_name} is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError(f"{dataset_name} contains no records.")

    missing_values = frame[REQUIRED_COLUMNS].isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        raise ValueError(
            f"{dataset_name} contains missing values: {missing_values.to_dict()}"
        )

    duplicate_ids = int(frame[ID_COLUMN].duplicated().sum())
    if duplicate_ids:
        raise ValueError(f"{dataset_name} contains {duplicate_ids} duplicate customer IDs.")
    if not frame[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(f"{dataset_name} target '{TARGET_COLUMN}' must contain only 0 or 1.")
    if frame[TARGET_COLUMN].nunique() != 2:
        raise ValueError(f"{dataset_name} must contain both target classes (0 and 1).")

    distribution = frame[TARGET_COLUMN].value_counts().sort_index().to_dict()
    print(
        f"{dataset_name}: {len(frame)} records; "
        f"{TARGET_COLUMN} distribution (0=retained, 1=churned): {distribution}"
    )
    return frame
