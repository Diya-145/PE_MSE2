import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_monitor.data import TARGET_COLUMN, load_validated_data
from mlops_monitor.evaluation import calculate_metrics
from mlops_monitor.modeling import train_candidate
from mlops_monitor.quality_gate import MIN_CANDIDATE_F1, assess_candidate


EVALUATION_FRACTION = 0.25
RANDOM_STATE = 42


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _log_mlflow(
    run_id: str,
    run_dir: Path,
    report_path: Path,
    decision: dict[str, Any],
) -> str:
    if importlib.util.find_spec("mlflow") is None:
        return "not installed; skipped"

    import mlflow

    mlflow.set_tracking_uri((run_dir.parent.parent / "mlruns").resolve().as_uri())
    mlflow.set_experiment("customer-churn-model-monitoring")
    with mlflow.start_run(run_name=run_id):
        mlflow.log_param("training_rows", decision["training_rows"])
        mlflow.log_param("new_batch_rows", decision["new_batch_rows"])
        mlflow.log_param("gate_metric", "f1")
        mlflow.log_param("minimum_candidate_f1", MIN_CANDIDATE_F1)
        mlflow.log_metric("champion_f1", decision["champion_holdout_metrics"]["f1"])
        mlflow.log_metric("candidate_f1", decision["candidate_holdout_metrics"]["f1"])
        mlflow.set_tag("quality_gate", decision["status"])
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(run_dir / "candidate.joblib"))
    return "logged"


def run_monitoring_workflow(
    train_data_path: Path,
    new_data_path: Path,
    champion_path: Path,
    artifacts_dir: Path,
) -> Path:
    if not champion_path.is_file():
        raise FileNotFoundError(
            f"Champion model not found at {champion_path}; run the bootstrap command first."
        )

    training_data = load_validated_data(train_data_path, "Original training data")
    new_data = load_validated_data(new_data_path, "Incoming labelled batch")
    evaluation_rows = math.ceil(len(new_data) * EVALUATION_FRACTION)
    if (
        new_data[TARGET_COLUMN].value_counts().min() < 2
        or evaluation_rows < 2
        or len(new_data) - evaluation_rows < 2
    ):
        raise ValueError(
            "Incoming labelled batch is too small for a stratified adaptation/evaluation split."
        )
    adaptation_data, evaluation_data = train_test_split(
        new_data,
        test_size=EVALUATION_FRACTION,
        random_state=RANDOM_STATE,
        stratify=new_data[TARGET_COLUMN],
    )

    champion = joblib.load(champion_path)
    full_batch_metrics = calculate_metrics(champion, new_data)
    champion_holdout_metrics = calculate_metrics(champion, evaluation_data)
    candidate, training_rows = train_candidate(training_data, adaptation_data)
    candidate_holdout_metrics = calculate_metrics(candidate, evaluation_data)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = run_dir / "candidate.joblib"
    joblib.dump(candidate, candidate_path)

    decision: dict[str, Any] = {
        "run_id": run_id,
        **assess_candidate(champion_holdout_metrics, candidate_holdout_metrics),
        "champion_model": str(champion_path),
        "candidate_model": str(candidate_path),
        "champion_holdout_metrics": champion_holdout_metrics,
        "candidate_holdout_metrics": candidate_holdout_metrics,
        "new_batch_rows": len(new_data),
        "adaptation_rows": len(adaptation_data),
        "evaluation_rows": len(evaluation_data),
        "training_rows": training_rows,
    }
    report = {
        "run_id": run_id,
        "incoming_batch": {
            "rows": len(new_data),
            "target_distribution": new_data[TARGET_COLUMN]
            .value_counts()
            .sort_index()
            .to_dict(),
            "champion_metrics": full_batch_metrics,
            "champion_holdout_metrics": champion_holdout_metrics,
        },
        "candidate_holdout_metrics": candidate_holdout_metrics,
        "split": {
            "adaptation_rows": len(adaptation_data),
            "evaluation_rows": len(evaluation_data),
            "evaluation_fraction": EVALUATION_FRACTION,
            "random_state": RANDOM_STATE,
        },
    }
    report_path = run_dir / "evaluation_report.json"
    decision_path = run_dir / "quality_gate_decision.json"
    _write_json(report_path, report)
    _write_json(decision_path, decision)

    mlflow_status = _log_mlflow(run_id, run_dir, report_path, decision)
    decision["mlflow"] = mlflow_status
    _write_json(decision_path, decision)

    print(f"Evaluation report: {report_path}")
    print(f"Candidate model: {candidate_path}")
    print(
        "Champion on incoming batch: "
        + ", ".join(f"{key}={value:.3f}" for key, value in full_batch_metrics.items())
    )
    print(
        "Holdout F1: "
        f"champion={champion_holdout_metrics['f1']:.3f}, "
        f"candidate={candidate_holdout_metrics['f1']:.3f}"
    )
    print(f"Quality gate: {decision['status']}")
    print("Champion was not changed.")
    print(f"MLflow: {mlflow_status}")
    return run_dir
