import argparse
import json
import os
from pathlib import Path
from typing import Any

from mlops_monitor.quality_gate import assess_candidate


METRIC_NAMES = {"accuracy", "precision", "recall", "f1"}


def _validate_metrics(metrics: Any, label: str) -> None:
    if not isinstance(metrics, dict) or not METRIC_NAMES.issubset(metrics):
        raise ValueError(f"{label} must contain accuracy, precision, recall, and f1.")
    for metric in METRIC_NAMES:
        value = metrics[metric]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{label}.{metric} must be a number between 0 and 1.")


def validate_run(runs_dir: Path, champion_path: Path) -> tuple[Path, dict[str, Any]]:
    if not champion_path.is_file():
        raise FileNotFoundError(f"Champion model artifact does not exist: {champion_path}")
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Model runs directory does not exist: {runs_dir}")
    tracking_database = runs_dir.parent / "mlflow.db"
    if not tracking_database.is_file() or tracking_database.stat().st_size == 0:
        raise FileNotFoundError(f"MLflow tracking database was not persisted: {tracking_database}")

    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No model run directories found in {runs_dir}")
    run_dir = max(run_dirs, key=lambda path: path.name)

    report_path = run_dir / "evaluation_report.json"
    decision_path = run_dir / "quality_gate_decision.json"
    if not report_path.is_file() or not decision_path.is_file():
        raise FileNotFoundError(f"Model run is missing its report or gate decision: {run_dir}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    incoming = report.get("incoming_batch", {})
    if not isinstance(incoming.get("rows"), int) or incoming["rows"] <= 0:
        raise ValueError("Evaluation report must include a positive incoming-batch row count.")
    _validate_metrics(incoming.get("champion_metrics"), "incoming_batch.champion_metrics")
    _validate_metrics(
        decision.get("champion_holdout_metrics"), "champion_holdout_metrics"
    )
    _validate_metrics(
        decision.get("candidate_holdout_metrics"), "candidate_holdout_metrics"
    )

    candidate_path = Path(decision.get("candidate_model", ""))
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate model artifact does not exist: {candidate_path}")
    if not candidate_path.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("Candidate model must be saved inside its timestamped run directory.")
    if decision.get("champion_updated_automatically") is not False:
        raise ValueError("The workflow must explicitly record that it did not replace the champion.")

    expected_decision = assess_candidate(
        decision["champion_holdout_metrics"], decision["candidate_holdout_metrics"]
    )
    if decision.get("status") != expected_decision["status"]:
        raise ValueError("Recorded quality-gate status does not match the reported F1 metrics.")
    if decision.get("mlflow") != "logged":
        raise ValueError("The CI model run must be recorded in MLflow.")

    return run_dir, decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a completed MLOps run.")
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("champion_path", type=Path)
    args = parser.parse_args()

    run_dir, decision = validate_run(args.runs_dir, args.champion_path)
    report_path = run_dir / "evaluation_report.json"
    candidate_path = Path(decision["candidate_model"])
    summary = (
        "## MLOps model run\n\n"
        f"- **Quality gate:** `{decision['status']}`\n"
        f"- **Champion holdout F1:** {decision['champion_holdout_metrics']['f1']:.3f}\n"
        f"- **Candidate holdout F1:** {decision['candidate_holdout_metrics']['f1']:.3f}\n"
        f"- **Evaluation report:** `{report_path}`\n"
        f"- **Candidate model:** `{candidate_path}`\n"
        "- **MLflow tracking:** logged\n"
        "- **Champion auto-promotion:** disabled\n"
    )
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary_file:
            summary_file.write(summary)

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as output_file:
            output_file.write(f"quality_gate={decision['status']}\n")

    print(summary)


if __name__ == "__main__":
    main()
