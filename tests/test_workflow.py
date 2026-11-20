import json
from pathlib import Path

import pandas as pd
import pytest

from mlops_monitor.data import REQUIRED_COLUMNS, load_validated_data
from mlops_monitor.modeling import bootstrap_champion
from mlops_monitor.quality_gate import assess_candidate
from mlops_monitor.workflow import run_monitoring_workflow


def test_load_validated_data_rejects_duplicate_ids(tmp_path: Path) -> None:
    rows = [
        [1, 10, 50.0, 1, "monthly", "electronic", "yes", 0],
        [1, 20, 60.0, 2, "annual", "credit_card", "no", 1],
    ]
    path = tmp_path / "duplicate.csv"
    pd.DataFrame(rows, columns=REQUIRED_COLUMNS).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate customer IDs"):
        load_validated_data(path, "Test data")


def test_quality_gate_only_approves_non_regressing_candidate() -> None:
    champion_metrics = {"f1": 0.70}
    eligible = assess_candidate(champion_metrics, {"f1": 0.71})
    regressing = assess_candidate(champion_metrics, {"f1": 0.69})
    below_minimum = assess_candidate({"f1": 0.60}, {"f1": 0.64})

    assert eligible["status"] == "eligible_for_manual_promotion"
    assert regressing["status"] == "retain_champion"
    assert below_minimum["status"] == "retain_champion"
    assert eligible["champion_updated_automatically"] is False


def test_workflow_saves_candidate_report_and_retains_champion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_data = pd.read_csv(Path(__file__).parents[1] / "data" / "train.csv")
    incoming_data = pd.read_csv(
        Path(__file__).parents[1] / "data" / "incoming_batch.csv"
    )
    train_path = tmp_path / "train.csv"
    incoming_path = tmp_path / "incoming.csv"
    champion_path = tmp_path / "champion.joblib"
    source_data.to_csv(train_path, index=False)
    incoming_data.to_csv(incoming_path, index=False)
    bootstrap_champion(train_path, champion_path)
    original_model_bytes = champion_path.read_bytes()

    monkeypatch.setattr("mlops_monitor.workflow._log_mlflow", lambda *args: "not installed")
    run_dir = run_monitoring_workflow(
        train_path, incoming_path, champion_path, tmp_path / "runs"
    )

    report = json.loads((run_dir / "evaluation_report.json").read_text())
    decision = json.loads((run_dir / "quality_gate_decision.json").read_text())
    assert (run_dir / "candidate.joblib").is_file()
    assert report["incoming_batch"]["rows"] == len(incoming_data)
    assert decision["champion_updated_automatically"] is False
    assert champion_path.read_bytes() == original_model_bytes
