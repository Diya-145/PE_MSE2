import argparse
from pathlib import Path

from mlops_monitor.modeling import bootstrap_champion
from mlops_monitor.workflow import run_monitoring_workflow


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor and retrain a churn model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap", help="Train the initial champion from the original training data."
    )
    bootstrap_parser.add_argument(
        "--train-data", type=Path, default=ROOT / "data" / "train.csv"
    )
    bootstrap_parser.add_argument(
        "--champion-path", type=Path, default=ROOT / "artifacts" / "champion.joblib"
    )

    run_parser = subparsers.add_parser(
        "run", help="Evaluate the champion and train/evaluate a candidate."
    )
    run_parser.add_argument(
        "--train-data", type=Path, default=ROOT / "data" / "train.csv"
    )
    run_parser.add_argument(
        "--new-data", type=Path, default=ROOT / "data" / "incoming_batch.csv"
    )
    run_parser.add_argument(
        "--champion-path", type=Path, default=ROOT / "artifacts" / "champion.joblib"
    )
    run_parser.add_argument(
        "--artifacts-dir", type=Path, default=ROOT / "artifacts" / "runs"
    )

    args = parser.parse_args()
    if args.command == "bootstrap":
        bootstrap_champion(args.train_data, args.champion_path)
    else:
        run_monitoring_workflow(
            args.train_data, args.new_data, args.champion_path, args.artifacts_dir
        )


if __name__ == "__main__":
    main()
