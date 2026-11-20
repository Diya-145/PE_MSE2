# DrishtiAI MLOps Practical Assessment

A runnable, from-scratch model-quality monitoring and retraining example for
binary customer-churn classification. The sample data is synthetic and included
in `data/`; replace it with your own project data using the same column contract
or update the feature definitions in `mlops_monitor/workflow.py`.

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python scripts/generate_demo_data.py  # optional; regenerates the included data
python -m mlops_monitor bootstrap
python -m mlops_monitor run
pytest
```

The first command trains and saves the initial champion to
`artifacts/champion.joblib`. The workflow then validates the incoming labelled
batch, evaluates the champion, trains a separate candidate, and writes its
artifacts under a timestamped `artifacts/runs/<run-id>/` directory:

- `evaluation_report.json`: incoming-batch and holdout metrics.
- `candidate.joblib`: candidate model artifact.
- `quality_gate_decision.json`: decision, comparison, and MLflow status.

The workflow prints the generated paths and gate result. To use different data
or paths, pass `--train-data`, `--new-data`, `--champion-path`, or
`--artifacts-dir` to the corresponding command. Both CSVs must have unique
`customer_id` values, nonmissing fields, and these columns:
`tenure_months`, `monthly_charges`, `support_calls`, `contract_type`,
`payment_method`, `paperless_billing`, and binary target `churn`.

## Assessment workflow

1. **Prepare the data:** Each dataset is checked for required columns, missing
   values, duplicate IDs, valid target values, row count, and target
   distribution. Invalid input stops the run with a clear error.
2. **Evaluate the champion:** The saved pipeline predicts the entire new
   labelled batch. Accuracy, precision, recall, and F1 are saved in the report.
3. **Retrain a candidate:** A stratified 75/25 split reserves an untouched
   evaluation partition. The original training data is combined with the other
   75% of the incoming batch; the same preprocessing/model pipeline is fitted
   and saved separately as the candidate.
4. **Apply the quality gate:** Champion and candidate are compared on the same
   untouched 25% holdout. The candidate is eligible for manual promotion only
   if its F1 is at least 0.65 and no lower than the champion's holdout F1.
   F1 is selected because churn detection needs a balance between finding
   churners (recall) and avoiding unnecessary interventions (precision), even
   if the classes are imbalanced. The report records all four metrics for
   context. The gate never replaces the champion automatically.
5. **Track the run:** MLflow logging is attempted when the optional dependency
   is installed; otherwise the report records that MLflow was skipped. Install
   it with `python -m pip install -e ".[mlflow]"`.

An eligible candidate is **not** automatically promoted. Review its report and
decision, then deliberately decide whether to promote it; this keeps the
champion immutable during monitoring.