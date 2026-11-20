# Project Evaluation Report

## Student Information

| Item | Details |
|---|---|
| Student | **Diya Kataria** |
| Student ID | **202401100600076** |
| Project | **DrishtiAI — MLOps Model Quality Monitoring and Retraining** |
| Evaluation date | **20 November 2026** |
| Evaluation dataset | `data/incoming_batch.csv` |

## 1. Executive Summary

This project implements an end-to-end machine-learning operations workflow for
binary customer-churn classification. It validates incoming labelled data,
evaluates a saved champion model, trains a candidate model, compares both
models on a held-out portion of the incoming batch, applies an F1-based quality
gate, and records the run and its artifacts.

The workflow completed successfully on the supplied data. On the 200-row
holdout, the candidate reached an F1 score of **0.720**, compared with **0.730**
for the champion. Although the candidate exceeded the minimum F1 threshold of
0.65, it did not match the champion, so the quality gate correctly returned
**`retain_champion`**. The champion was not automatically replaced.

The implementation demonstrates a reproducible monitoring and retraining
pipeline, input checks, model comparison, artifact handling, and MLflow
tracking. The included datasets are synthetic, so the reported scores are
demonstration results and should not be interpreted as evidence of performance
on real customer populations.

## 2. Project Goal and Scope

The project addresses the operational question: **when a new labelled batch
arrives, should a newly trained model replace the current model?**

It includes:

- Data validation for required fields, missing values, duplicate customer
  identifiers, and valid binary target values.
- A preprocessing and classification pipeline for numeric and categorical
  features.
- Champion evaluation on the incoming batch.
- Candidate retraining with an adaptation/evaluation split.
- A quality gate based on candidate F1 and champion comparison.
- Timestamped run artifacts and optional MLflow tracking.
- Automated tests and a GitHub Actions workflow for CI and model-run
  validation.

This repository is a monitoring and model-artifact workflow. It does **not**
provide a live prediction API, cloud deployment, or automatic model
promotion.

## 3. Data and Methodology

### Input data

The workflow used:

| Dataset | Rows | Retained (`churn=0`) | Churned (`churn=1`) |
|---|---:|---:|---:|
| Original training data | 1,400 | 912 | 488 |
| Incoming labelled batch | 800 | 532 | 268 |

The model uses `tenure_months`, `monthly_charges`, `support_calls`,
`contract_type`, `payment_method`, and `paperless_billing`. The
`customer_id` column is checked for uniqueness but is not used as a model
feature. The target is `churn`.

The repository's data-generation script creates synthetic data. Results from
this dataset are useful for demonstrating the workflow, but do not establish
real-world predictive validity, fairness, or business impact.

### Model and evaluation process

1. The champion is a scikit-learn pipeline using standard scaling for numeric
   features, one-hot encoding for categorical features, and a
   class-weight-balanced logistic regression classifier.
2. The incoming batch is split with stratification: **600 rows** for candidate
   adaptation/training and **200 rows** for evaluation. The split uses random
   state 42.
3. The candidate is trained on the original 1,400 training rows plus the 600
   adaptation rows, for **2,000 training rows** in total.
4. Champion and candidate metrics are calculated on the same held-out
   evaluation rows.
5. Candidate eligibility requires F1 of at least **0.65** and F1 no lower than
   the champion's holdout F1. Promotion, even when eligible, is manual.

## 4. Evaluation Results

### Champion on the complete incoming batch

These metrics describe the champion's predictions across all 800 incoming
records.

| Metric | Result |
|---|---:|
| Accuracy | 0.753 |
| Precision | 0.609 |
| Recall | 0.731 |
| F1 score | 0.664 |

### Champion and candidate on the 200-row holdout

| Metric | Champion | Candidate |
|---|---:|---:|
| Accuracy | 0.800 | 0.790 |
| Precision | 0.667 | 0.651 |
| Recall | 0.806 | 0.806 |
| F1 score | **0.730** | **0.720** |

### Quality-gate decision

**Decision: `retain_champion`**

The candidate's F1 of 0.720 is above the 0.65 minimum, but is approximately
0.010 lower than the champion's holdout F1 of 0.730. The candidate therefore
does not satisfy the non-regression rule. Retaining the champion is consistent
with the configured gate. The workflow recorded that automatic promotion is
disabled.

## 5. Implementation and Project Assessment

| Area | Evidence and evaluation |
|---|---|
| Data handling | Required columns, empty input, missing values, duplicate IDs, and target validity are checked. Validation fails explicitly for invalid input. |
| Machine-learning pipeline | Preprocessing is composed with the classifier in a scikit-learn pipeline, keeping training and prediction transformations together. |
| Evaluation design | The candidate is evaluated on a stratified holdout excluded from candidate training; both models are compared on the same rows. Accuracy, precision, recall, and F1 are reported. |
| Model governance | The F1 threshold and non-regression rule are explicit. The champion remains unchanged unless a person makes a promotion decision. |
| Reproducibility and artifacts | Runs use timestamped directories and save the candidate model, evaluation report, and quality-gate decision. The completed run was logged to MLflow. |
| Testing | The project test suite completed successfully: **3 passed**. The generated run also passed `scripts/validate_run.py`, which checks the report, metrics, quality-gate decision, and candidate artifact. |
| CI/CD | A GitHub Actions workflow is configured for dependency checks, Python compilation, tests, model evaluation, and artifact validation. The workflow configuration was inspected; a remote GitHub Actions run was not part of this evaluation. |
| Deployment scope | The project packages and validates model artifacts but has no deployed service or live inference endpoint. This limitation is stated in the project documentation. |

## 6. Strengths

- Clear separation between data validation, modeling, evaluation, and gate
  logic.
- Candidate evaluation uses a held-out split and compares the candidate
  directly with the champion.
- The model artifact and decision are preserved for review rather than
  silently replacing the champion.
- The run is reproducible with a fixed split seed and records metrics and
  tracking status.
- Tests cover core validation, quality-gate behavior, and end-to-end artifact
  creation.

## 7. Limitations and Suggested Improvements

- **Synthetic data:** Evaluate on a representative, appropriately governed
  real-world dataset before making operational decisions.
- **Metric uncertainty:** A 200-row holdout can produce variable estimates.
  Add confidence intervals or repeated/time-based validation when suitable.
- **Broader model review:** Include a confusion matrix and class-specific
  analysis, and assess calibration and relevant fairness criteria before
  production use.
- **Data drift:** The current workflow checks input validity and model quality
  but does not calculate feature-distribution drift or alert on it.
- **Deployment and operations:** A prediction service, monitoring alerts,
  access controls, and a documented rollback process would be needed for
  production deployment.

These are recommended extensions, not failures of the workflow's stated
monitoring-and-artifact-generation scope.

## 8. Reproducibility

From the repository root, the evaluation can be reproduced with:

```bash
python -m mlops_monitor run
pytest -q
python scripts/validate_run.py artifacts/runs artifacts/champion.joblib
```

The evaluated run is `20261120T094606931985Z`. Its machine-readable results and
candidate artifact are stored under
`artifacts/runs/20261120T094606931985Z/`.

## 9. Overall Conclusion

The project successfully demonstrates an end-to-end MLOps monitoring and
candidate-evaluation workflow. In this run, the candidate did not outperform
the champion on holdout F1, and the system correctly retained the existing
champion. The implementation and validation results support the project's
stated scope as a practical demonstration. Further validation on
representative data and additional operational safeguards would be required
before real-world deployment.

**Evaluator's grade/comments:** To be completed by the evaluator.
