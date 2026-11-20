from typing import Any


MIN_CANDIDATE_F1 = 0.65


def assess_candidate(
    champion_metrics: dict[str, float], candidate_metrics: dict[str, float]
) -> dict[str, Any]:
    eligible = (
        candidate_metrics["f1"] >= MIN_CANDIDATE_F1
        and candidate_metrics["f1"] >= champion_metrics["f1"]
    )
    return {
        "status": "eligible_for_manual_promotion" if eligible else "retain_champion",
        "champion_updated_automatically": False,
        "gate": {
            "metric": "f1",
            "minimum_candidate_f1": MIN_CANDIDATE_F1,
            "rule": (
                "Candidate F1 on the untouched evaluation split must be at least "
                f"{MIN_CANDIDATE_F1:.2f} and no lower than champion F1 on the same split."
            ),
        },
    }
