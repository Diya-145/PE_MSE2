import csv
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "customer_id",
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "contract_type",
    "payment_method",
    "paperless_billing",
    "churn",
]


def create_batch(path: Path, count: int, seed: int, id_start: int) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for offset in range(count):
            tenure = rng.randint(1, 72)
            monthly_charges = round(rng.uniform(30, 120), 2)
            support_calls = min(rng.choices(range(6), weights=[28, 25, 20, 14, 8, 5])[0], 5)
            contract = rng.choices(
                ["monthly", "annual", "two_year"], weights=[60, 25, 15]
            )[0]
            payment = rng.choice(["electronic", "bank_transfer", "credit_card"])
            paperless = rng.choice(["yes", "no"])

            score = (
                -2.0
                + 0.035 * (monthly_charges - 70)
                - 0.045 * (tenure - 24)
                + 0.55 * support_calls
                + {"monthly": 0.9, "annual": -0.45, "two_year": -1.0}[contract]
                + (0.4 if payment == "electronic" else 0)
                + (0.1 if paperless == "yes" else 0)
            )
            probability = 1 / (1 + math.exp(-score))
            churn = int(rng.random() < probability)
            writer.writerow(
                {
                    "customer_id": id_start + offset,
                    "tenure_months": tenure,
                    "monthly_charges": monthly_charges,
                    "support_calls": support_calls,
                    "contract_type": contract,
                    "payment_method": payment,
                    "paperless_billing": paperless,
                    "churn": churn,
                }
            )


def main() -> None:
    create_batch(ROOT / "data" / "train.csv", count=1400, seed=13, id_start=1)
    create_batch(
        ROOT / "data" / "incoming_batch.csv", count=800, seed=2026, id_start=10001
    )
    print("Generated data/train.csv (1,400 rows) and data/incoming_batch.csv (800 rows).")


if __name__ == "__main__":
    main()
