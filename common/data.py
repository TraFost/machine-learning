"""Shared input fields and data loading for all algorithms."""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LABEL = "purchased_within_30_days"


def read_leads(name: str) -> list[dict[str, str]]:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise SystemExit("First run: uv run python split_data.py")
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def features(leads: list[dict[str, str]]) -> list[dict[str, int | str]]:
    # Explicitly choose inputs. Neither lead ID nor purchase outcome is an input.
    return [
        {
            "pricing_interest": int(lead["pricing_interest"]),
            "demo_requested": int(lead["demo_requested"]),
            "purchase_intent": lead["purchase_intent"],
        }
        for lead in leads
    ]


