"""Shared top-K calculation for fair comparisons, including tied scores."""

from common.data import LABEL


def ranking_summary(rows: list[dict], k: int = 10) -> tuple[list[dict], int, float]:
    ranked = sorted(
        rows,
        key=lambda row: (-row["predicted_purchase_probability"], row["lead_id"]),
    )
    k = min(k, len(ranked))
    cutoff = ranked[k - 1]["predicted_purchase_probability"]
    above = [row for row in ranked if row["predicted_purchase_probability"] > cutoff]
    tied = [row for row in ranked if row["predicted_purchase_probability"] == cutoff]
    slots = k - len(above)
    expected = sum(int(row[LABEL]) for row in above) + slots * sum(
        int(row[LABEL]) for row in tied
    ) / len(tied)
    actual = sum(int(row[LABEL]) for row in ranked[:k])
    return ranked, actual, expected


