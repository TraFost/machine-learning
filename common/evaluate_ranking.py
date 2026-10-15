"""Step 5: inspect the highest-ranked validation leads without using test data."""

import csv
from pathlib import Path

from common.data import LABEL, read_leads


def main(results_dir: Path) -> None:
    source = results_dir / "validation_predictions.csv"
    if not source.exists():
        raise SystemExit("First run the algorithm training module.")
    with source.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    validation = read_leads("validation")
    expected = {row["lead_id"]: row for row in validation}
    if len(rows) != len(expected) or {row["lead_id"] for row in rows} != set(expected):
        raise SystemExit("Predictions do not match validation data. Run the algorithm training module again.")
    for row in rows:
        if any(row[key] != value for key, value in expected[row["lead_id"]].items()):
            raise SystemExit("Validation data changed. Run the algorithm training module again.")
        probability = float(row["predicted_purchase_probability"])
        if not 0 <= probability <= 1:
            raise SystemExit("Purchase probabilities must be between 0 and 1.")

    # Never use actual purchase labels to choose or break ties in the calling list.
    ranked = sorted(rows, key=lambda row: (-float(row["predicted_purchase_probability"]), row["lead_id"]))
    k = min(10, len(ranked))
    if not k:
        raise SystemExit("No validation leads found.")
    selected = ranked[:k]
    total_buyers = sum(int(row[LABEL]) for row in rows)
    found_buyers = sum(int(row[LABEL]) for row in selected)
    cutoff = float(selected[-1]["predicted_purchase_probability"])
    above = [row for row in ranked if float(row["predicted_purchase_probability"]) > cutoff]
    tied = [row for row in ranked if float(row["predicted_purchase_probability"]) == cutoff]
    available = k - len(above)
    above_buyers = sum(int(row[LABEL]) for row in above)
    tied_buyers = sum(int(row[LABEL]) for row in tied)
    expected_buyers = above_buyers + available * tied_buyers / len(tied)
    random_buyers = k * total_buyers / len(rows)

    lines = [
        "# Validation ranking: contact the top 10 leads",
        "",
        f"There are {len(rows)} validation leads and {total_buyers} actual buyers.",
        f"We select {k} leads using predicted probability, highest first.",
        "When scores tie, lead ID decides the displayed order. Purchase outcomes never decide the order.",
        "",
        "| Lead | Predicted purchase chance | Actually bought? |",
        "|---|---:|---|",
    ]
    for row in selected:
        lines.append(f"| {row['lead_id']} | {float(row['predicted_purchase_probability']):.1%} | {'Yes' if int(row[LABEL]) else 'No'} |")
    lines.extend([
        "",
        f"This displayed list contains {found_buyers} buyers among {k} leads.",
        f"Precision@{k}: {found_buyers / k:.1%}. This is the buyer rate in the selected list.",
        f"Buyers found: {found_buyers} of {total_buyers} actual buyers.",
        "",
        "## Check the tied scores",
        "",
        f"At the cutoff, {len(tied)} leads share a {cutoff:.1%} score, but only {available} places remain.",
    ])
    if available < len(tied):
        minimum = above_buyers + max(0, available - (len(tied) - tied_buyers))
        maximum = above_buyers + min(available, tied_buyers)
        lines.append(f"Different selections within this tie could give {minimum} to {maximum} buyers in the top {k}.")
    else:
        lines.append("All leads tied at this cutoff fit in the selected list.")
    lines.extend([
        f"With random selection inside the tie, the expected buyer count is {expected_buyers:.2f}.",
        "An expected count is an average across possible selections, not a fraction of a person.",
        "",
        "## Compare with random selection",
        "",
        f"Randomly choosing {k} of these {len(rows)} leads would find {random_buyers:.2f} buyers on average.",
    ])
    if random_buyers:
        lines.append(f"The model's tie-adjusted expected buyer rate is {expected_buyers / random_buyers:.2f} times the random-selection rate. This is lift@{k}.")
    lines.extend([
        "",
        "This checks ordering, not whether predicted percentages are accurate.",
        "It uses observed validation outcomes for evaluation; those outcomes were unknown to the model.",
        "It does not prove contacting these leads causes more purchases.",
        "This small synthetic validation group is not enough for a reliable business conclusion.",
        "The test set was not opened. No model was trained or changed.",
    ])
    report = "\n".join(lines) + "\n"
    output = results_dir / "validation_ranking.md"
    output.write_text(report, encoding="utf-8")
    with (results_dir / "validation_ranked.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["rank", "selected_top_10", *ranked[0]])
        writer.writeheader()
        for position, row in enumerate(ranked, start=1):
            writer.writerow({"rank": position, "selected_top_10": int(position <= k), **row})
    print(report)
    print(f"Saved: {output}")


