"""Step 2: keep separate examples for learning, selection, and final testing."""

import csv
import random

from prepare_data import DATA_PATH


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit("First run: uv run python prepare_data.py")

    with DATA_PATH.open(newline="", encoding="utf-8") as file:
        leads = list(csv.DictReader(file))

    if len(leads) < 5:
        raise SystemExit("At least 5 examples are required for this exercise.")

    all_ids = {lead["lead_id"] for lead in leads}
    if len(all_ids) != len(leads):
        raise SystemExit("Duplicate lead IDs found. Check the input data.")

    # Shuffle with a fixed seed so this exercise is repeatable.
    # Random splitting suits these independent synthetic examples.
    random.Random(42).shuffle(leads)
    train_end = int(len(leads) * 0.6)
    validation_end = int(len(leads) * 0.8)
    groups = {
        "train": leads[:train_end],
        "validation": leads[train_end:validation_end],
        "test": leads[validation_end:],
    }

    seen_ids = set()
    for name, group in groups.items():
        group_ids = {lead["lead_id"] for lead in group}
        if seen_ids & group_ids:
            raise RuntimeError("A lead appears in more than one group.")
        seen_ids.update(group_ids)

    if seen_ids != all_ids:
        raise RuntimeError("Some leads are missing from the split.")

    print("STEP 2: separate the examples before learning\n")
    print(f"{'Group':<14} {'Leads':>6}  Purpose")
    purposes = {
        "train": "Teach the model using these leads and their purchase outcomes.",
        "validation": "Check predictions on these leads while deciding which model to keep.",
        "test": "Check predictions on these leads after selecting the model.",
    }
    for name, group in groups.items():
        path = DATA_PATH.parent / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(leads[0]))
            writer.writeheader()
            writer.writerows(group)
        print(f"{name:<14} {len(group):>6}  {purposes[name]}")

    print("\nCheck passed: each original lead appears in exactly one group.")
    print("All three files keep the same columns, including the outcome label.")
    print("Only the training group will be used to teach the model.")
    print("For the other groups, the model will predict using lead information only.")
    print("We will compare those predictions with the recorded purchase outcomes.")
    print("\nSaved data/train.csv, data/validation.csv, and data/test.csv.")
    print("No model trained yet.")


if __name__ == "__main__":
    main()
