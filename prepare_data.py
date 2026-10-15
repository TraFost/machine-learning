"""Step 1: create and inspect synthetic leads. No model is trained here."""

import csv
import random
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "synthetic_leads.csv"
SEED = 42
LEAD_COUNT = 200


def generate_leads(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    leads = []
    for lead_id in range(1, count + 1):
        pricing = rng.random() < 0.55
        demo = rng.random() < 0.35
        intent = rng.choices(["low", "medium", "high"], weights=[45, 35, 20])[0]

        # An invented buying process, not a real business finding.
        # We sample outcomes with chance instead of labeling from a point score.
        # The future learner will receive features and labels, not this probability.
        chance = {"low": 0.06, "medium": 0.18, "high": 0.38}[intent]
        if pricing:
            chance += 0.03
        if demo:
            chance += 0.16
        if demo and intent == "high":
            chance += 0.12

        leads.append(
            {
                "lead_id": f"L-{lead_id:03}",
                "pricing_interest": int(pricing),
                "demo_requested": int(demo),
                "purchase_intent": intent,
                "purchased_within_30_days": int(rng.random() < chance),
            }
        )
    return leads


def main() -> None:
    leads = generate_leads(LEAD_COUNT, SEED)
    DATA_PATH.parent.mkdir(exist_ok=True)
    with DATA_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(leads[0]))
        writer.writeheader()
        writer.writerows(leads)

    print("STEP 1: inspect the examples before training")
    print("Question: after the first conversation, will a lead buy within 30 days?")
    print("All rows are synthetic. Each outcome window is assumed complete.\n")
    print(f"{'Lead':<8} {'Pricing?':<10} {'Demo?':<8} {'Intent':<9} {'Bought?':<8}")
    for lead in leads[:12]:
        print(
            f"{lead['lead_id']:<8} "
            f"{('yes' if lead['pricing_interest'] else 'no'):<10} "
            f"{('yes' if lead['demo_requested'] else 'no'):<8} "
            f"{lead['purchase_intent']:<9} "
            f"{('yes' if lead['purchased_within_30_days'] else 'no'):<8}"
        )
    buyers = sum(lead["purchased_within_30_days"] for lead in leads)
    print(f"\nTotal: {len(leads)} leads; {buyers} buyers; {len(leads) - buyers} nonbuyers.")
    print("\nHigh-intent leads do not always buy:")
    for outcome in [1, 0]:
        example = next(
            (
                lead
                for lead in leads
                if lead["purchase_intent"] == "high"
                and lead["purchased_within_30_days"] == outcome
            ),
            None,
        )
        if example:
            print(f"  {example['lead_id']}: high intent, bought={'yes' if outcome else 'no'}")
    print(f"\nCSV saved: {DATA_PATH}")
    print("No model trained yet. No training or test split created yet.")


if __name__ == "__main__":
    main()
