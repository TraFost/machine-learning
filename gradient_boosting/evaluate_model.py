"""Check gradient-boosted trees validation probabilities against the baseline."""

from pathlib import Path

from common.evaluate_model import main


if __name__ == "__main__":
    main(Path(__file__).parent / "results", "Gradient-boosted trees")
