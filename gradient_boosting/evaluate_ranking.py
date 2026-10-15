"""Check gradient-boosted trees ranking on validation leads."""

from pathlib import Path

from common.evaluate_ranking import main


if __name__ == "__main__":
    main(Path(__file__).parent / "results")
