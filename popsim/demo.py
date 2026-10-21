"""Reproduce one pricing-calibration split offline, using bundled aggregate data.

    uv run python -m popsim.demo
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from popsim.calibrate import predict


def main() -> dict[str, float]:
    example = json.loads(Path(__file__).with_name("example_pricing.json").read_text())
    cells = pd.DataFrame(example["cells"]).set_index(["product", "quintile"])
    surveyed = np.array(example["surveyed"])
    held = ~cells.index.get_level_values(0).isin(surveyed)
    guesses = {"survey alone": predict(cells.survey, None, surveyed, True),
               "survey + twins": predict(cells.survey, cells.twin, surveyed, True)}
    errors = {name: float(np.sqrt(((guess - cells.human[held]) ** 2).mean())) for name, guess in guesses.items()}
    print("Pricing calibration: one fixed split of 10 surveyed and 30 held-out products")
    print("300-person calibration survey; pre-existing profiles for the 2,058-person panel")
    for name, error in errors.items():
        print(f"  {name}: {error:.2f} percentage points RMSE")
    print("This illustrates the workflow; the full analysis averages 500 splits.")
    print(f"Source: {example['source']} at {example['revision']}; CC-BY-4.0")
    return errors


if __name__ == "__main__":
    main()
