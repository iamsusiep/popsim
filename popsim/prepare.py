"""Download the pinned public survey data and upstream scorer; no model inference or GPU.

    uv run python -m popsim.prepare
    uv run python -m popsim.prepare --example  # also rebuild the bundled aggregate-only demo
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from popsim import calibrate, score, signal, twin, uses


def build_example() -> None:
    humans = uses.formatted(twin.dataset_file(f"{score.HUMANS}/responses_wave4_formatted.csv"))
    offers = uses.offers()
    rng = np.random.default_rng(0)
    survey = humans.iloc[rng.choice(len(humans), 300, replace=False)]
    values = {"human": calibrate.cells("pricing", humans, offers, pd.DataFrame()),
              "survey": calibrate.cells("pricing", survey, offers, pd.DataFrame())}
    twins = []
    for spec in uses.PUBLISHED.values():
        df = uses.formatted(twin.dataset_file(f"{score.SPECS}/{spec}/csv_comparison/csv_formatted/responses_llm_imputed_formatted.csv"))
        twins.append(calibrate.cells("pricing", df, offers, pd.DataFrame()))
    values["twin"] = pd.concat(twins, axis=1).mean(axis=1)
    frame = pd.DataFrame(values).reset_index()
    example = {
        "description": "One fixed 300-person survey and 10-product split; not the 500-draw headline estimate.",
        "dataset": twin.DATASET, "revision": twin.DATASET_REVISION,
        "source": f"https://huggingface.co/datasets/{twin.DATASET}",
        "license": "CC-BY-4.0; see NOTICE for attribution",
        "people": 2058, "survey_people": 300, "seed": 0,
        "surveyed": rng.choice(frame['product'].unique(), 10, replace=False).tolist(),
        "cells": frame.to_dict(orient="records"),
    }
    path = Path(__file__).with_name("example_pricing.json")
    path.write_text(json.dumps(example, indent=2))
    print(f"wrote aggregate-only example: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", action="store_true", help="rebuild the bundled example from the pinned data")
    args = parser.parse_args()
    paths = [f"{config}/chunks/{prefix}_{i:03d}.parquet"
             for config, prefix in (("wave_split", "wave_persona_chunk"), ("full_persona", "persona_chunk"))
             for i in range(1, twin.CHUNKS + 1)]
    paths += [f"{score.HUMANS}/responses_{wave}_formatted.csv" for wave in ("wave1_3", "wave4")]
    paths += ["raw_data/wave_4_numbers_anonymized.csv", "question_catalog_and_human_response_csv/wave1_3_response.csv"]
    paths += [f"{score.SPECS}/{spec}/csv_comparison/csv_formatted/responses_llm_imputed_formatted.csv" for spec in signal.SPECS]
    for i, path in enumerate(paths, 1):
        twin.dataset_file(path)
        print(f"[{i}/{len(paths)}] ready: {path}")
    twin.twin_code()
    print(f"scorer ready at {twin.CODE_REVISION}")
    if args.example:
        build_example()


if __name__ == "__main__":
    main()
