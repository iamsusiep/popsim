"""Person signal for every published twin run and for ours, with 95% intervals.

    uv run python -m popsim.signal results/summary-n2058

Person signal (see score.py) for the 13 published simulation specs, for JEV-9B J and P from a
scored run, and for the humans' retest, over the questions that run was scored on. Intervals
bootstrap the questions; "vs JEV-9B J" uses the same questions and shared participants. Writes
results/signal.json.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from popsim import score, twin

SPECS = (
    "Demographics Only - GPT4.1-mini",
    "Persona Summary - GPT4.1-mini",
    "Persona Summary - JSON Persona - GPT4.1-mini",
    "Text Persona - GPT4.1-mini",
    "Text Persona (Default Temperature) - GPT4.1-mini",
    "Text Persona (Reasoning) - GPT4.1-mini",
    "Text Persona (Repeating Questions) - GPT4.1-mini",
    "JSON Persona - GPT4.1-mini",
    "JSON Persona (Predicted Output) - GPT4.1-mini",
    "JSON Persona - GPT4.1",
    "JSON Persona (Predicted Output) - GPT4.1",
    "Text Persona - Gemini-Flash2.5",
    "LLM Finetuning (500 training samples) - GPT4.1-mini",
)
JEV = "JEV-9B J"
DRAWS = 5000


def main(run: Path) -> None:
    columns = score.signal_columns(run)
    csvs = {score.RETEST: run / "J" / "csv", JEV: run / "J" / "csv", "JEV-9B P": run / "P" / "csv"}
    for spec in SPECS:
        csvs[spec] = twin.ROOT / "results" / "published" / spec.replace(" ", "_") / "csv"
        csvs[spec].mkdir(parents=True, exist_ok=True)
        source = twin.dataset_file(f"{score.SPECS}/{spec}/csv_comparison/csv_formatted/responses_llm_imputed_formatted.csv")
        shutil.copyfile(source, csvs[spec] / "responses_llm_imputed_formatted.csv")
        score._human_csvs(csvs[spec])
    table = pd.DataFrame({name: score.per_question(path, columns, "wave4" if name == score.RETEST else "llm_imputed") for name, path in csvs.items()})

    draws = np.random.default_rng(0).integers(0, len(columns), (DRAWS, len(columns)))

    def interval(values: pd.Series) -> list[float]:
        v = values.to_numpy()
        return [round(float(v.mean()), 4), *np.percentile(v[draws].mean(axis=1), [2.5, 97.5]).round(4).tolist()]

    results = {}
    for name in table:
        shared = score._load(csvs[JEV], "llm_imputed").index.intersection(score._load(csvs[name], "llm_imputed").index)
        difference = None
        if name not in (JEV, score.RETEST):
            difference = interval(score.per_question(csvs[JEV], columns, pids=shared)
                                  - score.per_question(csvs[name], columns, pids=shared))
        results[name] = {
            "people": len(score._load(csvs[name], "llm_imputed")),
            "signal": interval(table[name]),
            "paired_people": len(shared),
            "vs JEV-9B J": difference,
        }
    out = twin.ROOT / "results" / "signal.json"
    out.write_text(json.dumps({"run": str(run), "questions": len(columns), "arms": results}, indent=2))

    print(f"person signal over {len(columns)} questions, 95% intervals over questions; JEV-9B J from {run}")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["signal"][0]):
        m, lo, hi = r["signal"]
        diff = "" if r["vs JEV-9B J"] is None else "  J minus this {:+.3f} [{:+.3f}, {:+.3f}]".format(*r["vs JEV-9B J"])
        print(f"  {name:52s} {r['people']:5d} people  {m:.3f} [{lo:.3f}, {hi:.3f}]{diff}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
