"""Survey a few, simulate the rest: do twins plus a small human survey predict the units nobody
was asked about?

A client surveys real people on k units and runs twins on all of them. The twins are calibrated
on the surveyed units and scored on the rest, against what the survey alone predicts there.

    pricing  unit = product; cells = its buy rate (%) at 5 price levels
    polling  unit = policy or technology item; cells = its 7 segment gaps / answer range
    A/B      unit = wording contrast; cell = its mean difference / answer range

Fitting uses only fixed questionnaire ranges, never human SDs. Human SDs convert prediction
errors to SD / Cohen's d units only after predictions have been made.

Calibration is a least-squares fit on the surveyed cells: value = a[cell group] + b * twin value,
where the group is the price level or the segment. Without twins (b = 0) that is the surveyed
units' average for each group. A/B contrasts have no common direction, so there it is
value = b * twin value, against predicting no effect.

The survey is either all 2,058 participants or a random 300 of them, as a client would field;
errors are always against all 2,058. Surveyed units (and people) are drawn at random, 500 times
per setting. "beats" is the share of draws where adding the twins lowered the error.
Rich profiles for the full panel already exist. This counts additional calibration responses,
not the total human data needed to construct twins, and tests new units within the same panel.

    uv run python -m popsim.calibrate
"""

from __future__ import annotations

import json
from functools import cache

import numpy as np
import pandas as pd

from popsim import score, twin, uses

SURVEYED = {"pricing": (5, 10, 20), "polling": (3, 6, 9), "ab": (4, 8)}
PEOPLE = (300, 2058)
SPLITS = 500
ENSEMBLE = "ensemble of the published twins"
NONE = {"pricing": "survey average, no twins", "polling": "survey average, no twins", "ab": "no effect, no twins"}


@cache
def contrast_ranges() -> dict[str, float]:
    ranges = score.question_ranges()
    return {name: ranges[a.upper()][1] - ranges[a.upper()][0] for name, (a, _) in uses.CONTRASTS.items()}


def cells(use: str, df: pd.DataFrame, offers: pd.DataFrame, demo: pd.DataFrame) -> pd.Series:
    """One arm's values, indexed by (unit, group), without access to any other arm's answers."""

    if use == "pricing":
        return 100 * uses._buys(df, offers).groupby(["product", "quintile"]).buy.mean()
    if use == "polling":
        return pd.Series({
            (item, name): (df.loc[a, item].mean() - df.loc[b, item].mean()) / (hi - lo)
            for name, a, b in uses._groups(demo, df.index)
            for item, (lo, hi) in uses.ITEMS.items()
        })
    spans = contrast_ranges()
    return pd.Series({(name, "all"): (df[b].mean() - df[a].mean()) / spans[name]
                      for name, (a, b) in uses.CONTRASTS.items()})


def error_scales(use: str, humans: pd.DataFrame, index: pd.MultiIndex) -> pd.Series:
    """Evaluation units only. These factors must never be supplied to the predictor."""

    if use == "pricing":
        return pd.Series(1.0, index=index)
    if use == "polling":
        scales = {item: (hi - lo) / humans[item].std() for item, (lo, hi) in uses.ITEMS.items()}
    else:
        scales = {name: contrast_ranges()[name] / uses._pooled_sd(humans, a, b)
                  for name, (a, b) in uses.CONTRASTS.items()}
    return pd.Series([scales[unit] for unit, _ in index], index=index)


def predict(truth: pd.Series, sim: pd.Series | None, surveyed: np.ndarray, intercepts: bool) -> pd.Series:
    """Values of the cells outside `surveyed`, fitted on the cells inside it."""

    inside = truth.index.get_level_values(0).isin(surveyed)
    columns = []
    if intercepts:
        columns.append(pd.get_dummies(truth.index.get_level_values(1)).set_axis(truth.index).astype(float))
    if sim is not None:
        columns.append(sim.rename("twin").to_frame())
    if not columns:
        return pd.Series(0.0, index=truth.index[~inside])
    X = pd.concat(columns, axis=1)
    coef, *_ = np.linalg.lstsq(X[inside].to_numpy(), truth[inside].to_numpy(), rcond=None)
    return pd.Series(X[~inside].to_numpy() @ coef, index=truth.index[~inside])


def evaluate(use: str, truth: pd.Series, arms: dict[str, pd.Series], survey_cells, people: int, n: int,
             scales: pd.Series) -> dict[str, dict]:
    """Mean RMSE over held-out cells for each predictor and number of surveyed units, and how often
    each predictor beats the survey alone. `survey_cells(ids)` gives the humans' cells among `ids`."""

    units = truth.index.get_level_values(0).unique().to_numpy()
    intercepts = use != "ab"
    rng = np.random.default_rng(0)
    out = {}
    for k in SURVEYED[use]:
        errors: dict[str, list[float]] = {}
        for _ in range(SPLITS):
            surveyed = rng.choice(units, k, replace=False)
            survey = truth if people >= n else survey_cells(rng.choice(n, people, replace=False)).reindex(truth.index)
            held = ~truth.index.get_level_values(0).isin(surveyed)
            guesses = {NONE[use]: predict(survey, None, surveyed, intercepts)}
            for name, sim in arms.items():
                guesses[f"{name}, raw"] = sim[held]
                if name != score.RETEST:
                    guesses[f"{name}, calibrated"] = predict(survey, sim, surveyed, intercepts)
            for name, guess in guesses.items():
                residual = (guess - truth[held]) * scales[held]
                if not np.isfinite(residual).all():
                    raise ValueError(f"{name}: missing or nonfinite evaluation cells")
                errors.setdefault(name, []).append(float(np.sqrt((residual ** 2).mean())))
        base = np.array(errors[NONE[use]])
        out[f"{people} people, {k} units"] = {
            name: {"error": float(np.mean(e)), "beats": float(np.mean(np.array(e) < base))} for name, e in errors.items()
        }
    return out


def main() -> None:
    humans = uses.formatted(twin.dataset_file(f"{score.HUMANS}/responses_wave4_formatted.csv"))
    retest = uses.formatted(twin.dataset_file(f"{score.HUMANS}/responses_wave1_3_formatted.csv"))
    published = {
        name: uses.formatted(twin.dataset_file(f"{score.SPECS}/{spec}/csv_comparison/csv_formatted/responses_llm_imputed_formatted.csv"))
        for name, spec in uses.PUBLISHED.items()
    }
    offers, demo = uses.offers(), uses.demographics()
    units = {"pricing": "products, buy-rate error in pp", "polling": "items, gap error in SDs", "ab": "contrasts, effect error in d"}

    results = {}
    for use, label in units.items():
        truth = cells(use, humans, offers, demo)
        scales = error_scales(use, humans, truth.index)
        twins = {name: cells(use, df, offers, demo).reindex(truth.index) for name, df in published.items()}
        arms = {score.RETEST: cells(use, retest, offers, demo).reindex(truth.index), **twins}
        arms[ENSEMBLE] = pd.concat(twins.values(), axis=1).mean(axis=1)
        arms |= {name: cells(use, df, offers, demo).reindex(truth.index) for name, df in uses.jev_arms(len(humans)).items()}
        survey_cells = lambda rows: cells(use, humans.iloc[rows], offers, demo)
        table = {}
        for people in PEOPLE:
            table |= evaluate(use, truth, arms, survey_cells, people, len(humans), scales)
        results[use] = table
        print(f"\n{use}: error on units nobody surveyed ({label}) / share of draws beating the survey alone")
        print(pd.DataFrame({setting: {name: f"{r['error']:.2f} / {r['beats']:.0%}" for name, r in rows.items()} for setting, rows in table.items()}).to_string())

    out = twin.ROOT / "results" / "calibrate.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
