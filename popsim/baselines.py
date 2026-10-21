"""CPU-only comparison: learn baselines on 300 humans, evaluate on the remaining humans.

These baselines know the same survey items on different people. They do not establish
performance on entirely new questions. The demographic model is fixed ridge regression on
seven categorical demographic attributes (lambda=10, no tuning), rounded to answer levels.

    uv run python -m popsim.baselines
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from popsim import score, twin, uses


def split(people: pd.DataFrame, n_train: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = np.random.default_rng(0).choice(people.pid.to_numpy(), n_train, replace=False)
    return people[people.pid.isin(ids)], people[~people.pid.isin(ids)]


def demographic_predictions(train: pd.DataFrame, demo: pd.DataFrame, test_ids: pd.Index,
                            ranges: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """No held-out responses enter the fit or prediction. Unseen demographic levels map to zero."""

    x = pd.get_dummies(demo.loc[train.index].astype("string").fillna("missing"), dtype=float)
    z = pd.get_dummies(demo.loc[test_ids].astype("string").fillna("missing"), dtype=float).reindex(columns=x.columns, fill_value=0)
    x.insert(0, "intercept", 1.0)
    z.insert(0, "intercept", 1.0)
    penalty = 10 * np.eye(x.shape[1])
    penalty[0, 0] = 0
    out = {}
    for column, (lo, hi) in ranges.items():
        y = pd.to_numeric(train[column], errors="coerce").dropna()
        if y.empty:
            raise ValueError(f"no training answers for {column}")
        a = x.loc[y.index].to_numpy()
        coef = np.linalg.solve(a.T @ a + penalty, a.T @ y.to_numpy())
        out[column] = np.rint(z.to_numpy() @ coef).clip(lo, hi)
    return pd.DataFrame(out, index=test_ids)


def write_predictions(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    csvs = out / "csv"
    csvs.mkdir(parents=True, exist_ok=True)
    data = df.rename_axis("TWIN_ID").reset_index()
    with (csvs / "responses_llm_imputed_formatted.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data.columns)
        writer.writerow(["generated prediction"] * len(data.columns))
        data.to_csv(f, index=False, header=False)
    score._human_csvs(csvs)
    return score._score(csvs, out)


def main() -> None:
    people = twin.participants()
    train, test = split(people)
    assert set(train.pid).isdisjoint(test.pid)
    out = twin.ROOT / "results" / "baselines"
    humans = uses.formatted(twin.dataset_file(f"{score.HUMANS}/responses_wave4_formatted.csv")).rename(columns=str.upper)
    ranges = {c: r for c, r in score.question_ranges().items() if c not in {"Q164", "Q166", "Q168", "Q170"}}
    columns = [c for c in ranges if not c.endswith("_Q295")]
    tables, paths = {}, {}
    for mode in ("majority", "draw"):
        name = f"{mode}, 300 training humans"
        paths[name] = out / mode
        tables[name] = score.evaluate(score.population(test, train, mode), paths[name])

    name = "demographic ridge, 300 training humans"
    predictions = demographic_predictions(humans.loc[train.pid], uses.demographics(), pd.Index(test.pid), ranges)
    # Missingness encodes which randomized question version was shown; no answer values are copied.
    predictions = predictions.where(humans.loc[test.pid, predictions.columns].notna())
    paths[name] = out / "demographic"
    tables[name] = write_predictions(predictions, paths[name])

    for name, spec in uses.PUBLISHED.items():
        paths[name] = out / spec.replace(" ", "_")
        tables[name] = score.published(spec, list(test.pid), paths[name])
    for name, df in uses.jev_arms(len(people)).items():
        paths[name] = out / name.replace(" ", "_").replace(",", "")
        tables[name] = write_predictions(df.loc[test.pid], paths[name])

    results = {}
    for name, table in tables.items():
        kept = table.drop(index=score.SKIPPED_TASK, errors="ignore")
        results[name] = {
            "accuracy": float(kept["llm vs. wave1_3 Accuracy"].mean()),
            "person_signal": score.person_signal(paths[name] / "csv", columns),
        }
    first = next(iter(tables.values())).drop(index=score.SKIPPED_TASK, errors="ignore")
    results[score.RETEST] = {
        "accuracy": float(first["wave4 vs. wave1_3 Accuracy"].mean()),
        "person_signal": score.person_signal(next(iter(paths.values())) / "csv", columns, "wave4"),
    }
    report = {"seed": 0, "train_pids": train.pid.tolist(), "test_pids": test.pid.tolist(),
              "signal_questions": len(columns), "arms": results}
    (out / "scores.json").write_text(json.dumps(report, indent=2))
    print(f"\n{len(train)} training humans; {len(test)} disjoint evaluation humans; {len(columns)} signal questions")
    print(pd.DataFrame(results).T.round(4).to_string())
    print(f"wrote {out / 'scores.json'}")


if __name__ == "__main__":
    main()
