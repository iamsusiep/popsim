"""Score saved distributions with the Twin authors' converter and scorer, and with population-share measures.

    uv run python -m popsim.score results/summary-n200.parquet

A distribution becomes an answer two ways, as in jeve: J takes the most likely option (what a
point-predicting twin does); P draws one, seeded per participant and question.

Two baselines know the population but not the person, from every other participant's wave-4
answers to the same version of each question: everyone gets the most common answer ("majority"),
or a draw from the answer shares. These target-population oracle diagnostics have access to
other test participants' labels. For separate training/test samples, run popsim.baselines.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from popsim import twin

HUMANS = "LLM_simulation_results/GPT4.1-mini-simulation-llm-vs-human"
SPECS = "LLM_simulation_results/llm_simulations_all_specifications"
# The Twin authors' own GPT-4.1-mini twins: the one given the same persona form as our run, and the
# full-persona ones (their headline, and its temperature-1 variant, the sampling analogue of P).
MATCHED = {
    "demographics": "Demographics Only - GPT4.1-mini",
    "summary": "Persona Summary - GPT4.1-mini",
    "full": "Text Persona - GPT4.1-mini",
}
FULL = {"GPT-4.1-mini, full": "Text Persona - GPT4.1-mini", "GPT-4.1-mini, full, temp 1": "Text Persona (Default Temperature) - GPT4.1-mini"}
SKIPPED_TASK = "anchoring and adjustment"  # open number entries: no options to decide between
RETEST, RANDOM = "humans, retest", "random"
POPULATION = {"majority": "majority answer (target-population oracle)", "draw": "population draw (target-population oracle)"}


def population(people: pd.DataFrame, everyone: pd.DataFrame, mode: str) -> dict[int, list[dict]]:
    """Answer blocks for `people` from everyone else's answers to the same question wording (or,
    where no one else saw that wording, the same question): "majority" or "draw"."""

    by_wording, by_key, own = defaultdict(Counter), defaultdict(Counter), {}
    for row in everyone.itertuples():
        blocks = json.loads(row.wave4_Q_wave4_A)
        picks = twin.human_picks(blocks)
        for key, q in twin.questions(blocks).items():
            answer = q["criteria"][str(picks[key] + 1)]
            by_wording[key, q["instructions"]][answer] += 1
            by_key[key][answer] += 1
            own[row.pid, key] = Counter({answer: 1})
    out = {}
    for row in people.itertuples():
        blocks = json.loads(row.wave4_Q_wave4_A)
        picks = {}
        for key, q in twin.questions(blocks).items():
            options = list(q["criteria"].values())
            mine = own.get((row.pid, key), Counter())
            counts = (by_wording[key, q["instructions"]] - mine) or (by_key[key] - mine)
            if mode == "majority":
                answer = counts.most_common(1)[0][0]
            else:
                answer = random.Random(f"{row.pid}:{key}").choices(list(counts), weights=list(counts.values()))[0]
            picks[key] = options.index(answer)
        out[row.pid] = twin.fill(blocks, picks)
    return out


def pick(probs: list[float], mode: str, key: str) -> int:
    if mode == "J":
        return int(np.argmax(probs))
    roll = random.Random(key).random()
    return min(int(np.searchsorted(np.cumsum(probs), roll, side="right")), len(probs) - 1)


def _run(script: Path, *args: object) -> None:
    subprocess.run([sys.executable, "-I", str(script), *map(str, args)], check=True)


def _human_csvs(csvs: Path) -> None:
    for name in ("responses_wave1_3_formatted.csv", "responses_wave4_formatted.csv"):
        (csvs / name).unlink(missing_ok=True)  # an earlier copy may carry the cache's read-only mode
        shutil.copyfile(twin.dataset_file(f"{HUMANS}/{name}"), csvs / name)


def _score(csvs: Path, out: Path) -> pd.DataFrame:
    # Legacy positional form: the script's --csv-dir flag is shadowed by a positional of the same name.
    _run(twin.twin_code() / "evaluation/mad_accuracy_evaluation.py", csvs, out / "mad.xlsx", out / "accuracy.png")
    return pd.read_excel(out / "mad.xlsx", sheet_name="Accuracy - task level").set_index("Task")


def evaluate(blocks_by_pid: dict[int, list[dict]], out: Path) -> pd.DataFrame:
    """The Twin converter and scorer, unmodified, on our answer blocks. Returns the task table."""

    answers, csvs = out / "answer_blocks", out / "csv"
    answers.mkdir(parents=True, exist_ok=True)
    csvs.mkdir(exist_ok=True)
    for pid, blocks in blocks_by_pid.items():
        (answers / f"pid_{pid}_wave4_Q_wave4_A.json").write_text(json.dumps(blocks))
    formatted = csvs / "responses_llm_imputed_formatted.csv"
    config = {
        "waves": {
            "llm_imputed": {
                "input_pattern": str(answers / "pid_{pid}_wave4_Q_wave4_A.json"),
                "output_csv_formatted": str(formatted),
            }
        },
        "benchmark_csv": str(twin.dataset_file("raw_data/wave_4_numbers_anonymized.csv")),
        "column_mapping": str(twin.twin_code() / "evaluation/column_mapping.csv"),
    }
    (out / "json2csv.yaml").write_text(yaml.safe_dump(config))
    _run(twin.twin_code() / "evaluation/json2csv.py", "--config", out / "json2csv.yaml", "--format")
    if not formatted.exists():  # the converter logs failures and carries on
        raise RuntimeError(f"json2csv wrote no {formatted}")
    _human_csvs(csvs)
    return _score(csvs, out)


def published(spec: str, pids: list[int], out: Path) -> pd.DataFrame:
    """One of the Twin authors' published simulations, scored on the same participants."""

    csvs = out / "csv"
    csvs.mkdir(parents=True, exist_ok=True)
    keep = set(map(str, pids))
    source = f"{SPECS}/{spec}/csv_comparison/csv_formatted/responses_llm_imputed_formatted.csv"
    with open(twin.dataset_file(source), newline="") as f:
        header, description, *rows = csv.reader(f)
    with open(csvs / "responses_llm_imputed_formatted.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([header, description, *(r for r in rows if r[header.index("TWIN_ID")] in keep)])
    _human_csvs(csvs)
    return _score(csvs, out)


def question_ranges() -> dict[str, tuple[int, int]]:
    """The Twin scorer's answer columns and their ranges."""

    sys.path.insert(0, str(twin.twin_code() / "evaluation"))
    from mad_accuracy_evaluation import get_default_column_ranges

    return get_default_column_ranges()


def aggregate(csvs: Path, null_draws: int = 200) -> dict:
    """Population-share measures: per question, Spearman between the answer-share vectors and their RMSE
    in percentage points, against wave-4 humans; the median Spearman and the mean RMSE over
    questions, kept apart for yes/no (where Spearman is just +1 or -1) and multi-option items.

    `null` predicts each question with the human shares of a different question with as many
    options: what the metric gives for free."""

    def load(name: str) -> pd.DataFrame:
        return pd.read_csv(csvs / name, skiprows=[1]).rename(columns=str.upper).set_index("TWIN_ID")

    w13, w4, sim = (load(f"responses_{w}_formatted.csv") for w in ("wave1_3", "wave4", "llm_imputed"))
    ids = w4.index.intersection(sim.index)
    w13, w4, sim = w13.loc[ids], w4.loc[ids], sim.loc[ids]

    def shares(s: pd.Series, lo: int, hi: int) -> np.ndarray:
        s = s.dropna().round()
        return np.array([(s == k).mean() for k in range(lo, hi + 1)])

    items = []
    for col, (lo, hi) in question_ranges().items():
        if hi - lo + 1 > 10 or col not in sim or sim[col].isna().all():
            continue
        truth = shares(w4[col], lo, hi)
        if truth.sum() < 0.98:  # answers fall outside the declared levels: not a k-option item
            continue
        sd = w4[col].std()
        items.append(
            {
                "k": hi - lo + 1,
                "truth": truth,
                "retest": shares(w13[col], lo, hi),
                "sim": shares(sim[col], lo, hi),
                "sd_ratio": sim[col].std() / sd if sd > 0 else np.nan,
            }
        )

    def rho(a: np.ndarray, b: np.ndarray) -> float:
        return float(spearmanr(a, b).statistic) if np.std(a) > 0 and np.std(b) > 0 else np.nan

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(100 * np.sqrt(np.mean((a - b) ** 2)))

    rng = np.random.default_rng(0)
    out = {}
    for label, group in (("binary", [i for i in items if i["k"] == 2]), ("multi", [i for i in items if i["k"] >= 3])):
        row = {"questions": len(group)}
        for name in ("retest", "sim"):
            row[name] = {
                "median_spearman": float(np.nanmedian([rho(i[name], i["truth"]) for i in group])),
                "rmse_pp": float(np.mean([rmse(i[name], i["truth"]) for i in group])),
            }
        draws = []
        for _ in range(null_draws):
            pairs = []
            for i in group:
                others = [o for o in group if o["k"] == i["k"] and o is not i]
                if others:
                    pairs.append((others[rng.integers(len(others))]["truth"], i["truth"]))
            draws.append((np.nanmedian([rho(a, b) for a, b in pairs]), np.mean([rmse(a, b) for a, b in pairs])))
        row["null"] = {"median_spearman": float(np.mean([d[0] for d in draws])), "rmse_pp": float(np.mean([d[1] for d in draws]))}
        row["uniform"] = {"rmse_pp": float(np.mean([rmse(np.full(i["k"], 1 / i["k"]), i["truth"]) for i in group]))}
        row["sim_sd_ratio"] = float(np.nanmedian([i["sd_ratio"] for i in group]))
        out[label] = row
    return out


def _load(csvs: Path, wave: str) -> pd.DataFrame:
    return pd.read_csv(csvs / f"responses_{wave}_formatted.csv", skiprows=[1]).rename(columns=str.upper).set_index("TWIN_ID")


def answered(csvs: Path) -> set[str]:
    sim = _load(csvs, "llm_imputed")
    return {c for c in sim.columns if sim[c].notna().any()}


def signal_columns(out: Path) -> list[str]:
    """The questions person signal is taken over: answered by every arm of a scored run, varying among the humans."""

    w13 = _load(out / "J" / "csv", "wave1_3")
    everywhere = set.intersection(*(answered(csvs) for csvs in out.glob("*/csv")))
    # Pricing columns k_Q295 are positions, each a different product and price per person: knowing the offer
    # alone predicts the answer there, so person signal uses only questions that are the same for everyone.
    return [c for c in question_ranges() if c in everywhere and c in w13 and w13[c].std() > 0 and not c.endswith("_Q295")]


def per_question(csvs: Path, columns: list[str], wave: str = "llm_imputed", pids=None) -> pd.Series:
    """For each question, the correlation across people between the arm's answer and the person's
    own wave 1-3 answer. A constant answer counts as 0."""

    truth, sim = _load(csvs, "wave1_3"), _load(csvs, wave)
    ids = truth.index.intersection(sim.index).intersection(_load(csvs, "llm_imputed").index)  # the arm's people
    if pids is not None:
        ids = ids.intersection(pids)
    rs = {}
    for c in columns:
        a, b = sim.loc[ids, c].astype(float), truth.loc[ids, c].astype(float)
        ok = a.notna() & b.notna()
        rs[c] = float(np.corrcoef(a[ok], b[ok])[0, 1]) if a[ok].std() > 0 and b[ok].std() > 0 else 0.0
    return pd.Series(rs)


def person_signal(csvs: Path, columns: list[str], wave: str = "llm_imputed") -> float:
    """Mean per-question association across people; includes demographic group patterns."""

    return float(per_question(csvs, columns, wave).mean())


def main(path: Path) -> None:
    results = pd.read_parquet(path)
    pids = list(dict.fromkeys(results.pid))
    everyone = twin.participants()
    people = everyone.head(len(pids)).set_index("pid")
    if list(people.index) != pids:
        raise RuntimeError("results are not the first n participants in dataset order")
    out = path.with_suffix("")
    form = results.persona.iloc[0]

    baselines = {f"GPT-4.1-mini, {form}": MATCHED[form], **FULL}  # for `full` the two keys coincide
    arms = {name: spec.replace(" ", "_") for name, spec in baselines.items()}
    tables, aggregates = {}, {}
    for name, spec in baselines.items():
        tables[name] = published(spec, pids, out / arms[name])
        aggregates[name] = aggregate(out / arms[name] / "csv")
    for mode in ("J", "P"):
        blocks = {}
        for pid, group in results.groupby("pid", sort=False):
            picks = {r.key: pick(r.probs, mode, f"{pid}:{r.key}") for r in group.itertuples()}
            blocks[pid] = twin.fill(json.loads(people.at[pid, "wave4_Q_wave4_A"]), picks)
        name = f"JEV-9B {mode}"
        arms[name] = mode
        tables[name] = evaluate(blocks, out / mode)
        aggregates[name] = aggregate(out / mode / "csv")
    for mode, name in POPULATION.items():
        arms[name] = mode
        tables[name] = evaluate(population(people.reset_index(), everyone, mode), out / mode)
        aggregates[name] = aggregate(out / mode / "csv")
    columns = signal_columns(out)
    signal ={RETEST: person_signal(out / "J" / "csv", columns, "wave4")}
    signal |= {name: person_signal(out / sub / "csv", columns) for name, sub in arms.items()}

    first = next(iter(tables.values()))
    tasks = pd.DataFrame(
        {
            RETEST: first["wave4 vs. wave1_3 Accuracy"],
            RANDOM: first["random_baseline vs wave1_3 Accuracy"],
            **{name: t["llm vs. wave1_3 Accuracy"] for name, t in tables.items()},
        }
    )
    kept = tasks.drop(index=SKIPPED_TASK, errors="ignore")
    scores = {
        "results": str(path),
        "participants": len(pids),
        "persona": form,
        "arms": arms,
        "tasks": tasks.round(4).to_dict(orient="index"),
        "mean_accuracy": kept.mean().round(4).to_dict(),
        "normalized_accuracy": (kept.mean() / kept[RETEST].mean()).round(4).to_dict(),
        "person_signal": {name: round(r, 4) for name, r in signal.items()},
        "aggregate": aggregates,
    }
    (out / "scores.json").write_text(json.dumps(scores, indent=2))

    print(f"\n{len(pids)} participants, persona={scores['persona']}; mean over {len(kept)} tasks ({SKIPPED_TASK} skipped)")
    print(
        pd.DataFrame(
            {"accuracy": scores["mean_accuracy"], "vs retest": scores["normalized_accuracy"], f"person signal ({len(columns)} questions)": scores["person_signal"]}
        ).to_string()
    )
    for label in ("multi", "binary"):
        print(f"\n{label}-option items vs. wave-4 shares (median Spearman / mean RMSE pp / twin SD ratio)")
        any_agg = next(iter(aggregates.values()))[label]
        print(f"  {RETEST:28s} {any_agg['retest']['median_spearman']:.2f} / {any_agg['retest']['rmse_pp']:.1f}")
        print(f"  {'null: wrong question':28s} {any_agg['null']['median_spearman']:.2f} / {any_agg['null']['rmse_pp']:.1f}")
        print(f"  {'uniform: 1/k each':28s}   -  / {any_agg['uniform']['rmse_pp']:.1f}")
        for name, agg in aggregates.items():
            a = agg[label]
            print(f"  {name:28s} {a['sim']['median_spearman']:.2f} / {a['sim']['rmse_pp']:.1f} / {a['sim_sd_ratio']:.2f}")
    print(f"\nwrote {out / 'scores.json'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
