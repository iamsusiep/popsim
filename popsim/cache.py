"""Every prompt the decider has answered, with its answer, so nothing is ever asked twice.

    uv run python -m popsim.cache stats
    uv run python -m popsim.cache import results/summary-n200.parquet ...

A decision is stored under the hash of (model, revision, exact prompt text), so a changed
template or model is a new key. The prompt is kept as its two parts, the state (persona) and the
question, and `prompt()` rebuilds the exact text with the decider's template; storing whole
prompts would repeat each persona once per question. Raw log-probabilities are kept alongside the
calibrated distribution, so recalibrating never needs the GPU.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd

from popsim import twin
from popsim.decider import MODEL, MODEL_REVISION, options_of, render

PATH = twin.ROOT / "cache" / "decisions.sqlite"
SCHEMA = """
create table if not exists states (hash text primary key, text text not null);
create table if not exists questions (hash text primary key, json text not null);
create table if not exists decisions (
    key text primary key,
    model text not null,
    revision text not null,
    state text not null references states (hash),
    question text not null references questions (hash),
    probs text not null,     -- calibrated distribution, in the question's option order
    logprobs text,           -- the model's restricted log-probabilities; null when imported from a run that kept only probs
    input_tokens integer,
    cached_tokens integer,
    source text not null,
    created text not null
);
"""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Store:
    def __init__(self, path: Path = PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    @staticmethod
    def key(state: str, question: dict) -> str:
        return _hash("\0".join((MODEL, MODEL_REVISION, render(state, question))))

    def get(self, keys: list[str]) -> dict[str, list[float]]:
        """Stored distributions for whichever of `keys` have been answered."""

        out = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            marks = ",".join("?" * len(chunk))
            out |= {k: json.loads(p) for k, p in self.db.execute(f"select key, probs from decisions where key in ({marks})", chunk)}
        return out

    def put(self, decisions: list[tuple[str, dict, dict]], source: str) -> None:
        """Save (state, question, answer) triples; an answer already stored is kept as it was."""

        now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        with self.db:
            for state, question, answer in decisions:
                ids, _ = options_of(question)
                probs = answer["probabilities"] if "probabilities" in answer else {"false": 1 - answer["noul"], "true": answer["noul"]}
                text = json.dumps(question, ensure_ascii=False)
                usage = answer.get("usage", {})
                self.db.execute("insert or ignore into states values (?, ?)", (_hash(state), state))
                self.db.execute("insert or ignore into questions values (?, ?)", (_hash(text), text))
                self.db.execute(
                    "insert or ignore into decisions values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        self.key(state, question),
                        MODEL,
                        MODEL_REVISION,
                        _hash(state),
                        _hash(text),
                        json.dumps([probs[k] for k in ids]),
                        json.dumps([answer["logprobs"][k] for k in ids]) if "logprobs" in answer else None,
                        usage.get("input_tokens"),
                        usage.get("cached_tokens"),
                        source,
                        now,
                    ),
                )

    def prompt(self, key: str) -> str:
        """The exact text the model was given for a stored decision."""

        state, question = self.db.execute(
            "select s.text, q.json from decisions d join states s on s.hash = d.state "
            "join questions q on q.hash = d.question where d.key = ?",
            (key,),
        ).fetchone()
        return render(state, json.loads(question))

    def stats(self) -> pd.DataFrame:
        return pd.read_sql(
            "select source, count(*) as decisions, count(logprobs) as with_logprobs, sum(input_tokens) as input_tokens, "
            "min(created) as first, max(created) as last from decisions group by source order by first",
            self.db,
        )


def import_results(path: Path, store: Store) -> int:
    """Load a results parquet written before the cache existed (probabilities only)."""

    results = pd.read_parquet(path)
    pids = list(dict.fromkeys(results.pid))
    people = twin.participants(len(pids)).set_index("pid")
    if list(people.index) != pids:
        raise RuntimeError(f"{path} is not the first n participants in dataset order")
    form = results.persona.iloc[0]
    states = {pid: twin.persona(people.loc[pid], form) for pid in pids}
    decisions = []
    for r in results.itertuples():
        criteria = {str(i + 1): option for i, option in enumerate(r.options)}
        question = {"type": "choice", "instructions": r.instructions, "criteria": criteria}  # as twin.questions builds it
        decisions.append((states[r.pid], question, {"probabilities": dict(zip(criteria, map(float, r.probs)))}))
    store.put(decisions, source=f"import:{path.name}")
    return len(decisions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stats")
    load = sub.add_parser("import")
    load.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    store = Store()
    if args.command == "import":
        for path in args.paths:
            print(f"{path}: {import_results(path, store)} decisions")
    print(store.stats().to_string(index=False))
