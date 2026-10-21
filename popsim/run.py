"""Ask the decider every wave-4 question about each participant, and save the distributions.

    uv run python -m popsim.run --n 200 --persona summary

One request per participant: the persona is the state, each answerable item a `choice` question.
Questions already in the decision cache (popsim/cache.py) are not asked again, and each batch is
saved to the cache as it returns. Writes results/<persona>-n<n>.parquet with one row per
(participant, question).
"""

from __future__ import annotations

import argparse
import json
import time

import modal
import pandas as pd

from popsim import cache, twin
from popsim.decider import Decider, app


def requests_for(people: pd.DataFrame, form: str) -> list[dict]:
    return [
        {"state": twin.persona(row, form), "questions": twin.questions(json.loads(row.wave4_Q_wave4_A))}
        for row in people.itertuples()
    ]


def context_length(requests: list[dict]) -> int:
    """Room for the longest prompt, generously: ~3 characters a token, plus the question."""

    longest = max(len(r["state"]) + max(len(q["instructions"]) for q in r["questions"].values()) for r in requests)
    return -(-(longest // 3 + 1024) // 1024) * 1024


def ask(store: cache.Store, requests: list[dict], batch: int, source: str) -> None:
    """Send the requests to the decider in batches, saving each batch's answers as it returns."""

    batches = [requests[i : i + batch] for i in range(0, len(requests), batch)]
    max_model_len = context_length(requests)
    print(f"asking {sum(len(r['questions']) for r in requests)} questions in {len(batches)} calls, max_model_len={max_model_len}")
    started, tokens, cached = time.monotonic(), 0, 0
    with modal.enable_output(), app.run():
        decider = Decider(max_model_len=max_model_len)
        for done, (sent, responses) in enumerate(zip(batches, decider.decide.map(batches), strict=True), 1):
            store.put(
                [(r["state"], q, response["answers"][qid]) for r, response in zip(sent, responses, strict=True) for qid, q in r["questions"].items()],
                source,
            )
            tokens += sum(r["usage"]["input_tokens"] for r in responses)
            cached += sum(r["usage"]["cached_tokens"] for r in responses)
            print(f"  call {done}/{len(batches)} saved, {(time.monotonic() - started) / 60:.1f} min", flush=True)
    print(f"prompt tokens {tokens:,}, served from prefix cache {cached:,} ({cached / max(tokens, 1):.0%})")


def main(n: int, form: str, batch: int) -> None:
    people = twin.participants(n)
    requests = requests_for(people, form)
    store = cache.Store()
    keys = [{qid: store.key(r["state"], q) for qid, q in r["questions"].items()} for r in requests]
    known = store.get([k for ks in keys for k in ks.values()])
    missing = [
        {"state": r["state"], "questions": {qid: q for qid, q in r["questions"].items() if ks[qid] not in known}}
        for r, ks in zip(requests, keys, strict=True)
    ]
    missing = [r for r in missing if r["questions"]]
    total = sum(map(len, keys))
    print(f"{len(requests)} participants, {total} questions: {len(known)} in the cache, {total - len(known)} to ask")
    if missing:
        ask(store, missing, batch, source=f"{form}-n{n}")
        known = store.get([k for ks in keys for k in ks.values()])

    rows = []
    for row, request, ks in zip(people.itertuples(), requests, keys, strict=True):
        for key, question in request["questions"].items():
            rows.append(
                {
                    "pid": row.pid,
                    "persona": form,
                    "key": key,
                    "instructions": question["instructions"],
                    "options": list(question["criteria"].values()),
                    "probs": known[ks[key]],
                }
            )
    out = twin.ROOT / "results" / f"{form}-n{n}.parquet"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    print(f"wrote {out}: {len(rows)} decisions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, required=True, help="first n participants in dataset order")
    parser.add_argument("--persona", choices=twin.PERSONAS, required=True)
    parser.add_argument("--batch", type=int, default=25, help="participants per Modal call")
    args = parser.parse_args()
    main(args.n, args.persona, args.batch)
