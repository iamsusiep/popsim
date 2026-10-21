"""Twin-2K-500 as typed decisions.

Each participant's wave-4 questions (their own randomised versions) become `choice` questions
about that participant. Picked options are written back into the dataset's answer-block JSON, so
the Twin authors' own converter and scorer evaluate the result unchanged.
"""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DATASET = "LLM-Digital-Twin/Twin-2K-500"
DATASET_REVISION = "f883165a3026fde855dfd448e0cd16443ab257b6"
CODE = "https://github.com/tianyipeng-lab/Digital-Twin-Simulation"
CODE_REVISION = "f1eed510c9a4fb47aaa9bb46e178aaa2d9224623"
CHUNKS = 7
ROOT = Path(__file__).resolve().parent.parent

PERSONAS = ("demographics", "summary", "full")
FRAME = "This person answered the survey question below. Which answer did they give?"
SLIDER_STEPS = 10  # a 0-100 slider becomes 11 options: 0, 10, ..., 100
# Open number entries whose range the question itself fixes ("how many of your next 20 coffee
# purchases"): asked like sliders. The anchoring estimates have no such range and are skipped.
BOUNDED_ENTRIES = {"QID181": (0, 20), "QID182": (0, 20)}


def dataset_file(path: str) -> Path:
    return Path(hf_hub_download(DATASET, path, repo_type="dataset", revision=DATASET_REVISION))


def twin_code() -> Path:
    """The Twin authors' repository (Apache-2.0) at a pinned commit, for its converter and scorer."""

    path = ROOT / ".cache" / "Digital-Twin-Simulation"
    if not path.exists():
        subprocess.run(["git", "clone", "--quiet", CODE, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--quiet", CODE_REVISION], check=True)
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != CODE_REVISION:
        raise RuntimeError(f"scorer at {path} is {revision}; expected {CODE_REVISION}. Use a separate clean checkout.")
    return path


def participants(n: int | None = None) -> pd.DataFrame:
    """The first n participants: pid, persona texts, and wave-4 answer blocks as a JSON string.

    The two parquet configs chunk participants differently, so they are joined on pid."""

    split = []
    for i in range(1, CHUNKS + 1):
        split.append(
            pd.read_parquet(
                dataset_file(f"wave_split/chunks/wave_persona_chunk_{i:03d}.parquet"),
                columns=["pid", "wave1_3_persona_text", "wave4_Q_wave4_A"],
            )
        )
        if n is not None and sum(map(len, split)) >= n:
            break
    people = pd.concat(split, ignore_index=True).head(n)

    summaries = []
    for i in range(1, CHUNKS + 1):
        chunk = pd.read_parquet(
            dataset_file(f"full_persona/chunks/persona_chunk_{i:03d}.parquet"), columns=["pid", "persona_summary"]
        )
        summaries.append(chunk.astype({"pid": people.pid.dtype}))  # this config stores pid as a string
        if people.pid.isin(pd.concat(summaries).pid).all():
            break
    return people.merge(pd.concat(summaries), on="pid", how="left", validate="one_to_one")


def persona(row, form: str) -> str:
    """`full` is everything from waves 1-3 (~24k tokens), `summary` the dataset's own summary of it
    (~3k), `demographics` that summary's opening demographics section (~200)."""

    if form == "full":
        return row.wave1_3_persona_text
    if form == "summary":
        return row.persona_summary
    if form == "demographics":
        return row.persona_summary.split("\n\nThe person's Big 5")[0]
    raise ValueError(f"unknown persona form {form!r}; expected one of {PERSONAS}")


def levels(question: dict) -> list[int]:
    """The answer levels of a slider or a bounded number entry."""

    if question["QuestionType"] == "Slider":
        lo, hi = question["Range"]["Min"], question["Range"]["Max"]
    else:
        lo, hi = BOUNDED_ENTRIES[question["QuestionID"]]
    return [round(lo + (hi - lo) * k / SLIDER_STEPS) for k in range(SLIDER_STEPS + 1)]


def _nearest(values: list[int], value: str) -> int:
    return min(range(len(values)), key=lambda k: abs(values[k] - float(value)))


def _choice(text: str, options: list[str]) -> dict:
    return {
        "type": "choice",
        "instructions": f"{FRAME}\n\n{text}",
        "criteria": {str(i + 1): option for i, option in enumerate(options)},
    }


def questions(blocks: list[dict]) -> dict[str, dict]:
    """One typed question per answerable item: an MC question, a matrix row, a slider statement.

    A block's descriptive text is prepended to each of its questions. Open number entries are
    asked only when bounded (BOUNDED_ENTRIES)."""

    out: dict[str, dict] = {}
    for block in blocks:
        items = block.get("Questions", [])
        intro = "\n\n".join(q["QuestionText"] for q in items if q["QuestionType"] == "DB")
        for q in items:
            qid, kind = q["QuestionID"], q["QuestionType"]
            text = f"{intro}\n\n{q['QuestionText']}" if intro else q["QuestionText"]
            if kind == "MC":
                out[qid] = _choice(text, q["Options"])
            elif kind == "Matrix":
                for r, row in enumerate(q["Rows"]):
                    out[f"{qid}#{r}"] = _choice(f"{text}\n\n{row}", q["Columns"])
            elif kind == "Slider":
                options = [str(v) for v in levels(q)]
                for s, statement in enumerate(q["Statements"]):
                    out[f"{qid}#{s}"] = _choice(f"{text}\n\n{statement}" if statement else text, options)
            elif kind == "TE" and qid in BOUNDED_ENTRIES:
                out[qid] = _choice(text, [str(v) for v in levels(q)])
    return out


def fill(blocks: list[dict], picks: dict[str, int]) -> list[dict]:
    """A copy of the blocks with every answer replaced by the picked option (0-based index into
    that question's options). Unbounded number entries are left unanswered."""

    blocks = copy.deepcopy(blocks)
    for block in blocks:
        for q in block.get("Questions", []):
            qid, kind = q["QuestionID"], q["QuestionType"]
            if kind == "MC":
                i = picks[qid]
                q["Answers"] = {"SelectedByPosition": i + 1, "SelectedText": q["Options"][i]}
            elif kind == "Matrix":
                rows = [picks[f"{qid}#{r}"] for r in range(len(q["Rows"]))]
                q["Answers"] = {
                    "SelectedByPosition": [i + 1 for i in rows],
                    "SelectedText": [q["Columns"][i] for i in rows],
                }
            elif kind == "Slider":
                values = levels(q)
                q["Answers"] = {"Values": [str(values[picks[f"{qid}#{s}"]]) for s in range(len(q["Statements"]))]}
            elif kind == "TE":
                q["Answers"] = {"Text": str(levels(q)[picks[qid]])} if qid in BOUNDED_ENTRIES else {}
    return blocks


def human_picks(blocks: list[dict]) -> dict[str, int]:
    """The participant's own wave-4 answers as picks; numbers snap to the nearest level."""

    picks: dict[str, int] = {}
    for block in blocks:
        for q in block.get("Questions", []):
            qid, kind, answers = q["QuestionID"], q["QuestionType"], q.get("Answers", {})
            if kind == "MC":
                picks[qid] = answers["SelectedByPosition"] - 1
            elif kind == "Matrix":
                for r, position in enumerate(answers["SelectedByPosition"]):
                    picks[f"{qid}#{r}"] = position - 1
            elif kind == "Slider":
                for s, value in enumerate(answers["Values"]):
                    picks[f"{qid}#{s}"] = _nearest(levels(q), value)
            elif kind == "TE" and qid in BOUNDED_ENTRIES:
                picks[qid] = _nearest(levels(q), answers["Text"])
    return picks
