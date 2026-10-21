"""Browse scored runs: headline numbers, task accuracy, answer shares per question, and each
person's decisions next to their real answers.

    uv run streamlit run popsim/app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from popsim import score, twin

st.set_page_config(page_title="popsim", layout="wide")
folder = Path(st.sidebar.text_input("results folder", str(twin.ROOT / "results")))
runs = sorted(p for p in folder.glob("*.parquet") if (p.with_suffix("") / "scores.json").exists())
if not runs:
    st.info(f"No scored runs in {folder}. Try the offline example with `python -m popsim.demo`, or see README for the full workflow.")
    st.stop()
path = st.sidebar.selectbox("run", runs, format_func=lambda p: p.stem)
out = path.with_suffix("")
scores = json.loads((out / "scores.json").read_text())


@st.cache_data
def decisions(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def formatted(out: Path, arm: str, wave: str) -> pd.DataFrame:
    return pd.read_csv(out / arm / "csv" / f"responses_{wave}_formatted.csv", skiprows=[1]).set_index("TWIN_ID")


@st.cache_data
def blocks(n: int) -> dict[int, list]:
    return {r.pid: json.loads(r.wave4_Q_wave4_A) for r in twin.participants(n).itertuples()}


st.title(f"{scores['participants']} participants · persona: {scores['persona']}")

st.subheader("Individual accuracy (Twin-2K-500 scorer)")
st.caption(
    f"Mean over 16 tasks ({score.SKIPPED_TASK} skipped). 'vs retest' divides by the humans' own two-week retest, the usual way twin accuracy is reported. "
    "'person signal' is the mean per-question correlation with each person's own earlier answer, over the questions that are the same for everyone "
    "(pricing is left out: each person saw different products at different prices). Demographic group patterns can contribute to it. "
    "'majority answer' and 'population draw' are target-population oracle diagnostics with access to other test participants' answers. "
    "Use python -m popsim.baselines for a separate-sample comparison."
)
st.dataframe(pd.DataFrame({"accuracy": scores["mean_accuracy"], "vs retest": scores["normalized_accuracy"], "person signal": scores["person_signal"]}))
st.bar_chart(pd.DataFrame(scores["tasks"]).T, horizontal=True, stack=False, height=600)

st.subheader("Population shares")
rows = []
for kind in ("multi", "binary"):
    any_arm = next(iter(scores["aggregate"].values()))[kind]
    rows.append({"items": kind, "arm": score.RETEST, **any_arm["retest"]})
    rows.append({"items": kind, "arm": "null: wrong question", **any_arm["null"]})
    rows.append({"items": kind, "arm": "uniform: 1/k each", **any_arm["uniform"]})
    for arm, agg in scores["aggregate"].items():
        rows.append({"items": kind, "arm": arm, **agg[kind]["sim"], "twin SD / human SD": agg[kind]["sim_sd_ratio"]})
st.dataframe(pd.DataFrame(rows), hide_index=True)

humans = formatted(out, "J", "wave4")
column = st.selectbox("question column", [c for c in humans.columns if humans[c].nunique() <= 11 and humans[c].notna().any()])
shares = {"humans, wave 4": humans[column].value_counts(normalize=True)}
for arm, sub in scores["arms"].items():
    sim = formatted(out, sub, "llm_imputed")
    shares[arm] = sim.loc[sim.index.intersection(humans.index), column].value_counts(normalize=True)
st.bar_chart(pd.DataFrame(shares).fillna(0).sort_index(), stack=False)

st.subheader("One person")
df = decisions(path)
pid = st.selectbox("participant", list(dict.fromkeys(df.pid)))
human = twin.human_picks(blocks(scores["participants"])[pid])
person = df[df.pid == pid].copy()
person["question"] = person.instructions.str.removeprefix(twin.FRAME).str.strip().str.slice(0, 160)
person["human answer"] = [r.options[human[r.key]] for r in person.itertuples()]
person["J"] = [r.options[score.pick(r.probs, "J", "")] for r in person.itertuples()]
person["P"] = [r.options[score.pick(r.probs, "P", f"{pid}:{r.key}")] for r in person.itertuples()]
st.dataframe(
    person[["key", "question", "probs", "human answer", "J", "P"]],
    column_config={"probs": st.column_config.BarChartColumn("distribution", y_min=0, y_max=1)},
    hide_index=True,
    width="stretch",
)

uses_path = folder / "uses.json"
if uses_path.exists():
    uses = json.loads(uses_path.read_text())
    st.header("Industry questions (python -m popsim.uses)")
    st.caption(
        "Pricing: would they buy this product at this price. A/B: do wording changes move answers the way they moved humans' (Cohen's d). "
        "Polling: segment gaps on policy and technology items, in item SDs; gap ratio 1 is the right size."
    )
    population = st.radio("participants", [name for name in ("all", "pilot") if uses[name]], format_func={"all": "all 2,058 (published twins, JEV-9B summary)", "pilot": "JEV-9B pilot 200"}.get, horizontal=True)
    for use in ("pricing", "ab", "polling"):
        table = {arm: {k: v for k, v in r[use].items() if k != "effects"} for arm, r in uses[population].items() if use in r}
        st.subheader(use)
        st.dataframe(pd.DataFrame(table).T)
    st.subheader("A/B effects by contrast")
    st.dataframe(pd.DataFrame({arm: r["ab"]["effects"] for arm, r in uses[population].items() if "effects" in r.get("ab", {})}))

calibrated_path = folder / "calibrate.json"
if calibrated_path.exists():
    calibrated = json.loads(calibrated_path.read_text())
    st.header("Survey a few, simulate the rest (python -m popsim.calibrate)")
    st.caption(
        "A survey covers k units (products, policy items, A/B tests); twins cover all of them and are calibrated on the surveyed ones. "
        "Error is on the units nobody surveyed: buy rate in pp, segment gaps in SDs, effects in d. "
        "Fits use fixed questionnaire ranges; human SDs enter evaluation only. Rich profiles for the full panel are assumed to exist. "
        "'beats' is the share of 500 random surveys where adding twins lowered the error."
    )
    use = st.radio("use case", list(calibrated), horizontal=True)
    for metric in ("error", "beats"):
        st.dataframe(pd.DataFrame({setting: {name: r[metric] for name, r in rows.items()} for setting, rows in calibrated[use].items()}))
