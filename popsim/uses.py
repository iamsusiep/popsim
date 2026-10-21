"""Three questions a business asks of simulated respondents, scored against the real wave-4
answers of the same people:

    pricing  would they buy this product at this price? Each person saw each of 40 grocery
             products once, at a random price from about 0 to 2x its median.
    A/B      does changing the wording move the answers, and by how much? 16 contrasts from the
             between-subject bias experiments, as standardized effects (Cohen's d).
    polling  how far apart are segments (party, age, income, ...) on 10 policy and 8 technology
             items? Gaps in units of the item's standard deviation.

"humans, retest" is the same people's answers from waves 1-3 (same products, prices and
conditions): the ceiling for any twin.

    uv run python -m popsim.uses
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from popsim import score, twin

PUBLISHED = {
    **{f"GPT-4.1-mini, {form}": spec for form, spec in score.MATCHED.items()},
    "GPT-4.1-mini, full, temp 1": "Text Persona (Default Temperature) - GPT4.1-mini",
    "GPT-4.1-mini, full, reasoning": "Text Persona (Reasoning) - GPT4.1-mini",
    "GPT-4.1, JSON persona": "JSON Persona - GPT4.1",
    "Gemini 2.5 Flash, full": "Text Persona - Gemini-Flash2.5",
}
TRUTH = "humans, wave 4"
OFFER = re.compile(r"category: .*?\. Suppose.*?category: (?P<product>.*?)\. The product is priced at: \$(?P<price>[\d,.]+)\.", re.S)
PRICING = {f"{k}_Q295": k for k in range(1, 41)}  # formatted column -> position of QID9_k
CONTRASTS = {  # name: (control column, treatment column); each person saw one condition
    "disease: loss vs gain frame": ("Q157", "Q158"),
    "base rate: 70 vs 30 engineers": ("Form A _1", "Q156_1"),
    "outcome bias: died vs succeeded": ("Q161", "Q162"),
    "less is more: lose 5c vs nothing": ("Q171", "Q172"),
    "less is more: lose 25c vs nothing": ("Q171", "Q173"),
    "proportion dominance 1: B vs A": ("Q174", "Q175"),
    "proportion dominance 1: C vs A": ("Q174", "Q176"),
    "proportion dominance 2: B vs A": ("Q177", "Q178"),
    "proportion dominance 2: C vs A": ("Q177", "Q179"),
    "sunk cost: paid $50 card vs none": ("Q181", "Q182"),
    "drive to save $10: on $250 vs $30": ("Q183", "Q184"),
    "willingness to accept vs pay": ("Q189", "Q190"),
    "pay for partial vs full protection": ("Q189", "Q191"),
    "Allais: form 2 vs form 1": ("Q192", "Q193"),
    "myside: ban German car vs Ford": ("Q194", "Q195"),
    "Linda: feminist bank teller vs teller": ("Q159_3", "Q160_3"),
}
ITEMS = {  # column: (lowest, highest) answer code
    **{f"False Cons. self _{i}": (1, 5) for i in range(1, 11)},  # policy support
    **{f"nonseparabilty bene _{i}": (1, 7) for i in range(1, 5)},  # technology benefit
    **{f"nonseparability ris _{i}": (1, 7) for i in range(1, 5)},  # technology risk
}
SEGMENTS = {  # name: (wave 1-3 demographic question, codes of group A, codes of group B); gap is A - B
    "Democrat - Republican": ("QID20", {2}, {1}),
    "liberal - conservative": ("QID22", {4, 5}, {1, 2}),
    "age 18-29 - 65+": ("QID13", {1}, {4}),
    "income <$30k - $100k+": ("QID21", {1}, {5}),
    "bachelor's+ - less": ("QID14", {5, 6}, {1, 2, 3, 4}),
    "women - men": ("QID12", {2}, {1}),
    "Black - White": ("QID15", {2}, {1}),
}
CLEAR = 2.58  # |t| in all 2,058 humans for an effect or gap to count as clearly signed


def formatted(path) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=[1]).set_index("TWIN_ID")


def offers() -> pd.DataFrame:
    """Every pricing question as shown: pid, position, product, price, price relative to the
    product's median price, and that relative price's quintile."""

    rows = []
    for i in range(1, twin.CHUNKS + 1):
        path = twin.dataset_file(f"wave_split/chunks/wave_persona_chunk_{i:03d}.parquet")
        for r in pd.read_parquet(path, columns=["pid", "wave4_Q_wave4_A"]).itertuples():
            for block in json.loads(r.wave4_Q_wave4_A):
                for q in block.get("Questions", []):
                    if q["QuestionID"].startswith("QID9_"):
                        m = OFFER.search(q["QuestionText"])
                        price = float(m["price"].replace(",", ""))
                        rows.append({"pid": int(r.pid), "pos": int(q["QuestionID"][5:]), "product": m["product"], "price": price})
    out = pd.DataFrame(rows)
    out["rel"] = out.price / out.groupby("product").price.transform("median")
    out["quintile"] = pd.qcut(out.rel, 5, labels=False)
    return out


def demographics() -> pd.DataFrame:
    path = twin.dataset_file("question_catalog_and_human_response_csv/wave1_3_response.csv")
    columns = sorted({col for col, _, _ in SEGMENTS.values()})
    return pd.read_csv(path, usecols=["pid", *columns]).set_index("pid")


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if a[ok].std() > 0 and b[ok].std() > 0 else np.nan


def _slope(rel: pd.Series, buy: pd.Series) -> float:
    x = rel - rel.mean()
    return float((x * buy).sum() / (x**2).sum())


def _buys(df: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    answers = df[list(PRICING)].rename(columns=PRICING).rename_axis(index="pid", columns="pos").stack().rename("code")
    return offers.merge(answers.reset_index(), on=["pid", "pos"]).assign(buy=lambda d: (d.code == 1).astype(float))


def pricing(sim: pd.DataFrame, truth: pd.DataFrame, offers: pd.DataFrame) -> dict:
    d = _buys(sim, offers).merge(_buys(truth, offers)[["pid", "pos", "buy"]], on=["pid", "pos"], suffixes=("", "_truth"))
    curve = d.groupby("quintile")[["buy", "buy_truth"]].mean()
    within = d.assign(rel=d.rel - d.groupby("product").rel.transform("mean"))  # product fixed effects
    products = d.groupby("product")
    slopes = products.apply(lambda p: pd.Series({"sim": _slope(p.rel, p.buy), "truth": _slope(p.rel, p.buy_truth)}))
    return {
        "buy rate": d.buy.mean(),
        "demand curve error (pp)": 100 * np.sqrt(((curve.buy - curve.buy_truth) ** 2).mean()),
        "price slope": _slope(within.rel, within.buy),
        "products r(buy rate)": _corr(products.buy.mean(), products.buy_truth.mean()),
        "products r(price slope)": _corr(slopes.sim, slopes.truth),
    }


def _pooled_sd(df: pd.DataFrame, a: str, b: str) -> float:
    x, y = df[a].dropna(), df[b].dropna()
    return float(np.sqrt(((len(x) - 1) * x.var() + (len(y) - 1) * y.var()) / (len(x) + len(y) - 2)))


def effects(sim: pd.DataFrame, truth: pd.DataFrame) -> pd.Series:
    """Each contrast's mean difference over the humans' pooled SD, so that twins whose answers
    vary less are not credited with bigger effects."""

    return pd.Series({name: (sim[b].mean() - sim[a].mean()) / _pooled_sd(truth, a, b) for name, (a, b) in CONTRASTS.items()})


def clear_effects(humans: pd.DataFrame) -> list[str]:
    out = []
    for name, (a, b) in CONTRASTS.items():
        x, y = humans[a].dropna(), humans[b].dropna()
        if abs(y.mean() - x.mean()) / np.sqrt(x.var() / len(x) + y.var() / len(y)) > CLEAR:
            out.append(name)
    return out


def ab(sim: pd.DataFrame, truth: pd.DataFrame, clear: list[str]) -> dict:
    d, t = effects(sim, truth), effects(truth, truth)
    return {
        "r(effects)": _corr(d, t),
        "sign right (clear effects)": float((np.sign(d[clear]) == np.sign(t[clear])).mean()),
        "effect error (d)": float((d - t).abs().mean()),
        "effects": d.round(3).to_dict(),
    }


def _groups(demo: pd.DataFrame, ids: pd.Index):
    demo = demo.reindex(ids)
    for name, (col, a, b) in SEGMENTS.items():
        yield name, ids[demo[col].isin(a).to_numpy()], ids[demo[col].isin(b).to_numpy()]


def gaps(sim: pd.DataFrame, truth: pd.DataFrame, demo: pd.DataFrame) -> pd.Series:
    out = {}
    for name, a, b in _groups(demo, sim.index):
        for item in ITEMS:
            out[f"{name} | {item}"] = (sim.loc[a, item].mean() - sim.loc[b, item].mean()) / truth[item].std()
    return pd.Series(out)


def clear_gaps(humans: pd.DataFrame, demo: pd.DataFrame) -> list[str]:
    out = []
    for name, a, b in _groups(demo, humans.index):
        for item in ITEMS:
            x, y = humans.loc[a, item].dropna(), humans.loc[b, item].dropna()
            if abs(x.mean() - y.mean()) / np.sqrt(x.var() / len(x) + y.var() / len(y)) > CLEAR:
                out.append(f"{name} | {item}")
    return out


def polling(sim: pd.DataFrame, truth: pd.DataFrame, demo: pd.DataFrame, clear: list[str]) -> dict:
    """Topline: each item's mean as net support, -1 (all lowest) to +1 (all highest). Gaps: the
    segment differences, and how much the twins stretch or shrink them (ratio 1 = right size)."""

    net = lambda df: pd.Series({c: (df[c].mean() - (lo + hi) / 2) / ((hi - lo) / 2) for c, (lo, hi) in ITEMS.items()})
    g, t = gaps(sim, truth, demo), gaps(truth, truth, demo)
    return {
        "topline r": _corr(net(sim), net(truth)),
        "topline error": float((net(sim) - net(truth)).abs().mean()),
        "gaps r": _corr(g, t),
        "gap ratio": float((g * t).sum() / (t**2).sum()),
        "gap sign right (clear gaps)": float((np.sign(g[clear]) == np.sign(t[clear])).mean()),
        "gap error (SD)": float((g - t).abs().mean()),
    }


def evaluate(arms: dict[str, pd.DataFrame], humans: pd.DataFrame, offers, demo, clear_ab, clear_poll) -> dict:
    """Every use case for every arm, each against the humans among that arm's participants."""

    out = {}
    for name, sim in {TRUTH: humans, **arms}.items():
        truth = humans.loc[sim.index]
        out[name] = {
            "participants": len(sim),
            "pricing": pricing(sim, truth, offers),
            "ab": ab(sim, truth, clear_ab),
            "polling": polling(sim, truth, demo, clear_poll),
        }
    no_effect = effects(humans, humans)
    out["baseline: no effect / no gap"] = {
        "ab": {"effect error (d)": float(no_effect.abs().mean())},
        "polling": {"gap error (SD)": float(gaps(humans, humans, demo).abs().mean())},
    }
    return out


def jev_arms(n: int) -> dict[str, pd.DataFrame]:
    """Our scored JEV-9B runs over the first n participants, each as J and P."""

    arms = {}
    for path in sorted((twin.ROOT / "results").glob(f"*-n{n}/scores.json")):
        form = json.loads(path.read_text())["persona"]
        for mode in ("J", "P"):
            arms[f"JEV-9B {mode}, {form}"] = formatted(path.parent / mode / "csv" / "responses_llm_imputed_formatted.csv")
    return arms


def pilot_arms(humans_retest: pd.DataFrame, published: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The JEV-9B pilots, with the human retest and the matched GPT-4.1-mini twins on the same people."""

    arms = jev_arms(200)
    if not arms:
        return {}
    ids = next(iter(arms.values())).index
    matched = {f"GPT-4.1-mini, {form}": published[f"GPT-4.1-mini, {form}"].loc[ids] for form in score.MATCHED}
    return {score.RETEST: humans_retest.loc[ids], **matched, **arms}


def main() -> None:
    humans = formatted(twin.dataset_file(f"{score.HUMANS}/responses_wave4_formatted.csv"))
    retest = formatted(twin.dataset_file(f"{score.HUMANS}/responses_wave1_3_formatted.csv"))
    published = {
        name: formatted(twin.dataset_file(f"{score.SPECS}/{spec}/csv_comparison/csv_formatted/responses_llm_imputed_formatted.csv"))
        for name, spec in PUBLISHED.items()
    }
    shown, demo = offers(), demographics()
    clear_ab, clear_poll = clear_effects(humans), clear_gaps(humans, demo)
    pilots = pilot_arms(retest, published)
    results = {
        "all": evaluate({score.RETEST: retest, **published, **jev_arms(len(humans))}, humans, shown, demo, clear_ab, clear_poll),
        "pilot": evaluate(pilots, humans, shown, demo, clear_ab, clear_poll) if pilots else {},
        "clear effects": clear_ab,
        "clear gaps": len(clear_poll),
    }
    out = twin.ROOT / "results" / "uses.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))

    for population, table in (("all 2,058 participants", results["all"]), ("JEV-9B pilot participants", results["pilot"])):
        if not table:
            continue
        for use in ("pricing", "ab", "polling"):
            rows = {arm: {k: v for k, v in r[use].items() if k != "effects"} for arm, r in table.items() if use in r}
            print(f"\n{use}, {population}")
            print(pd.DataFrame(rows).T.round(3).to_string())
    effects_table = pd.DataFrame({arm: r["ab"]["effects"] for arm, r in results["all"].items() if "effects" in r.get("ab", {})})
    print(f"\nA/B effects (Cohen's d), all participants; clearly signed in humans: {len(clear_ab)} of {len(CONTRASTS)}")
    print(effects_table.round(2).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
