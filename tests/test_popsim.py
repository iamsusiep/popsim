import json
import math

import numpy as np
import pandas as pd
import pytest

from popsim import cache, calibrate, score, twin, uses
from popsim.decider import distribution, options_of, render


def test_render_matches_model_card_template():
    question = {
        "type": "choice",
        "instructions": "Supplier response for this scenario.",
        "criteria": {"issue_warning": None, "renegotiate": None, "dual_source": None, "maintain": None},
    }
    assert render("SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.", question) == (
        "[kind] choice\n"
        "[state] SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.\n"
        "[question] Supplier response for this scenario.\n"
        "[options]\n"
        "A) issue_warning\nB) renegotiate\nC) dual_source\nD) maintain\n"
        "[decision]:"
    )
    assert render("s", {"type": "noul", "instructions": "q?"}).endswith("[options]\nfalse\ntrue\n[decision]:")


def test_distribution_applies_bias_and_temperature():
    assert distribution([math.log(0.2), math.log(0.8)], [0.0, 0.0], 1.0) == pytest.approx([0.2, 0.8])
    # a bias of log 4 on the first option makes 0.2 * 4 = 0.8 against 0.8: even
    assert distribution([math.log(0.2), math.log(0.8)], [math.log(4), 0.0], 1.0) == pytest.approx([0.5, 0.5])
    assert distribution([0.0, math.log(3)], [0.0, 0.0], 2.0) == pytest.approx([1 / (1 + math.sqrt(3)), math.sqrt(3) / (1 + math.sqrt(3))])


def test_choice_rejects_more_options_than_verbalizers():
    with pytest.raises(ValueError):
        options_of({"type": "choice", "instructions": "", "criteria": {str(i): None for i in range(17)}})


@pytest.fixture(scope="module")
def people():
    return twin.participants(100)


@pytest.mark.integration
def test_every_answerable_item_is_asked_and_round_trips(people):
    for row in people.itertuples():
        blocks = json.loads(row.wave4_Q_wave4_A)
        asked, human = twin.questions(blocks), twin.human_picks(blocks)
        assert asked.keys() == human.keys()
        assert all(2 <= len(q["criteria"]) <= 16 for q in asked.values())
        refilled = twin.fill(blocks, human)
        for block, again in zip(blocks, refilled, strict=True):
            for q, q2 in zip(block.get("Questions", []), again.get("Questions", []), strict=True):
                if q["QuestionType"] in ("MC", "Matrix"):
                    assert q2["Answers"]["SelectedByPosition"] == q["Answers"]["SelectedByPosition"]
                elif q["QuestionType"] == "Slider":
                    for v, v2 in zip(q["Answers"]["Values"], q2["Answers"]["Values"], strict=True):
                        assert abs(float(v) - float(v2)) <= 5
                elif q["QuestionID"] in twin.BOUNDED_ENTRIES:
                    assert abs(float(q["Answers"]["Text"]) - float(q2["Answers"]["Text"])) <= 1


@pytest.mark.integration
def test_persona_forms_are_nested(people):
    for row in people.itertuples():
        demo, summary, full = (twin.persona(row, f) for f in twin.PERSONAS)
        assert summary.startswith(demo) and len(demo) < len(summary) < len(full)


@pytest.mark.integration
def test_structured_full_personas_exclude_target_question_ids():
    """Check the source split across the entire panel; this does not audit summary generation."""

    def question_ids(value):
        if isinstance(value, dict):
            own = {value["QuestionID"]} if "QuestionID" in value else set()
            return own.union(*(question_ids(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(question_ids(v) for v in value))
        return set()

    for i in range(1, twin.CHUNKS + 1):
        rows = pd.read_parquet(twin.dataset_file(f"wave_split/chunks/wave_persona_chunk_{i:03d}.parquet"),
                               columns=["pid", "wave1_3_persona_json", "wave4_Q_wave4_A"])
        for row in rows.itertuples():
            assert question_ids(json.loads(row.wave1_3_persona_json)).isdisjoint(
                question_ids(json.loads(row.wave4_Q_wave4_A))), row.pid


@pytest.mark.integration
def test_oracle_twin_reproduces_human_retest(people, tmp_path):
    """Answering with each person's own wave-4 answers, through our question mapping and the Twin
    converter and scorer, must give the human retest accuracy: exactly where nothing is binned,
    closely where numbers snap to 11 levels."""

    blocks = {}
    for row in people.itertuples():
        b = json.loads(row.wave4_Q_wave4_A)
        blocks[row.pid] = twin.fill(b, twin.human_picks(b))
    table = score.evaluate(blocks, tmp_path)
    assert score.SKIPPED_TASK not in table.index and len(table) == 16
    for task, r in table.iterrows():
        tolerance = 0.03 if task in ("base rate", "false consensus", "sunk cost fallacy") else 0.0
        assert r["llm vs. wave1_3 Accuracy"] == pytest.approx(r["wave4 vs. wave1_3 Accuracy"], abs=tolerance), task


@pytest.mark.integration
def test_pricing_offers_line_up_with_formatted_answers(people):
    """Column k_Q295 of the formatted CSVs is the k-th pricing question each person saw (QID9_k),
    whose product and price vary by person: the parsed offers must carry the same answers."""

    offers = uses.offers()
    assert len(offers) == 2058 * 40 and offers.groupby(["pid", "product"]).size().max() == 1
    humans = uses.formatted(twin.dataset_file(f"{score.HUMANS}/responses_wave4_formatted.csv"))
    seen = offers[offers.pid.isin(people.pid)]
    for row in people.itertuples():
        for block in json.loads(row.wave4_Q_wave4_A):
            for q in block.get("Questions", []):
                if q["QuestionID"].startswith("QID9_"):
                    assert humans.at[row.pid, f"{q['QuestionID'][5:]}_Q295"] == q["Answers"]["SelectedByPosition"]
    buys = uses._buys(humans.loc[people.pid], seen)
    assert len(buys) == len(seen) and set(buys.code) == {1, 2}


def test_calibration_fits_surveyed_units_and_predicts_the_rest():
    cells = pd.MultiIndex.from_product([["a", "b", "c", "d"], [1, 2]])  # (unit, group)
    sim = pd.Series(np.arange(8.0), index=cells)
    truth = 10 + 2 * sim + 5 * (cells.get_level_values(1) == 2)  # twins right up to scale and a group shift
    guess = calibrate.predict(truth, sim, np.array(["a", "b"]), intercepts=True)
    assert list(guess.index) == list(cells[4:]) and guess.to_numpy() == pytest.approx(truth[4:].to_numpy())
    # without twins: each group's average over the surveyed units (a, b)
    assert calibrate.predict(truth, None, np.array(["a", "b"]), intercepts=True).to_numpy() == pytest.approx([12, 19, 12, 19])
    assert (calibrate.predict(truth, None, np.array(["a"]), intercepts=False) == 0).all()


def test_cache_keeps_prompts_and_answers_once(tmp_path):
    store = cache.Store(tmp_path / "decisions.sqlite")
    question = {"type": "choice", "instructions": "Which?", "criteria": {"1": "yes", "2": "no"}}
    answer = {"probabilities": {"1": 0.7, "2": 0.3}, "logprobs": {"1": -0.2, "2": -1.1}, "usage": {"input_tokens": 9, "cached_tokens": 0}}
    key = store.key("persona A", question)
    assert store.key("persona B", question) != key and store.get([key]) == {}
    store.put([("persona A", question, answer)], source="test")
    store.put([("persona A", question, {"probabilities": {"1": 0.0, "2": 1.0}})], source="again")  # never overwritten
    assert store.get([key]) == {key: [0.7, 0.3]}
    assert store.prompt(key) == render("persona A", question)
    assert store.stats().decisions.sum() == 1
