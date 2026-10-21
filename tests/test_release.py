import json

import numpy as np
import pandas as pd
import pytest

from popsim import baselines, calibrate, score, twin, uses


def blocks(answer):
    return [{"Questions": [{"QuestionID": "Q1", "QuestionType": "MC", "QuestionText": "Choose a drink.",
                            "Options": ["Tea", "Water"], "Answers": {"SelectedByPosition": answer}}]}]


def test_population_predictions_do_not_read_test_answers():
    train = pd.DataFrame({"pid": [1, 2, 3], "wave4_Q_wave4_A": [json.dumps(blocks(a)) for a in [1, 1, 2]]})
    test = pd.DataFrame({"pid": [4], "wave4_Q_wave4_A": [json.dumps(blocks(2))]})
    before = score.population(test, train, "majority")
    test.loc[0, "wave4_Q_wave4_A"] = json.dumps(blocks(1))
    assert score.population(test, train, "majority") == before
    assert twin.human_picks(before[4]) == {"Q1": 0}


def test_polling_cells_use_question_ranges_not_human_variance(monkeypatch):
    monkeypatch.setattr(uses, "ITEMS", {"item": (1, 5)})
    monkeypatch.setattr(uses, "SEGMENTS", {"A - B": ("group", {1}, {2})})
    demo = pd.DataFrame({"group": [1, 1, 2, 2]}, index=[1, 2, 3, 4])
    sim = pd.DataFrame({"item": [5, 3, 2, 2]}, index=demo.index)
    cells = calibrate.cells("polling", sim, pd.DataFrame(), demo)
    assert cells.loc[("item", "A - B")] == 0.5


def test_held_out_labels_cannot_change_calibrated_predictions():
    index = pd.MultiIndex.from_product([["train1", "train2", "held"], ["group"]])
    sim = pd.Series([1., 2., 3.], index=index)
    survey = pd.Series([4., 7., 100.], index=index)
    before = calibrate.predict(survey, sim, np.array(["train1", "train2"]), True)
    survey.loc[("held", "group")] = -1000
    pd.testing.assert_series_equal(before, calibrate.predict(survey, sim, np.array(["train1", "train2"]), True))
    assert before.iloc[0] == pytest.approx(10.)


def test_calibration_can_predict_when_held_out_labels_are_absent():
    index = pd.MultiIndex.from_product([["a", "b", "c"], [1, 2]])
    sim = pd.Series(np.arange(6.), index=index)
    survey = 2 + 3 * sim
    survey.loc["c"] = np.nan
    predictions = calibrate.predict(survey, sim, np.array(["a", "b"]), True)
    assert predictions.tolist() == pytest.approx([14., 17.])


def test_baseline_split_is_disjoint_reproducible_and_covers_every_person():
    people = pd.DataFrame({"pid": np.arange(30)})
    train, test = baselines.split(people, 10)
    assert set(train.pid).isdisjoint(test.pid)
    assert set(train.pid) | set(test.pid) == set(people.pid)
    pd.testing.assert_frame_equal(train, baselines.split(people, 10)[0])


def test_demographic_model_handles_missing_answers_and_unseen_categories():
    train = pd.DataFrame({"item": [1., 1., 5., 5., np.nan]}, index=[1, 2, 3, 4, 5])
    demo = pd.DataFrame({"group": ["a", "a", "b", "b", "a", "a", "b", "new"]}, index=range(1, 9))
    predicted = baselines.demographic_predictions(train, demo, pd.Index([6, 7, 8]), {"item": (1, 5)})
    assert np.isfinite(predicted.to_numpy()).all()
    assert predicted.item.between(1, 5).all()
    assert list(predicted.index) == [6, 7, 8]


def test_synthetic_question_mapping_covers_all_supported_types():
    source = blocks(1)
    source[0]["Questions"] += [
        {"QuestionID": "M", "QuestionType": "Matrix", "QuestionText": "Ratings", "Rows": ["a", "b"],
         "Columns": ["low", "high"], "Answers": {"SelectedByPosition": [1, 2]}},
        {"QuestionID": "S", "QuestionType": "Slider", "QuestionText": "Amount", "Range": {"Min": 0, "Max": 100},
         "Statements": ["x"], "Answers": {"Values": ["43"]}},
        {"QuestionID": "QID181", "QuestionType": "TE", "QuestionText": "Out of 20?", "Answers": {"Text": "7"}},
        {"QuestionID": "free", "QuestionType": "TE", "QuestionText": "An unbounded estimate", "Answers": {"Text": "123"}},
    ]
    picks = twin.human_picks(source)
    assert picks.keys() == twin.questions(source).keys()
    filled = twin.fill(source, picks)
    assert twin.human_picks(filled) == picks
    assert filled[0]["Questions"][-1]["Answers"] == {}
    assert source[0]["Questions"][-1]["Answers"] == {"Text": "123"}


def test_signal_can_compare_only_shared_participants(tmp_path):
    for wave, values in (("wave1_3", [1, 2, 3, 4]), ("wave4", [1, 2, 3, 4]), ("llm_imputed", [1, 2, 3, -10])):
        frame = pd.DataFrame({"TWIN_ID": [1, 2, 3, 4], "ITEM": values})
        path = tmp_path / f"responses_{wave}_formatted.csv"
        path.write_text("TWIN_ID,ITEM\ndescription,description\n" + frame.to_csv(index=False, header=False))
    assert score.per_question(tmp_path, ["ITEM"], pids=pd.Index([1, 2, 3])).iloc[0] == pytest.approx(1.)
    assert score.per_question(tmp_path, ["ITEM"]).iloc[0] < 0


def test_bundled_demo_runs_without_dataset_or_model(monkeypatch):
    from popsim import demo

    def no_download(*args, **kwargs):
        raise AssertionError("offline demo must not download data")

    monkeypatch.setattr(twin, "dataset_file", no_download)
    errors = demo.main()
    assert 0 < errors["survey + twins"] < errors["survey alone"]
