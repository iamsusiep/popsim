"""Export compact result snapshots and refresh the reviewed tables in RESULTS.md.

Run after popsim.baselines, popsim.calibrate and popsim.signal. No inference is performed.
    uv run python -m popsim.report
"""

import hashlib
import json

from popsim import calibrate, twin
from popsim.decider import MODEL, MODEL_REVISION


def table(headers, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in [headers, ["---"] * len(headers), *rows])


def main() -> None:
    out = twin.ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    sources = {
        "calibration.json": "calibrate.json", "baselines.json": "baselines/scores.json",
        "person-signal.json": "signal.json", "use-cases.json": "uses.json",
        **{f"{run}.json": f"{run}/scores.json" for run in ("summary-n2058", "summary-n200", "demographics-n200", "full-n200")},
    }
    data = {}
    for name, source in sources.items():
        data[name] = json.loads((twin.ROOT / "results" / source).read_text())
        encoded = json.dumps(data[name], indent=2).replace(str(twin.ROOT) + "/", "")
        for old in ("majority answer", "population draw"):
            encoded = encoded.replace(f'"{old}"', f'"{old} (target-population oracle)"')
        (out / name).write_text(encoded + "\n")

    baseline = data["baselines.json"]
    baseline_table = table(["arm", "accuracy", "cross-person correlation"], [
        [name, f"{r['accuracy']:.4f}", f"{r['person_signal']:.4f}"] for name, r in baseline["arms"].items()
    ])
    signals = data["person-signal.json"]["arms"]
    signal_table = table(["run", "people", "correlation [95% interval]", "paired people", "JEV J minus run [95% interval]"], [
        [name, r["people"], "{:.3f} [{:.3f}, {:.3f}]".format(*r["signal"]), r["paired_people"],
         "—" if r["vs JEV-9B J"] is None else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*r["vs JEV-9B J"])]
        for name, r in sorted(signals.items(), key=lambda x: -x[1]["signal"][0])
    ])
    calibration = data["calibration.json"]
    settings = {"pricing": "300 people, 10 units", "polling": "300 people, 6 units", "ab": "300 people, 8 units"}
    summary_rows = []
    for use, setting in settings.items():
        rows = calibration[use][setting]
        ensemble = rows[f"{calibrate.ENSEMBLE}, calibrated"]
        summary_rows.append([use, setting, f"{rows[calibrate.NONE[use]]['error']:.3f}",
                             f"{ensemble['error']:.3f}", f"{ensemble['beats']:.1%}",
                             f"{rows['humans, retest, raw']['error']:.3f}"])
    calibration_table = table(["use", "calibration survey", "survey alone", "ensemble calibrated", "beats survey", "human retest"], summary_rows)
    all_names = [name for name in calibration["pricing"][settings["pricing"]] if name.endswith(", raw") and not name.startswith("humans")]
    details = []
    for raw in all_names:
        name = raw.removesuffix(", raw")
        values = []
        for use in settings:
            rows = calibration[use][settings[use]]
            values += [f"{rows[raw]['error']:.3f}", f"{rows[name + ', calibrated']['error']:.3f}"]
        details.append([name, *values])
    calibration_table += "\n\n" + table(["predictor", "pricing raw", "pricing calibrated", "polling raw", "polling calibrated", "A/B raw", "A/B calibrated"], details)

    path = twin.ROOT / "RESULTS.md"
    text = path.read_text()
    for name, content in (("baselines", baseline_table), ("signal", signal_table), ("calibration", calibration_table)):
        start, end = f"<!-- START:{name} -->", f"<!-- END:{name} -->"
        before, remainder = text.split(start, 1)
        _, after = remainder.split(end, 1)
        text = before + start + "\n" + content + "\n" + end + after
    path.write_text(text)
    files = [*sorted(out.glob("*.json")), twin.ROOT / "popsim" / "example_pricing.json", twin.ROOT / "uv.lock"]
    manifest = {
        "dataset": twin.DATASET, "dataset_revision": twin.DATASET_REVISION,
        "scorer_revision": twin.CODE_REVISION, "model": MODEL, "model_revision": MODEL_REVISION,
        "calibration": "fixed questionnaire ranges for fitting; target SDs only for evaluation",
        "baseline_split": "seed 0; 300 training participants; 1758 separate evaluation participants",
        "sha256": {str(p.relative_to(twin.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name != "manifest.json"},
        "source_sha256": {str(p.relative_to(twin.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((twin.ROOT / "popsim").glob("*.py"))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote snapshots and provenance to {out}; refreshed RESULTS.md")


if __name__ == "__main__":
    main()
