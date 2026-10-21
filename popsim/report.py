"""Export current results and regenerate the README, tables and calibration chart.

    uv run python -m popsim.report                   # refresh from local results
    uv run python -m popsim.report --from-artifacts  # render checked-in data offline
"""

import argparse
import hashlib
import json

from matplotlib import rc_context
from matplotlib.figure import Figure

from popsim import calibrate, twin
from popsim.decider import MODEL, MODEL_REVISION

CASES = {
    "pricing": ("Pricing", "300 people, 10 units", "10 / 40 products", "Buy-rate RMSE (pp)"),
    "polling": ("Polling gaps", "300 people, 6 units", "6 / 18 items", "Gap RMSE (item SD)"),
    "ab": ("A/B effects", "300 people, 8 units", "8 / 16 contrasts", "Effect RMSE (pooled SD)"),
}


def table(headers, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in [headers, ["---"] * len(headers), *rows])


def errors(data, use):
    rows = data[use][CASES[use][1]]
    return [rows[name]["error"] for name in (calibrate.NONE[use],
            f"{calibrate.ENSEMBLE}, raw", f"{calibrate.ENSEMBLE}, calibrated")]


def calibration_figure(data):
    fig = Figure(figsize=(12, 4.1), facecolor="white")
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.27, top=0.72, wspace=0.82)
    fig.suptitle("Error on held-out products and questions", x=0.035, y=0.97,
                 ha="left", fontsize=19, fontweight="bold", color="#172b3a")
    fig.text(0.035, 0.865, "Lower is better · mean of 500 seeded splits", fontsize=11, color="#52616b")
    for ax, (use, (title, _, surveyed, metric)) in zip(axes, CASES.items()):
        values = errors(data, use)
        labels = ["Zero effect" if use == "ab" else "Survey alone", "Raw twins", "Calibrated twins"]
        ax.barh(range(3), values, height=0.54, color=["#aebac4", "#83b9ba", "#126b71"])
        ax.set_yticks(range(3), labels, fontsize=10, color="#273d4a")
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.3)
        ax.set_title(f"{title}\n{surveyed} surveyed", loc="left", fontsize=11, pad=12, color="#172b3a")
        ax.set_xlabel(metric, fontsize=9, labelpad=8, color="#52616b")
        ax.tick_params(axis="both", length=0, labelsize=9, colors="#52616b")
        ax.xaxis.grid(True, color="#e5ebee", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, value in enumerate(values):
            label = f"{value:.2f}" if use == "pricing" else f"{value:.3f}"
            ax.text(value + max(values) * 0.04, i, label, va="center", fontsize=10, color="#172b3a")
    fig.text(0.035, 0.105, "Twin-2K-500 · 2,058-person panel with existing profiles · 300 calibration respondents",
             fontsize=10, color="#52616b")
    fig.text(0.035, 0.04, "Twins = ensemble of seven published runs. Source: artifacts/calibration.json",
             fontsize=9, color="#52616b")
    return fig


def refresh(path, sections):
    text = path.read_text()
    for name, content in sections.items():
        start, end = f"<!-- START:{name} -->", f"<!-- END:{name} -->"
        before, remainder = text.split(start, 1)
        _, after = remainder.split(end, 1)
        text = before + start + "\n" + content + "\n" + end + after
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-artifacts", action="store_true", help="render without local runs or data downloads")
    args = parser.parse_args()
    out = twin.ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    sources = {
        "calibration.json": "calibrate.json", "baselines.json": "baselines/scores.json",
        "person-signal.json": "signal.json", "use-cases.json": "uses.json",
        "summary-n2058.json": "summary-n2058/scores.json",
    }
    if not args.from_artifacts:
        for name, source in sources.items():
            data = json.loads((twin.ROOT / "results" / source).read_text())
            if name == "use-cases.json":
                data.pop("pilot", None)
            encoded = json.dumps(data, indent=2).replace(str(twin.ROOT) + "/", "")
            for old in ("majority answer", "population draw"):
                encoded = encoded.replace(f'"{old}"', f'"{old} (target-population oracle)"')
            (out / name).write_text(encoded + "\n")
    baseline = json.loads((out / "baselines.json").read_text())["arms"]
    calibration = json.loads((out / "calibration.json").read_text())
    baseline_table = table(["Arm", "Similarity", "Correlation"], [
        [name, f"{r['accuracy']:.3f}", f"{r['person_signal']:.3f}"] for name, r in baseline.items()
    ])
    calibration_table = table(["Task (RMSE unit)", "Surveyed", "Survey only¹", "Raw twins", "Calibrated twins"], [
        [f"{title} ({metric.split('(')[1].rstrip(')')})", surveyed,
         *[f"{value:.3f}" for value in errors(calibration, use)]]
        for use, (title, _, surveyed, metric) in CASES.items()
    ])
    calibration_table += "\n\n¹ Survey averages for pricing/polling; zero effect for A/B. Twins = published-model ensemble."
    refresh(twin.ROOT / "RESULTS.md", {"baselines": baseline_table, "calibration": calibration_table})
    price, poll, ab = (errors(calibration, use) for use in CASES)
    demo = baseline["demographic ridge, 300 training humans"]["person_signal"]
    jev = baseline["JEV-9B J, summary"]["person_signal"]
    highlights = (
        f"- **Pricing:** {price[0]:.2f} → {price[2]:.2f} pp RMSE ({1 - price[2] / price[0]:.0%} lower than survey alone).\n"
        f"- **Polling:** {poll[0]:.3f} → {poll[2]:.3f} SD RMSE. **A/B:** {ab[1]:.3f} raw → {ab[2]:.3f} calibrated.\n"
        f"- **Individual prediction:** demographic baseline correlation ({demo:.3f}) matches JEV-9B ({jev:.3f});\n"
        "  the current metric does not establish individual-specific knowledge."
    )
    refresh(twin.ROOT / "README.md", {"highlights": highlights})
    chart = twin.ROOT / "docs" / "calibration.svg"
    chart.parent.mkdir(exist_ok=True)
    with rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "popsim"}):
        calibration_figure(calibration).savefig(chart, metadata={"Date": None})
    chart.write_text("\n".join(line.rstrip() for line in chart.read_text().splitlines()) + "\n")
    files = [*(out / name for name in sources), twin.ROOT / "popsim" / "example_pricing.json", twin.ROOT / "uv.lock", chart]
    manifest = {
        "dataset": twin.DATASET, "dataset_revision": twin.DATASET_REVISION,
        "scorer_revision": twin.CODE_REVISION, "model": MODEL, "model_revision": MODEL_REVISION,
        "calibration": "fixed questionnaire ranges for fitting; target SDs only for evaluation",
        "baseline_split": "seed 0; 300 training participants; 1758 separate evaluation participants",
        "sha256": {str(p.relative_to(twin.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "source_sha256": {str(p.relative_to(twin.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((twin.ROOT / "popsim").glob("*.py"))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("refreshed README.md, RESULTS.md, docs/calibration.svg and artifacts/manifest.json")


if __name__ == "__main__":
    main()
