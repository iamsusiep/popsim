# PopSim

Reproducible evaluation and calibration of synthetic survey panels against real human responses.
Experimental research tooling built on [Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500).

PopSim compares individual-answer similarity, population answer shares, pricing curves, survey
wording effects and demographic gaps. It also tests whether a small additional human survey can
calibrate twins on some products or questions and improve predictions on other units.

The optional inference path runs the open [JEV-9B](https://huggingface.co/autotrust/JEV-9B) model
on Modal. The evaluation also uses the dataset authors' published GPT-4.1-mini, GPT-4.1 and
Gemini 2.5 Flash outputs. No model API key is needed to reproduce the CPU analyses.

## Try it without a GPU

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From this checkout:

```bash
uv sync --locked
uv run python -m popsim.demo
uv run pytest
```

Installation downloads Python dependencies. After installation, the demo and default tests run
offline with no model, dataset download, credentials or paid service. The demo contains only
aggregate product-level data and reproduces **one fixed split**: 17.23 pp pricing error for the
survey alone and 9.09 pp after calibration. It illustrates the workflow, not the 500-split estimate.

## Reproduce the full CPU analyses

The first preparation downloads the pinned public dataset and scorer; it needs network access
and Git. Later commands can use `HF_HUB_OFFLINE=1` once preparation is complete.

```bash
uv run python -m popsim.prepare
uv run pytest -m integration
uv run python -m popsim.baselines
uv run python -m popsim.uses
uv run python -m popsim.calibrate
```

`baselines` uses 300 people for training and the other 1,758 for evaluation. It compares the
majority answer, a population draw and a fixed demographic ridge regression with published twins.
`calibrate` uses 500 seeded splits of surveyed units and people. Fits use fixed questionnaire
ranges; held-out human standard deviations enter only the final error calculation.

Published-model analyses work without our local JEV runs. If scored JEV outputs are present in
`results/`, they are included automatically. The compact [artifacts](artifacts/) contain the
recorded full comparisons, corrected calibration results, split IDs and a revision/hash manifest.
Individual response files, billing snapshots and the large decision cache are not bundled.

## What the current evidence supports

- On a separate 1,758-person evaluation sample, majority answers learned from 300 people score
  0.744 on the upstream accuracy measure, above all tested twins. This baseline knows human
  responses to the same survey questions; it does not predict wholly unseen questions.
- A demographic ridge baseline reaches 0.104 cross-person correlation, comparable to JEV-9B J
  and the full-persona GPT-4.1-mini run (both about 0.102). The current correlation metric does
  not establish individual-specific knowledge beyond demographic patterns.
- With 300 calibration respondents and 10 of 40 products surveyed, the published-model ensemble
  reduces held-out pricing RMSE from 17.26 to 8.87 pp. With 6 of 18 polling items surveyed,
  calibration reduces gap RMSE from 0.491 to 0.265 SD after removing target-statistic access.
- These savings assume rich profiles already exist for the 2,058-person panel. The 300-person
  figure counts additional calibration responses, not all data needed to construct the twins.

See [RESULTS.md](RESULTS.md) for definitions, all comparisons and limitations. Results are from
one US panel and these particular model/prompt configurations. The upstream "accuracy" is
normalized absolute-error similarity, not the fraction of exactly correct answers. Historical
leave-one-out majority/draw results are explicitly **target-population oracle diagnostics**.

## Optional JEV-9B inference — incurs GPU charges

Authenticate a Modal account using `uv run modal setup` before these commands. The GPU runner
uses one H100 and saves every completed batch; the local demo and CPU analyses do not invoke it.

```bash
uv run modal run popsim/decider.py
uv run python -m popsim.run --n 200 --persona summary
uv run python -m popsim.score results/summary-n200.parquet
uv run python -m popsim.run --n 2058 --persona summary
uv run python -m popsim.score results/summary-n2058.parquet
uv run python -m popsim.signal results/summary-n2058
uv run streamlit run popsim/app.py
```

`--persona` accepts `demographics`, `summary` or `full`. Each model output is a distribution:
**J** takes its most likely option; **P** draws one option with a fixed seed per person/question.
The cache key includes the model revision and exact rendered prompt. Raw log-probabilities are
stored for new runs; imported pilot runs contain probabilities only. `python -m popsim.cache stats`
shows the local cache. Historical Modal charges are observations, not a guarantee of current cost.

Anchoring is excluded from all 16-task comparisons because it requires unbounded numeric answers.
Sliders and bounded entries use 11 answer levels. The original converter and scorer run unmodified
at a pinned commit, with oracle mapping checks in the integration suite.

## Refresh release artifacts

To refresh every historical snapshot, also reproduce and score the 200-person `demographics`
and `full` pilots using the same `run` / `score` commands with those persona names. Then, after
the scored JEV runs and CPU analyses:

```bash
uv run python -m popsim.prepare --example
uv run python -m popsim.report
uv build
```

`report` writes local aggregate snapshots and regenerates the baseline, correlation and calibration
tables in RESULTS.md. It does not publish anything. The demo is included in the built wheel.
GitHub CI runs the default tests and demo with an empty offline dataset cache, then builds the package.

## Related work and attribution

PopSim builds on [Digital-Twin-Simulation](https://github.com/tianyipeng-lab/Digital-Twin-Simulation)
and Jev-style typed decisions. Related work includes
[Jev Synthetic Survey](https://github.com/jjd-lab/jev-synthetic-survey),
[Digital Twins as Funhouse Mirrors](https://arxiv.org/abs/2509.19088), and
[Valid Survey Simulations with Limited Human Data](https://aclanthology.org/2026.acl-long.498/).
The contribution is a small, inspectable evaluation workflow and reproducible comparisons.

Original code: [Apache-2.0](LICENSE). Twin-2K-500 and derived data: CC-BY-4.0.
Upstream scorer and JEV-9B: Apache-2.0. See [NOTICE](NOTICE) for attribution and the distinction
between code and data licenses.
