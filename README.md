# PopSim

Evaluate synthetic survey respondents against real people, then test whether a small human
survey can improve their predictions. Experimental research tooling built on
[Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500).

## Latest results

![Survey-only, raw-twin and calibrated-twin errors for pricing, polling and A/B effects](docs/calibration.svg)

<!-- START:highlights -->
- **Pricing:** 17.26 → 8.87 pp RMSE (49% lower than survey alone).
- **Polling:** 0.491 → 0.265 SD RMSE. **A/B:** 0.538 raw → 0.541 calibrated.
- **Individual prediction:** demographic baseline correlation (0.104) matches JEV-9B (0.102);
  the current metric does not establish individual-specific knowledge.
<!-- END:highlights -->

These results use one 2,058-person US panel with **existing rich profiles**. The 300-person
survey is additional calibration data. [Results and limits →](RESULTS.md)

## How it works

```mermaid
flowchart LR
    A[Existing profiles] --> B[Synthetic answers]
    B --> D[Fit calibration]
    C[Small human survey] --> D
    D --> E[Predict held-out items]
    E --> F[Compare with humans]
```

PopSim uses published GPT-4.1-mini, GPT-4.1 and Gemini 2.5 Flash responses, or generates
answers with JEV-9B. A simple linear calibration fits surveyed items and predicts the rest.
Separate comparisons test answer similarity and correlation against human-trained baselines.

## Try it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/iamsusiep/popsim.git
cd popsim
uv sync --locked
uv run python -m popsim.demo
uv run pytest
```

After dependency installation, the demo and default tests work offline without credentials
or a GPU. The demo uses one fixed pricing split; the chart averages 500 splits.

[Reproduce the analyses or run JEV-9B →](docs/REPRODUCE.md)

## Sources and license

Built on [Digital-Twin-Simulation](https://github.com/tianyipeng-lab/Digital-Twin-Simulation)
and [JEV-9B](https://huggingface.co/autotrust/JEV-9B). Related work:
[Jev Synthetic Survey](https://github.com/jjd-lab/jev-synthetic-survey),
[Funhouse Mirrors](https://arxiv.org/abs/2509.19088),
[Limited Human Data](https://aclanthology.org/2026.acl-long.498/).

Code: [Apache-2.0](LICENSE). Derived data: CC-BY-4.0. [Attribution](NOTICE).
