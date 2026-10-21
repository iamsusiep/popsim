# Results

Current evaluation on Twin-2K-500. Tables are generated from the checked-in
[artifacts](artifacts/); revisions and hashes are in the [manifest](artifacts/manifest.json).

## Calibration

300 people answer a subset of items; a seven-run published-model ensemble predicts the rest.
Targets are wave-4 response aggregates. Values are mean held-out RMSE over 500 seeded splits.
Lower is better.

<!-- START:calibration -->
| Task (RMSE unit) | Surveyed | Survey only¹ | Raw twins | Calibrated twins |
| --- | --- | --- | --- | --- |
| Pricing (pp) | 10 / 40 products | 17.263 | 12.429 | 8.865 |
| Polling gaps (item SD) | 6 / 18 items | 0.491 | 0.361 | 0.265 |
| A/B effects (pooled SD) | 8 / 16 contrasts | 0.715 | 0.538 | 0.541 |

¹ Survey averages for pricing/polling; zero effect for A/B. Twins = published-model ensemble.
<!-- END:calibration -->

Pricing and polling improve after calibration. For A/B effects, the raw ensemble is already
better than predicting zero; calibration adds no improvement. All models and survey sizes
are in [calibration.json](artifacts/calibration.json).

## Predictions for separate people

Baselines train on 300 people; every arm is evaluated on the same 1,758 other people (seed 0).
Similarity averages 16 tasks; correlation averages 78 non-pricing questions. Higher is better.

<!-- START:baselines -->
| Arm | Similarity | Correlation |
| --- | --- | --- |
| majority, 300 training humans | 0.744 | 0.000 |
| draw, 300 training humans | 0.680 | 0.001 |
| demographic ridge, 300 training humans | 0.742 | 0.104 |
| GPT-4.1-mini, demographics | 0.685 | 0.063 |
| GPT-4.1-mini, summary | 0.681 | 0.084 |
| GPT-4.1-mini, full | 0.723 | 0.102 |
| GPT-4.1-mini, full, temp 1 | 0.718 | 0.097 |
| GPT-4.1-mini, full, reasoning | 0.708 | 0.099 |
| GPT-4.1, JSON persona | 0.712 | 0.103 |
| Gemini 2.5 Flash, full | 0.691 | 0.106 |
| JEV-9B J, summary | 0.637 | 0.102 |
| JEV-9B P, summary | 0.627 | 0.068 |
| humans, retest | 0.816 | 0.479 |
<!-- END:baselines -->

**Similarity** is 1 minus normalized absolute error against the person's waves 1–3 answers,
not exact-match accuracy. **Correlation** uses those same answers across people and can reflect
demographics. JEV **J** takes the most likely answer; **P** samples with a fixed seed.
Human retest compares wave-4 answers with waves 1–3.

Majority answers exceed the tested twins on similarity. Demographic ridge matches several twins
on correlation, so individual-specific knowledge beyond demographics remains unestablished.
This is one split, not a significance test. [Scores and split IDs](artifacts/baselines.json).

## Method and limits

- **Existing profiles:** calibration assumes rich profiles for all 2,058 participants. The 300
  respondents count additional survey effort. Evaluation holds out items within this panel,
  includes calibration participants, and does not test a new population.
- **Calibration:** least squares fits a scale plus an intercept per price level or demographic
  segment; A/B fits only a scale. Fixed questionnaire ranges define fit units. Held-out human
  SDs are used only to score errors. Some A/B contrasts share controls.
- **Baseline information:** majority/draw use same-question training answers, matching wording
  when available. Ridge uses seven categorical demographics with fixed lambda=10. These
  baselines have different information from twins and do not model pricing offers.
- **Scope:** one US panel and the listed model/prompt configurations. JEV answers one item at a
  time; published prompts differ. Structured persona/target question IDs are disjoint, but
  supplied summary generation is not fully audited. Anchoring is excluded; sliders and bounded
  numeric answers use 11 levels.

Additional current diagnostics: [cross-person correlations](artifacts/person-signal.json)
(question-bootstrap intervals only), [use-case metrics](artifacts/use-cases.json), and
[full-panel scores](artifacts/summary-n2058.json) (oracle rows explicitly labeled).
[Reproduction commands →](docs/REPRODUCE.md)
