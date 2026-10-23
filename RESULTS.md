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
on correlation, so individual-specific knowledge beyond demographics remains unestablished for
zero-shot `argmax` predictions. [Scores and split IDs](artifacts/baselines.json).

## Probing JEV-9B hidden representations

We captured `JEV-9B` activations across layers `[7, 15, 19, 23, 27, 31]` at the end of `[state]`
and at the `[decision]:` token on the 18 polling items (`uses.ITEMS`). Ridge probes and all
supervised controls train on the 300 calibration participants (5-fold CV) and evaluate on the
same 1,758 held-out test participants.

<!-- START:probes -->
| Arm (18 polling items) | r (W1–3) | r (W4) | Partial r (given demo) | Similarity | Gap RMSE (SD) | Gap ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Demographic ridge (rounded, lam=10) | 0.270 | 0.269 | -0.010 | 0.797 | 0.133 | 0.900 |
| Demographic ridge (continuous, CV) | 0.316 | 0.317 | 0.000 | 0.792 | 0.117 | 0.835 |
| Persona TF-IDF ridge (300 train) | 0.180 | 0.182 | 0.053 | 0.779 | 0.238 | 0.253 |
| JEV-9B P (sampled, zero-shot) | 0.211 | 0.210 | 0.096 | 0.672 | 0.235 | 1.332 |
| JEV-9B J (argmax, zero-shot) | 0.263 | 0.254 | 0.138 | 0.699 | 0.300 | 1.654 |
| JEV-9B E[Y] (prob expectation, zero-shot) | 0.316 | 0.312 | 0.185 | 0.717 | 0.227 | 1.318 |
| JEV-9B probs ridge (300 train) | 0.304 | 0.301 | 0.171 | 0.796 | 0.142 | 0.813 |
| Demographics + JEV probs (concat ridge) | 0.343 | 0.340 | 0.152 | 0.796 | 0.107 | 0.857 |
| Probe [state] Layer 19 (best [state]) | 0.122 | 0.119 | 0.057 | 0.774 | 0.277 | 0.096 |
| Probe [decision] Layer 7 | 0.073 | 0.067 | 0.033 | 0.772 | 0.288 | 0.046 |
| Probe [decision] Layer 15 | 0.358 | 0.355 | 0.208 | 0.799 | 0.100 | 0.906 |
| Probe [decision] Layer 23 | 0.370 | 0.370 | 0.226 | 0.800 | 0.102 | 0.897 |
| Probe [decision] Layer 31 | 0.366 | 0.366 | 0.222 | 0.800 | 0.102 | 0.894 |
| Probe [decision] Layer 23 (PCA top-128) | 0.368 | 0.366 | 0.225 | 0.801 | 0.092 | 0.957 |
| Residual Probe (JEV E[Y] + Layer 23 probe) | 0.371 | 0.372 | 0.227 | 0.801 | 0.105 | 0.981 |
| Decomp A: Mean [decision] from other 17 Qs | 0.369 | 0.368 | 0.224 | 0.799 | 0.101 | 0.860 |
| Shuffle Check: Within-Demo-Stratum Permute | 0.276 | 0.276 | 0.103 | 0.791 | 0.112 | 0.875 |
| Shuffle Check: Global Persona Permute | 0.011 | 0.011 | 0.011 | 0.755 | 0.310 | 0.013 |
| Humans, retest (wave 1-3 ceiling) | 1.000 | 0.657 | 1.000 | 1.000 | 0.037 | 1.000 |
<!-- END:probes -->

Details are in [probes.json](artifacts/probes.json):

- **Two-stage bottleneck:** Replacing `argmax` (`J`, `r = 0.263`) with probability-weighted
  expected value `E[Y]` reaches `0.316` (`partial r = 0.185` given demographics) without training
  labels. Probing Layer 23 at `[decision]:` reaches **`0.370`** (`0.371` for the residual probe,
  with `gap_ratio = 0.981` vs. `1.654` for `JEV-9B J`). Over 5,000 question-bootstrap draws, the
  Layer 23 probe beats `JEV-9B probs ridge` by `+0.066` (`[+0.034, +0.103]`, 17/18 items) and
  continuous `Demographic ridge` by `+0.053` (`[+0.034, +0.072]`, 15/18 items).
- **All 78 signal items:** Probing at the end of `[state]` before the question gets `0.122`.
  Averaging a person's Layer 23 `[decision]:` vector across the other 17 questions (`Decomp A`)
  gets `0.369`. Using that mean `[decision]:` vector as a fixed person embedding across all 78
  signal questions reaches **`0.173` correlation** (`0.326` disattenuated by human retest
  reliability), compared with `0.102` (`0.154`) for `JEV-9B J` and `0.135` (`0.247`) for
  continuous `Demographic ridge`.
- **Where the output head fails:** On mainstream partisan policy items (`_1..7, 9`), `JEV-9B E[Y]`
  already matches most of the internal signal (`0.537` vs. `0.548` for Layer 23). On `Wealth Tax`
  (`_8`), `Medicare Vouchers` (`_10`), and the 8 technology benefit/risk items (`QID288`, `QID289`),
  the output head misweights its own latent PCs. It assigns large partisan (`PC2 = -0.407`) and
  emotional-stability (`PC3 = +0.600`) weights to technology benefit, and a `-0.750` anxiety
  weight (`PC4`) to technology risk. Real human responses show near-zero loading on those axes
  (`-0.022` to `+0.051`) and instead track interpersonal warmth (`PC5`) and compassion (`PC7`).
- **Person dependence and limits:** Shuffling personas within identical demographic strata
  (`party × ideology × gender × age`) drops Layer 23 correlation from `0.370` to `0.276`. The
  probe requires per-question calibration labels (`N = 300`): when target questions are held out
  entirely (leave-one-question-out `0.312` or leave-family-out `0.294`), a shared probe does not
  beat zero-shot `JEV-9B E[Y]` (`0.316`).

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
(question-bootstrap intervals only), [hidden-state probing and PC interpretation](artifacts/probes.json),
[use-case metrics](artifacts/use-cases.json), and [full-panel scores](artifacts/summary-n2058.json)
(oracle rows explicitly labeled). [Reproduction commands →](docs/REPRODUCE.md)
