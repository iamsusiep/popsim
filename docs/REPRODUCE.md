# Reproduce

Run from the checkout after `uv sync --locked`.

## CPU analyses

Preparation downloads the pinned public dataset and upstream scorer; it requires Git and
network access. Subsequent runs can use `HF_HUB_OFFLINE=1`.

```bash
uv run python -m popsim.prepare
uv run pytest -m integration
uv run python -m popsim.baselines
uv run python -m popsim.uses
uv run python -m popsim.calibrate
uv run python -m popsim.probes
```

These commands use published model outputs (and captured JEV-9B activations in `cache/probes/` for `popsim.probes`) and need no API key or GPU. Scored local JEV
outputs are included when present. Individual responses and caches stay in ignored directories.

## Optional JEV-9B inference

**Incurs Modal GPU charges.** Authenticate, run all 2,058 participants, then score:

```bash
uv run modal setup
uv run python -m popsim.run --n 2058 --persona summary
uv run python -m popsim.score results/summary-n2058.parquet
uv run python -m popsim.signal results/summary-n2058
```

`--persona` also accepts `demographics` or `full`. Decisions are cached by model revision and
prompt; interrupted runs resume. Re-run the CPU analyses to include new JEV outputs.
Use `uv run streamlit run popsim/app.py` to explore local runs.

## Refresh documentation

Regenerate the chart and tables from checked-in artifacts, offline:

```bash
uv run python -m popsim.report --from-artifacts
```

After running the CPU analyses, full-panel JEV scoring and `signal`, replace the snapshots:

```bash
uv run python -m popsim.report
```

The report exports the current full-panel evaluation; 200-person pilots are excluded.
`uv run python -m popsim.prepare --example` rebuilds the bundled one-split demo.
`uv build` builds the package. CI checks the offline tests, demo and build.
