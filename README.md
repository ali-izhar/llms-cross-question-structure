# Stochastic Sampling is Epistemically Shallow

**The Dimensionality Gap Between Temperature Variation and Model Diversity in LLMs**

Izhar Ali, Rowan University.
Accepted at the **2nd Workshop on Epistemic Intelligence in Machine Learning (EIML@ICML 2026)**, Seoul.

[Paper (PDF)](paper/main.pdf) · [LaTeX source](paper/main.tex)

---

## TL;DR

When an LLM gives different answers on repeated runs, self-consistency turns
that variation into a per-question uncertainty estimate. But does the same
variation reveal *cross-question* structure — related questions flipping
together, the way a diverse ensemble does?

We compare two regimes on the same MMLU questions:

- **Within-model:** one model run K=100 times at τ=1.
- **Across-model:** an ensemble of 24 LLMs run once each at τ=0.

A Marchenko–Pastur random-matrix test separates signal from sampling noise on
both sides. The headline result:

| | within | across |
|---|---|---|
| above-noise eigenvalues (d) | ≤ 1 across 5 families × 3 benchmarks | 4 (vs. ≤ 1 under a matched-difficulty Bernoulli null in 500 MC draws) |

Per-question uncertainty lives *within* a model (split-half r = 0.994);
multi-dimensional, cross-question uncertainty lives *between* them. Two peer
models outperform 100-sample self-consistency at 1/40th the cost.

## Repository layout

```
paper/                 LaTeX source, bibliography, ICML style files, compiled PDF
  figures/             camera-ready figures (PDF)
src/                   experiment runners (within-model probing, ensembles, GSM8K)
  analysis.py          MP / Tracy–Widom / Horn parallel analysis
  stats.py             core statistics helpers (+ tests)
scripts/               figure generation + post-hoc analyses
results/               .npy correctness matrices, per-model analysis JSONs,
                       figure inputs (raw run logs are .gitignored)
```

## Reproducing the figures

The committed `results/` directory contains every artifact the figures read
from. To regenerate the five paper figures from the existing data:

```bash
uv sync                                              # install deps
uv run python scripts/make_hero_figure.py            # Fig. 1 (hero)
uv run python scripts/make_epsilon_sweep_figure.py   # Fig. 2 (eps robustness)
uv run python scripts/make_selective_prediction_figure.py
uv run python scripts/make_supplementary_figure.py
```

PDFs land in `paper/figures/`.

## Reproducing the data

The full pipeline needs a GPU (a single H100 / GB10 is enough; the 24-model
ensemble is the heavy step) and Hugging Face access for the gated models
(Llama-3, Gemma-2). Set `HF_TOKEN` in a `.env` file before running.

```bash
uv run python src/run_within_batch.py        # K=100 within-model on MMLU
uv run python src/run_ensemble_expanded.py   # 24-model τ=0 ensemble
uv run python src/run_within_batch_hellaswag.py
uv run python src/run_gsm8k_medium.py        # GSM8K CoT
```

Each runner checkpoints incrementally, so an interrupted run can be resumed
in place.

## Citation

```bibtex
@inproceedings{ali2026dimgap,
  title     = {Stochastic Sampling is Epistemically Shallow:
               The Dimensionality Gap Between Temperature Variation
               and Model Diversity in {LLM}s},
  author    = {Ali, Izhar},
  booktitle = {2nd Workshop on Epistemic Intelligence in Machine Learning
               (EIML@ICML)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
