# Where Cross-Question Structure in LLM Errors Comes From

Code, data and LaTeX source for the paper *Where Cross-Question Structure in LLM Errors Comes From: Shared Seeds,
Shared Prompts, and Answer-Letter Preference* (Izhar Ali, Rowan University; EIML@ICML 2026).
[PDF](paper/main.pdf)

For each run we record a binary correctness matrix (K samples or models × N questions) and count the eigenvalues
of its question correlation matrix that exceed those produced by independent answers or, across models, by a
single ability per model. Within one model, independent requests show no detectable direction; a seed or
exemplar set shared across one round of samples creates directions. Across 24 greedily decoded models, the first
direction follows ability and the second is answer-letter preference: it follows the displayed letter when the
options are rotated, and it rests mostly on one model. Scoring a question correct only if it is answered
correctly in all four rotations leaves no excess at the second eigenvalue (with low power against a second
ability of typical strength); a marginal excess at the third and fourth comes mostly from two near-duplicate
model pairs, partly from the letter-biased model. Options should be rotated before an ensemble's error structure
is read as epistemic uncertainty.

Earlier versions are in git history.

## Data

Benchmarks: MMLU (500 questions, `cais/mmlu`), HellaSwag (200 items, `Rowan/hellaswag` validation), GSM8K (first
100 test problems, chain of thought).

| condition | shared within one pass |
|---|---|
| `iid` | nothing; every (question, pass) is an independent sample |
| `sharedseed` (S1) | the sampling noise (one seed per pass) |
| `sharedprompt` (S2) | a 5-exemplar prompt prefix (one exemplar set per pass) |
| `greedyrot` (R2) | greedy decoding (T = 0) on 4 cyclic rotations of the options |

| model | runs |
|---|---|
| Qwen2.5-7B-Instruct | MMLU iid K=100 at T=1.0, K=30 at T=0.5 and 1.5; S1 K=100; S2 K=100; HellaSwag iid K=50; GSM8K iid K=30 (256 and 1024 tokens) |
| Mistral-7B-Instruct-v0.3, Meta-Llama-3-8B-Instruct | MMLU iid K=100; HellaSwag iid K=50 (Llama also S2 K=100) |
| SmolLM2-1.7B-Instruct, Phi-3-mini-4k-instruct | MMLU iid K=100 |
| 24 models (`src/plan.py`, `ENSEMBLE_MODELS`) | R2, two replicates (A, B) |

Every raw completion, seed and setting is saved. [`results/PROTOCOL.md`](results/PROTOCOL.md) is the
collection protocol (prompts, decoding, determinism, parser, rotations, integrity checks);
`results/manifest.md` lists every run with its sha256. The analysis numbers are in
[`results/analysis/SUMMARY.md`](results/analysis/SUMMARY.md), which the paper cites.

## Layout

```
src/                  data collection: question freezing, prompts, vLLM collection, scoring, manifest, tests
scripts/              analysis drivers, figure scripts, shared statistics, tests
results/              per-run .npz + .json, questions_*.json, ensemble .npz, manifest, PROTOCOL.md
  r2/, r2_rep/        R2 replicates A and B
  analysis/           analysis outputs and SUMMARY.md
  provenance/         files from earlier versions that current code reads (see its README)
  pins/               tokenizer pins and reference prompt hashes read by src/collect.py
  determinism/        three-start engine-determinism summaries (PROTOCOL.md)
  logs/               run status, event notes and the rescore record (other run logs are not versioned)
paper/                LaTeX source, bibliography, style files, figures
```

## Reproducing

The analysis environment is `uv sync` (Python ≥ 3.12; the `dev` group adds pytest and Hypothesis); the analysis
runs on CPU.

Data collection needs a GPU and a container with vLLM 0.23.0. The scripts use the local image `gb10-vllm`
(NVIDIA GB10, aarch64, sm_121; torch 2.11.0+cu130, transformers 5.12.1, tokenizers 0.22.2). On other hardware,
replace the image name in `src/run_all.sh` and `src/run_v3.sh`. Models are read from the local Hugging Face cache
in offline mode.

**1. Data collection** (outputs in `results/`):

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 .venv/bin/python src/prepare_questions.py
nohup bash src/run_all.sh > results/logs/run_all.log 2>&1 &              # sampled runs
nohup bash src/run_r2_replicates.sh > results/logs/run_r2.log 2>&1 &     # R2, replicates A and B
.venv/bin/python src/rescore.py --skip-condition greedy --backup <dir> --log results/logs/rescore_v31.json
.venv/bin/python src/r2_analysis.py
.venv/bin/python src/build_manifest.py
```

Both drivers run `src/collect.py` one model and one container at a time through `src/docker_run.sh`, and resume
after an interruption. One model: `src/docker_run.sh gb10-vllm python src/collect.py --model Qwen/Qwen2.5-7B-Instruct`.
`run_all.sh` also makes greedy runs, which R2 rotation 0 replaces. The reference prompt hashes are built in a
separate venv (`src/reference_hashes.py`, see its docstring).

**2. Analysis** (outputs in `results/analysis/`; in this order):

```bash
cd scripts
uv run python -c "import v2lib; v2lib.extract_all()"
for s in item01_within item02_shared item03_power item04_dispersion item05_split_half \
         item06_answer_identity item07_across item08_09_sub_ensembles_family item10_pairs \
         item11_effective_rank item12_temperature_gsm8k item13_unparseable item14_integrity; do
  uv run python $s.py
done
uv run python v3_across.py B_extra
uv run python v3_across.py C --rep A
uv run python v3_across.py C --rep B
uv run python v3_across.py near
uv run python v3_rotation_letters.py
uv run python v3_loo.py
uv run python v3_replicate_compare.py
uv run python v3_softpref.py all        # SUMMARY section F (hours)
uv run python v3_G_familywise.py all    # section G
uv run python v3_H_followup.py all      # section H
uv run python v3_I_llama.py all         # section I
uv run python v3_J_power2pl.py          # section J (about 30 min on 16 processes)
uv run python make_summary_v3.py        # SUMMARY.md up to section E; keeps the hand-written sections F-J
```

Sections F-I checkpoint to `results/analysis/logs_*/ckpt/` (not versioned). Their `assemble` part reads the
checkpoints, so run `all` (or `run`, then `assemble`).

**3. Figures** (outputs in `paper/figures/`):

```bash
for f in main power mechanism letters; do uv run python scripts/make_fig_$f.py; done
```

**4. Tests and paper:**

```bash
uv run python -m pytest -q
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## License

MIT; see [LICENSE](LICENSE).
