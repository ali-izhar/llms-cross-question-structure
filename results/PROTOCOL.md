# Data-collection protocol

This file states how every run in `results/` was collected and scored. Code: `src/` (`prepare_questions.py`,
`plan.py`, `collect.py`, `canonical_tok.py`, `scoring.py`, `rescore.py`, `run_all.sh`, `run_v3.sh`,
`r2_analysis.py`, `build_manifest.py`). The files from earlier versions that current outputs still read are in
`results/provenance/` (see its README).

## Relation to the submitted version

The data of the submitted (April) version had four defects, corrected here: (1) `check_answer` scored a wrong
bare letter as correct when the gold text contained it; (2) callers scored `answer.upper().startswith(letter)`,
so "Based on ..." counted as B, and the HellaSwag fallback read the article "a" as A; (3) the question loader
matched by a 100-character prefix, so later runs used different questions at up to 66 of 500 positions; (4) the
prompt protocol differed between runs and raw outputs were not saved.

The first corrected collection (v2) built prompts with `AutoTokenizer`, which in the gb10-vllm container
(transformers 5.12.1) tokenized three models wrongly: DeepSeek-LLM-7B-chat (Metaspace pre-tokenizer: spaces and
newlines dropped), Yi-1.5-6B-Chat (`<|im_start|>` not matched as an added token) and OLMo-2-1124-7B-Instruct
(ByteLevel without the model's split regex). Their v2 greedy runs (500/500 prompts non-canonical) are in
`results/provenance/v2_broken_tokenization/`. The other 21 models and all sampled runs had canonical ids at every
prompt. The current data (v3) use canonical tokenization, parser v3.1, deterministic engine flags, and the R2
rotation runs below; R2 rotation 0 of replicate A replaces the v2 greedy MMLU ensemble for all 24 models.

## Questions

Frozen once by `src/prepare_questions.py` (host CPU).

| file | content | identity |
|---|---|---|
| `questions_mmlu.json` | the 500 questions of the original Qwen2.5-7B run | resolved by the fixed `load_questions_from_saved`, then mapped to the exact `cais/mmlu` `all`/`test` row (question, subject, choices, answer); the row equals the original stratified sampler's row at 500/500 positions; no placeholder fallback allowed |
| `questions_hellaswag.json` | the 200 items of `results/provenance/questions_hellaswag.json` | `Rowan/hellaswag` validation row; `ind` is not unique (433 duplicates), so the row is matched by `ind` + exact `ctx` + exact endings + label; exactly one row per item; stores `row`, `ind`, `source_id` |
| `questions_gsm8k.json` | first 100 `openai/gsm8k` `main`/`test` problems | rows 0..99; identical to the submitted version's `questions_gsm8k_medium.json` |
| `questions_mmlu_dev.json` | 285 `cais/mmlu` `all`/`dev` items | exemplar pool for S2 |

## Prompts

The same protocol for every model:

* One user message whose text is `mmlu_utils.build_mmlu_prompt` (MMLU), `hellaswag_utils.build_hellaswag_prompt`
  (HellaSwag) or `COT_PROMPT` in `src/collect.py` (GSM8K; verbatim from the submitted version's
  `src/run_gsm8k_medium.py`).
* The text is rendered with the model's own chat template (`add_generation_prompt=True`) and tokenized with
  `add_special_tokens=False` (the template carries its own BOS). A model without a chat template gets the raw
  text, tokenized with the tokenizer's default special tokens (phi-2: `add_special_tokens=True`). The prompt mode
  is recorded per run (`prompt_mode`) and in the manifest; the rendered prompt of question 0 is saved
  (`rendered_prompt_q0_pass0`).
* The Qwen chat template inserts its default system message ("You are Qwen, ...") when only a user message is
  given; this is the model's own template and is kept.
* S2 (shared prompt perturbation): the same single user message, with 5 dev exemplars inserted after the
  instruction line: "Here are some example questions with their answers.", then each exemplar as
  `Question/A./B./C./D./Answer: X`, then "Now answer this question." and the target question. The exemplar set of
  pass k is `default_rng(10000+k).choice(285, 5, replace=False)` (any subject); every model uses the same sets.

**Canonical tokenization** (`src/canonical_tok.py`). The chat template from the snapshot (`tokenizer_config.json`
or `chat_template.jinja`) is rendered with jinja2 in the HF environment (trim_blocks, lstrip_blocks, sandbox,
raise_exception, strftime_now), with the template clock pinned to 2026-10-10 12:00 (granite-3.1 and Llama-3.2
print the date). Ids = `tokenizers.Tokenizer.from_file(<snapshot>/tokenizer.json).encode(text,
add_special_tokens=False)`. vLLM receives the ids as `TokensPrompt`, so it adds nothing, and the returned
`prompt_token_ids` are re-checked. Before the engine loads and again before generation, every prompt is asserted:

1. ids == canonical encoding;
2. every special added token in the text is one id, as often as it occurs;
3. decode(ids) == text, exactly or under one pinned allowance per model: `metaspace_space_after_special`
   (zephyr), `added_token_literal` (DeepSeek: "÷" etc. are added tokens; the ByteLevel decoder prints U+FFFD),
   `added_token_strip` (Phi-3/3.5: `<|user|>`/`<|end|>`/`<|assistant|>` have rstrip, the tokenizer's own
   behaviour);
4. the per-prompt hash equals an independent reference: transformers 4.46.3 `apply_chat_template(tokenize=True)`,
   Yi via the slow SentencePiece tokenizer, same pinned date (`src/reference_hashes.py`,
   `results/dryrun_v3/reference_hashes.json`);
5. the `tokenizer.json` sha256 and the snapshot revision equal the pins in `results/dryrun_v3/prompt_check.json`.

Completion texts are decoded with the canonical tokenizer (`texts`); vLLM's own text is kept as `texts_vllm`.

## Generation

vLLM 0.23.0 in the `gb10-vllm` container (torch 2.11.0+cu130, transformers 5.12.1, tokenizers 0.22.2, jinja2
3.1.6), started by `src/docker_run.sh` (`--memory=100g --memory-swap=100g`, offline HF cache). One container (one
engine) per model, in sequence; the process exits between models. The reference hashes use a separate venv
(transformers 4.46.3, tokenizers 0.20.3, sentencepiece 0.2.2).

* Engine: `dtype=bfloat16` for every model (some checkpoints ship fp16/fp32 weights; Mistral-Small-22B also runs
  in bf16), `tensor_parallel_size=1`, `gpu_memory_utilization` 0.25 (≤2B), 0.30 (3–4B), 0.40 (6–9B), 0.48
  (Mistral-Small-22B), `max_model_len=min(3072, model limit)` (phi-2: 2048), engine `seed=0`,
  `generation_config="vllm"` (the model's `generation_config.json` defaults are not applied), prefix caching on,
  `max_num_seqs=128`, `tokenizer_mode=hf`. A watchdog (`src/memwatch.sh`, log `logs/mem.log`) enforces a 100 GB
  total-used cap.
* SamplingParams: `n=1, temperature=T, top_p=1.0, top_k=0` (0 = disabled in vLLM 0.23; -1 is mapped to 0),
  `min_p=0, repetition_penalty=1.0, presence_penalty=0, frequency_penalty=0`, `max_tokens=8` (MMLU, HellaSwag) or
  `256` (GSM8K), `logprobs=20` (multiple-choice tasks), `skip_special_tokens=True`, no stop strings beyond EOS.
  Greedy = `temperature=0` (no seed).
* Logprobs are vLLM's `raw_logprobs` (log-softmax of the logits before temperature): the model's distribution,
  the same at every temperature.

**Deterministic flags (v3 runs, `collect.py --plan v3`).** vLLM 0.23 selects kernels by timing at every engine
start, so two starts on the same prompts can give different first-token logprobs, in discrete steps (phi-2: 0.127
nats; 2/20 letters flipped). The v3 runs therefore use:

- env: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`,
  `VLLM_BATCH_INVARIANT=1`;
- engine: `enforce_eager=True` (no torch.compile, no CUDA graphs), `kernel_config={"enable_flashinfer_autotune":
  False}`, `async_scheduling=False`, `num_gpu_blocks_override=2048` (32,768 tokens);
- bf16; all 4 rotations of a model in one `generate` call.

Both are recorded per run (`deterministic`, `engine_kwargs`). `run_v3.sh` drops `VLLM_BATCH_INVARIANT`, then the
block override, only for a model that fails with them (`logs/model_status.tsv`); no R2 model needed this.

Three-start tests (20 questions x 4 rotations): with eager mode and no autotune only (`results/dryrun_det/`),
phi-2 and Yi are identical, Qwen2.5-7B replies are identical but logprobs differ (0.375), and DeepSeek differs in
3/80 letters. With the full flag set (`results/dryrun_det2/`), DeepSeek is identical (max diff 0) and Qwen2.5-7B
replies are identical, but one start differs in 39/80 cells (top-1 logprob up to 0.019, tokens above -5 up to
0.48). Generation is therefore not fully deterministic across engine starts, and R2 is run as two replicates.

## Conditions and seeds

| condition | what is shared inside pass k | seeds |
|---|---|---|
| `iid` (self-consistency) | nothing: every (question, pass) is an independent sample | `seed[k,i] = base + k*N + i` |
| `sharedseed` (S1) | the sampling noise | `seed[k,i] = base + k` for all i |
| `sharedprompt` (S2) | the 5-exemplar prompt prefix | independent, as `iid` |
| `greedy` | n/a | none |

`base = (crc32(run_id) & 0x7FFFFFFF) << 20` (`src/collect.py::run_seed_base`). All seeds are stored (`seeds`, K×N).

**How vLLM uses the seed (S1).** In vLLM 0.23 V1, a request with `seed` gets its own
`torch.Generator(device="cuda").manual_seed(seed)` (`gpu_model_runner.py`). Sampling with any seeded request goes
through `TopKTopPSampler.forward_native` → `random_sample`: at each decode step the request's row of a
vocabulary-sized tensor `q` is filled by `q[i].exponential_(generator=gen_i)` and the token is `argmax(probs / q)`
(the exponential race, equivalent to Gumbel-max), with `probs = softmax(logits / T)`. Two requests with the same
seed and model therefore draw the same exponential vector at every step, regardless of batch position or prompt:
in S1 every question of pass k sees the same noise vector over the vocabulary. Across passes the seeds differ.

**Check.** At T=1 the first token is `argmax_j(log p_j + g_j)`. Within one pass, a common noise vector g must
satisfy every question's choice; this is a system of difference constraints over the top-20 tokens, feasible iff
its graph has no negative cycle. Qwen2.5-7B MMLU: S1 is feasible in 100/100 passes, the iid run in 0/100. (The
check script, `verify_shared_seed.py`, is in git history.)

## R2: rotated options, two replicates

24 models x 500 MMLU questions x 4 rotations, greedy (`results/run_r2_replicates.sh`). Rotation r shows original
option j at position (j + r) mod 4; rotation 0 is the original order. Each run (`mmlu_greedyrot_<model>`) stores,
per (rotation, question): `parsed` (shown letter), `chosen_option` (original index, -1 = unparseable), `gold_rot`,
`correct`, `letter_logprob_first` (shown positions) and `letter_logprob_first_by_option` (re-indexed to the
original options).

Replicates with separate engine starts, run model by model, A then B: A = `results/r2/` (primary),
B = `results/r2_rep/`. Rotation 0 of A is the greedy MMLU ensemble (`results/mmlu_ensemble_greedy.npz`), so all
24 rows come from one pipeline. Ensembles: `results/mmlu_ensemble_rotations.npz` (A) and
`results/mmlu_ensemble_rotations_rep.npz` (B): `correct` [24, 500, 4], `chosen_option` [24, 500, 4], `parsed`,
`letter_logprob_first_by_option`.

`src/r2_analysis.py` writes `results/r2/a_vs_b.json` (replicate agreement per model and rotation, flips on near
ties), `results/r2/near_ties.json` (near ties per model and rotation, replicate A) and
`results/r2/v2_vs_r2_rotation0.json` (v2 greedy, re-parsed with v3.1, vs A rotation 0; v2 inputs in
`results/provenance/v2_greedy/`).

**Near tie.** A cell (model, question, rotation) is a near tie when its top-two first-token letter logprobs differ
by less than 0.13 nats. `near_ties.json` counts replicate A only; `a_vs_b.json` and
`results/analysis/v3_C5_near_ties.json` use "gap < 0.13 in A or in B". The bf16 first-token logprobs lie on a grid
of about 0.125 nats: in replicate A, 97% of the letter gaps below 0.6 are within 0.01 nats of a multiple of 0.125
(a few lie on 0.0625 steps). "Gap < 0.13" thus selects the gaps 0 and ~0.125 (one grid step): 1,291 of the 46,527
finite gaps lie in [0.12, 0.13), and 46 in (0.005, 0.12).

## Scoring

**Multiple choice: parser v3.1** (`src/scoring.py`, scorer sha256[:16] `f804dbf14ef19530`). The rule is the
module docstring of `src/scoring.py`, with the four clarifications below. In short:

- Stage 1: the leading uppercase letter X after the optional wrappers "Answer:", "(", "**". Its tail decides:
  - reject a hedge or list: "A or D", "A, C", "A/B", "B C D";
  - reject a negation (words in any case): is/are/was/were not|incorrect|wrong|false,
    does/do/did/would/should/could/will not, can not, the n't forms, cannot/can't/won't; exception: the tail is
    option X's own text;
  - accept end, punctuation, or a newline;
  - reject any other character right after X, i.e. one that is not whitespace, punctuation (Unicode P*) or a
    symbol (S*): a letter, a digit, a combining mark, a control character (reason code `letter_then_alnum`);
  - after a space: accept punctuation, "is"/"would be", or option X's text. Otherwise B/C/D are accepted, and A
    only before is/was/would/because/since/as/for/seems/appears; else `article_ambiguous` ("A patient ...").
- Stage 1b: a standalone lowercase letter ("b", "b)", "(b)", "b.").
- Stage 2: in the first line, "answer is X", "answer would be X", "answer: X", or "Option X"/"Choice X" at the
  line start. The phrase words match in any case ("ANSWER IS", "option", "CHOICE"); X must be uppercase.
- Unparseable = wrong, counted. Each cell stores a reason code (`parse_reason`).

The docstring is not edited, because `scorer_sha256` (recorded in every run) is the sha256 of the whole file.
Where it is less exact than the code, the code's behaviour is:

1. Stage 2: "Option"/"Choice" at the line start, and "answer is", "answer would be", "answer:", match in any case.
2. Stage 1: the negation words match in any case ("D IS NOT", "B Does not").
3. Stage 1: the docstring says "a letter or a digit"; the code rejects any character after X that is not
   whitespace, P* or S* (`letter_then_alnum`).
4. GSM8K (`extract_final_number`): see below.

Tests: `src/test_scoring.py` (pytest + Hypothesis).

**GSM8K.** The value is the number of the last match of "#### <number>" (`####`, optional whitespace, a number);
a `####` without a number after it is skipped. Only if there is no such match is the last number in the text
used. Commas and `$` are ignored; correct if `|x - gold| < 1e-6`; no number = unparseable (counted).

**Rescore.** The sampled runs were re-scored from their saved texts with v3.1:
`src/rescore.py --skip-condition greedy --backup <dir>` (log `results/logs/rescore_v31.json`). No sampled run
changed by more than 0.5 accuracy points; the largest changes are Mistral-7B-v0.3 MMLU iid 80 cells (+0.10 pts),
Phi-3-mini MMLU iid 45 cells (+0.07), Mistral-v0.3 HellaSwag 11 cells (-0.09).

## Saved per run

`results/<task>_<condition>[-T<temp>]_<model-slug>.npz` + `.json`:

`correct` (K×N int8), `texts` (raw completion text, every sample), `parsed`, `finish_reason`, `n_gen_tokens`,
`cum_logprob`, `seeds`, `question_rows`, `gold`; for multiple-choice tasks also `letter_logprob_first` (K×N×4:
logsumexp over the top-20 tokens at generated position 0 whose text is the letter, NaN if absent),
`letter_logprob_at_letter` (the same at the first generated token that is a letter; `letter_token_pos`),
`top_ids_first`/`top_lp_first` (top-21 at position 0; token strings in `<run_id>.tokens.json`); for S2 also
`exemplar_dev_rows` (K×5). v3 runs also save the prompt ids (flat + offsets), a per-prompt hash, the generated
ids, and the tokenizers/jinja2/transformers versions. The JSON holds every setting, the model revision hash,
versions, runtime, accuracy, unparseable rate, `n_A_word_replies` (replies "A" + space + lowercase word), and
`N_b` (questions with 0.05 < p̂ < 0.95).

Pass k of the matrix is the k-th sample of each question. Batches of passes are checkpointed to
`<run_id>.partial.npz`; a restart resumes there with the same seeds.

`results/manifest.json` holds the sha256 of every A run (`runs`), every B run (`runs_rep_B`) and the three
ensemble files (`ensemble_files_sha256`); `results/manifest.md` lists every run and the R2 table (accuracy per
rotation, unparseable %, `article_ambiguous`, letter shares, near ties).

## Runs

| model | runs |
|---|---|
| Qwen2.5-7B-Instruct | MMLU iid K=100 T=1.0; S1 K=100; S2 K=100; MMLU iid K=30 at T=0.5, 1.5; HellaSwag iid K=50; GSM8K iid K=30 (CoT, 256 tokens; also 1024 tokens, below) |
| Mistral-7B-Instruct-v0.3, Meta-Llama-3-8B-Instruct | MMLU iid K=100; HellaSwag iid K=50 (Llama also S2 K=100) |
| SmolLM2-1.7B-Instruct, Phi-3-mini-4k-instruct | MMLU iid K=100 |
| all 24 ensemble models (`src/plan.py`) | R2: MMLU greedy, 4 rotations, replicates A and B |

**GSM8K at 1024 tokens** (`gsm8k_iidlong`, `max_tokens=1024`). At the specified 256 tokens, 55% of Qwen's CoT
replies are truncated (`length_truncated_rate`), so the last number is often an intermediate value (accuracy 0.51
vs 0.90 at 1024). The 256-token run is kept as specified.

## Integrity checks and outcomes

* Prompts: the reference hash agrees at 48,000/48,000 R2 prompts (24 models x 4 rotations x 500); in every R2 run
  all 2000 prompts equal it (N = 500, K = 4, parser v3.1). `src/verify_prompts.py` re-checked every prompt of
  every sampled run (all passes, including the 100 S2 exemplar sets): all canonical. The sampled runs were
  therefore not re-run.
* R2 completion: all 48 runs (24 models x replicates A and B) finished on the first attempt with the full flag
  set. Mean accuracy over the 24 models in replicate A: 0.586 / 0.584 / 0.575 / 0.573 for rotations 0-3.
* Replicates: 23 of 24 models are identical in every letter. Qwen2-7B differs in 6 cells (3 each in rotations 2
  and 3) and in 16 texts; all 6 flips are near ties (6 flips in 48,000 cells). Letter logprobs differ between A
  and B in 7 models (max 1.12 for Qwen2-7B; 0.50 for Qwen2.5-7B and Mistral-7B-v0.2; 0.23 for DeepSeek; about 0.1
  for Llama-3.2-1B and Meta-Llama-3-8B; Ministral-8B in 4 cells, at most 2e-4) and are identical in the other 17
  (`a_vs_b.json`, `max_abs_diff_letter_logprob`).
* Near ties per rotation range from 0-3 cells (Qwen2.5-3B) to 83-104 (phi-2) (`near_ties.json`).
* v2 greedy vs A rotation 0 (letter agreement): 0.940-0.998 for the 21 models with canonical v2 prompts; OLMo-2
  0.844, Yi-1.5 0.880, DeepSeek 0.448 for the three without (DeepSeek accuracy 0.348 -> 0.454 with canonical
  prompts).
* Memory: peak used 82.4 GB over the sampled collection and 86.2 GB over R2 (cap 100 GB; in R2 the guard never acted).
  A first dry run at `gpu_memory_utilization=0.7` reached 103 GiB and was stopped before writing output.

## Known limits and incidents

* Generation is not bit-invariant across engine starts or batch compositions (floating-point reduction order).
  Seeds fix the noise, not the kernels; R2 measures the effect with two replicates.
* bf16 for every model, whatever the checkpoint's dtype.
* Llama-3.1-8B-Instruct: engine init hit a transient CUDA illegal-instruction error during CUDA-graph capture;
  the in-process retry used `tokenizer_mode=auto` (recorded in `engine_kwargs`). Prompts are token ids, so only
  detokenization is affected.
* deepseek-llm-7b-chat: vLLM's detokenizer leaves byte-level markers (`Ġ`, `Ċ`) in 105/500 greedy texts
  (e.g. `B.ĠThe`); replacing them changes no parsed letter.
* No model needed the HF-transformers fallback (container `gb10-prm-sk`, same prompts and settings, recorded as
  `backend`).
