# results/provenance

Byte-identical copies of files from earlier versions that current code reads to reproduce a current output.
Earlier versions are in git history.

| file(s) | what it is | read by |
|---|---|---|
| `questions_Qwen_Qwen2.5-7B-Instruct.json` | question list (100-char prefix, subject, gold letter) of the original Qwen2.5-7B MMLU run | `src/prepare_questions.py` (defines `results/questions_mmlu.json`), `scripts/item14_integrity.py`, `src/test_mmlu_utils.py` |
| `questions_hellaswag.json` | the 200 HellaSwag items of the original runs | `src/prepare_questions.py` (defines `results/questions_hellaswag.json`), `scripts/item14_integrity.py` |
| `questions_{mistralai_Mistral-7B-Instruct-v0.3, meta-llama_Meta-Llama-3-8B-Instruct, HuggingFaceTB_SmolLM2-1.7B-Instruct, microsoft_Phi-3-mini-4k-instruct}.json` | question lists of the other old MMLU runs | `scripts/item14_integrity.py` |
| `correctness_matrix_*.npy` (11 files) | K x N correctness of the old within-model runs (MMLU x5, HellaSwag x3, GSM8K 256 tokens, Qwen T = 0.5 / 1.5) | `scripts/item14_integrity.py`; the five MMLU ones also `scripts/test_v2lib.py`, `scripts/test_horn_vs_mp.py` |
| `semantic_full_correctness.npy` | correctness of the old second Qwen2.5-7B letter run | `scripts/item14_integrity.py` |
| `ensemble_expanded_correctness.npy`, `ensemble_expanded_models.json` | the old 24-model greedy matrix and its model order | `scripts/item10_pairs.py` (`old_data_reproduction`), `scripts/item14_integrity.py`, `scripts/test_v2lib.py`, `src/test_null_models.py`; the model order also `src/build_manifest.py` (docstring) |
| `v2_greedy/mmlu_greedy_<model>.npz` (21) | v2 greedy MMLU replies of the 21 models with canonical v2 prompts | `src/r2_analysis.py` (`results/r2/v2_vs_r2_rotation0.json`, which `src/build_manifest.py` copies into the manifest) |
| `v2_broken_tokenization/mmlu_greedy_<model>.npz` (3) | v2 greedy MMLU replies of DeepSeek-LLM-7B, Yi-1.5-6B, OLMo-2-7B (non-canonical prompt ids) | `src/r2_analysis.py` |

Output fields that name these files (`old_file` in `results/analysis/item14_integrity_diff.json`, `v2_source` in
`results/r2/v2_vs_r2_rotation0.json` and `results/manifest.*`) give the `archive/results/...` paths the files had when
the outputs were made.
