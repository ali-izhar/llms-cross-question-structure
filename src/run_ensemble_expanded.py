"""
Expanded ensemble experiment: 16 models (6 original + 10 new) run deterministically
with consistent chat-template methodology.

Methodological change from original run_ensemble.py: uses tokenizer.apply_chat_template()
for all models, which is the standard for instruction-tuned MMLU evaluation. The original
raw-prompt approach worked for Qwen/Mistral/Phi-2 (which generalize to non-chat contexts)
but fails for newer chat-strict models (SmolLM2, OLMo-2, Phi-3.5). For scientific
consistency we re-run the 6 original models here too.

Saves:
- ensemble_expanded_correctness.npy     (M × N matrix, grown incrementally)
- ensemble_expanded_models.json         (model name list in order)
- ensemble_expanded_metrics.json        (per-model metrics + current ensemble stats)
- ensemble_expanded_stream.jsonl        (one JSON line per model, for real-time monitoring)
"""

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Load .env (HF_TOKEN for gated models)
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT.parent / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).parent))
from mmlu_utils import check_answer, build_mmlu_prompt, load_questions_from_saved

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS = Path(__file__).parent.parent / "results"

# All 16 models to run (6 original + 10 new), in run order.
# Small models first so we fail fast if something's wrong before burning time on 7Bs.
MODELS = [
    # -- smaller / lighter (fast sanity signal) --
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",       # NEW - HF, FineWeb-Edu + Cosmopedia
    "Qwen/Qwen2.5-0.5B-Instruct",                # ORIG - Alibaba
    "google/gemma-2-2b-it",                      # NEW - Google Gemma-2
    "Qwen/Qwen2.5-1.5B-Instruct",                # ORIG
    "microsoft/phi-2",                           # ORIG - Microsoft Phi-2
    "microsoft/Phi-3-mini-4k-instruct",          # NEW - Microsoft Phi-3
    "microsoft/Phi-3.5-mini-instruct",           # NEW - Microsoft Phi-3.5
    "Qwen/Qwen2.5-3B-Instruct",                  # ORIG
    # -- 7B class --
    "Qwen/Qwen2-7B-Instruct",                    # NEW - older Qwen gen
    "Qwen/Qwen2.5-7B-Instruct",                  # ORIG
    "mistralai/Mistral-7B-Instruct-v0.2",        # NEW - older Mistral
    "mistralai/Mistral-7B-Instruct-v0.3",        # ORIG
    "HuggingFaceH4/zephyr-7b-beta",              # NEW - Mistral DPO
    "teknium/OpenHermes-2.5-Mistral-7B",         # NEW - Mistral GPT-4 distill
    "allenai/OLMo-2-1124-7B-Instruct",           # NEW - AllenAI fully open
    # -- 8-9B --
    "meta-llama/Llama-3.1-8B-Instruct",          # NEW - Meta Llama-3.1 (gated but accessible)
    "google/gemma-2-9b-it",                      # NEW - Google Gemma-2 9B (gated)
    # -- ROUND 2: more same-size diversity --
    "01-ai/Yi-1.5-6B-Chat",                      # NEW - 01.AI Yi family
    "deepseek-ai/deepseek-llm-7b-chat",          # NEW - DeepSeek family
    "ibm-granite/granite-3.1-8b-instruct",       # NEW - IBM Granite
    "internlm/internlm2_5-7b-chat",              # NEW - Shanghai AI Lab (custom code)
    # -- ROUND 3: newer Mistral gated models (license accepted) --
    "mistralai/Ministral-8B-Instruct-2410",      # NEW - newer Mistral architecture (Oct 2024)
    "mistralai/Mistral-Small-Instruct-2409",     # NEW - Mistral Small 22B
    # -- ROUND 4: Meta Llama + Google Gemma + Cohere Aya (all gated, tokens refreshed) --
    "meta-llama/Llama-3.2-1B-Instruct",          # NEW - Meta small
    "meta-llama/Llama-3.2-3B-Instruct",          # NEW - Meta mid
    "meta-llama/Meta-Llama-3-8B-Instruct",       # NEW - Meta Llama-3 (original)
    "CohereForAI/aya-23-8B",                     # NEW - Cohere multilingual
    # SKIPPED: google/gemma-2-27b-it — same family as Gemma-2-2B/9B, 2hr for ~1 dim, low marginal value
]


def build_chat_prompt(tokenizer, question: str, choices: list[str]) -> str:
    """Build a chat-formatted prompt. Wraps build_mmlu_prompt content in the model's
    chat template. Falls back to raw content if the tokenizer has no chat_template."""
    user_content = build_mmlu_prompt(question, choices)
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning(f"apply_chat_template failed, falling back to raw: {e}")
    return user_content


def run_single_model(model_name: str, questions: list[dict]) -> tuple[np.ndarray, dict]:
    """Run one model deterministically with chat template. Returns correctness + metrics."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    t_load = time.time()
    logger.info(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # GB10 has unified memory (121 GiB shared CPU/GPU). Without this hint,
    # device_map="auto" conservatively flags big models for CPU offload, which
    # routes inference through CPU kernels (~16× slower) despite weights living
    # in the same physical RAM. Forcing a large GPU memory budget keeps all
    # layers on GPU kernels. "100GiB" leaves ~20 GiB headroom for system/activations.
    max_memory = {0: "100GiB", "cpu": "20GiB"}

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map="auto", max_memory=max_memory,
        )
    except Exception as e:
        logger.warning(f"Standard load failed, trying trust_remote_code=True: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map="auto", max_memory=max_memory,
            trust_remote_code=True,
        )
    model.eval()

    load_time = time.time() - t_load
    n_params = sum(p.numel() for p in model.parameters())
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))

    N = len(questions)
    correctness = np.zeros(N, dtype=np.int8)
    extracted = []
    no_output = 0

    t_infer = time.time()
    for i, q in enumerate(tqdm(questions, desc=model_name.split("/")[-1], file=sys.stdout)):
        prompt = build_chat_prompt(tokenizer, q["question"], q["choices"])
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        if not answer:
            no_output += 1

        pred_letter = answer[:1].upper() if answer else "?"
        correct = (
            answer.upper().startswith(q["correct_letter"])
            or check_answer(answer, q["correct_text"])
        )
        correctness[i] = int(correct)
        extracted.append(pred_letter)

    infer_time = time.time() - t_infer
    accuracy = float(correctness.mean())

    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    metrics = {
        "model": model_name,
        "n_parameters": int(n_params),
        "has_chat_template": has_chat_template,
        "load_time_sec": round(load_time, 1),
        "inference_time_sec": round(infer_time, 1),
        "accuracy": accuracy,
        "no_output_count": no_output,
        "extracted_letter_counts": {
            letter: extracted.count(letter) for letter in ["A", "B", "C", "D", "?"]
        },
    }

    logger.info(
        f"  -> {model_name}: acc={accuracy:.3f}, "
        f"params={n_params/1e9:.2f}B, "
        f"load={load_time:.0f}s, infer={infer_time:.0f}s, "
        f"chat_template={has_chat_template}, no_output={no_output}"
    )
    return correctness, metrics


def compute_ensemble_metrics(C: np.ndarray, names: list[str]) -> dict:
    """Given an M × N correctness matrix, compute key ensemble statistics."""
    M, N = C.shape
    out = {"M": M, "N": N, "models": names}

    out["accuracies"] = {n: float(C[i].mean()) for i, n in enumerate(names)}

    if M >= 2:
        R = np.corrcoef(C)
        off = R[np.triu_indices(M, k=1)]
        out["pairwise_corr"] = {
            "mean": float(off.mean()),
            "min": float(off.min()),
            "max": float(off.max()),
        }

    col_sum = C.sum(axis=0)
    out["all_correct"] = int((col_sum == M).sum())
    out["all_wrong"] = int((col_sum == 0).sum())
    out["disagreement_questions"] = int(((col_sum > 0) & (col_sum < M)).sum())

    disagree = (col_sum > 0) & (col_sum < M)
    if disagree.sum() > 0 and M >= 3:
        C_dis = C[:, disagree].astype(np.float64)
        # Question-level covariance spectrum (what the paper reports)
        # Compute via M × M Gram matrix (shares nonzero eigenvalues)
        C_centered = C_dis - C_dis.mean(axis=0, keepdims=True)
        gram = C_centered @ C_centered.T
        eig = np.sort(np.linalg.eigvalsh(gram))[::-1]
        eig = eig[eig > 1e-10]
        if len(eig) > 0:
            p = eig / eig.sum()
            H = -np.sum(p * np.log(p + 1e-12))
            out["d_across"] = float(np.exp(H))
            out["eigenvalues"] = [float(v) for v in eig]
            out["top_eigenvalue_variance_fraction"] = float(p[0])

    return out


def main():
    t_start = time.time()

    logger.info("=" * 70)
    logger.info("EXPANDED ENSEMBLE EXPERIMENT — chat-template methodology, 16 models")
    logger.info("=" * 70)

    questions = load_questions_from_saved(
        RESULTS / "questions_Qwen_Qwen2.5-7B-Instruct.json"
    )
    N = len(questions)
    logger.info(f"Loaded {N} questions")
    logger.info(f"Running {len(MODELS)} models sequentially")

    # Checkpoint / output files
    CKPT_MATRIX = RESULTS / "ensemble_expanded_correctness.npy"
    CKPT_MODELS = RESULTS / "ensemble_expanded_models.json"
    CKPT_METRICS = RESULTS / "ensemble_expanded_metrics.json"
    STREAM_LOG = RESULTS / "ensemble_expanded_stream.jsonl"

    # Resume support: if checkpoint exists, skip already-run models
    if CKPT_MATRIX.exists() and CKPT_MODELS.exists():
        C_all = np.load(CKPT_MATRIX)
        with open(CKPT_MODELS) as f:
            done_models = json.load(f)
        logger.info(f"Resuming: {len(done_models)} models already completed")
    else:
        C_all = np.zeros((0, N), dtype=np.int8)
        done_models = []

    all_per_model_metrics = []

    for idx, model_name in enumerate(MODELS):
        if model_name in done_models:
            logger.info(f"[{idx+1}/{len(MODELS)}] SKIP (already done): {model_name}")
            continue

        logger.info("-" * 70)
        logger.info(f"[{idx+1}/{len(MODELS)}] {model_name}")
        logger.info("-" * 70)

        try:
            correctness, metrics = run_single_model(model_name, questions)
        except Exception as e:
            logger.error(f"FAILED {model_name}: {e}", exc_info=True)
            with open(STREAM_LOG, "a") as f:
                f.write(json.dumps({
                    "model": model_name, "status": "failed", "error": str(e)[:500],
                    "timestamp": time.time(),
                }) + "\n")
            continue

        # Append and checkpoint
        C_all = np.vstack([C_all, correctness.reshape(1, -1)])
        done_models.append(model_name)
        all_per_model_metrics.append(metrics)

        np.save(CKPT_MATRIX, C_all)
        with open(CKPT_MODELS, "w") as f:
            json.dump(done_models, f, indent=2)

        # Compute and save ensemble metrics at this M
        current = compute_ensemble_metrics(C_all, done_models)
        with open(CKPT_METRICS, "w") as f:
            json.dump({
                "per_model": all_per_model_metrics,
                "current_ensemble": current,
            }, f, indent=2)

        # Stream entry
        entry = {
            "idx": idx + 1,
            "model": model_name,
            "status": "ok",
            "accuracy": metrics["accuracy"],
            "no_output": metrics["no_output_count"],
            "M_after": len(done_models),
            "d_across": current.get("d_across"),
            "disagreement": current.get("disagreement_questions"),
            "elapsed_min": round((time.time() - t_start) / 60, 1),
            "timestamp": time.time(),
        }
        with open(STREAM_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

        d_str = f"{entry['d_across']:.2f}" if entry["d_across"] is not None else "N/A (M<3)"
        logger.info(
            f">>> M={len(done_models)}: "
            f"d_across={d_str}, "
            f"disagreement={entry['disagreement']}/{N}, "
            f"cumulative time={entry['elapsed_min']:.1f} min"
        )

    # Final summary
    total = (time.time() - t_start) / 60
    logger.info("=" * 70)
    logger.info(f"DONE. Total time: {total:.1f} min. Final M={len(done_models)}")
    final = compute_ensemble_metrics(C_all, done_models)
    d_final = final.get("d_across")
    logger.info(f"Final d_across = {d_final:.3f}" if d_final is not None else "Final d_across = N/A")
    eig_str = [f"{v:.1f}" for v in final.get("eigenvalues", [])[:10]]
    logger.info(f"Top eigenvalues: {eig_str}")
    logger.info(f"Disagreement questions: {final.get('disagreement_questions')}/{N}")


if __name__ == "__main__":
    main()
