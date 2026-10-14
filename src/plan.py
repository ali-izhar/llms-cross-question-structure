"""Run plan: the runs of each model (one container per model), run ids, and the option rotation."""

from __future__ import annotations

ENSEMBLE_MODELS = [
    "HuggingFaceTB/SmolLM2-1.7B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/phi-2", "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.3", "HuggingFaceH4/zephyr-7b-beta",
    "allenai/OLMo-2-1124-7B-Instruct", "01-ai/Yi-1.5-6B-Chat", "deepseek-ai/deepseek-llm-7b-chat",
    "ibm-granite/granite-3.1-8b-instruct", "mistralai/Ministral-8B-Instruct-2410", "google/gemma-2-2b-it",
    "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-2-9b-it", "mistralai/Mistral-Small-Instruct-2409",
    "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct",
]

QWEN7 = "Qwen/Qwen2.5-7B-Instruct"
MISTRAL3 = "mistralai/Mistral-7B-Instruct-v0.3"
LLAMA3 = "meta-llama/Meta-Llama-3-8B-Instruct"
SMOL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
PHI3 = "microsoft/Phi-3-mini-4k-instruct"


def R(task: str, condition: str, K: int, temperature: float) -> dict:
    return {"task": task, "condition": condition, "K": K, "temperature": temperature}


GREEDY = R("mmlu", "greedy", 1, 0.0)
# greedy MMLU on the 4 cyclic option rotations; pass k = rotation k
GREEDYROT = R("mmlu", "greedyrot", 4, 0.0)


def v3_runs(model: str) -> list[dict]:
    if model not in ENSEMBLE_MODELS:
        raise KeyError(f"{model} is not in ENSEMBLE_MODELS (no v3 runs)")
    # rotation 0 is the greedy run: one deterministic pipeline for all 24 rows
    return [GREEDYROT]


def positions_to_options(values, r: int) -> list:
    """Per-position values (A-D at rotation r) by original option: out[j] = values[(j + r) mod 4]."""
    return [values[(j + r) % 4] for j in range(4)]


def rotate(choices: list, r: int) -> list:
    """Rotation r moves option j to position (j + r) mod 4 (r = 0: original order)."""
    out = [None] * 4
    for j, c in enumerate(choices):
        out[(j + r) % 4] = c
    return out


PLAN: dict[str, list[dict]] = {
    QWEN7: [
        R("mmlu", "iid", 100, 1.0),
        R("mmlu", "sharedseed", 100, 1.0),
        R("mmlu", "sharedprompt", 100, 1.0),
        R("mmlu", "iid", 30, 0.5),
        R("mmlu", "iid", 30, 1.5),
        R("hellaswag", "iid", 50, 1.0),
        R("gsm8k", "iid", 30, 1.0),
        R("gsm8k", "iidlong", 30, 1.0),  # supplementary, max_tokens=1024
        GREEDY,
    ],
    MISTRAL3: [R("mmlu", "iid", 100, 1.0), R("hellaswag", "iid", 50, 1.0), GREEDY],
    LLAMA3: [R("mmlu", "iid", 100, 1.0), R("hellaswag", "iid", 50, 1.0),
             R("mmlu", "sharedprompt", 100, 1.0), GREEDY],
    SMOL: [R("mmlu", "iid", 100, 1.0), GREEDY],
    PHI3: [R("mmlu", "iid", 100, 1.0), GREEDY],
}
for _m in ENSEMBLE_MODELS:
    PLAN.setdefault(_m, [GREEDY])

MODEL_ORDER = [QWEN7, MISTRAL3, LLAMA3, SMOL, PHI3] + [m for m in ENSEMBLE_MODELS
                                                     if m not in (QWEN7, MISTRAL3, LLAMA3, SMOL, PHI3)]


def slug(model: str) -> str:
    return model.replace("/", "_")


def run_id(model: str, run: dict) -> str:
    cond = run["condition"] if run["condition"] in ("greedy", "greedyrot") else f"{run['condition']}-T{run['temperature']}"
    return f"{run['task']}_{cond}_{slug(model)}"
