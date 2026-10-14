"""Item 14: old data (results/provenance/) vs the v2 runs: question and gold-letter identity, accuracy,
N_b and pass-rate correlation per run, and the 24-model greedy matrix.

The `old_file` labels in the output keep the files' original archive/results/ paths.
Output: results/analysis/item14_integrity_diff.json
"""

from __future__ import annotations

import json

import v2lib  # noqa: F401  (threads, sys.path)
import numpy as np
from v2lib import ENSEMBLE_NAMES, ENSEMBLE_NPY, EPS, LABEL, ROOT, V2, matrix_path, write_json

from null_models import borderline_mask

OLD = ROOT / "results" / "provenance"
SLUGS = {"mmlu_iid-T1.0_Qwen_Qwen2.5-7B-Instruct": "Qwen_Qwen2.5-7B-Instruct",
         "mmlu_iid-T1.0_mistralai_Mistral-7B-Instruct-v0.3": "mistralai_Mistral-7B-Instruct-v0.3",
         "mmlu_iid-T1.0_meta-llama_Meta-Llama-3-8B-Instruct": "meta-llama_Meta-Llama-3-8B-Instruct",
         "mmlu_iid-T1.0_HuggingFaceTB_SmolLM2-1.7B-Instruct": "HuggingFaceTB_SmolLM2-1.7B-Instruct",
         "mmlu_iid-T1.0_microsoft_Phi-3-mini-4k-instruct": "microsoft_Phi-3-mini-4k-instruct"}
OLD_MATRIX = {**{r: f"correctness_matrix_{s}.npy" for r, s in SLUGS.items()},
              "hellaswag_iid-T1.0_Qwen_Qwen2.5-7B-Instruct": "correctness_matrix_hellaswag_Qwen_Qwen2.5-7B-Instruct.npy",
              "hellaswag_iid-T1.0_mistralai_Mistral-7B-Instruct-v0.3": "correctness_matrix_hellaswag_mistralai_Mistral-7B-Instruct-v0.3.npy",
              "hellaswag_iid-T1.0_meta-llama_Meta-Llama-3-8B-Instruct": "correctness_matrix_hellaswag_meta-llama_Meta-Llama-3-8B-Instruct.npy",
              "gsm8k_iid-T1.0_Qwen_Qwen2.5-7B-Instruct": "correctness_matrix_gsm8k_medium_Qwen_Qwen2.5-7B-Instruct.npy",
              "mmlu_iid-T0.5_Qwen_Qwen2.5-7B-Instruct": "correctness_matrix_tau0.5.npy",
              "mmlu_iid-T1.5_Qwen_Qwen2.5-7B-Instruct": "correctness_matrix_tau1.5.npy"}


def summary(C: np.ndarray) -> dict:
    return {"K": int(C.shape[0]), "N": int(C.shape[1]), "accuracy": float(C.mean()),
            "N_b": int(borderline_mask(C, EPS).sum())}


def main() -> None:
    true = json.loads((V2 / "questions_mmlu.json").read_text())
    q_ids = []
    for r, s in SLUGS.items():
        old = json.loads((OLD / f"questions_{s}.json").read_text())
        dq = [i for i, (o, t) in enumerate(zip(old, true)) if o["question"][:100] != t["question"][:100]]
        dg = [i for i, (o, t) in enumerate(zip(old, true)) if o["correct_letter"] != t["gold_letter"]]
        q_ids.append({"run": LABEL[r], "old_file": f"archive/results/questions_{s}.json", "n_positions": len(old),
                      "n_question_prefix100_differs": len(dq), "n_gold_letter_differs": len(dg),
                      "note": ("old files keep only a 100-char prefix, so a prefix-collision question "
                               "(same prefix, different row) is invisible here; a differing gold letter "
                               "at an identical prefix is its detectable signature")})
    runs = {}
    for r, f in OLD_MATRIX.items():
        Co = np.load(OLD / f).astype(np.int8)
        Cn = np.load(matrix_path(r)).astype(np.int8)
        rec = {"label": LABEL[r], "old_file": f"archive/results/{f}", "old": summary(Co), "v2": summary(Cn)}
        if Co.shape[1] == Cn.shape[1]:
            rec["pass_rate_corr_old_vs_v2"] = float(np.corrcoef(Co.mean(0), Cn.mean(0))[0, 1])
        runs[r] = rec
    sem = np.load(OLD / "semantic_full_correctness.npy")
    runs["old_qwen_letter_run_semantic_full"] = {"old": summary(sem), "note": "old second Qwen K=100 run with letters (paper App. answer identity)"}
    # HellaSwag item identity.
    hs_true = json.loads((V2 / "questions_hellaswag.json").read_text())
    hs_old = json.loads((OLD / "questions_hellaswag.json").read_text())
    hs = {"n": len(hs_true), "n_ind_differs": sum(o["ind"] != t["ind"] for o, t in zip(hs_old, hs_true)),
          "n_gold_differs": sum(o["correct_letter"] != t["gold_letter"] for o, t in zip(hs_old, hs_true))}
    # 24-model greedy.
    names = json.loads(ENSEMBLE_NAMES.read_text())
    old_names = json.loads((OLD / "ensemble_expanded_models.json").read_text())
    assert names == old_names
    Eo = np.load(OLD / "ensemble_expanded_correctness.npy").astype(np.int8)
    En = np.load(ENSEMBLE_NPY).astype(np.int8)
    ens = {n: {"old_acc": float(Eo[m].mean()), "v2_acc": float(En[m].mean()),
               "agreement": float((Eo[m] == En[m]).mean())} for m, n in enumerate(names)}
    write_json("item14_integrity_diff", {
        "mmlu_question_identity_old_runs_vs_true_set": q_ids, "hellaswag_identity": hs,
        "within_model_runs_old_vs_v2": runs, "ensemble_greedy_old_vs_v2": ens,
        "ensemble_split_columns": {"old": int(((Eo.sum(0) > 0) & (Eo.sum(0) < 24)).sum()),
                                   "v2": int(((En.sum(0) > 0) & (En.sum(0) < 24)).sum())},
        "caveat": ("Old and v2 runs differ in prompt protocol (raw vs chat template), decoding defaults "
                   "(top-k 20/50 and repetition penalty in old runs; none in v2), scorer, and for the later "
                   "old runs the question set; differences are not attributable to one cause."),
    })
    for q in q_ids:
        print(q)
    for r, d in runs.items():
        print(d.get("label", r), d["old"], d.get("v2"), d.get("pass_rate_corr_old_vs_v2"))
    print(hs)


if __name__ == "__main__":
    main()
