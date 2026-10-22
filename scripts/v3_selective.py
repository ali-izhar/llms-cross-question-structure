"""K: selective prediction on the corrected data (spec: results/analysis/specs/K_selective.md).

Does an ensemble of two peer models (greedy, agreement with the target's greedy answer) predict whether the
target's greedy MMLU answer is correct better than 100-sample self-consistency (SC_100), and at what cost?
Primary target Qwen2.5-7B; secondary: the other four models with a K = 100 sampled MMLU run.
Also reproduces the 546a25c numbers from the 546a25c artifacts with the 546a25c (label-using) definitions.
Output: results/analysis/v3_selective.json
"""

from __future__ import annotations

import io
import itertools
import json
import subprocess

import v2lib  # noqa: F401
import numpy as np
from scipy.stats import norm, rankdata
from v2lib import ROOT, V2, letters_to_codes, write_json

from null_models import family_of

SEED = 20261022
N_BOOT = 4000
N_SUBSETS = 2000
N_SC_SUBSETS = 200
KS_SC = (2, 5, 10, 20, 50, 100)
KS_ENS = (1, 2, 3, 5, 10)
OLD_COMMIT = "546a25c"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
TARGETS = {  # model -> sampled run id
    QWEN: "mmlu_iid-T1.0_Qwen_Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct": "mmlu_iid-T1.0_meta-llama_Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3": "mmlu_iid-T1.0_mistralai_Mistral-7B-Instruct-v0.3",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "mmlu_iid-T1.0_HuggingFaceTB_SmolLM2-1.7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct": "mmlu_iid-T1.0_microsoft_Phi-3-mini-4k-instruct",
}
ORIGINAL_PAIR = ["meta-llama/Llama-3.1-8B-Instruct", "google/gemma-2-9b-it"]
PARAMS_B = {  # as in 546a25c:scripts/selective_prediction.py
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 1.7, "Qwen/Qwen2.5-0.5B-Instruct": 0.5, "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
    "microsoft/phi-2": 2.7, "microsoft/Phi-3-mini-4k-instruct": 3.8, "microsoft/Phi-3.5-mini-instruct": 3.8,
    "Qwen/Qwen2.5-3B-Instruct": 3.0, "Qwen/Qwen2-7B-Instruct": 7.0, "Qwen/Qwen2.5-7B-Instruct": 7.0,
    "mistralai/Mistral-7B-Instruct-v0.2": 7.0, "mistralai/Mistral-7B-Instruct-v0.3": 7.0,
    "HuggingFaceH4/zephyr-7b-beta": 7.0, "allenai/OLMo-2-1124-7B-Instruct": 7.0, "01-ai/Yi-1.5-6B-Chat": 6.0,
    "deepseek-ai/deepseek-llm-7b-chat": 7.0, "ibm-granite/granite-3.1-8b-instruct": 8.0,
    "mistralai/Ministral-8B-Instruct-2410": 8.0, "google/gemma-2-2b-it": 2.6,
    "meta-llama/Llama-3.1-8B-Instruct": 8.0, "google/gemma-2-9b-it": 9.2,
    "mistralai/Mistral-Small-Instruct-2409": 22.0, "meta-llama/Llama-3.2-1B-Instruct": 1.2,
    "meta-llama/Llama-3.2-3B-Instruct": 3.2, "meta-llama/Meta-Llama-3-8B-Instruct": 8.0,
}


# ── metrics ──────────────────────────────────────────────────────────
def auroc(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """AUROC of score(s) s (N,) or (N, J) for binary y (N,); ties count 1/2 (Mann-Whitney)."""
    S = s.reshape(len(y), -1)
    r = rankdata(S, axis=0)
    n1 = y.sum()
    n0 = len(y) - n1
    a = (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return a if s.ndim == 2 else a[0]


def aupr(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision, step-wise as sklearn: sum over distinct thresholds of (R_n - R_{n-1}) P_n."""
    o = np.argsort(-s, kind="mergesort")
    ys, ss = y[o], s[o]
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]               # last index of each distinct score
    tp = np.cumsum(ys)[last]
    prec = tp / (last + 1)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def delong(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> tuple[float, float]:
    """DeLong test for two correlated AUROCs (Sun & Xu 2014 structural components). Returns (z, two-sided p)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    V10, V01 = [], []
    for s in (s1, s2):
        x, z = s[pos], s[neg]
        cmp = (x[:, None] > z[None, :]) + 0.5 * (x[:, None] == z[None, :])
        V10.append(cmp.mean(axis=1))
        V01.append(cmp.mean(axis=0))
    S10, S01 = np.cov(np.array(V10)), np.cov(np.array(V01))
    var = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n
    d = V10[0].mean() - V10[1].mean()
    if var <= 0:
        return 0.0, 1.0
    zz = d / np.sqrt(var)
    return float(zz), float(2 * norm.sf(abs(zz)))


def ci(a: np.ndarray) -> list[float]:
    return np.percentile(a, [2.5, 97.5]).tolist()


# ── scores ───────────────────────────────────────────────────────────
def modal_agreement(codes: np.ndarray) -> np.ndarray:
    """codes (K, N) in -1..3 -> max over options of count / K (unparseable counts in K, never the mode)."""
    return np.stack([(codes == j).mean(axis=0) for j in range(4)]).max(axis=0)


def cost(models: list[str]) -> float:
    return float(sum(PARAMS_B[m] for m in models) / 7.0)


def p1_largest_family_diverse(target: str, names: list[str], k: int) -> list[str]:
    """Label-free: peers outside the target's family, by parameter count (ties: id), one per family per round."""
    cand = sorted((m for m in names if family_of(m) != family_of(target)), key=lambda m: (-PARAMS_B[m], m.lower()))
    fams: dict[str, list[str]] = {}
    for m in cand:
        fams.setdefault(family_of(m), []).append(m)
    order = [m for rnd in itertools.zip_longest(*fams.values()) for m in rnd if m is not None]
    return order[:k]


def l1_family_diverse_by_acc(target: str, names: list[str], acc: dict, k: int) -> list[str]:
    """Label-using (546a25c rule): best-accuracy member per family, other families first, round-robin."""
    fams: dict[str, list[str]] = {}
    for m in names:
        if m != target:
            fams.setdefault(family_of(m), []).append(m)
    for f in fams:
        fams[f].sort(key=lambda m: -acc[m])
    order_f = sorted(fams, key=lambda f: (f == family_of(target), -acc[fams[f][0]]))
    order = [m for rnd in itertools.zip_longest(*(fams[f] for f in order_f)) for m in rnd if m is not None]
    return order[:k]


def l2_topk_by_acc(target: str, names: list[str], acc: dict, k: int) -> list[str]:
    return sorted((m for m in names if m != target), key=lambda m: -acc[m])[:k]


# ── reproduction of 546a25c ──────────────────────────────────────────
def git_bytes(path: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{OLD_COMMIT}:{path}"], check=True,
                          capture_output=True).stdout


def reproduce_old() -> dict:
    sc = np.load(io.BytesIO(git_bytes("results/correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy")))
    ens = np.load(io.BytesIO(git_bytes("results/ensemble_expanded_correctness.npy")))
    names = json.loads(git_bytes("results/ensemble_expanded_models.json"))
    old = json.loads(git_bytes("results/selective_prediction.json"))["headline"]
    y = ens[names.index(QWEN)].astype(int)
    out = {"note": "546a25c artifacts, 546a25c definitions: SC = max(p, 1-p) with p = fraction of samples CORRECT; "
                   "Ens = fraction of peers CORRECT (both use gold labels at inference; the paper text describes "
                   "answer agreement instead). Only 0/1 correctness was saved at 546a25c, so the stated "
                   "answer-agreement scores cannot be computed on the old data.",
           "target_accuracy": float(y.mean())}
    for K in KS_SC:
        p = sc[:K].mean(axis=0)
        out[f"SC{K}_code_def"] = float(auroc(y, np.maximum(p, 1 - p)))
        out[f"SC{K}_oracle"] = float(auroc(y, p))
    s1 = ens[names.index(ORIGINAL_PAIR[0])].astype(float)
    s2 = ens[[names.index(m) for m in ORIGINAL_PAIR]].mean(axis=0)
    p100 = sc.mean(axis=0)
    sc100 = np.maximum(p100, 1 - p100)
    out["Ens1_Llama31_code_def"] = float(auroc(y, s1))
    out["Ens2_Llama31_Gemma9_code_def"] = float(auroc(y, s2))
    out["delong_Ens1_vs_SC100"] = delong(y, s1, sc100)
    out["delong_Ens2_vs_SC100"] = delong(y, s2, sc100)
    out["paper_values"] = {"SC100": 0.712, "Ens1": 0.749, "Ens2": 0.807, "Ens1_delong_p": "~0.2",
                           "Ens2_delong_p": "<0.005"}
    out["saved_546a25c_headline"] = {k: v for k, v in old.items() if "AUROC" in k}
    return out


# ── one target ───────────────────────────────────────────────────────
def analyse_target(target: str, run_id: str, E: dict, rng: np.random.Generator, full: bool) -> dict:
    names, CO, CR = E["names"], E["chosen"], E["correct"]
    t = names.index(target)
    y = CR[t, :, 0].astype(int)
    a = CO[t, :, 0]
    acc = {m: float(CR[j, :, 0].mean()) for j, m in enumerate(names)}
    peers = [m for m in names if m != target]
    agree = (CO == a[None, :, None]) & (a[None, :, None] >= 0)               # (M, N, 4) content agreement
    codes = letters_to_codes(np.load(V2 / f"{run_id}.npz")["parsed"])        # (100, N)
    corr_s = np.load(V2 / f"{run_id}.npz")["correct"]
    assert codes.shape == (100, 500)

    S: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}

    def add(name: str, s: np.ndarray, c: float, models: list[str] | None = None, **kw) -> None:
        S[name] = s.astype(float)
        meta[name] = {"cost": c, "models": models, **kw}

    for K in KS_SC:
        add(f"SC{K}", modal_agreement(codes[:K]), float(K), kind="label_free")
        add(f"SC{K}_greedyagree", (codes[:K] == a[None]).mean(axis=0) * (a >= 0), float(K), kind="label_free")
        p = corr_s[:K].mean(axis=0)
        add(f"SC{K}_codedef_LABEL", np.maximum(p, 1 - p), float(K), kind="label_using")
        add(f"SC{K}_oracle_LABEL", p, float(K), kind="label_using")

    def ens_block(rule: str, models: list[str], kind: str) -> None:
        idx = [names.index(m) for m in models]
        k = len(models)
        add(f"{rule}_k{k}_rot0", agree[idx, :, 0].mean(axis=0), cost(models), models, kind=kind)
        for r in (1, 2, 3):
            add(f"{rule}_k{k}_rot{r}", agree[idx, :, r].mean(axis=0), cost(models), models, kind=kind)
        add(f"{rule}_k{k}_rot4", agree[idx].mean(axis=(0, 2)), 4 * cost(models), models, kind=kind)
        if kind == "label_using":
            add(f"{rule}_k{k}_codedef_LABEL", CR[idx, :, 0].mean(axis=0), cost(models), models, kind=kind)

    for k in KS_ENS:
        ens_block("P1", p1_largest_family_diverse(target, names, k), "label_free")
        ens_block("L1acc", l1_family_diverse_by_acc(target, names, acc, k), "label_using")
        ens_block("L2topacc", l2_topk_by_acc(target, names, acc, k), "label_using")
    ens_block("P2all", peers, "label_free")
    if target == QWEN:
        ens_block("L0orig", ORIGINAL_PAIR[:1], "label_using_post_hoc")
        ens_block("L0orig", ORIGINAL_PAIR, "label_using_post_hoc")

    names_s = list(S)
    Smat = np.stack([S[n] for n in names_s], axis=1)                          # (N, J)
    point = auroc(y, Smat)
    scores = {}
    for j, n in enumerate(names_s):
        scores[n] = {"AUROC": float(point[j]), "AUPR": aupr(y, Smat[:, j]), **meta[n]}

    # rotation-robust mean over r = 1..3 as a single score per ensemble rule (AUROC averaged over r)
    for n in [n for n in names_s if n.endswith("_rot1")]:
        base = n[:-5]
        scores[base + "_rotmean123"] = {"AUROC": float(np.mean([scores[f"{base}_rot{r}"]["AUROC"] for r in (1, 2, 3)])),
                                        "cost": scores[n]["cost"], "models": scores[n]["models"], "kind": scores[n]["kind"]}

    # all pairs / singles (P3) and random subsets
    pidx = [names.index(m) for m in peers]
    pairs = list(itertools.combinations(pidx, 2))
    pair_rot0 = np.stack([agree[list(p), :, 0].mean(axis=0) for p in pairs], axis=1)
    pair_rotm = [np.stack([agree[list(p), :, r].mean(axis=0) for p in pairs], axis=1) for r in (1, 2, 3)]
    sing_rot0 = agree[pidx, :, 0].T.astype(float)
    tf = family_of(target)
    fam_ok = np.array([family_of(names[i]) != family_of(names[j]) and tf not in (family_of(names[i]), family_of(names[j]))
                       for i, j in pairs])
    sc100 = S["SC100"]
    p3 = {}

    def dist(A: np.ndarray, mask: np.ndarray | None = None, with_delong: bool = True) -> dict:
        au = auroc(y, A)
        if mask is not None:
            au, A = au[mask], A[:, mask]
        d = {"n": int(au.size), "mean": float(au.mean()), "q05_q50_q95": np.percentile(au, [5, 50, 95]).tolist(),
             "frac_above_SC100": float((au > scores["SC100"]["AUROC"]).mean())}
        if with_delong:
            res = [delong(y, A[:, j], sc100) for j in range(A.shape[1])]
            d["frac_sig_above_SC100"] = float(np.mean([(z > 0) and (p < 0.05) for z, p in res]))
            d["frac_sig_below_SC100"] = float(np.mean([(z < 0) and (p < 0.05) for z, p in res]))
        return d

    p3["singles_rot0"] = dist(sing_rot0)
    p3["pairs_rot0"] = dist(pair_rot0)
    p3["pairs_rot0_family_diverse_P3prime"] = dist(pair_rot0, fam_ok)
    p3["pairs_rotmean123_meanAUROC"] = float(np.mean([auroc(y, A).mean() for A in pair_rotm]))
    for k in (3, 5, 10):
        subs = [rng.choice(pidx, k, replace=False) for _ in range(N_SUBSETS)]
        p3[f"random_k{k}_rot0"] = dist(np.stack([agree[s, :, 0].mean(axis=0) for s in subs], axis=1),
                                       with_delong=full)
        p3[f"random_k{k}_rot0"]["mean_cost"] = float(np.mean([cost([names[i] for i in s]) for s in subs]))
    p3["pairs_mean_cost"] = float(np.mean([cost([names[i], names[j]]) for i, j in pairs]))
    sc_sub = {}
    for K in KS_SC[:-1]:
        sc_sub[f"SC{K}_random_subsets_meanAUROC"] = float(np.mean(
            [auroc(y, modal_agreement(codes[rng.choice(100, K, replace=False)])) for _ in range(N_SC_SUBSETS)]))

    # bootstrap (paired over questions): named scores, the mean pair AUROC (rot0, rot1-3 mean)
    key_cmp = [n for n in names_s if n.startswith(("P1_k2", "P1_k1", "P2all", "L0orig", "L1acc_k2", "L2topacc_k2"))
               and not n.endswith("codedef_LABEL")] + ["SC10", "SC100_greedyagree", "SC100_codedef_LABEL"]
    if "L1acc_k2_codedef_LABEL" in S:
        key_cmp.append("L1acc_k2_codedef_LABEL")
    boot_au = np.empty((N_BOOT, len(names_s)))
    boot_pair0 = np.empty(N_BOOT)
    boot_pairr = np.empty(N_BOOT)
    N = len(y)
    for b in range(N_BOOT):
        ii = rng.integers(0, N, N)
        yb = y[ii]
        boot_au[b] = auroc(yb, Smat[ii])
        if full:
            boot_pair0[b] = auroc(yb, pair_rot0[ii]).mean()
            boot_pairr[b] = np.mean([auroc(yb, A[ii]).mean() for A in pair_rotm])
    j100 = names_s.index("SC100")
    for j, n in enumerate(names_s):
        scores[n]["AUROC_CI95"] = ci(boot_au[:, j])
    comps = {}
    for n in key_cmp:
        j = names_s.index(n)
        z, p = delong(y, S[n], sc100)
        diff = boot_au[:, j] - boot_au[:, j100]
        comps[f"{n}_vs_SC100"] = {"delta": float(point[j] - point[j100]), "delta_CI95": ci(diff),
                                  "delong_z": z, "delong_p": p, "cost_factor_SC100_over": 100.0 / meta[n]["cost"]}
    for base in ("P1_k2", "P1_k1", "P2all_k23", "L0orig_k2", "L0orig_k1", "L1acc_k2"):
        if f"{base}_rot1" not in S:
            continue
        js = [names_s.index(f"{base}_rot{r}") for r in (1, 2, 3)]
        diff = boot_au[:, js].mean(axis=1) - boot_au[:, j100]
        comps[f"{base}_rotmean123_vs_SC100"] = {
            "delta": float(point[js].mean() - point[j100]), "delta_CI95": ci(diff),
            "delong_p_per_rotation": [delong(y, S[f"{base}_rot{r}"], sc100)[1] for r in (1, 2, 3)]}
        j0 = names_s.index(f"{base}_rot0")
        comps[f"{base}_rot0_minus_rotmean123"] = {"delta": float(point[j0] - point[js].mean()),
                                                  "delta_CI95": ci(boot_au[:, j0] - boot_au[:, js].mean(axis=1))}
    if full:
        comps["P3_meanpair_rot0_vs_SC100"] = {"delta": p3["pairs_rot0"]["mean"] - float(point[j100]),
                                              "delta_CI95": ci(boot_pair0 - boot_au[:, j100])}
        comps["P3_meanpair_rotmean123_vs_SC100"] = {"delta": p3["pairs_rotmean123_meanAUROC"] - float(point[j100]),
                                                    "delta_CI95": ci(boot_pairr - boot_au[:, j100])}

    # letter-preference diagnostic: peer agreement on the target's WRONG answers, rot 0 vs rot 1-3
    wrong = (y == 0) & (a >= 0)
    diag = {"n_target_wrong_parsed": int(wrong.sum()),
            "mean_peer_agreement_on_wrong_rot0": float(agree[pidx][:, wrong, 0].mean()),
            "mean_peer_agreement_on_wrong_rot123": float(agree[pidx][:, wrong, 1:].mean()),
            "mean_peer_agreement_on_right_rot0": float(agree[pidx][:, y == 1, 0].mean()),
            "mean_peer_agreement_on_right_rot123": float(agree[pidx][:, y == 1, 1:].mean())}

    def passes(c: dict, need_delong: bool) -> bool:
        ok = c["delta"] > 0 and c["delta_CI95"][0] > 0
        return bool(ok and (c["delong_p"] < 0.05 if need_delong else True))

    decision = {"i_P1_Ens2_rot0": passes(comps["P1_k2_rot0_vs_SC100"], True),
                "iii_P1_Ens2_rotmean123": passes(comps["P1_k2_rotmean123_vs_SC100"], False),
                "P1_Ens2_cost_factor": 100.0 / meta["P1_k2_rot0"]["cost"]}
    if full:
        decision["ii_P3_meanpair_rot0"] = passes(comps["P3_meanpair_rot0_vs_SC100"], False)
        decision["extra_P3_meanpair_rotmean123"] = passes(comps["P3_meanpair_rotmean123_vs_SC100"], False)
    return {"target": target, "sampled_run": run_id, "accuracy_greedy_rot0": float(y.mean()), "n_correct": int(y.sum()),
            "target_greedy_unparseable": int((a < 0).sum()), "scores": scores, "comparisons_vs_SC100": comps,
            "P3_subsets": p3, "SC_random_subsets": sc_sub, "letter_confound_diagnostic": diag,
            "decision_conditions": decision}


def load_ensemble(path: str) -> dict:
    d = np.load(V2 / path)
    return {"names": [str(m) for m in d["models"]], "chosen": d["chosen_option"].astype(int),
            "correct": d["correct"].astype(int), "gold": d["gold_idx"]}


def main() -> None:
    rng = np.random.default_rng(SEED)
    old = reproduce_old()
    print("546a25c reproduction:", {k: round(v, 3) for k, v in old.items() if isinstance(v, float)})
    EA = load_ensemble("mmlu_ensemble_rotations.npz")
    EB = load_ensemble("mmlu_ensemble_rotations_rep.npz")
    out = {"meta": {"spec": "results/analysis/specs/K_selective.md", "seed": SEED, "n_boot": N_BOOT,
                    "n_random_subsets": N_SUBSETS, "post_hoc": True, "doc": (__doc__ or "").strip().splitlines()[0]},
           "reproduction_546a25c": old, "targets": {}}
    for target, run in TARGETS.items():
        res = analyse_target(target, run, EA, rng, full=True)
        out["targets"][target] = res
        sc = res["scores"]
        print(target, "acc", round(res["accuracy_greedy_rot0"], 3), "SC100", round(sc["SC100"]["AUROC"], 3),
              "P1k2", round(sc["P1_k2_rot0"]["AUROC"], 3), sc["P1_k2_rot0"]["models"],
              "P1k2 rot123", round(sc["P1_k2_rotmean123"]["AUROC"], 3), res["decision_conditions"])
    rb = analyse_target(QWEN, TARGETS[QWEN], EB, rng, full=False)
    out["replicate_B_qwen"] = {"scores": {k: rb["scores"][k] for k in ("SC100", "P1_k2_rot0", "P1_k2_rotmean123",
                                                                        "L0orig_k2_rot0", "L1acc_k2_rot0")},
                               "comparisons_vs_SC100": rb["comparisons_vs_SC100"],
                               "decision_conditions": rb["decision_conditions"]}
    write_json("v3_selective", out)


if __name__ == "__main__":
    main()
