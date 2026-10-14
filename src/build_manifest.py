"""Build results/manifest.json + manifest.md (per-run sha256 and summaries) and the ensemble matrices.

mmlu_ensemble_rotations[_rep].npz: correct / chosen_option / parsed [M, 500, 4 rotations], gold_idx [500];
mmlu_ensemble_greedy.npz = rotation 0. Unchanged ensemble files are not rewritten, so their sha256 is stable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plan import ENSEMBLE_MODELS, GREEDYROT, PLAN, run_id  # noqa: E402

KEYS = ["run_id", "model", "task", "condition", "K", "N", "temperature", "accuracy", "unparseable_rate",
        "length_truncated_rate", "N_b_eps0.05", "n_always_correct", "n_always_wrong", "prompt_mode",
        "backend", "model_revision", "container_image", "vllm_version", "torch_version",
        "transformers_version", "sampling_params", "seed_scheme", "seed_base", "engine_kwargs",
        "runtime_sec", "engine_load_sec", "first_token_letter_logprob_coverage", "parsed_letter_counts",
        "logprobs_mode", "dry_run", "memory_settings", "scorer_sha256", "n_A_word_replies",
        "parse_reason_counts", "tokenization", "accuracy_by_rotation", "unparseable_by_rotation",
        "letter_share_by_rotation", "deterministic", "parser_version"]


def _sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _savez_if_changed(path: Path, **arrays) -> bool:
    """Save unless the file holds exactly these arrays (a rewrite changes the zip time stamp); True if written."""
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            same = set(z.files) == set(arrays) and all(
                z[k].dtype == np.asarray(v).dtype and np.array_equal(z[k], np.asarray(v)) for k, v in arrays.items())
        if same:
            return False
    np.savez_compressed(path, **arrays)
    return True


def build_rotation_ensemble(src: Path, d: Path, out_name: str):
    """Stack the rotation runs of one replicate; assert the rotation mapping in every cell."""
    rot_rows, rot_names = [], []
    for m in ENSEMBLE_MODELS:
        p = src / f"{run_id(m, GREEDYROT)}.npz"
        if p.exists():
            rot_rows.append(np.load(p))
            rot_names.append(m)
    if not rot_rows:
        return None, None, []
    q = json.loads((d / "questions_mmlu.json").read_text())
    gidx = np.array([x["gold_idx"] for x in q])
    for m, z in zip(rot_names, rot_rows, strict=True):
        K, N = z["correct"].shape
        assert K == 4 and N == len(q), m
        for r in range(4):
            assert all(str(z["gold_rot"][r, i]) == "ABCD"[(gidx[i] + r) % 4] for i in range(N)), (m, r)
            exp = np.array([("ABCD".index(L) - r) % 4 if L else -1 for L in z["parsed"][r]])
            assert (z["chosen_option"][r] == exp).all(), (m, r)
            assert (z["correct"][r] == (z["chosen_option"][r] == gidx)).all(), (m, r)
    CR = np.stack([z["correct"].T for z in rot_rows]).astype(np.int8)          # [M, N, 4]
    CH = np.stack([z["chosen_option"].T for z in rot_rows]).astype(np.int8)
    PR = np.stack([z["parsed"].T for z in rot_rows])
    LP = np.stack([z["letter_logprob_first_by_option"].transpose(1, 0, 2) for z in rot_rows])
    _savez_if_changed(d / out_name, correct=CR, chosen_option=CH, parsed=PR,
                      letter_logprob_first_by_option=LP.astype(np.float32),
                      gold_idx=gidx.astype(np.int8), models=np.array(rot_names),
                      source_dir=np.array(str(src.relative_to(d))),
                      rotation_rule=np.array("rotation r shows option j at position (j + r) mod 4"))
    info = {"file": out_name, "source_dir": str(src.relative_to(d)), "M": len(rot_names), "N": int(CR.shape[1]),
            "models": rot_names, "mean_accuracy": float(CR.mean()),
            "mean_accuracy_by_rotation": [float(CR[:, :, r].mean()) for r in range(4)],
            "missing_models": [m for m in ENSEMBLE_MODELS if m not in rot_names]}
    return info, CR, rot_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE.parent / "results"))
    ap.add_argument("--r2-dir", default=None, help="R2 rotation runs, replicate A (default: <dir>/r2)")
    ap.add_argument("--rep-dir", default=None, help="R2 replicate B (default: <dir>/r2_rep)")
    args = ap.parse_args()
    d = Path(args.dir)
    r2 = Path(args.r2_dir) if args.r2_dir else d / "r2"
    r2b = Path(args.rep_dir) if args.rep_dir else d / "r2_rep"
    img = {}
    for name in ("gb10-vllm", "gb10-prm-sk"):
        try:
            img[name] = subprocess.check_output(["docker", "image", "inspect", name, "--format", "{{.Id}}"],
                                                text=True).strip()
        except Exception:
            img[name] = None

    runs, missing = [], []
    # sampled runs (results/) + rotation runs (results/r2/); rotation 0 replaces the v2 greedy runs
    expected = [(model, r, d) for model, plan in PLAN.items() for r in plan if r["condition"] != "greedy"]
    expected += [(m, GREEDYROT, r2) for m in ENSEMBLE_MODELS]
    for model, r, dd in expected:
        rid = run_id(model, r)
        js = dd / f"{rid}.json"
        if not (js.exists() and (dd / f"{rid}.npz").exists()):
            missing.append(rid)
            continue
        meta = json.loads(js.read_text())
        ci = meta.get("container_image")
        if ci in img:
            meta["container_image"] = f"{ci} ({img[ci]})"
        runs.append({k: meta.get(k) for k in KEYS} | {
            "npz": str((dd / f"{rid}.npz").relative_to(d)),
            "npz_sha256": _sha256(dd / f"{rid}.npz"), "json_sha256": _sha256(js)})
    # replicate B (separate engine starts)
    runs_b, missing_b = [], []
    for m in ENSEMBLE_MODELS:
        rid = run_id(m, GREEDYROT)
        nb, jb = r2b / f"{rid}.npz", r2b / f"{rid}.json"
        if not (nb.exists() and jb.exists()):
            missing_b.append(rid)
            continue
        mb = json.loads(jb.read_text())
        runs_b.append({"run_id": rid, "model": m, "replicate": "B", "npz": str(nb.relative_to(d)),
                       "npz_sha256": _sha256(nb), "json_sha256": _sha256(jb),
                       "accuracy_by_rotation": mb.get("accuracy_by_rotation"),
                       "scorer_sha256": mb.get("scorer_sha256"), "parser_version": mb.get("parser_version")})

    rot, CR, rot_names = build_rotation_ensemble(r2, d, "mmlu_ensemble_rotations.npz")
    rot_b, _, _ = build_rotation_ensemble(r2b, d, "mmlu_ensemble_rotations_rep.npz")

    # across-model greedy ensemble = replicate A rotation 0 (original option order)
    ens = None
    if rot_names:
        C = CR[:, :, 0].copy()
        p = C.mean(axis=0)
        ens = {"file": "mmlu_ensemble_greedy.npz", "source": f"R2 replicate A rotation 0 ({r2.relative_to(d)}/mmlu_greedyrot_*.npz)",
               "M": len(rot_names), "N": C.shape[1], "models": rot_names, "mean_accuracy": float(C.mean()),
               "N_b_eps0.05": int(((p > 0.05) & (p < 0.95)).sum()),
               "missing_models": [m for m in ENSEMBLE_MODELS if m not in rot_names]}
        _savez_if_changed(d / "mmlu_ensemble_greedy.npz", correct=C, models=np.array(rot_names),
                          source=np.array("R2 rotation 0"))
    nt_file = r2 / "near_ties.json"
    near = json.loads(nt_file.read_text())["models"] if nt_file.exists() else {}

    events = (d / "logs" / "events.log")
    ens_files = {f: _sha256(d / f) for f in ("mmlu_ensemble_greedy.npz", "mmlu_ensemble_rotations.npz",
                                             "mmlu_ensemble_rotations_rep.npz") if (d / f).exists()}
    manifest = {"runs": runs, "missing_runs": missing, "runs_rep_B": runs_b, "missing_runs_rep_B": missing_b,
                "ensemble_files_sha256": ens_files, "ensemble": ens, "ensemble_rotations": rot,
                "ensemble_rotations_rep": rot_b, "images": img,
                "r2_a_vs_b": json.loads((r2 / "a_vs_b.json").read_text()) if (r2 / "a_vs_b.json").exists() else None,
                "v2_greedy_vs_r2_rotation0": json.loads((r2 / "v2_vs_r2_rotation0.json").read_text())
                if (r2 / "v2_vs_r2_rotation0.json").exists() else None,
                "near_ties": near,
                "protocol": "PROTOCOL.md",
                "events": events.read_text().splitlines() if events.exists() else [],
                "model_status": (d / "logs" / "model_status.tsv").read_text().splitlines()
                if (d / "logs" / "model_status.tsv").exists() else []}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))

    lines = ["| model | task | condition | T | K | N | acc | unparse % | N_b | prompt | backend | runtime s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {r['model']} | {r['task']} | {r['condition']} | {r['temperature']} | {r['K']} | {r['N']} | "
                     f"{r['accuracy']:.4f} | {100 * r['unparseable_rate']:.2f} | {r['N_b_eps0.05']} | "
                     f"{r['prompt_mode']} | {r['backend']} | {r['runtime_sec']} |")
    if ens:
        lines.append(f"| ensemble ({ens['M']} models) | mmlu | greedy across models | 0 | {ens['M']} | {ens['N']} | "
                     f"{ens['mean_accuracy']:.4f} | - | {ens['N_b_eps0.05']} | - | - | - |")
    if rot:
        lines += ["", "## R2: greedy MMLU on the 4 option rotations (pass r = rotation r)", "",
                  "| model | acc r0 | acc r1 | acc r2 | acc r3 | acc mean | unparse % (r0-r3) | article_ambiguous | "
                  "letter share A/B/C/D % (all rotations) | near ties r0-r3 (gap < 0.13) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in runs:
            if r["condition"] != "greedyrot":
                continue
            a = r["accuracy_by_rotation"]
            u = r["unparseable_by_rotation"]
            sh = r["letter_share_by_rotation"]
            share = "/".join(f"{100 * np.mean([x[L] for x in sh]):.0f}" for L in "ABCD")
            aa = (r.get("parse_reason_counts") or {}).get("article_ambiguous", 0)
            nt = near.get(r["model"], {}).get("near_tie_by_rotation", "-")
            nt = "/".join(map(str, nt)) if isinstance(nt, list) else nt
            lines.append(f"| {r['model']} | " + " | ".join(f"{x:.3f}" for x in a) + f" | {np.mean(a):.3f} | "
                         + "/".join(f"{100 * x:.1f}" for x in u) + f" | {aa} | {share} | {nt} |")
    ab_file = r2 / "a_vs_b.json"
    if ab_file.exists():
        ab = json.loads(ab_file.read_text())
        lines += ["", f"## R2 replicate A vs B (separate engine starts; near tie = letter gap < {ab['eps']})", "",
                  "| model | letter agreement r0-r3 | correct agreement r0-r3 | letter flips | flips on near ties | "
                  "near-tie rate (all cells) | texts differ | max letter-logprob diff |",
                  "|---|---|---|---|---|---|---|---|"]
        for m, v in ab["models"].items():
            mx = v["max_abs_diff_letter_logprob"]
            lines.append(f"| {m} | " + "/".join(f"{x:.3f}" for x in v["letter_agreement_by_rotation"]) + " | "
                         + "/".join(f"{x:.3f}" for x in v["correct_agreement_by_rotation"])
                         + f" | {v['n_flips']} | {v['n_flips_on_near_tie']} | {v['near_tie_rate_all_cells']:.3f} | "
                         f"{v['n_texts_differ']} | {mx if mx is None else round(mx, 4)} |")
        lines.append(f"| total | | | {ab['total_flips']} | {ab['total_flips_on_near_tie']} | | | |")
    v2f = r2 / "v2_vs_r2_rotation0.json"
    if v2f.exists():
        v2 = json.loads(v2f.read_text())
        lines += ["", "## v2 greedy (re-parsed with v3.1) vs R2 replicate A rotation 0 (same questions, original order)", "",
                  "| model | letter agreement | letters differ | correct agreement | acc v2 | acc A r0 | v2 source |",
                  "|---|---|---|---|---|---|---|"]
        for m, v in v2.items():
            lines.append(f"| {m} | {v['letter_agreement']:.3f} | {v['n_letter_differ']} | {v['correct_agreement']:.3f} | "
                         f"{v['acc_v2_greedy_v31']:.3f} | {v['acc_r2_r0']:.3f} | {v['v2_source']} |")
    (d / "manifest.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"missing runs: {missing}")


if __name__ == "__main__":
    main()
