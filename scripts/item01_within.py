"""Item 1: within-model MP test per iid run (also the base rows of items 2 and 12).

Borderline columns (eps = 0.05): N_b, lambda_+, lambda_max, TW z with its H1 Monte Carlo null,
strict / buffered MP counts, Horn counts, and the eps sweep.
Output: results/analysis/item01_within_table.json
"""

from __future__ import annotations

import zlib

import v2lib  # noqa: F401  (sets threads and sys.path)
import numpy as np
from v2lib import EPS, LABEL, RUNS, load_meta, matrix_path, write_json

import archive_claims
import horn_vs_mp
import make_epsilon_sweep_figure as epsfig
import tw_z_diagnostic as twd

N_MC = 1000
N_SURR = 200
N_EPS_NULL = 1000


def run_seed(run_id: str) -> int:
    return zlib.crc32(run_id.encode())


def within_stats(run_id: str) -> dict:
    path = matrix_path(run_id)
    C = np.load(path).astype(np.int8)                       # (K, N)
    K, N = C.shape
    meta = load_meta(run_id)
    seed = run_seed(run_id)
    model = twd.load_model(LABEL[run_id], str(path))      # absolute path overrides results/
    C_B = horn_vs_mp.borderline_filter(C.astype(np.float64), EPS)
    Nb = C_B.shape[1]
    p95, d_contig, d_any, eig = horn_vs_mp.horn_pa(C_B, N_SURR, seed)
    gamma = Nb / K
    lp = (1 + np.sqrt(gamma)) ** 2
    lp_km1 = (1 + np.sqrt(Nb / (K - 1))) ** 2
    sigma_tw = (1 + np.sqrt(gamma)) * (1 + 1 / np.sqrt(gamma)) ** (1 / 3) * K ** (-2 / 3)
    zs = twd.mc_h1_filter(model, np.random.default_rng(seed + 1), n_mc=N_MC)
    sweep = archive_claims.tw_z_eps_sweep(C, LABEL[run_id], n_draws=N_EPS_NULL, seed=seed + 2,
                                          epsilons=epsfig.EPSILONS)
    for row in sweep["by_eps"]:
        margin, s_tw, nb = epsfig.margin_and_tw(C, row["eps"])
        assert nb == row["N_b"]
        row["margin"] = float(margin)
        row["sigma_tw"] = float(s_tw)
    rec = {
        "run_id": run_id, "label": LABEL[run_id], "task": meta["task"], "condition": meta["condition"],
        "temperature": meta["temperature"], "K": int(K), "N": int(N),
        "accuracy": float(C.mean()), "accuracy_meta": meta["accuracy"],
        "unparseable_rate": meta["unparseable_rate"],
        "N_b": int(Nb), "N_b_meta": meta.get("N_b_eps0.05"), "gamma": gamma,
        "lambda_plus": float(lp), "lambda_plus_K_minus_1": float(lp_km1),
        "lambda_max": float(eig[0]), "top_eigs": eig[:5].tolist(),
        "margin": float(lp - eig[0]), "margin_K_minus_1": float(lp_km1 - eig[0]),
        "sigma_tw": float(sigma_tw), "tw_z": float(model["z_obs"]),
        "mc_null_z": {**twd.summary_stats(zs), "percentile_of_obs": twd.percentile_of(model["z_obs"], zs)},
        "mp_count_strict": horn_vs_mp.mp_signal(eig, lp, 1.0),
        "mp_count_buffered": horn_vs_mp.mp_signal(eig, lp, 1.05),
        "mp_count_strict_K_minus_1": horn_vs_mp.mp_signal(eig, lp_km1, 1.0),
        "horn_p95_rank1": float(p95[0]), "horn_count_contiguous": int(d_contig),
        "horn_count_any_above": int(d_any),
        "eps_sweep": sweep["by_eps"],
        "low_power_note": (f"N_b = {Nb} is small; the edge test has little power here (see item 3)."
                           if Nb < 30 else None),
    }
    assert abs(rec["accuracy"] - meta["accuracy"]) < 1e-4, (run_id, rec["accuracy"], meta["accuracy"])
    assert meta.get("N_b_eps0.05") in (None, Nb), (run_id, Nb, meta.get("N_b_eps0.05"))
    print(f"{LABEL[run_id]:40s} acc={rec['accuracy']:.4f} Nb={Nb:3d} l+={lp:.3f} lmax={eig[0]:.3f} "
          f"z={rec['tw_z']:+.2f} pct={rec['mc_null_z']['percentile_of_obs']:.3f} "
          f"MP={rec['mp_count_strict']} buf={rec['mp_count_buffered']} Horn={d_contig}")
    return rec


def main() -> None:
    rows = [within_stats(r) for r, _, g in RUNS if g == "iid"]
    write_json("item01_within_table", {
        "meta": {"eps": EPS, "n_mc_null_z": N_MC, "n_horn_surrogates": N_SURR,
                 "n_eps_sweep_null": N_EPS_NULL, "eps_sweep": epsfig.EPSILONS,
                 "seeds": "crc32(run_id) (+1 MC null, +2 eps-sweep null)",
                 "edge": "lambda_+ = (1+sqrt(N_b/K))^2; *_K_minus_1 fields use N_b/(K-1)",
                 "mc_null": "H1 of scripts/tw_z_diagnostic.py: independent Bernoulli(f_i) over all N columns, borderline filter re-applied; percentile = fraction of null z <= observed z"},
        "runs": rows,
    })


if __name__ == "__main__":
    main()
