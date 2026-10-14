"""
Tests for scripts/v3_H_followup.py against stored step-1 null draws and the stored G1 result.
Run: uv run python -m pytest scripts/test_v3_H_followup.py
"""

from __future__ import annotations

import itertools
import json
import os
import re
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import numpy.testing as npt
import pytest
from scipy.special import expit

import v3_across as V3
import v3_G_familywise as G
import v3_H_followup as H
import v3_softpref as V
from v2lib import OUT

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ctx():
    return H.real_context()


@pytest.mark.parametrize("null", ["2pl_post", "rasch_jml"])
def test_h1_simulator_reproduces_paper_null_draws(ctx, null):
    """Captured matrices reproduce the stored step-1 draws exactly; each is 391 split 0/1 columns."""
    X, E = H.null_matrices(ctx, null, V.SEED_FAMILY, 30, f"t_{null}")
    stored = np.load(OUT / "raw_v3_F_family_eigs.npz")[f"full_ge4__{null}"][:30]
    npt.assert_array_equal(E, stored)
    assert X.shape == (30, 24, 391)
    assert set(np.unique(X).tolist()) <= {0, 1}
    for k in range(30):
        npt.assert_array_equal(V3.corr_eigs(X[k].astype(np.float64)), stored[k])
        c = X[k].mean(axis=0)
        assert np.all((c > 0) & (c < 1))
    assert V3.corr_eigs is not None and V3.corr_eigs.__name__ == "corr_eigs"      # patch restored


def test_h1_refit_on_a_captured_dataset_uses_all_columns(ctx):
    X, _ = H.null_matrices(ctx, "rasch_jml", H.seed_h(99, 1), 2, "t2")
    c = V3.Context(X[0])
    assert int(c.keep.sum()) == 391


def test_h2_enumeration():
    names = V.variant_data("full")["names"]
    subs = H.h2_subsets(names)
    pairs = [s for s in subs if s["kind"] == "pair"]
    refs = [s for s in subs if s["kind"] == "reference"]
    assert len(pairs) == 4 and len(refs) == 190 and len(subs) == 194
    for p in H.P1 + H.P2:
        assert names.count(p) == 1
    for s in pairs:
        assert len(s["keep"]) == 22
        assert sum(n in s["removed"] for n in H.P1) == 1 and sum(n in s["removed"] for n in H.P2) == 1
    others = [n for n in names if n not in H.P1 + H.P2]
    assert len(others) == 20
    assert {frozenset(s["removed"]) for s in refs} == {frozenset(c) for c in itertools.combinations(others, 2)}
    for s in refs:
        assert len(s["keep"]) == 22 and not set(s["removed"]) & set(H.P1 + H.P2)
        assert s["keep"] == [n for n in names if n in s["keep"]]
    assert len({tuple(s["keep"]) for s in subs}) == 194


def test_g1_full_data_reproduced_by_same_code_path():
    """G1 all-4 p_T 0.026 / 0.041 from score_context + run_nulls + familywise with the G1 seed."""
    ref = json.loads((OUT / "v3_G_familywise.json").read_text())["G1_familywise"]["by_score_and_null"]["ge4"]
    d = V.variant_data("full")
    c = G.score_context(d, "ge4", H.NULLS)
    E = G.run_nulls(c, list(H.NULLS), G.seed_g(G.STREAM_G["g1"], G.score_code("ge4")), G.N_G1, "tG1")
    for nm in H.NULLS:
        fw = G.familywise(c.obs, E[nm])
        assert fw["p_T"] == ref[nm]["p_T"]
        assert fw["valid"] == ref[nm]["valid"]
    assert round(ref["2pl_post"]["p_T"], 3) == 0.026 and round(ref["rasch_jml"]["p_T"], 3) == 0.041


def test_seed_namespace_is_new():
    lits = set()
    for p in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "src").glob("*.py")):
        if p.name in ("v3_H_followup.py", "test_v3_H_followup.py"):
            continue
        lits |= {int(m.group(0).replace("_", "")) for m in re.finditer(r"\b\d[\d_]{3,}\b", p.read_text())}
    assert G.SEED_G in lits and V.SEED_F in lits                     # the scan sees the earlier namespaces
    assert all(abs(H.SEED_H - s) > 10 ** 8 for s in lits if s > 10 ** 6)
    assert 0 < H.SEED_H < 2 ** 32
    keys = [(H.STREAM_H["h1_truth"], t + 1) for t in range(2)]
    keys += [(H.STREAM_H["h1_null"], t + 1, r + 1) for t in range(2) for r in range(H.N_H1)]
    keys += [(H.STREAM_H["h2_null"], i + 1) for i in range(194)]
    assert len(keys) == len(set(keys))
    with pytest.raises(ValueError):
        H.seed_h(0)


def test_expected_2pl_explicit_loop(ctx):
    fit = ctx.fit_2pl()
    keep = ctx.keep
    P = H.expected_2pl(fit, keep)
    a, b = fit.node_params()
    idx = np.flatnonzero(keep)
    rng = np.random.default_rng(0)
    for i in rng.choice(idx.size, 5, replace=False):
        w = np.exp(fit.log_post[idx[i]])
        w /= w.sum()
        for m in (0, 7, 23):
            npt.assert_allclose(P[m, i], np.sum(w * expit(a * (fit.theta[m] - b))), rtol=1e-12)
    assert P.shape == (24, 391) and np.all((P > 0) & (P < 1))


def test_items_cover_jobs_once():
    h1, h2 = H.h1_items(), H.h2_items()
    assert len(h1) == 600 and len(h2) == 194
    assert len({repr(j) for _, j in h1 + h2}) == 794
