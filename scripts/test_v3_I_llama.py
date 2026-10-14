"""Tests for v3_I_llama.py against the stored step-1 draws (raw_v3_F_family_eigs.npz), the stored G1 result,
item07's per-model accuracy, explicit enumeration, and a scan of the repo's seed literals.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import numpy.testing as npt
import pytest

import v3_G_familywise as G
import v3_H_followup as H
import v3_I_llama as I
import v3_softpref as V
from v2lib import OUT

ROOT = Path(__file__).resolve().parents[1]


def test_23_model_data():
    d, full = I.data_23(), V.variant_data("full")
    assert len(d["names"]) == 23 and I.LLAMA_1B not in d["names"]
    assert full["names"].count(I.LLAMA_1B) == 1
    assert d["names"] == [n for n in full["names"] if n != "meta-llama/Llama-3.2-1B-Instruct"]
    a = G.subset_rows(full, d["names"])
    for k in ("CR", "chosen", "shown_choice"):
        npt.assert_array_equal(a[k], d[k])


def test_i2_subsets():
    names = V.variant_data("full")["names"]
    subs = I.i2_subsets(names)
    assert len(subs) == 4 and len({tuple(s["keep"]) for s in subs}) == 4
    for s in subs:
        assert len(s["keep"]) == 21 and len(s["removed"]) == 3 and I.LLAMA_1B in s["removed"]
        assert sum(n in s["removed"] for n in H.P1) == 1 and sum(n in s["removed"] for n in H.P2) == 1
        assert set(s["removed"]) <= set(names)
        assert s["keep"] == [n for n in names if n not in s["removed"]]


@pytest.mark.parametrize("score,nulls", [("rot0", ["3pl_post", "letter_post", "kg_uniform4", "2pl_post", "rasch_jml"]),
                                         ("ge4", ["2pl_post", "rasch_jml", "3pl_post", "ba_a4", "kg_letter4"])])
def test_23_model_code_path_reproduces_stored_draws(score, nulls):
    """23-model score_context + run_nulls with the step-1 seed reproduce the stored no_llama1b draws (first 20)."""
    z = np.load(OUT / "raw_v3_F_family_eigs.npz")
    T = json.loads((OUT / "v3_F_softpref.json").read_text())["step1_construction_family"]["table"]["no_llama1b"]
    ctx = G.score_context(I.data_23(), score, V.family_nulls(score))
    npt.assert_allclose(ctx.obs[:4], T[score]["top_eigs"], rtol=1e-12)
    for nm in nulls:
        E = G.run_nulls(ctx, [nm], V.SEED_FAMILY, 20, f"t_{score}_{nm}")[nm]
        npt.assert_allclose(E, z[f"no_llama1b_{score}__{nm}"][:20], rtol=1e-12, atol=1e-12)


def test_full_data_g1_reproduced():
    ref = json.loads((OUT / "v3_G_familywise.json").read_text())["G1_familywise"]["by_score_and_null"]["ge4"]
    d = V.variant_data("full")
    c = G.score_context(d, "ge4", I.I2_NULLS)
    E = G.run_nulls(c, list(I.I2_NULLS), G.seed_g(G.STREAM_G["g1"], G.score_code("ge4")), G.N_G1, "tG1")
    for nm in I.I2_NULLS:
        assert G.familywise(c.obs, E[nm])["p_T"] == ref[nm]["p_T"]
    assert round(ref["2pl_post"]["p_T"], 3) == 0.026 and round(ref["rasch_jml"]["p_T"], 3) == 0.041


def test_stored_reference_matches_hostile_round4():
    r = I.stored_reference()
    for sc, d in I.HOSTILE_REF.items():
        for nm, p in d.items():
            assert round(r[sc][nm]["p_T"], 3) == p, (sc, nm, r[sc][nm]["p_T"])


def test_seed_namespace_is_new():
    lits = set()
    for p in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "src").glob("*.py")):
        if p.name in ("v3_I_llama.py", "test_v3_I_llama.py"):
            continue
        lits |= {int(m.group(0).replace("_", "")) for m in re.finditer(r"\b\d[\d_]{3,}\b", p.read_text())}
    assert {G.SEED_G, H.SEED_H, V.SEED_F} <= lits
    assert all(abs(I.SEED_I - s) > 10 ** 8 for s in lits if s > 10 ** 6)
    assert 0 < I.SEED_I < 2 ** 32
    keys = [it for it in I.all_items()]
    assert len({repr(k) for k in keys}) == len(keys) == len(I.i1_items()) + 4 + 1
    with pytest.raises(ValueError):
        I.seed_i(0)


def test_accuracy_matches_item07_and_hostile_count():
    names = V.variant_data("full")["names"]
    acc = I.accuracy_rot0(names)
    ref = json.loads((OUT / "item07_across_model.json").read_text())["observed"]["accuracy_per_model"]
    for n in names:
        assert acc[n] == pytest.approx(ref[n], abs=1e-12)
    x = I.i2_from_h2()
    assert x["with_llama1b"]["summary"]["n"] == 19
    assert x["accuracy_matched_rot0_accuracy"]["hostile_band"]["summary"]["n"] == 15
