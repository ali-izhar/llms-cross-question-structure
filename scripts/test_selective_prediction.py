"""
Tests for scripts/selective_prediction.py.

Equivalence tests against hand-computed AUROC values, plus Hypothesis
property tests on the helper functions.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import numpy.testing as npt
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from selective_prediction import (  # noqa: E402
    PARAMS_B,
    QWEN7B,
    QWEN7B_PARAMS,
    compute_predictor_metrics,
    cost_units,
    family_diverse_models,
    family_of,
    topk_other_models,
)

PROJECT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"


# ── 1. Equivalence test against sklearn directly ────────────────────────
def test_compute_predictor_metrics_matches_sklearn():
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, size=200)
    signal = rng.normal(size=200)
    auroc, aupr = compute_predictor_metrics(target, signal)
    npt.assert_allclose(auroc, roc_auc_score(target, signal),
                        rtol=1e-12, atol=1e-12)


# ── 2. Hypothesis property tests ────────────────────────────────────────
@given(
    target=hnp.arrays(np.int64, shape=st.integers(20, 200),
                      elements=st.integers(0, 1)),
    signal=hnp.arrays(np.float64, shape=st.integers(20, 200),
                      elements=st.floats(-10, 10, allow_nan=False,
                                         allow_infinity=False)),
)
@settings(max_examples=50, deadline=None)
def test_auroc_in_unit_interval_or_nan(target, signal):
    # truncate to common length
    n = min(len(target), len(signal))
    target = target[:n]
    signal = signal[:n]
    auroc, aupr = compute_predictor_metrics(target, signal)
    if not np.isnan(auroc):
        assert 0.0 <= auroc <= 1.0, f"AUROC {auroc} out of [0,1]"
        assert 0.0 <= aupr <= 1.0, f"AUPR {aupr} out of [0,1]"


def test_auroc_perfect_signal():
    target = np.array([0, 0, 1, 1])
    signal = np.array([0.1, 0.2, 0.8, 0.9])
    auroc, aupr = compute_predictor_metrics(target, signal)
    npt.assert_allclose(auroc, 1.0, atol=1e-12)
    npt.assert_allclose(aupr, 1.0, atol=1e-12)


def test_auroc_inverted_signal():
    target = np.array([0, 0, 1, 1])
    signal = np.array([0.9, 0.8, 0.2, 0.1])
    auroc, _ = compute_predictor_metrics(target, signal)
    npt.assert_allclose(auroc, 0.0, atol=1e-12)


def test_auroc_random_signal_around_half():
    rng = np.random.default_rng(42)
    target = rng.integers(0, 2, size=10_000)
    signal = rng.normal(size=10_000)  # independent of target
    auroc, _ = compute_predictor_metrics(target, signal)
    # central limit ~ 0.5 ± 0.01 for n=10k
    assert 0.48 <= auroc <= 0.52


# ── 3. cost_units consistency ──────────────────────────────────────────
def test_cost_units_qwen7b_is_one():
    npt.assert_allclose(cost_units([QWEN7B]), 1.0, atol=1e-12)


def test_cost_units_two_qwen05b():
    cost = cost_units(["Qwen/Qwen2.5-0.5B-Instruct",
                       "Qwen/Qwen2.5-0.5B-Instruct"])
    npt.assert_allclose(cost, 2 * 0.5 / QWEN7B_PARAMS, atol=1e-12)


def test_cost_units_additive():
    a = cost_units(["Qwen/Qwen2.5-0.5B-Instruct"])
    b = cost_units(["meta-llama/Llama-3.1-8B-Instruct"])
    ab = cost_units(["Qwen/Qwen2.5-0.5B-Instruct",
                     "meta-llama/Llama-3.1-8B-Instruct"])
    npt.assert_allclose(a + b, ab, atol=1e-12)


# ── 4. family_of correctness ──────────────────────────────────────────
def test_family_of_zephyr_is_mistral():
    # Zephyr-7b-beta is a Mistral-7B fine-tune.
    assert family_of("HuggingFaceH4/zephyr-7b-beta") == "mistral"


def test_family_of_examples():
    assert family_of("Qwen/Qwen2.5-7B-Instruct") == "qwen"
    assert family_of("meta-llama/Llama-3.1-8B-Instruct") == "llama"
    assert family_of("microsoft/Phi-3-mini-4k-instruct") == "phi"
    assert family_of("google/gemma-2-9b-it") == "gemma"


# ── 5. ensemble selection ──────────────────────────────────────────────
def test_topk_excludes_target():
    accs = {m: 0.5 for m in PARAMS_B}
    accs[QWEN7B] = 1.0
    chosen = topk_other_models(list(PARAMS_B), accs, k=3)
    assert QWEN7B not in chosen
    assert len(chosen) == 3


def test_family_diverse_first_is_non_qwen():
    accs = {m: 0.5 for m in PARAMS_B}
    chosen = family_diverse_models(list(PARAMS_B), accs, k=1)
    assert family_of(chosen[0]) != "qwen"


def test_family_diverse_picks_distinct_families_first():
    accs = {m: 0.5 for m in PARAMS_B}
    chosen = family_diverse_models(list(PARAMS_B), accs, k=4)
    families = [family_of(m) for m in chosen]
    # First 4 picks should all be distinct families.
    assert len(set(families)) == 4


# ── 6. End-to-end smoke: SC_100 oracle AUROC matches direct calc ──────
def test_sc100_oracle_auroc_matches_direct():
    sc = np.load(RESULTS / "correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy")
    ens = np.load(RESULTS / "ensemble_expanded_correctness.npy")
    with open(RESULTS / "ensemble_expanded_models.json") as f:
        models = json.load(f)
    target = ens[models.index(QWEN7B)].astype(np.int64)
    signal = sc.mean(axis=0)
    expected = roc_auc_score(target, signal)
    # Re-run script and load result
    subprocess.run([sys.executable, str(PROJECT / "scripts" /
                                       "selective_prediction.py")],
                   check=True, capture_output=True)
    with open(RESULTS / "selective_prediction.json") as f:
        out = json.load(f)
    sc100 = next(r for r in out["SC_K"] if r["K_or_k"] == 100)
    npt.assert_allclose(sc100["AUROC_oracle_upper_bound"], expected,
                        rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    # Minimal in-process runner so it works without pytest.
    import inspect
    failures = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                # hypothesis-decorated functions take no args at top-level
                sig = inspect.signature(fn)
                if any(p.default is inspect.Parameter.empty
                       and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                      inspect.Parameter.POSITIONAL_ONLY)
                       for p in sig.parameters.values()):
                    fn()  # will be filled by hypothesis
                else:
                    fn()
                print(f"  OK   {name}")
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                failures.append((name, e))
    if failures:
        sys.exit(f"\n{len(failures)} test(s) failed.")
    print("\nAll tests passed.")
