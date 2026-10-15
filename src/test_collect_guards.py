"""Guards of src/collect.py on the host with a vllm stub: non-v3 outputs raise, reference hashes catch a
self-consistent wrong tokenizer, unpinned decode kinds fail, and stale partial files are not resumed.
Uses the pins in results/pins/ and the local HF cache (skips if a model is not cached).
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
try:
    import vllm  # noqa: F401
except ImportError:
    sys.path.insert(0, str(HERE / "test_stubs"))
sys.path.insert(0, str(HERE))
import collect  # noqa: E402
from plan import GREEDY, GREEDYROT, run_id  # noqa: E402

RESULTS = HERE.parent / "results"
CTX = {t: json.loads((RESULTS / f"questions_{t}.json").read_text()) for t in ("mmlu", "mmlu_dev", "hellaswag", "gsm8k")}
NOLOG = lambda m: None  # noqa: E731


def _prompter_and_pins(model):
    from huggingface_hub.errors import LocalEntryNotFoundError

    from canonical_tok import CanonicalPrompter

    try:
        b = CanonicalPrompter(model)
    except (FileNotFoundError, LocalEntryNotFoundError) as e:
        pytest.skip(f"{model} not available: {e}")
    if not (collect.PINS_DIR / "reference_hashes.json").exists():
        pytest.skip("pins not built")
    return b, collect.load_pins(model, collect.PINS_DIR)


def test_precheck_passes_with_pins(tmp_path):
    b, pins = _prompter_and_pins("microsoft/phi-2")
    r = collect.do_run(None, None, b, "microsoft/phi-2", GREEDYROT, CTX, tmp_path, 10, None, NOLOG, pins)
    pc = r["prompt_check"]
    assert pc["n_prompts_checked"] == 40 and pc["n_equal_to_reference_hash"] == 40


def test_v2_final_output_raises(tmp_path):
    m = "microsoft/phi-2"
    b, pins = _prompter_and_pins(m)
    rid = run_id(m, GREEDYROT)
    np.savez(tmp_path / f"{rid}.npz", correct=np.zeros((4, 10), np.int8))
    (tmp_path / f"{rid}.json").write_text(json.dumps({"run_id": rid, "accuracy": 0.5}))  # v2: no tokenization
    with pytest.raises(RuntimeError, match="not a complete v3 output"):
        collect.do_run(None, None, b, m, GREEDYROT, CTX, tmp_path, 10, None, NOLOG, pins)


@pytest.mark.parametrize("field,value", [("dry_run", True), ("N", 20), ("parser_version", "v3"),
                                         ("model_revision", "other"), ("ref", 499),
                                         ("deterministic", {"on": False})])
def test_is_v3_output_requires_every_condition(tmp_path, field, value):
    pins = {"revision": "rev1"}
    good = {"N": 500, "dry_run": False, "deterministic": {"on": True}, "parser_version": collect.PARSER_VERSION, "model_revision": "rev1",
            "tokenization": {"pins": {"revision": "rev1"},
                             "prompt_check": {"n_prompts_checked": 500, "n_equal_to_reference_hash": 500}}}
    js = tmp_path / "r.json"
    js.write_text(json.dumps(good))
    assert collect.is_v3_output(js, 500, pins)
    bad = copy.deepcopy(good)
    if field == "ref":
        bad["tokenization"]["prompt_check"]["n_equal_to_reference_hash"] = value
    else:
        bad[field] = value
    js.write_text(json.dumps(bad))
    assert not collect.is_v3_output(js, 500, pins)


def test_swapped_tokenizer_fails_reference_hash(tmp_path):
    # OLMo-2 with the v2 defect: ByteLevel without the model's split regex. The tokenizer is
    # self-consistent (check_prompt passes: same text, same specials, exact decode), so only the
    # independent reference hash can catch it.
    import tokenizers

    m = "allenai/OLMo-2-1124-7B-Instruct"
    b, pins = _prompter_and_pins(m)
    tj = json.loads((b.snap / "tokenizer.json").read_text())
    tj["pre_tokenizer"] = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}
    b.tok = tokenizers.Tokenizer.from_str(json.dumps(tj))
    collect.do_run(None, None, b, m, GREEDY, CTX, tmp_path, 25, None, NOLOG, None)  # without pins: passes (q21 differs)
    with pytest.raises(AssertionError, match="independent reference"):
        collect.do_run(None, None, b, m, GREEDY, CTX, tmp_path, 25, None, NOLOG, pins)


def test_unpinned_decode_kind_fails(tmp_path):
    m = "microsoft/phi-2"   # pinned kind: exact
    b, pins = _prompter_and_pins(m)
    pins = {**pins, "allowed_decode_kinds": {"added_token_strip"}}
    with pytest.raises(AssertionError, match="not pinned"):
        collect.do_run(None, None, b, m, GREEDYROT, CTX, tmp_path, 3, None, NOLOG, pins)


class _NoGenerate:
    def generate(self, *a, **k):
        raise AssertionError("generate was called: the partial file was accepted")


@pytest.mark.parametrize("extra,match", [({}, "another scorer"),
                                         ({"scorer_sha256": "deadbeef", "prompt_hash": "x"}, "another scorer"),
                                         ({"scorer_sha256": "SCORER", "prompt_hash": "x"}, "other prompt ids")])
def test_partial_with_other_scorer_or_prompts_raises(tmp_path, extra, match):
    m = "microsoft/phi-2"
    b, pins = _prompter_and_pins(m)
    sha = hashlib.sha256((HERE / "scoring.py").read_bytes()).hexdigest()[:16]
    extra = {k: (sha if v == "SCORER" else v) for k, v in extra.items()}
    rid = run_id(m, GREEDYROT)
    np.savez(tmp_path / f"{rid}.partial.npz", K=4, N=5, k_done=2, runtime_sec=1.0,
             correct=np.zeros((4, 5), np.int8), **extra)
    with pytest.raises(RuntimeError, match=match):
        collect.do_run(_NoGenerate(), None, b, m, GREEDYROT, CTX, tmp_path, 5, None, NOLOG, pins)


def test_v3_runs_rejects_unknown_model():
    from plan import v3_runs

    with pytest.raises(KeyError):
        v3_runs("org/not-in-ensemble")
