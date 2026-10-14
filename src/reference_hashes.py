"""Reference hashes of the MMLU prompt ids from an independent tokenizer (transformers 4.46.3; Yi: slow SentencePiece).

    uv venv tf4 && uv pip install -p tf4 transformers==4.46.3 tokenizers==0.20.3 sentencepiece==0.2.2 jinja2 protobuf huggingface_hub
    HF_HUB_OFFLINE=1 tf4/bin/python src/reference_hashes.py --out results/dryrun_v3/reference_hashes.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mmlu_utils import build_mmlu_prompt  # noqa: E402
from plan import ENSEMBLE_MODELS, rotate  # noqa: E402

PIN = datetime.datetime(2026, 10, 10, 12, 0, 0)
SLOW = {"01-ai/Yi-1.5-6B-Chat"}


def ids_sha(ids) -> str:
    """Same format as canonical_tok.ids_sha."""
    return hashlib.sha256((",".join(map(str, ids))).encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import huggingface_hub
    import tokenizers
    import transformers
    import transformers.utils.chat_template_utils as ctu
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    class _Pinned(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(PIN.year, PIN.month, PIN.day, PIN.hour, PIN.minute, PIN.second)

    ctu.datetime = _Pinned  # strftime_now() in templates (granite-3.1, Llama-3.2)
    qs = json.loads((HERE.parent / "results" / "questions_mmlu.json").read_text())
    out = {"method": "transformers apply_chat_template(tokenize=True); Yi: use_fast=False (SentencePiece)",
           "versions": {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
                        "huggingface_hub": huggingface_hub.__version__},
           "pinned_date": PIN.isoformat(), "hash": "sha256(','.join(ids))[:16]", "models": {}}
    try:
        import sentencepiece
        out["versions"]["sentencepiece"] = sentencepiece.__version__
    except ImportError:
        pass
    for m in a.models or ENSEMBLE_MODELS:
        t0 = time.time()
        slow = m in SLOW
        tok = AutoTokenizer.from_pretrained(m, local_files_only=True, use_fast=not slow)
        has_t = bool(getattr(tok, "chat_template", None))
        rows = [q["row"] for q in qs]
        hashes = {}
        for r in range(4):
            hs = []
            for q in qs:
                u = build_mmlu_prompt(q["question"], rotate(q["choices"], r))
                if has_t:
                    ids = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=True,
                                                  add_generation_prompt=True)
                else:
                    ids = tok(u)["input_ids"]
                hs.append(ids_sha(ids))
            hashes[f"r{r}"] = hs
        out["models"][m] = {"revision": Path(snapshot_download(m, local_files_only=True)).name,
                            "tokenizer_class": type(tok).__name__, "slow": slow,
                            "prompt_mode": "chat_template" if has_t else "raw",
                            "question_rows": rows, "hashes": hashes}
        print(f"{m}: {type(tok).__name__} {round(time.time() - t0, 1)}s", flush=True)
    Path(a.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
