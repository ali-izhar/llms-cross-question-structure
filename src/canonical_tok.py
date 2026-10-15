"""Canonical prompt ids: the model's own chat template (jinja2, HF environment, pinned clock) + tokenizer.json.

Avoids the transformers tokenizer classes, which mis-load DeepSeek-LLM, Yi-1.5 and OLMo-2 in transformers 5.12.1.
check_prompt asserts canonical ids, one id per special token, and decode == text up to DECODE_ALLOWANCES.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

PINNED_DATE = _dt.datetime(2026, 10, 10, 12, 0, 0)  # template clock = v2 collection date (granite-3.1, Llama-3.2)

# Exact, documented decode differences:
#  metaspace_space_after_special: Metaspace decode shows one extra space after a special token.
#  added_token_literal: plain characters listed as added tokens (DeepSeek "÷") decode to U+FFFD; decode piecewise.
#  added_token_strip: lstrip/rstrip added tokens (Phi-3 <|user|>) absorb adjacent whitespace; strip it from the text.
DECODE_ALLOWANCES = ("metaspace_space_after_special", "added_token_literal", "added_token_strip")


def snapshot_dir(model: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, local_files_only=True))


def _tok_str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, dict):
        return v.get("content")
    return str(v)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256((",".join(map(str, ids))).encode()).hexdigest()[:16]


class CanonicalPrompter:
    def __init__(self, model: str, snap: Path | None = None):
        import tokenizers

        self.model = model
        self.snap = Path(snap) if snap else snapshot_dir(model)
        tj = self.snap / "tokenizer.json"
        if not tj.exists():
            raise FileNotFoundError(f"{model}: no tokenizer.json in {self.snap}")
        self.tok = tokenizers.Tokenizer.from_file(str(tj))
        self.tokenizer_json_sha256 = sha256_file(tj)
        cfg_p = self.snap / "tokenizer_config.json"
        self.cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
        smap_p = self.snap / "special_tokens_map.json"
        smap = json.loads(smap_p.read_text()) if smap_p.exists() else {}
        self.special = {k: _tok_str(self.cfg.get(k)) or _tok_str(smap.get(k))
                        for k in ("bos_token", "eos_token", "unk_token", "pad_token")}
        self.template, self.template_source = self._load_template()
        self.mode = "chat_template" if self.template else "raw"
        self._env = None
        tj_json = json.loads(tj.read_text())
        self.added = {a["content"]: a["id"] for a in tj_json.get("added_tokens", [])}
        self.special_added = {a["content"]: a["id"] for a in tj_json.get("added_tokens", []) if a.get("special")}
        self.strip_flags = {a["content"]: (bool(a.get("lstrip")), bool(a.get("rstrip")))
                            for a in tj_json.get("added_tokens", []) if a.get("lstrip") or a.get("rstrip")}
        pre = (json.dumps(tj_json.get("pre_tokenizer"), ensure_ascii=False)
               + json.dumps(tj_json.get("normalizer"), ensure_ascii=False))
        self.is_metaspace = "Metaspace" in pre or "▁" in pre
        self.versions = self._versions()

    @staticmethod
    def _versions() -> dict:
        out = {}
        for name in ("tokenizers", "jinja2", "transformers", "huggingface_hub"):
            try:
                out[name] = __import__(name).__version__
            except Exception:
                out[name] = None
        return out

    def _load_template(self) -> tuple[str | None, str | None]:
        jf = self.snap / "chat_template.jinja"
        if jf.exists():
            return jf.read_text(), "chat_template.jinja"
        t = self.cfg.get("chat_template")
        if isinstance(t, list):
            named = {d["name"]: d["template"] for d in t}
            return named.get("default"), "tokenizer_config.json[default]"
        if t:
            return t, "tokenizer_config.json"
        return None, None

    def _jinja(self):
        if self._env is None:
            import jinja2
            import jinja2.ext
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            def raise_exception(message):
                raise jinja2.exceptions.TemplateError(message)

            def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
                return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                                  sort_keys=sort_keys)

            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=[jinja2.ext.loopcontrols])
            env.filters["tojson"] = tojson
            env.globals["raise_exception"] = raise_exception
            env.globals["strftime_now"] = lambda fmt: PINNED_DATE.strftime(fmt)
            self._env = env.from_string(self.template)
        return self._env

    def render(self, user_text: str) -> str:
        if not self.template:
            return user_text
        return self._jinja().render(messages=[{"role": "user", "content": user_text}],
                                    add_generation_prompt=True, **self.special)

    def render_transformers(self, hf_tok, user_text: str) -> str:
        """The v2 rendering (transformers apply_chat_template), same pinned clock."""
        if not self.template:
            return user_text
        return hf_tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                          add_generation_prompt=True,
                                          strftime_now=lambda fmt: PINNED_DATE.strftime(fmt))

    def encode_text(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=(self.mode == "raw")).ids)

    def ids(self, user_text: str) -> list[int]:
        return self.encode_text(self.render(user_text))

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self.tok.decode(list(map(int, ids)), skip_special_tokens=skip_special_tokens)

    def apply_strip_flags(self, text: str) -> str:
        """The text with the whitespace removed that lstrip/rstrip added tokens absorb."""
        import re

        for content, (ls, rs) in self.strip_flags.items():
            c = re.escape(content)
            if rs:
                text = re.sub(c + r"\s+", content, text)
            if ls:
                text = re.sub(r"\s+" + c, content, text)
        return text

    def _matches(self, dec: str, ids, text: str) -> bool:
        return (dec == text or (self.is_metaspace and _equal_up_to_space_after_special(dec, text, self.added))
                or self.decode_piecewise(ids) == text)

    def decode_piecewise(self, ids) -> str:
        """Added-token ids as their literal content; runs of other ids through the decoder."""
        by_id = {v: k for k, v in self.added.items()}
        out, run = [], []
        for t in map(int, ids):
            if t in by_id:
                if run:
                    out.append(self.tok.decode(run, skip_special_tokens=False))
                    run = []
                out.append(by_id[t])
            else:
                run.append(t)
        if run:
            out.append(self.tok.decode(run, skip_special_tokens=False))
        return "".join(out)

    def token_text(self, tid: int) -> str:
        return self.tok.decode([int(tid)], skip_special_tokens=False)

    def check_prompt(self, text: str, ids: list[int], hf_tok=None) -> dict:
        """Verify one prompt. Returns a dict; `ok` False means the prompt must not be sent."""
        res: dict = {"ok": True, "problems": []}
        canon = self.encode_text(text)
        if list(ids) != canon:
            res["ok"] = False
            res["problems"].append("ids != canonical encode(text)")
        # every special added token in the text is one id, as often as it occurs
        for content, tid in self.special_added.items():
            n_txt = text.count(content)
            if n_txt and list(ids).count(tid) != n_txt:
                res["ok"] = False
                res["problems"].append(f"special token {content!r}: {n_txt} in text, "
                                       f"{list(ids).count(tid)} ids")
        dec = self.decode(ids, skip_special_tokens=False)
        if dec == text:
            res["decode"] = "exact"
        elif self.is_metaspace and _equal_up_to_space_after_special(dec, text, self.added):
            res["decode"] = "metaspace_space_after_special"
        elif self.decode_piecewise(ids) == text:
            res["decode"] = "added_token_literal"
        elif self.strip_flags and self._matches(dec, ids, self.apply_strip_flags(text)):
            res["decode"] = "added_token_strip"
        else:
            res["ok"] = False
            res["decode"] = "MISMATCH"
            res["problems"].append(f"decode != text: {_first_diff(dec, text)}")
        if hf_tok is not None:
            old = list(hf_tok(text, add_special_tokens=(self.mode == "raw"))["input_ids"])
            res["v2_pipeline_equal"] = old == list(ids)
        return res


def _equal_up_to_space_after_special(dec: str, text: str, added: dict) -> bool:
    """dec equals text after removing single spaces that directly follow an added token in dec."""
    out, i = [], 0
    toks = sorted(added, key=len, reverse=True)
    while i < len(dec):
        hit = next((t for t in toks if t and dec.startswith(t, i)), None)
        if hit:
            out.append(hit)
            i += len(hit)
            if i < len(dec) and dec[i] == " " and not text[len("".join(out)):].startswith(" "):
                i += 1
            continue
        out.append(dec[i])
        i += 1
    return "".join(out) == text


def _first_diff(a: str, b: str) -> str:
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"at {n}: decoded {a[max(0, n - 20):n + 20]!r} vs text {b[max(0, n - 20):n + 20]!r}"
