"""Minimal vllm stub for the host tests of src/collect.py (the real vllm runs only in gb10-vllm)."""
__version__ = "stub"


class SamplingParams:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.stop = None
        self.stop_token_ids = None
        self.ignore_eos = False


class LLM:
    def __init__(self, **kw):
        raise SystemExit("STUB_ENGINE_REACHED")
