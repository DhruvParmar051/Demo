"""Local-only teacher. No API calls.

Primary backend: ``transformers`` with bf16/fp16 + ``device_map="auto"``.
Optional backend: ``vllm`` (used automatically when importable and CUDA
is available) for batched high-throughput generation.

Exports:
    LocalTeacher  -- ``.generate`` / ``.generate_batch``
    get_teacher   -- back-compat factory (always returns LocalTeacher)
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Iterable

logger = logging.getLogger(__name__)


_PRIMARY_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_FALLBACK_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def _pick_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalTeacher:
    """Batched local inference with optional vLLM acceleration + prompt cache."""

    def __init__(
        self,
        model_name: str | None = None,
        fallback_model: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        use_vllm: bool | None = None,
        cache_size: int = 4096,
        **_: Any,
    ) -> None:
        self.model_name = model_name or os.getenv("AEGIS_LOCAL_MODEL") or _PRIMARY_MODEL
        self.fallback_model = fallback_model or _FALLBACK_MODEL
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self._cache: dict[str, str] = {}
        self._cache_size = int(cache_size)

        want_vllm = use_vllm if use_vllm is not None else self._vllm_wanted()
        self._backend: str
        if want_vllm:
            try:
                self._init_vllm()
                self._backend = "vllm"
            except Exception as exc:
                logger.warning("vLLM unavailable (%s); falling back to transformers", exc)
                self._init_hf()
                self._backend = "hf"
        else:
            self._init_hf()
            self._backend = "hf"
        logger.info(
            "LocalTeacher ready: model=%s backend=%s device=%s",
            self.model_name,
            self._backend,
            getattr(self, "_device", "n/a"),
        )

    # ------------------------------------------------------------------
    # backend init
    # ------------------------------------------------------------------

    @staticmethod
    def _vllm_wanted() -> bool:
        try:
            import torch  # noqa: F401
            import vllm  # noqa: F401
            import torch as _t

            return _t.cuda.is_available()
        except Exception:
            return False

    def _init_vllm(self) -> None:
        from vllm import LLM, SamplingParams

        self._llm = LLM(
            model=self.model_name,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,
        )
        self._sampling = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )

    def _init_hf(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = _pick_dtype()
        device = _pick_device()
        self._device = device

        def _load(name: str):
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"
            kwargs: dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": True}
            if device == "cuda":
                kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
            if device != "cuda":
                model = model.to(device)
            model.eval()
            return tok, model

        try:
            self._tok, self._model = _load(self.model_name)
        except Exception as exc:
            logger.warning("Primary %s failed (%s); loading fallback %s",
                           self.model_name, exc, self.fallback_model)
            self.model_name = self.fallback_model
            self._tok, self._model = _load(self.model_name)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **overrides: Any) -> str:
        return self.generate_batch([prompt], **overrides)[0]

    def generate_batch(
        self,
        prompts: list[str],
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        # cache lookup
        keys = [self._key(p) for p in prompts]
        outputs: list[str | None] = [self._cache.get(k) for k in keys]
        todo = [i for i, o in enumerate(outputs) if o is None]
        if not todo:
            return [o or "" for o in outputs]

        missing = [prompts[i] for i in todo]
        if self._backend == "vllm":
            fresh = self._gen_vllm(missing, temperature, max_new_tokens)
        else:
            fresh = self._gen_hf(missing, temperature, max_new_tokens)

        for idx, text in zip(todo, fresh):
            outputs[idx] = text
            self._put_cache(keys[idx], text)
        return [o or "" for o in outputs]

    # ------------------------------------------------------------------
    # backend-specific decode
    # ------------------------------------------------------------------

    def _gen_vllm(
        self, prompts: list[str], temperature: float | None, max_new_tokens: int | None
    ) -> list[str]:
        from vllm import SamplingParams

        sp = SamplingParams(
            temperature=self.temperature if temperature is None else float(temperature),
            top_p=self.top_p,
            max_tokens=self.max_new_tokens if max_new_tokens is None else int(max_new_tokens),
        )
        formatted = [self._format_chat(p) for p in prompts]
        outs = self._llm.generate(formatted, sp, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]

    def _gen_hf(
        self, prompts: list[str], temperature: float | None, max_new_tokens: int | None
    ) -> list[str]:
        import torch

        t = self.temperature if temperature is None else float(temperature)
        m = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        formatted = [self._format_chat(p) for p in prompts]
        enc = self._tok(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=3584,
        ).to(self._model.device)

        with torch.inference_mode():
            out = self._model.generate(
                **enc,
                max_new_tokens=m,
                do_sample=t > 0,
                temperature=max(t, 1e-5),
                top_p=self.top_p,
                pad_token_id=self._tok.pad_token_id,
                eos_token_id=self._tok.eos_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        texts = self._tok.batch_decode(new_tokens, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def _format_chat(self, prompt: str) -> str:
        if self._backend == "vllm":
            # vLLM wants raw text; rely on tokenizer's chat template via HF
            try:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                return tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return prompt
        return self._tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    # ------------------------------------------------------------------
    # cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(prompt: str) -> str:
        return hashlib.blake2b(prompt.encode("utf-8"), digest_size=16).hexdigest()

    def _put_cache(self, k: str, v: str) -> None:
        if len(self._cache) >= self._cache_size:
            # cheap FIFO eviction
            self._cache.pop(next(iter(self._cache)))
        self._cache[k] = v


def get_teacher(**kwargs: Any) -> LocalTeacher:
    """Back-compat factory. Always returns a ``LocalTeacher``."""
    return LocalTeacher(**kwargs)
