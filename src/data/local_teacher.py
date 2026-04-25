"""
AegisRAG - Local Teacher

Batched local inference with no external API calls.

Primary backend: HuggingFace ``transformers`` with bf16/fp16 and
``device_map="auto"``. Optional backend: ``vllm`` (selected automatically
when importable and CUDA is available) for high-throughput generation.

Exports
-------
LocalTeacher
    ``.generate(prompt)`` / ``.generate_batch(prompts)``
get_teacher
    Back-compat factory; always returns a ``LocalTeacher``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_PRIMARY_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_FALLBACK_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


# ---------------------------------------------------------------------------
# Device/dtype utilities
# ---------------------------------------------------------------------------

def _pick_dtype():
    """Return the best available torch dtype for inference."""
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def _pick_device() -> str:
    """Return the preferred device string."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# LocalTeacher
# ---------------------------------------------------------------------------

class LocalTeacher:
    """Batched local inference with optional vLLM acceleration and a prompt cache.

    Parameters
    ----------
    model_name : str or None
        HuggingFace model identifier. Falls back to the ``AEGIS_LOCAL_MODEL``
        environment variable, then to Qwen2.5-7B-Instruct.
    fallback_model : str or None
        Model loaded when the primary model fails to initialise.
    max_new_tokens : int
        Maximum tokens generated per call.
    temperature : float
        Sampling temperature (0 = greedy).
    top_p : float
        Nucleus sampling probability.
    use_vllm : bool or None
        Force vLLM on/off. ``None`` auto-detects based on CUDA availability.
    cache_size : int
        Maximum number of (prompt, response) pairs to keep in the FIFO cache.
    """

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
        if want_vllm:
            try:
                self._init_vllm()
                self._backend = "vllm"
            except Exception as exc:
                logger.warning("vLLM unavailable (%s); falling back to transformers.", exc)
                self._init_hf()
                self._backend = "hf"
        else:
            self._init_hf()
            self._backend = "hf"

        logger.info(
            "LocalTeacher ready — model=%s backend=%s device=%s",
            self.model_name,
            self._backend,
            getattr(self, "_device", "n/a"),
        )

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _vllm_wanted() -> bool:
        """Return True when vLLM is importable and CUDA is available."""
        try:
            import torch  # noqa: F401
            import vllm  # noqa: F401
            import torch as _t
            return _t.cuda.is_available()
        except Exception:
            return False

    def _init_vllm(self) -> None:
        from vllm import LLM, SamplingParams  # type: ignore

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
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

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
            logger.warning(
                "Primary model %s failed (%s); loading fallback %s.",
                self.model_name, exc, self.fallback_model,
            )
            self.model_name = self.fallback_model
            self._tok, self._model = _load(self.model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **overrides: Any) -> str:
        """Generate a single response for *prompt*."""
        return self.generate_batch([prompt], **overrides)[0]

    def generate_batch(
        self,
        prompts: list[str],
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Generate responses for a batch of prompts, using the cache where possible."""
        keys = [self._key(p) for p in prompts]
        outputs: list[str | None] = [self._cache.get(k) for k in keys]
        todo = [i for i, o in enumerate(outputs) if o is None]

        if not todo:
            return [o or "" for o in outputs]

        missing = [prompts[i] for i in todo]
        fresh = (
            self._gen_vllm(missing, temperature, max_new_tokens)
            if self._backend == "vllm"
            else self._gen_hf(missing, temperature, max_new_tokens)
        )

        for idx, text in zip(todo, fresh):
            outputs[idx] = text
            self._put_cache(keys[idx], text)

        return [o or "" for o in outputs]

    # ------------------------------------------------------------------
    # Backend-specific generation
    # ------------------------------------------------------------------

    def _gen_vllm(
        self,
        prompts: list[str],
        temperature: float | None,
        max_new_tokens: int | None,
    ) -> list[str]:
        from vllm import SamplingParams  # type: ignore

        sp = SamplingParams(
            temperature=self.temperature if temperature is None else float(temperature),
            top_p=self.top_p,
            max_tokens=self.max_new_tokens if max_new_tokens is None else int(max_new_tokens),
        )
        formatted = [self._format_chat(p) for p in prompts]
        outs = self._llm.generate(formatted, sp, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]

    def _gen_hf(
        self,
        prompts: list[str],
        temperature: float | None,
        max_new_tokens: int | None,
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

        new_tokens = out[:, enc["input_ids"].shape[1]:]
        return [s.strip() for s in self._tok.batch_decode(new_tokens, skip_special_tokens=True)]

    def _format_chat(self, prompt: str) -> str:
        """Apply the model's chat template to a raw prompt string."""
        if self._backend == "vllm":
            # vLLM needs the formatted string; load the tokenizer once for the template.
            try:
                from transformers import AutoTokenizer  # type: ignore
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
    # Prompt cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(prompt: str) -> str:
        """Return a 16-byte Blake2b hex digest of the prompt."""
        return hashlib.blake2b(prompt.encode("utf-8"), digest_size=16).hexdigest()

    def _put_cache(self, key: str, value: str) -> None:
        """Insert into cache with FIFO eviction when at capacity."""
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value


# ---------------------------------------------------------------------------
# Back-compat factory
# ---------------------------------------------------------------------------

def get_teacher(**kwargs: Any) -> LocalTeacher:
    """Return a :class:`LocalTeacher`. Accepts any ``LocalTeacher.__init__`` kwargs."""
    return LocalTeacher(**kwargs)
