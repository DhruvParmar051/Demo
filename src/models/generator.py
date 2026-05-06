"""
AegisRAG - Generator

Qwen2.5-7B-Instruct (GPU/MPS via transformers + bitsandbytes) or
Phi-3.5-mini-instruct (CPU fallback).  Optionally swappable to GGUF via
llama-cpp-python for edge inference.

Exposes:

- ``generate(prompt, ...) -> str``                    -- deterministic greedy
- ``stream(prompt, ...) -> Iterator[str]``            -- token-by-token
- ``generate_with_citations(query, context, ...)``    -- RAG-style wrapper
- ``parse_citations(answer, context)``                -- marker resolver
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator

from src.data.schema import Citation, RetrievalResult
from src.utils.config import get_config
from src.utils.device import get_device, get_device_string

logger = logging.getLogger(__name__)

# Stop sequences that prevent repetitive / out-of-role generation.
_STOP = [
    "<|user|>", "<|system|>", "<|context|>", "<|assistant|>",
    "\n\nIf you need further", "\n\nPlease provide more", "\n\nIf you have any other",
]

SYSTEM_PROMPT = (
    "You are AegisRAG, a grounded document-understanding assistant. "
    "Every factual claim you make must be immediately followed by a citation "
    "of the form [doc_id:start-end] pointing to the supplied context. "
    "If the context is insufficient to answer, say so explicitly and suggest "
    "escalation. Do not fabricate facts, tool calls, or citations."
)

_CITATION_RE = re.compile(r"\[([^\]:]+):(\d+)-(\d+)\]")


class Generator:
    """LLM generator with RAG prompt construction and citation parsing.

    Parameters
    ----------
    model_name : str or None
        HF identifier. Defaults to ``cfg.models.generator.model_name``.
    cpu_fallback_model : str or None
        Smaller model used when CUDA and MPS are both unavailable.
    adapter_path : str or None
        Path to a PEFT (LoRA/DoRA) adapter to apply on top of the base model.
    backend : str
        ``"hf"`` (transformers) or ``"gguf"`` (llama-cpp-python).
    gguf_path : str or None
        Path to the GGUF file when ``backend == "gguf"``.
    quantize_4bit : bool
        Enable bitsandbytes 4-bit NF4 loading on CUDA.
    warmup : bool
        Run a tiny warmup generation on construction.
    """

    def __init__(
        self,
        model_name: str | None = None,
        cpu_fallback_model: str | None = None,
        adapter_path: str | None = None,
        backend: str | None = None,
        gguf_path: str | None = None,
        quantize_4bit: bool = True,
        warmup: bool = False,
    ) -> None:
        cfg = get_config()
        self.cfg = cfg
        self.model_name = model_name or cfg.models.generator.name
        self.cpu_fallback_model = cpu_fallback_model or getattr(cfg.models.generator, "cpu_fallback_model", None)
        self.adapter_path = adapter_path
        
        # Resolve GGUF path: explicit arg > config field
        self.gguf_path = gguf_path or getattr(cfg.models.generator, "gguf_path", None)

        # Auto-select backend: if a GGUF file is configured and exists on disk,
        # default to "gguf"; otherwise fall back to "hf".
        if backend:
            self.backend = backend
        elif self.gguf_path and Path(self.gguf_path).exists():
            self.backend = "gguf"
        else:
            self.backend = "hf"
        self.quantize_4bit = quantize_4bit
        self.device = get_device(cfg.device.preferred_device)
        self.device_str = get_device_string(self.device)

        self._generation_kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": int(cfg.models.generator.max_new_tokens),
        }

        # Lazy-loaded.
        self._model = None
        self._tokenizer = None
        self._llama = None  # for GGUF

        if warmup:
            self._ensure_loaded()
            try:
                _ = self.generate("Warmup.", max_new_tokens=4)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Warmup failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_generation_kwargs(self, **kwargs: Any) -> None:
        """Force kwargs used by every subsequent generate/stream call."""
        self._generation_kwargs.update(kwargs)

    def generate(
        self,
        prompt: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        query: str | None = None,
        context: list[RetrievalResult] | None = None,
    ) -> str:
        """Greedy deterministic generation.

        Accepts either ``prompt`` directly or (``query``, ``context``) pair
        which will be formatted into the RAG prompt template.
        """
        full_prompt = self._build_prompt(prompt, query, context)
        self._ensure_loaded()
        tokens = max_new_tokens if max_new_tokens is not None else int(self._generation_kwargs["max_new_tokens"])
        if self.backend == "gguf":
            return self._generate_gguf(full_prompt, tokens, temperature)
        return self._generate_hf(full_prompt, tokens, temperature)

    def stream(
        self,
        query: str,
        context: list[RetrievalResult] | list[Citation] | None = None,
        max_new_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield tokens one at a time."""
        full_prompt = self._build_prompt(None, query, context)
        self._ensure_loaded()
        tokens = max_new_tokens if max_new_tokens is not None else int(self._generation_kwargs["max_new_tokens"])
        if self.backend == "gguf":
            yield from self._stream_gguf(full_prompt, tokens)
        else:
            yield from self._stream_hf(full_prompt, tokens)

    def generate_with_citations(
        self,
        query: str,
        context: list[RetrievalResult],
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[str, list[Citation]]:
        """Generate and parse citations out of the answer."""
        prompt = self._build_prompt(None, query, context, history=history)
        answer = self.generate(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        citations = self.parse_citations(answer, context)
        return answer, citations

    def parse_citations(
        self,
        answer: str,
        context: list[RetrievalResult],
    ) -> list[Citation]:
        """Extract `[doc_id:start-end]` markers and resolve to Citation objects."""
        lookup: dict[str, RetrievalResult] = {}
        for rr in context:
            lookup[rr.chunk.doc_id] = rr

        citations: list[Citation] = []
        seen: set[tuple[str, int, int]] = set()
        for match in _CITATION_RE.finditer(answer):
            doc_id = match.group(1)
            try:
                s, e = int(match.group(2)), int(match.group(3))
            except ValueError:
                continue
            key = (doc_id, s, e)
            if key in seen:
                continue
            seen.add(key)
            rr = lookup.get(doc_id)
            if rr is None:
                cited_text = ""
                source = ""
                chunk_id = ""
                page_number = None
                source_url = None
            else:
                text = rr.chunk.text
                start_off = max(0, s - rr.chunk.span_start)
                end_off = max(start_off, e - rr.chunk.span_start)
                cited_text = text[start_off:end_off] or text
                source = rr.chunk.source
                chunk_id = rr.chunk.chunk_id
                page_number = rr.chunk.page_number
                source_url = (rr.chunk.metadata or {}).get("source_url")
            citations.append(
                Citation(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    span_start=s,
                    span_end=e,
                    cited_text=cited_text,
                    source=source,
                    page_number=page_number,
                    source_url=source_url,
                )
            )
        return citations

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        prompt: str | None,
        query: str | None,
        context: Iterable[Any] | None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        if prompt is not None and query is None:
            return prompt

        ctx_blocks: list[str] = []
        if context:
            for i, item in enumerate(context):
                chunk = getattr(item, "chunk", None) or item
                doc_id = getattr(chunk, "doc_id", "")
                span_start = getattr(chunk, "span_start", 0)
                span_end = getattr(chunk, "span_end", span_start + len(getattr(chunk, "text", "")))
                text = getattr(chunk, "text", "") or getattr(chunk, "cited_text", "")
                ctx_blocks.append(
                    f"[{doc_id}:{span_start}-{span_end}] {text}"
                )
        context_str = "\n\n".join(ctx_blocks) if ctx_blocks else "(no context)"
        q = query or ""

        # Build prior-turn block from conversation history.
        history_str = ""
        if history:
            turns: list[str] = []
            for turn in history:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "user":
                    turns.append(f"<|user|>\n{content}")
                elif role == "assistant":
                    turns.append(f"<|assistant|>\n{content}")
            if turns:
                history_str = "\n".join(turns) + "\n"

        return (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|context|>\n{context_str}\n"
            f"{history_str}"
            f"<|user|>\n{q}\n"
            f"<|assistant|>\n"
        )

    # ------------------------------------------------------------------
    # Backend loaders
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self.backend == "gguf":
            if self._llama is None:
                self._load_gguf()
        else:
            if self._model is None:
                self._load_hf()

    def _choose_hf_model(self) -> str:
        model = (
            self.cpu_fallback_model if self.device_str == "cpu"
            else self.model_name
        ) or self.model_name
        if model is None:
            raise ValueError("No valid model configured.")
        return model

    def _load_hf(self) -> None:
        try:
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for the HF backend."
            ) from exc
        import torch  # type: ignore

        name = self._choose_hf_model()
        logger.info("Loading generator %s on %s", name, self.device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.device_str == "cuda" and self.quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["device_map"] = "auto"
            except Exception as exc:
                logger.warning("bitsandbytes 4-bit load failed (%s); "
                               "falling back to fp16.", exc)
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
        elif self.device_str == "mps":
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(name, **load_kwargs)
        if "device_map" not in load_kwargs:
            self._model.to(self.device)
        self._model.eval()

        if self.adapter_path:
            try:
                from peft import PeftModel  # type: ignore
                logger.info("Attaching PEFT adapter %s", self.adapter_path)
                self._model = PeftModel.from_pretrained(
                    self._model, self.adapter_path
                )
                self._model.eval()
            except ImportError:
                logger.warning("peft not installed; adapter not attached.")
            except Exception as exc:
                logger.warning("Failed to load adapter: %s", exc)

    def _load_gguf(self) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required for the GGUF backend."
            ) from exc
        if not self.gguf_path or not Path(self.gguf_path).exists():
            raise RuntimeError(
                f"GGUF model not found at {self.gguf_path}. "
                "Run scripts/convert_to_gguf.py first."
            )
        n_ctx = int(getattr(self.cfg.models.generator, "max_seq_length", 4096))

        # Resolve n_gpu_layers: env var > config > device-based default.
        # MPS (Apple Silicon) crashes with -1 on many quantised GGUF models.
        import os as _os
        _env_gpu = (
            _os.getenv("AEGIS_MODELS__GENERATOR__GGUF_N_GPU_LAYERS")
            or _os.getenv("AEGIS_GGUF_N_GPU_LAYERS")
        )
        if _env_gpu is not None:
            n_gpu_layers = int(_env_gpu)
        else:
            _cfg_n_gpu = getattr(self.cfg.models.generator, "gguf_n_gpu_layers", -1)
            if self.device_str == "mps":
                n_gpu_layers = 0   # Metal matmul crashes on many quant shapes
            elif self.device_str == "cuda":
                n_gpu_layers = int(_cfg_n_gpu)
            else:
                n_gpu_layers = 0

        self._llama = Llama(
            model_path=str(self.gguf_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            seed=42,
        )
        logger.info("GGUF loaded: n_gpu_layers=%d ctx=%d", n_gpu_layers, n_ctx)

    # ------------------------------------------------------------------
    # HF generation / streaming
    # ------------------------------------------------------------------

    def _generate_hf(
        self, prompt: str, max_new_tokens: int, temperature: float
    ) -> str:
        import torch  # type: ignore

        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            raise RuntimeError("HF model/tokenizer is not loaded.")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        gen_kwargs = dict(self._generation_kwargs)
        gen_kwargs["max_new_tokens"] = max_new_tokens

        if temperature <= 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0, inputs.input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _stream_hf(self, prompt: str, max_new_tokens: int) -> Iterator[str]:
        from transformers import TextIteratorStreamer  # type: ignore
        import torch  # type: ignore

        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            raise RuntimeError("HF model/tokenizer is not loaded.")

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_kwargs = dict(self._generation_kwargs)
        gen_kwargs["max_new_tokens"] = max_new_tokens
        gen_kwargs["streamer"] = streamer
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

        def _run() -> None:
            with torch.no_grad():
                model.generate(**inputs, **gen_kwargs)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            for tok in streamer:
                if tok:
                    yield tok
        finally:
            thread.join(timeout=0.01)

    # ------------------------------------------------------------------
    # GGUF generation / streaming
    # ------------------------------------------------------------------

    def _generate_gguf(
        self, prompt: str, max_new_tokens: int, temperature: float
    ) -> str:
        llama = self._llama
        if llama is None:
            raise RuntimeError("GGUF backend is not loaded.")
        out = llama(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            echo=False,
            stop=_STOP,
        )
        return out["choices"][0]["text"].strip()

    def _stream_gguf(self, prompt: str, max_new_tokens: int) -> Iterator[str]:
        llama = self._llama
        if llama is None:
            raise RuntimeError("GGUF backend is not loaded.")
        for chunk in llama(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            echo=False,
            stream=True,
            stop=_STOP,
        ):
            piece = chunk["choices"][0]["text"]
            if piece:
                yield piece
