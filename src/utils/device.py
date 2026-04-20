"""
Device detection and management utilities for AegisRAG.

Provides automatic device selection (CUDA > MPS > CPU), dtype mapping,
quantization config construction, and GPU memory estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def auto_detect_device() -> str:
    """
    Detect the best available compute device.

    Returns:
        "cuda" if NVIDIA GPU is available,
        "mps" if Apple Silicon GPU is available,
        "cpu" otherwise.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_device_string(preferred: Optional[str] = None) -> str:
    """Resolve a possibly-None / 'auto' preference into a concrete device string."""
    if preferred is None or preferred == "auto":
        return auto_detect_device()
    return preferred


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Return a :class:`torch.device` for the preferred or auto-detected backend.

    Accepts ``None`` or ``"auto"`` to mean "auto-detect".
    """
    return torch.device(_resolve_device_string(preferred))


def get_device_string(preferred: Optional[object] = None) -> str:
    """Return a canonical device string ("cuda" | "mps" | "cpu").

    Accepts ``None``, a preference string (including ``"auto"``), or a
    :class:`torch.device` instance — so call sites can pass whichever is
    convenient.
    """
    if isinstance(preferred, torch.device):
        # torch.device stringifies as e.g. "cuda:0"; take the type prefix.
        return str(preferred).split(":", 1)[0]
    if preferred is None or isinstance(preferred, str):
        return _resolve_device_string(preferred)
    # Fallback for anything that stringifies to a device name.
    return _resolve_device_string(str(preferred))


def get_torch_dtype(device: Optional[str] = None) -> torch.dtype:
    """
    Return the appropriate floating-point dtype for the given device.

    Args:
        device: One of "cuda", "mps", "cpu", or None (auto-detect).

    Returns:
        torch.bfloat16 or torch.float16 for CUDA, torch.float32 for MPS/CPU.
    """
    if device is None:
        device = auto_detect_device()

    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    # MPS has limited fp16 support; use float32 for stability
    return torch.float32


def get_quantization_config(
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_use_double_quant: bool = True,
) -> Optional[object]:
    """
    Build a BitsAndBytesConfig for QLoRA training, or return None
    if CUDA is not available (bitsandbytes requires CUDA).

    Args:
        load_in_4bit: Enable 4-bit quantization.
        load_in_8bit: Enable 8-bit quantization (mutually exclusive with 4-bit).
        bnb_4bit_quant_type: Quantization type ("nf4" or "fp4").
        bnb_4bit_compute_dtype: Compute dtype string ("float16" or "bfloat16").
        bnb_4bit_use_double_quant: Enable nested quantization.

    Returns:
        A BitsAndBytesConfig instance, or None if not on CUDA.
    """
    if not torch.cuda.is_available():
        logger.warning(
            "bitsandbytes quantization requires CUDA. Returning None."
        )
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.warning(
            "transformers not installed or BitsAndBytesConfig unavailable. "
            "Returning None."
        )
        return None

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(bnb_4bit_compute_dtype, torch.float16)

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


@dataclass
class DeviceManager:
    """
    Centralized device management for the AegisRAG pipeline.

    Handles device selection, dtype mapping, quantization config,
    and memory estimation.
    """

    preferred_device: str = "auto"
    gpu_memory_utilization: float = 0.85
    _resolved_device: Optional[str] = field(default=None, init=False, repr=False)

    @property
    def device(self) -> str:
        """Resolved device string."""
        if self._resolved_device is None:
            if self.preferred_device == "auto":
                self._resolved_device = auto_detect_device()
            else:
                self._resolved_device = self.preferred_device
        return self._resolved_device

    @property
    def torch_device(self) -> torch.device:
        """Return a torch.device object."""
        return torch.device(self.device)

    @property
    def dtype(self) -> torch.dtype:
        """Appropriate dtype for the current device."""
        return get_torch_dtype(self.device)

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def is_mps(self) -> bool:
        return self.device == "mps"

    @property
    def is_cpu(self) -> bool:
        return self.device == "cpu"

    def get_quantization_config(self, **kwargs) -> Optional[object]:
        """
        Build a BitsAndBytesConfig using project defaults or keyword overrides.
        Returns None on non-CUDA devices.
        """
        return get_quantization_config(**kwargs)

    # ------------------------------------------------------------------
    # Memory utilities
    # ------------------------------------------------------------------

    def get_gpu_memory_total(self) -> Optional[float]:
        """Total GPU memory in GB, or None if not on CUDA."""
        if not self.is_cuda:
            return None
        props = torch.cuda.get_device_properties(0)
        return props.total_mem / (1024 ** 3)

    def get_gpu_memory_free(self) -> Optional[float]:
        """Free GPU memory in GB, or None if not on CUDA."""
        if not self.is_cuda:
            return None
        free, _ = torch.cuda.mem_get_info(0)
        return free / (1024 ** 3)

    def get_gpu_memory_allocated(self) -> Optional[float]:
        """Currently allocated GPU memory in GB, or None if not on CUDA."""
        if not self.is_cuda:
            return None
        return torch.cuda.memory_allocated(0) / (1024 ** 3)

    def estimate_model_memory_gb(
        self,
        num_params_billions: float,
        bits: int = 16,
    ) -> float:
        """
        Rough estimate of model memory footprint in GB.

        Args:
            num_params_billions: Number of parameters in billions (e.g. 7.0).
            bits: Precision bits (4, 8, 16, 32).

        Returns:
            Estimated memory in GB.
        """
        bytes_per_param = bits / 8
        return num_params_billions * 1e9 * bytes_per_param / (1024 ** 3)

    def can_fit_model(
        self,
        num_params_billions: float,
        bits: int = 16,
    ) -> bool:
        """
        Check whether a model of the given size can fit in available GPU memory.

        Uses gpu_memory_utilization as the fraction of total memory to consider
        available (to leave headroom for activations and OS).
        """
        required = self.estimate_model_memory_gb(num_params_billions, bits)
        total = self.get_gpu_memory_total()
        if total is None:
            return True
        usable = total * self.gpu_memory_utilization
        fits = required <= usable
        if not fits:
            logger.warning(
                "Model needs ~%.1f GB but only ~%.1f GB usable "
                "(%.0f%% of %.1f GB total).",
                required, usable,
                self.gpu_memory_utilization * 100, total,
            )
        return fits

    def empty_cache(self) -> None:
        """Clear GPU cache if on CUDA or MPS."""
        if self.is_cuda:
            torch.cuda.empty_cache()
        elif self.is_mps:
            torch.mps.empty_cache()

    def synchronize(self) -> None:
        """Synchronize the device (useful for timing)."""
        if self.is_cuda:
            torch.cuda.synchronize()
        elif self.is_mps:
            torch.mps.synchronize()

    def summary(self) -> str:
        """Human-readable summary of the device state."""
        lines = [f"Device: {self.device}", f"Dtype: {self.dtype}"]
        if self.is_cuda:
            total = self.get_gpu_memory_total()
            free = self.get_gpu_memory_free()
            alloc = self.get_gpu_memory_allocated()
            lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
            lines.append(
                f"Memory: {alloc:.1f} GB allocated / "
                f"{free:.1f} GB free / {total:.1f} GB total"
            )
        return "\n".join(lines)
