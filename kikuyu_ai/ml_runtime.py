"""Small runtime helpers for memory-constrained local inference."""

from __future__ import annotations

import gc
import importlib.util
import os
from typing import Any


# The app uses PyTorch-only Transformers models. Disable optional framework
# probes before importing transformers so broken user-level TensorFlow installs
# do not break AutoProcessor/AutoTokenizer imports.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def configure_torch(torch: Any) -> None:
    """Keep CPU inference from over-allocating worker memory on mobile devices."""
    threads = int(os.environ.get("KIKUYU_TORCH_THREADS", "1") or "1")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, str(threads))
    try:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(threads, 2)))
    except RuntimeError:
        pass


def local_pretrained_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"local_files_only": True}
    if importlib.util.find_spec("accelerate") is not None:
        kwargs["low_cpu_mem_usage"] = True
    return kwargs


def release_torch_memory(torch: Any | None = None) -> None:
    gc.collect()
    if torch is not None:
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and callable(getattr(cuda, "empty_cache", None)):
            cuda.empty_cache()
