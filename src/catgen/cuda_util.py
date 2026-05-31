"""Release GPU memory between training runs on shared GPUs."""
from __future__ import annotations

import gc


def clear_cuda_cache() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
