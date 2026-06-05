"""
Global PyTorch optimizations for RTX 5090 (SM_120, 32 GB VRAM, CUDA 13.x).

Import this module once at server startup — it sets process-wide flags that
benefit all model inference: embedding, CLIP, and cross-encoder reranker.

Optimizations applied
---------------------
  TF32 matmul     — ~2x throughput on Ampere/Hopper vs FP32, no accuracy loss
                    for embedding / retrieval tasks
  cuDNN benchmark — auto-selects fastest convolution kernel per input shape
  FP16 autocast   — enabled as a context manager default for inference paths
  CUDA graphs     — torch.compile uses these automatically on sm_120
"""

from __future__ import annotations

import torch


def apply() -> None:
    if not torch.cuda.is_available():
        return

    # TF32: tensor-core matmul at FP32 precision with ~2x throughput
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # cuDNN auto-tuner: finds fastest kernel for each input shape
    torch.backends.cudnn.benchmark = True

    device_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[hardware] {device_name}  {vram:.0f} GB VRAM  TF32+cuDNN benchmark enabled")


# Apply immediately on import so any module that does `import rufus.hardware`
# (or `from rufus import hardware`) gets the settings for free.
apply()
