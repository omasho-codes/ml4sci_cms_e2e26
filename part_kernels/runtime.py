"""High-level runtime helpers for optimized ParticleTransformer workflows."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from .layers.optimized_model import OptimizedBlock, OptimizedParticleTransformer


def optimize_part_model(
    model: nn.Module,
    *,
    compile_mode: str | None = None,
    use_fused_blocks: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Apply the recommended ParT runtime optimizations to a model.

    Parameters
    ----------
    model:
        A weaver ``ParticleTransformer`` instance (or compatible).
    compile_mode:
        If not None, wraps the model in ``torch.compile(model, mode=...)``.
    use_fused_blocks:
        If True (default), replaces standard attention blocks with
        ``OptimizedBlock`` using fused Triton attention.

    Returns
    -------
    (model, stats):
        The optimized model and a small metadata dictionary.
    """
    opt = OptimizedParticleTransformer.from_pretrained(model)

    if not use_fused_blocks:
        opt.blocks = nn.ModuleList([copy.deepcopy(b) for b in model.blocks])

    stats: dict[str, Any] = {
        "fused_blocks": use_fused_blocks,
        "compile_mode": compile_mode,
        "params": sum(p.numel() for p in opt.parameters()),
    }

    if compile_mode is not None:
        opt = torch.compile(opt, mode=compile_mode)

    return opt, stats
