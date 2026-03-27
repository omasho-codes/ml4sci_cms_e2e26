"""part_kernels: Custom GPU kernels for the Particle Transformer.

Drop-in optimization for weaver's ``ParticleTransformer``.

Quick start (eval, fused kernels)::

    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from part_kernels import OptimizedParticleTransformer

    orig = ParticleTransformer(...).cuda()
    opt = OptimizedParticleTransformer.from_pretrained(orig).eval()

Best performance (compile)::

    from part_kernels import optimize_part_model

    orig = ParticleTransformer(...).cuda()
    model, stats = optimize_part_model(orig, compile_mode="reduce-overhead")
"""

from .autograd.attention import fused_attention_with_bias
from .layers.cuda_graph_wrapper import CUDAGraphInferenceWrapper
from .layers.fused_pair_mlp import FusedPairMLP
from .layers.optimized_model import (
    OptimizedBlock,
    OptimizedPairEmbed,
    OptimizedParticleTransformer,
)
from .runtime import optimize_part_model
from .triton.pairwise_kernel import fused_pairwise_lv_fts
