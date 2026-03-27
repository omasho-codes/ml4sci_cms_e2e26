"""lgatr_kernels: Custom GPU kernels for the L-GATr architecture.

Drop-in optimization for the official ``lgatr`` package.

Quick start (eager, 1.5x)::

    from lgatr_kernels import patch_lgatr, fuse_equi_linear_layers
    patch_lgatr()
    model = LorentzGATr(config).cuda()
    fuse_equi_linear_layers(model)

Best performance (compile, 2.5x)::

    from lgatr_kernels import optimize_lgatr_model
    model = LorentzGATr(config).cuda()
    model, stats = optimize_lgatr_model(
        model, use_compile_patches=True, compile_mode="reduce-overhead",
    )
"""

from .compile_patches import patch_lgatr_compile
from .layers import FusedEquiLinear, fuse_equi_linear_layers
from .primitives import patch_lgatr
from .runtime import LorentzGATrGraphWrapper, optimize_lgatr_model
