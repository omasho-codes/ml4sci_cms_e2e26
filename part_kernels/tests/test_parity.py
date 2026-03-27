"""
Numerical parity tests: custom Triton kernels vs reference PyTorch.

Run:
    python -m part_kernels.tests.test_parity       (from /root/e2e)
"""
import math
import sys

import torch

torch.manual_seed(42)

DEVICE = 'cuda'
EPS = 1e-8


# =====================================================================
# Helpers
# =====================================================================

def _ref_pairwise(v):
    """Reference pairwise_lv_fts using original weaver code."""
    from weaver.nn.model.ParticleTransformer import pairwise_lv_fts
    xi = v.unsqueeze(-1)
    xj = v.unsqueeze(-2)
    return pairwise_lv_fts(xi, xj, num_outputs=4, eps=EPS)


def _make_physical_4vec(N, P, device):
    """Generate physically valid 4-vectors (E > |p|)."""
    p3 = torch.randn(N, 3, P, device=device) * 10
    mass = torch.rand(N, 1, P, device=device) * 0.5 + 0.1
    E = (p3.square().sum(dim=1, keepdim=True) + mass.square()).sqrt()
    return torch.cat([p3, E], dim=1)


# =====================================================================
# 1.  Pairwise Lorentz features
# =====================================================================

def test_pairwise_kernel():
    from part_kernels.triton.pairwise_kernel import fused_pairwise_lv_fts

    for N, P in [(1, 16), (4, 32), (8, 64), (16, 128)]:
        v = _make_physical_4vec(N, P, DEVICE)

        ref = _ref_pairwise(v)
        opt = fused_pairwise_lv_fts(v, eps=EPS)

        assert ref.shape == opt.shape, f"Shape mismatch: {ref.shape} vs {opt.shape}"

        finite = ref.isfinite() & opt.isfinite()
        diff = (ref - opt).abs()
        max_diff = diff[finite].max().item() if finite.any() else 0.0
        rel_diff = (diff[finite] / (ref[finite].abs() + 1e-12)).max().item() if finite.any() else 0.0

        status = "PASS" if max_diff < 1e-2 else "FAIL"
        print(f"  pairwise N={N:3d} P={P:3d}  max_abs={max_diff:.2e}  max_rel={rel_diff:.2e}  [{status}]")
        if status == "FAIL":
            return False
    return True


# =====================================================================
# 2.  Fused attention with bias
# =====================================================================

def _ref_attention(Q, K, V, bias, pad_mask, scale, num_heads):
    """Reference attention using plain PyTorch."""
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if bias is not None:
        S = S + bias
    if pad_mask is not None:
        NH = Q.size(0)
        N = NH // num_heads
        expanded = pad_mask.unsqueeze(1).expand(N, num_heads, -1)
        expanded = expanded.reshape(NH, 1, Q.size(1))
        S = S.masked_fill(expanded.expand_as(S), float('-inf'))
    attn = torch.softmax(S, dim=-1)
    attn = torch.nan_to_num(attn)
    return torch.matmul(attn, V)


def test_attention_kernel():
    from part_kernels.autograd.attention import fused_attention_with_bias

    for N, H, P, D in [(2, 8, 32, 16), (4, 8, 64, 16), (8, 8, 128, 16)]:
        NH = N * H
        Q = torch.randn(NH, P, D, device=DEVICE, dtype=torch.float32)
        K = torch.randn(NH, P, D, device=DEVICE, dtype=torch.float32)
        V = torch.randn(NH, P, D, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(NH, P, P, device=DEVICE, dtype=torch.float32) * 0.1
        pad_mask = torch.zeros(N, P, device=DEVICE, dtype=torch.bool)
        pad_mask[:, -P // 4:] = True

        scale = 1.0 / math.sqrt(D)

        ref = _ref_attention(Q, K, V, bias, pad_mask, scale, H)
        opt = fused_attention_with_bias(Q, K, V, bias, pad_mask, scale, H)

        diff = (ref - opt).abs()
        max_diff = diff.max().item()
        rel_diff = (diff / (ref.abs() + 1e-12)).max().item()

        status = "PASS" if max_diff < 1e-2 else "FAIL"
        print(f"  attention N={N} H={H} P={P:3d} D={D}  max_abs={max_diff:.2e}  max_rel={rel_diff:.2e}  [{status}]")
        if status == "FAIL":
            return False
    return True


# =====================================================================
# 3.  Attention backward
# =====================================================================

def test_attention_backward():
    from part_kernels.autograd.attention import fused_attention_with_bias

    N, H, P, D = 2, 8, 32, 16
    NH = N * H
    scale = 1.0 / math.sqrt(D)

    Q = torch.randn(NH, P, D, device=DEVICE).requires_grad_(True)
    K = torch.randn(NH, P, D, device=DEVICE).requires_grad_(True)
    V = torch.randn(NH, P, D, device=DEVICE).requires_grad_(True)
    bias = (torch.randn(NH, P, P, device=DEVICE) * 0.1).requires_grad_(True)

    out = fused_attention_with_bias(Q, K, V, bias, None, scale, H)
    loss = out.sum()
    loss.backward()

    has_grads = all(t.grad is not None for t in [Q, K, V, bias])
    grads_finite = all(t.grad.isfinite().all() for t in [Q, K, V, bias]) if has_grads else False

    status = "PASS" if has_grads and grads_finite else "FAIL"
    print(f"  attention backward  grads_exist={has_grads}  grads_finite={grads_finite}  [{status}]")
    return status == "PASS"


# =====================================================================
# 4.  Full model parity
# =====================================================================

def test_model_parity():
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from part_kernels.layers.optimized_model import OptimizedParticleTransformer

    orig = ParticleTransformer(
        input_dim=17, num_classes=10,
        pair_input_dim=4, pair_extra_dim=0,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8, num_layers=2, num_cls_layers=1,
        trim=False, for_inference=False, use_amp=False,
    ).to(DEVICE).eval()

    opt = OptimizedParticleTransformer.from_pretrained(orig).to(DEVICE).eval()

    N, P = 4, 64
    x = torch.randn(N, 17, P, device=DEVICE)
    v = _make_physical_4vec(N, P, DEVICE)
    mask = torch.ones(N, 1, P, device=DEVICE, dtype=torch.float32)

    with torch.no_grad():
        ref_out = orig(x, v=v, mask=mask)
        opt_out = opt(x, v=v, mask=mask)

    diff = (ref_out - opt_out).abs()
    max_diff = diff.max().item()
    rel_diff = (diff / (ref_out.abs() + 1e-12)).max().item()

    status = "PASS" if max_diff < 0.05 else "FAIL"
    print(f"  model parity  max_abs={max_diff:.2e}  max_rel={rel_diff:.2e}  [{status}]")
    return status == "PASS"


# =====================================================================

def main():
    results = {}

    print("=== Pairwise kernel parity ===")
    results['pairwise'] = test_pairwise_kernel()

    print("\n=== Attention kernel parity ===")
    results['attention_fwd'] = test_attention_kernel()

    print("\n=== Attention backward ===")
    results['attention_bwd'] = test_attention_backward()

    print("\n=== Full model parity ===")
    results['model'] = test_model_parity()

    print("\n" + "=" * 50)
    all_pass = all(results.values())
    for name, ok in results.items():
        print(f"  {name:20s} {'PASS' if ok else 'FAIL'}")
    print("=" * 50)
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
