"""End-to-end LorentzGATr benchmark on QuarkGluon data.

Part 1 -- Kernel-level microbenchmarks:
  Each primitive (EquiLinear, geometric product, LayerNorm, gated GELU)
  is benchmarked baseline vs optimized at shapes matching the default
  LorentzGATr config (B=128, items=128, mv=8, s=16).

Part 2 -- Model-level tiers:
  1. baseline              -- stock lgatr from pip, zero patches
  2. +kernels              -- FusedEquiLinear + Triton GP (proven faster only)
  3. +compile (with patches)      -- above + compile-friendly patches + torch.compile

Each model tier runs in a **fresh subprocess** so global monkey-patches
from one tier never contaminate another.

Reports latency, speedup, and peak GPU memory throughout.

Usage:
    /root/.venv/bin/python lgatr_kernels/benchmarks/bench_e2e.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON = sys.executable
DATA_PATH = os.path.join(ROOT, "data/QuarkGluon/val.npz")
BS = 128


# -----------------------------------------------------------------------
# Worker script executed in a subprocess for each tier
# -----------------------------------------------------------------------

_WORKER = r'''
import gc, json, os, sys, time
ROOT = {root!r}
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "hybrid_transformer/MAEs/Hybrid_Transformer_Thanh_Nguyen"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda"
BS = {bs}
DATA_PATH = {data_path!r}
TIER = {tier}

def mark_step():
    fn = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if fn is not None:
        fn()

def clear():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()

def load_batch():
    raw = np.load(DATA_PATH)
    X = torch.from_numpy(raw["X"][:BS]).float().to(DEVICE)
    y = torch.from_numpy(raw["y"][:BS]).long().to(DEVICE)
    return X, y

def build_model():
    from src.models import LorentzGATr
    from src.configs import LGATrConfig
    cfg = LGATrConfig(
        num_classes=2, embed_dim=128, num_heads=8, num_layers=8,
        num_cls_layers=2, num_mlp_layers=0, hidden_dim=256,
        hidden_mv_channels=8, hidden_s_channels=16,
        attention={{}}, mlp={{}},
        dropout=0.1, expansion_factor=4, max_num_particles=128,
        num_particle_features=4, mask=False, weights=None, inference=False,
    )
    return LorentzGATr(cfg).to(DEVICE)

def bench_inference(model, x, warmup=15, timed=40):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            mark_step(); model(x)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(timed):
            mark_step(); torch.cuda.synchronize()
            t0 = time.perf_counter(); model(x); torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000

def bench_training(model, x, y, warmup=8, timed=20):
    model.train()
    for _ in range(warmup):
        mark_step(); model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y); loss.backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(timed):
        mark_step(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y); loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000

def measure_mem(fn):
    clear()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6

x, y = load_batch()
torch.manual_seed(42)

if TIER == 1:
    model = build_model()
elif TIER == 2:
    from lgatr_kernels.primitives import patch_lgatr
    from lgatr_kernels.layers import fuse_equi_linear_layers
    patch_lgatr()
    model = build_model()
    fuse_equi_linear_layers(model)
elif TIER == 3:
    from lgatr_kernels.primitives import patch_lgatr
    from lgatr_kernels.layers import fuse_equi_linear_layers
    from lgatr_kernels.compile_patches import patch_lgatr_compile
    patch_lgatr()
    patch_lgatr_compile()
    model = build_model()
    fuse_equi_linear_layers(model)
    model.eval()
    with torch.no_grad():
        model(x)
    model = torch.compile(model, mode="reduce-overhead")

params = sum(p.numel() for p in model.parameters())

infer_ms = bench_inference(model, x)

def _infer_mem_fn():
    model.eval()
    with torch.no_grad():
        model(x)
infer_mem = measure_mem(_infer_mem_fn)

if TIER == 3:
    del model; clear()
    torch.manual_seed(42)
    from lgatr_kernels.primitives import patch_lgatr
    from lgatr_kernels.layers import fuse_equi_linear_layers
    from lgatr_kernels.compile_patches import patch_lgatr_compile
    patch_lgatr(); patch_lgatr_compile()
    model = build_model(); fuse_equi_linear_layers(model)
    model.eval()
    with torch.no_grad():
        model(x)
    model = torch.compile(model, mode="reduce-overhead")

train_ms = bench_training(model, x, y)

def _train_mem_fn():
    model.train()
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
train_mem = measure_mem(_train_mem_fn)

print(json.dumps({{
    "tier": TIER,
    "params": params,
    "infer_ms": round(infer_ms, 2),
    "train_ms": round(train_ms, 2),
    "infer_mem": round(infer_mem, 1),
    "train_mem": round(train_mem, 1),
}}))
'''


# -----------------------------------------------------------------------
# Kernel-level microbenchmark worker (also runs in subprocess)
# -----------------------------------------------------------------------

_KERNEL_WORKER = r'''
import gc, json, os, sys, time
ROOT = {root!r}
sys.path.insert(0, ROOT)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda"
B, N, MV, S = 128, 128, 8, 16

def clear():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

def bench(fn, warmup=20, timed=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times)//2] * 1e6  # microseconds

def peak_mem(fn):
    clear(); fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6

results = []

# --- EquiLinear ---
from lgatr.layers.linear import EquiLinear
from lgatr_kernels.layers.fused_linear import FusedEquiLinear

el_base = EquiLinear(MV, MV, in_s_channels=S, out_s_channels=S).cuda().eval()
el_fused = FusedEquiLinear(el_base).cuda().eval()

mv_in = torch.randn(B, N, MV, 16, device=DEVICE)
sc_in = torch.randn(B, N, S, device=DEVICE)

with torch.no_grad():
    t_base = bench(lambda: el_base(mv_in, scalars=sc_in))
    t_fused = bench(lambda: el_fused(mv_in, scalars=sc_in))
    m_base = peak_mem(lambda: el_base(mv_in, scalars=sc_in))
    m_fused = peak_mem(lambda: el_fused(mv_in, scalars=sc_in))
results.append({{"name": "EquiLinear (fwd)", "base_us": round(t_base,1), "opt_us": round(t_fused,1), "base_mb": round(m_base,1), "opt_mb": round(m_fused,1)}})
del el_base, el_fused; clear()

# --- Geometric Product ---
from lgatr.primitives.bilinear import geometric_product as ref_gp
from lgatr_kernels.autograd.geometric_product import triton_geometric_product

gp_x = torch.randn(B, N, MV, 16, device=DEVICE)
gp_y = torch.randn(B, N, MV, 16, device=DEVICE)

with torch.no_grad():
    t_base = bench(lambda: ref_gp(gp_x, gp_y))
    t_opt = bench(lambda: triton_geometric_product(gp_x, gp_y))
    m_base = peak_mem(lambda: ref_gp(gp_x, gp_y))
    m_opt = peak_mem(lambda: triton_geometric_product(gp_x, gp_y))
results.append({{"name": "Geometric Product (fwd)", "base_us": round(t_base,1), "opt_us": round(t_opt,1), "base_mb": round(m_base,1), "opt_mb": round(m_opt,1)}})
del gp_x, gp_y; clear()

# --- Equivariant LayerNorm ---
from lgatr.primitives.normalization import equi_layer_norm as ref_ln
from lgatr_kernels.autograd.equi_layernorm import triton_equi_layer_norm

ln_x = torch.randn(B, N, MV, 16, device=DEVICE)

with torch.no_grad():
    t_base = bench(lambda: ref_ln(ln_x))
    t_opt = bench(lambda: triton_equi_layer_norm(ln_x))
    m_base = peak_mem(lambda: ref_ln(ln_x))
    m_opt = peak_mem(lambda: triton_equi_layer_norm(ln_x))
results.append({{"name": "Equi LayerNorm (fwd)", "base_us": round(t_base,1), "opt_us": round(t_opt,1), "base_mb": round(m_base,1), "opt_mb": round(m_opt,1)}})
del ln_x; clear()

# --- Gated GELU ---
from lgatr.primitives.nonlinearities import gated_gelu as ref_gelu
from lgatr_kernels.autograd.gated_gelu import triton_gated_gelu

gg_x = torch.randn(B, N, MV, 16, device=DEVICE)
gg_gates = gg_x[..., 0:1]

with torch.no_grad():
    t_base = bench(lambda: ref_gelu(gg_x, gg_gates))
    t_opt = bench(lambda: triton_gated_gelu(gg_x))
    m_base = peak_mem(lambda: ref_gelu(gg_x, gg_gates))
    m_opt = peak_mem(lambda: triton_gated_gelu(gg_x))
results.append({{"name": "Gated GELU (fwd)", "base_us": round(t_base,1), "opt_us": round(t_opt,1), "base_mb": round(m_base,1), "opt_mb": round(m_opt,1)}})

print(json.dumps(results))
'''


def run_kernel_microbench() -> list[dict]:
    code = _KERNEL_WORKER.format(root=ROOT)
    result = subprocess.run(
        [PYTHON, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print("  Kernel microbench FAILED")
        for line in result.stderr.strip().split("\n")[-8:]:
            print(f"    {line}")
        return []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("["):
            return json.loads(line)
    return []


def run_tier(tier: int) -> dict:
    code = _WORKER.format(root=ROOT, bs=BS, data_path=DATA_PATH, tier=tier)
    result = subprocess.run(
        [PYTHON, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  TIER {tier} FAILED (exit {result.returncode})")
        for line in result.stderr.strip().split("\n")[-10:]:
            print(f"    {line}")
        return {"tier": tier, "error": True}

    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)

    print(f"  TIER {tier}: no JSON output found")
    print(f"  stdout: {result.stdout[-500:]}")
    return {"tier": tier, "error": True}


def main():
    import torch

    print("=" * 78)
    print(f"  LorentzGATr Benchmark on QuarkGluon (bs={BS})")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Data: {DATA_PATH}")
    print(f"  Each tier runs in a FRESH subprocess (no cross-contamination)")
    print("=" * 78)

    # === Part 1: kernel-level microbenchmarks ===
    print("\n  --- Part 1: Kernel-Level Microbenchmarks ---")
    print(f"  Shape: B={BS}, items=128, mv_channels=8, s_channels=16\n")

    kernels = run_kernel_microbench()
    if kernels:
        print(f"  {'Kernel':<28s} {'Baseline':>10s} {'Optimized':>10s} {'Speedup':>8s} {'BaseMem':>8s} {'OptMem':>8s}")
        print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        for k in kernels:
            sp = f"{k['base_us']/k['opt_us']:.2f}x" if k['opt_us'] > 0 else "N/A"
            print(f"  {k['name']:<28s} {k['base_us']:>9.0f}us {k['opt_us']:>9.0f}us {sp:>8s} {k['base_mb']:>7.0f}MB {k['opt_mb']:>7.0f}MB")
    else:
        print("  (kernel microbench failed or returned no data)")

    # === Part 2: model-level tiers ===
    print("\n  --- Part 2: Model-Level Tiers ---\n")

    tier_names = {
        1: "1. baseline (stock lgatr)",
        2: "2. +kernels (fuse+GP)",
        3: "3. +compile (with patches)",
    }

    results = []
    for tier in [1, 2, 3]:
        print(f"\n  Running {tier_names[tier]} ...")
        r = run_tier(tier)
        r["name"] = tier_names[tier]
        results.append(r)
        if "error" not in r:
            print(f"    infer={r['infer_ms']:.2f}ms  train={r['train_ms']:.2f}ms  "
                  f"infer_mem={r['infer_mem']:.0f}MB  train_mem={r['train_mem']:.0f}MB")

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\n  All tiers failed.")
        return

    base = valid[0]

    print("\n" + "=" * 78)
    print("  RESULTS")
    print("=" * 78)
    print(f"\n  {'Tier':<32s} {'Infer':>8s} {'Spd':>6s} {'IMem':>7s} "
          f"{'Train':>8s} {'Spd':>6s} {'TMem':>7s}")
    print(f"  {'-'*32} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*6} {'-'*7}")
    for r in results:
        if "error" in r:
            print(f"  {r['name']:<32s} {'FAIL':>8s}")
            continue
        i_sp = f"{base['infer_ms']/r['infer_ms']:.1f}x"
        t_sp = f"{base['train_ms']/r['train_ms']:.1f}x"
        print(
            f"  {r['name']:<32s} {r['infer_ms']:>7.1f}ms {i_sp:>6s} {r['infer_mem']:>6.0f}MB "
            f"{r['train_ms']:>7.1f}ms {t_sp:>6s} {r['train_mem']:>6.0f}MB"
        )

    if valid:
        print(f"\n  Model: LorentzGATr, {valid[0]['params']:,} params")
    print(f"  Config: 8 LGATr blocks (mv=8, s=16), num_classes=2")
    print(f"  Data: QuarkGluon val, {BS} real jets, 128 particles, 4 features (pT, eta, phi, E)")
    print(f"  Isolation: each tier ran in a fresh Python subprocess")
    print()


if __name__ == "__main__":
    main()
