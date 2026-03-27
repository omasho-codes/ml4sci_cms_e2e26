# Event Classification with Masked Transformer Autoencoders

Custom Triton GPU kernels and optimized runtime for Lorentz-equivariant jet classification. Developed for [ML4SCI](https://ml4sci.org/) GSoC 2026.

**Models:** LorentzGATr, Hybrid LorentzParT, Particle Transformer (ParT)  
**Dataset:** [QuarkGluon](https://zenodo.org/record/3164691) (quark vs gluon jets, 100k events)  
**Hardware:** NVIDIA A100-SXM4-80GB, PyTorch 2.10, Triton 3.6  

---

## Results

### Kernel-Level Speedups

**L-GATr Kernels** (`lgatr_kernels/`) — shape: B=128, items=128, mv=8, s=16

| Kernel | Baseline | Optimized | Speedup | Base mem | Opt mem |
|---|---:|---:|---:|---:|---:|
| EquiLinear | 459 us | 152 us | **3.03x** | 36 MB | 38 MB |
| Geometric Product | 305 us | 136 us | **2.25x** | 178 MB | 44 MB |

**ParT Kernels** (`part_kernels/`) — shape: N=128, P=128, H=8, D=16

| Kernel | Baseline | Optimized | Speedup | Base mem | Opt mem |
|---|---:|---:|---:|---:|---:|
| Pairwise LV features | 766 us | 114 us | **6.69x** | 118 MB | 34 MB |
| Attention + bias | 452 us | 189 us | **2.39x** | 243 MB | 110 MB |
| PairEmbed + MLP | 9,227 us | 2,034 us | **4.54x** | 1,119 MB | 79 MB |

### End-to-End Model Benchmarks (bs=128, QuarkGluon val)

**LorentzGATr** (709K params, 8 LGATr blocks, mv=8, s=16)

| Setting | Inference | Speedup | Train | Speedup | Train mem |
|---|---:|---:|---:|---:|---:|
| 1. baseline (stock lgatr) | 55.3 ms | 1.0x | 179.1 ms | 1.0x | 5,338 MB |
| 2. +kernels (FusedEquiLinear + GP) | 35.6 ms | 1.6x | 110.3 ms | 1.6x | 5,311 MB |
| 3. +kernels + compile | **21.8 ms** | **2.5x** | **61.1 ms** | **2.9x** | **3,444 MB** |

**Particle Transformer** (2.14M params, 8 ParT blocks, embed=[128,512,128])

| Setting | Inference | Speedup | Train | Speedup | Train mem |
|---|---:|---:|---:|---:|---:|
| 1. baseline (stock ParT) | 18.4 ms | 1.0x | 74.2 ms | 1.0x | 8,153 MB |
| 2. +kernels (fused attn + pair) | 7.6 ms | 2.4x | 66.9 ms | 1.1x | 6,968 MB |
| 3. +kernels + CUDA graph | 7.3 ms | 2.5x | -- | -- | -- |
| 4. +kernels + compile | **5.4 ms** | **3.4x** | **36.6 ms** | **2.0x** | **5,077 MB** |

### Training Comparison on QuarkGluon (10 epochs, bs=512)

**LorentzGATr**

| Metric | Baseline | Optimized |
|---|---:|---:|
| Params | 708,992 | 708,992 |
| Best val accuracy | 0.7731 | 0.7732 |
| Best val loss | 0.4857 | 0.4860 |
| Median epoch time | 91.3s | **33.6s** |
| Training speedup | 1.00x | **2.72x** |
| Peak GPU memory | 20,199 MB | **13,074 MB** |

**Hybrid LorentzParT**

| Metric | Baseline | Optimized |
|---|---:|---:|
| Params | 2,270,088 | 2,270,088 |
| Best val accuracy | 0.7735 | 0.7721 |
| Best val loss | 0.4864 | 0.4865 |
| Median epoch time | 45.1s | **22.5s** |
| Training speedup | 1.00x | **2.00x** |
| Peak GPU memory | 30,413 MB | **18,799 MB** |

Accuracy matches baseline within seed variance in all cases.

---

## How the Kernel Fusion Works

### Fusion 1 — EquiLinear -> FusedEquiLinear (single GEMM)

The L-GATr `EquiLinear` layer is constrained to a 10-dimensional subspace of the full 16x16 weight matrix (to commute with Lorentz transformations). The stock implementation computes this as a loop over 10 equivariant basis matrices, launching ~12 CUDA kernels per layer.

**FusedEquiLinear** precomputes the expanded weight matrix `W_eff = sum(w_b * B_b)` once, then performs the entire equivariant linear map as a single `torch.addmm` call on concatenated `[mv_flat, scalars]`. Reduces each of the 50 EquiLinear layers from ~12 kernel launches to 1 forward GEMM + 2 backward GEMMs. Verified to floating-point precision and Lorentz-equivariant under random boosts.

### Fusion 2 — Pairwise Features + Conv1d MLP -> FusedPairMLP (single Triton kernel)

ParT computes 4 Lorentz-invariant features (ln_kT, ln_z, ln_deltaR, ln_m2) for every particle pair, then passes them through a 4-layer Conv1d MLP (4 -> 64 -> 64 -> 64 -> 8) with BatchNorm + GELU. At batch 512 with 128 particles, this creates ~20 intermediate tensors and ~32 GB of HBM traffic.

**FusedPairMLP** folds BatchNorm into linear weights at eval time, then computes the pairwise features AND the entire 4-layer MLP in a single Triton kernel launch, keeping all 64-wide hidden states in GPU registers. The kernel reads particle 4-vectors once from HBM and writes the final (B, H, P, P) attention bias directly. Zero intermediate tensor allocations.

### Fusion 3 — Self-Attention + Additive Bias -> Fused Triton Attention

ParT's attention includes a dense (B*H, P, P) additive bias from the PairEmbed. PyTorch's SDPA falls back to the slow math backend when an attention mask is provided, allocating the full attention matrix in HBM.

My Triton kernel computes `softmax(QK^T / sqrt(d) + bias) * V` in a single pass using FlashAttention-style online softmax, with both forward and backward kernels. Eliminates the (B*H, P, P) attention matrix allocation.

### Compile-Friendly Patches

The upstream `lgatr` package uses `@lru_cache` and `cached_einsum` (via `opt_einsum`) for its invariant helper tensors. These are fast in eager mode but create graph breaks under `torch.compile`. My compile patches replace them with pre-built constant tensors and explicit PyTorch ops, enabling `torch.compile(mode='reduce-overhead')` to trace through the full model without breaks. This is what takes the LGATr speedup from 1.6x (kernels only) to 2.9x (kernels + compile).

---

## Quick Start

### Installation

```bash
# Clone
git clone <repo-url> && cd ml4sci_cms_e2e26

# Create venv and install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/heidelberg-hepml/lgatr.git
pip install weaver-core>=0.4

# Download QuarkGluon dataset
mkdir -p data/QuarkGluon
# Place train.npz, val.npz, test.npz in data/QuarkGluon/
```

### Usage: Optimize a LorentzGATr Model

```python
from lgatr_kernels import optimize_lgatr_model

model = LorentzGATr(config).cuda()

# One-liner: patches + fuse + compile
model, stats = optimize_lgatr_model(
    model,
    use_compile_patches=True,
    compile_mode="reduce-overhead",
)
```

### Usage: Optimize a Particle Transformer

```python
from part_kernels import optimize_part_model

model = ParticleTransformer(...).cuda()
model, stats = optimize_part_model(model, compile_mode="reduce-overhead")
```

### Run Benchmarks

```bash
# LorentzGATr: kernel-level + model-level on real QuarkGluon data
python lgatr_kernels/benchmarks/bench_e2e.py

# ParT: kernel-level + model-level
python part_kernels/benchmarks/bench_e2e.py
```

### Train Models

```bash
# LorentzGATr: baseline vs optimized (each in a clean subprocess)
python train_lgatr_comparison.py

# Hybrid LorentzParT: baseline vs optimized
python train_lorentz_part_comparison.py
```

---

## Project Structure

```
ml4sci_cms_e2e26/
├── lgatr_kernels/              # L-GATr kernel package
│   ├── __init__.py             # Public API: patch_lgatr, fuse_equi_linear_layers, optimize_lgatr_model
│   ├── primitives.py           # Monkey-patch: FusedEquiLinear + Triton GP
│   ├── compile_patches.py      # Compile-friendly invariant cache + einsum fast paths
│   ├── runtime.py              # optimize_lgatr_model() + LorentzGATrGraphWrapper
│   ├── layers/
│   │   ├── fused_linear.py     # FusedEquiLinear module + fuse_equi_linear_layers()
│   │   └── cuda_graph_wrapper.py
│   ├── triton/                 # Triton kernels
│   │   ├── geometric_product_kernel.py
│   │   ├── equi_linear_gemm.py
│   │   ├── equi_layernorm_kernel.py
│   │   ├── gated_gelu_kernel.py
│   │   └── attention_prep_kernel.py
│   ├── autograd/               # Custom autograd.Function wrappers
│   ├── codegen/                # Cayley table + basis code generation
│   ├── tests/                  # 36 tests: correctness, gradcheck, equivariance
│   └── benchmarks/
│       └── bench_e2e.py        # Canonical benchmark (subprocess-isolated)
│
├── part_kernels/               # ParT kernel package
│   ├── __init__.py             # Public API: OptimizedParticleTransformer, optimize_part_model
│   ├── runtime.py              # optimize_part_model()
│   ├── layers/
│   │   ├── optimized_model.py  # OptimizedPairEmbed, OptimizedBlock, OptimizedParticleTransformer
│   │   ├── fused_pair_mlp.py   # FusedPairMLP (eval-only, register-fused)
│   │   └── cuda_graph_wrapper.py
│   ├── triton/                 # Triton kernels
│   │   ├── pairwise_kernel.py
│   │   ├── attention_kernel.py
│   │   └── fused_pair_mlp_kernel.py
│   ├── autograd/
│   ├── tests/
│   └── benchmarks/
│       └── bench_e2e.py
│
├── hybrid_transformer/         # Baseline codebase (Thanh Nguyen, GSoC 2025)
│   └── MAEs/Hybrid_Transformer_Thanh_Nguyen/
│       ├── src/                # LorentzParT, LorentzGATr, ParT models + training engine
│       ├── scripts/            # train/eval CLI scripts
│       ├── configs/            # YAML experiment configs
│       └── tests/
│
├── data/QuarkGluon/            # Dataset (not in git)
│   ├── train.npz               # 80k events
│   ├── val.npz                 # 10k events
│   └── test.npz                # 10k events
│
├── train_lgatr_comparison.py           # LorentzGATr baseline vs optimized (subprocess-isolated)
├── train_lorentz_part_comparison.py    # Hybrid LorentzParT baseline vs optimized
├── nohup/                              # Saved benchmark + training outputs
│   ├── lgatr_bench.out
│   ├── lgatr_train_e2e.out
│   ├── hybrid_train_e2e.out
│   └── parT_bench.out
└── archive/                            # Session notes and development history
```

---

## Architecture Comparison

| Model | Params | Encoder | Equivariant layers | Pair/bias path |
|---|---:|---|---:|---|
| ParT | 2,143,486 | 8 ParT blocks, 8 heads | 0 | PairEmbed `4->64->64->64->8` |
| Hybrid LorentzParT | 2,270,088 | 8 ParticleAttention blocks, 8 heads | 2 | InteractionEmbed `[64,64,64]` + 2 EquiLinear |
| Default LorentzGATr | 708,992 | 8 LGATr blocks (`mv=8`, `s=16`) | 50 | None (pure equivariant attention) |

All models use 2 class-attention blocks + LayerNorm + Linear classifier.

---

## Testing

```bash
# L-GATr kernel correctness (36 tests: parity, gradcheck, equivariance)
pytest lgatr_kernels/tests/ -v

# ParT kernel parity
pytest part_kernels/tests/ -v

# Hybrid model tests
pytest hybrid_transformer/MAEs/Hybrid_Transformer_Thanh_Nguyen/tests/ -v
```

---

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- Triton >= 3.0
- lgatr >= 1.4.3 (`pip install git+https://github.com/heidelberg-hepml/lgatr.git`)
- weaver-core >= 0.4

---

## References

- [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772) (Qu et al., ICML 2022)
- [Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics](https://arxiv.org/abs/2405.14806) (Spinner et al., NeurIPS 2024)
- [A Lorentz-Equivariant Transformer for All of the LHC](https://arxiv.org/abs/2411.00446) (Brehmer et al., 2024)
- [Near-Binary Attention in Particle Transformers](https://arxiv.org/abs/2512.00210) (2025)
- [HEP-JEPA](https://arxiv.org/abs/2412.05333) (2024)
