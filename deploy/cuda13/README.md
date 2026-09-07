# CUDA-13 install (Blackwell-Ultra: sm103, e.g. B300/GB300)

> **Status after merging upstream main (2026-09):** the root project is now
> CUDA-13 natively (torch 2.11 cu130, vLLM 0.28.0, transformer-engine 2.16,
> flashinfer 0.6.16.post3, megatron-bridge 0.7.0, megatron-core 0.20,
> transformers <= 5.16.1, ray 2.57), so the premise below no longer holds and
> the pins in this directory (vLLM 0.26.0, megatron-bridge 0.6.0, TE 2.11,
> flashinfer 0.6.13) are *older* than the root lock. The merged code already
> assumes the newer stack: `workers/megatron/model_bridges.py` imports
> `moe_experts_stored_packed` (megatron-bridge >= 0.7.0); on a 0.6.0 venv the
> import fails, every SkyRL bridge registration is skipped (a warning is
> logged), and Kimi K2.5-family checkpoints dispatch to megatron-bridge's VL
> bridge instead of the text-only one. Regenerate this manifest from the
> root lock (`regenerate.sh` now works directly on main's CUDA-13 lock; update
> `overrides.txt`/`install.sh` pins to match) and re-verify on B300 before
> deploying the merged branch. The text below describes the previously
> verified 0.26 stack.

The root project used to target CUDA 12.8 images. Blackwell-Ultra GPUs need the
CUDA-13 stack instead: PyPI torch 2.11 (the cu130 build), the PyPI vllm 0.26.0
wheel (its default build is CUDA 13.0; sm_100 cubins are family-compatible
with sm103), flashinfer's cu130 jit-cache (the only build with sm103 kernels),
the `transformer-engine-cu13` core, and `nixl-cu13`. `flash-attn` is omitted:
there is no torch-2.11/cu13 build, and a CUDA-less stub crashes
TransformerEngine (it version-gates, then unconditionally imports
`flash_attn_2_cuda`). Without it TE uses cuDNN fused attention and HF models
fall back to SDPA.

Version couplings that only bite on Hopper/Blackwell (capability >= 9.0, i.e.
never on upstream CI GPUs):

- `quack-kernels` must match `nvidia-cutlass-dsl`'s CuTeDSL API: quack 0.5.x
  with cutlass-dsl 4.6.0 crashes vLLM's ll_bf16 kernel warmup at engine boot
  (`module 'cutlass.cute.core' has no attribute 'ThrMma'`). vLLM 0.26.0's
  loose `quack-kernels>=0.4.0` does not protect against this; the pin here
  (0.6.1) matches the root lock's resolution.
- flashinfer is held at 0.6.13 across python/cubin/jit-cache: vLLM 0.26.0's
  metadata asks for 0.6.14, but cubin 0.6.14 was never released and
  flashinfer hard-errors on a version mismatch (see `overrides.txt`).

This directory ships that stack as a pinned export (`requirements-megatron.txt`)
instead of an extra because uv cannot express a default-preserving CUDA
variant inside one project today: `extra`/`group` markers do not bind in
sources or overrides, group-scoped sources require the packages in that group
(pulling multi-GB defaults into bare `uv sync`), and a sub-project wrapper
inherits the parent's source mappings. The canonical uv alternative — separate
`cuda12`/`cuda13` extras where *every* install names its variant — changes all
existing install commands and is left as an upstream decision.

## Install

```bash
# NVIDIA driver >= R580 required; CUDA >= 13.0 toolkit for the sdist builds.
export CUDA_HOME=/usr/local/cuda-13.0
bash deploy/cuda13/install.sh
```

The resulting venv is managed through the uv pip interface; do not run
`uv sync` against it (that re-applies the root CUDA-12 lock).

## Verified

2x8 B300 (driver R580, CUDA 13.3 toolkit): Kimi K2.7 (1T INT4) GRPO LoRA
training with TP2/CP8/EP16 and 2x TP8 vLLM engines completes multi-step
training (see `examples/train/megatron/run_megatron_dapo_kimi_k2.7_code_lora_int4_qat.sh`).

With vLLM 0.26.0 the same topology additionally runs DFlash speculative
decoding (drafter on FLASH_ATTN, bf16 KV via
`speculative_config.kv_cache_dtype: auto`) together with an fp8 KV cache on
the FLASHINFER_MLA target (~1.86x KV tokens at 262k context), verified
through the tinker API: weight sync, multi-adapter churn and sampling.

## Regenerating the pins

The pins are exported from the CUDA-13 resolution of the root project (a
branch whose root `pyproject.toml`/`uv.lock` are the CUDA-13 stack):

```bash
bash deploy/cuda13/regenerate.sh
```

`overrides.txt` mirrors the root project's `override-dependencies` (with the
CUDA variant flipped); keep it in sync when the root overrides change.
