# moe-stream

A Rust inference engine for GGUF MoE models that treats expert weights the
way a game engine treats textures: a fixed VRAM residency budget, experts
streamed in and out of the GPU on demand, all compute staying on the GPU,
and a hard-reserved arena for the KV cache so context is never evicted by
weights.

Built for consumer hardware first: developed and measured on an RX 7900 XTX
(24 GB VRAM, PCIe 4.0 x16), Linux, Vulkan (RADV). The goal is to run models
that are 2-4x the size of VRAM at usable decode speeds — with the same
exact-greedy output as if the whole model were resident.

## Current state (August 2026)

**Working:** Qwen3-MoE (`qwen3moe`) models, any size. Mixed quantization
(q4_K/q5_K/q6_K/q8_0), greedy decode, online expert repinning, telemetry.

```
Qwen3-Coder-30B-A3B (Q4_K_XL, 17.5 GB weights) on RX 7900 XTX:
  all experts in VRAM (18.6 GB) ......... 33.4 tok/s
  6 GiB budget + hot-expert pinning ..... 24.0 tok/s   (91% cache hits)
  6 GiB budget + online repinning ....... +9% on long generations
  experts in system memory only (2.7 GB VRAM) .. 17.75 tok/s
```

Output is greedy-identical across every placement strategy — placement
changes speed, never results.

**The target:** models that genuinely do not fit. Next up is
Qwen3-Next-80B-A3B (46 GB Q4_K_M, 48 layers, 512 experts/layer, top-10):
feasibility math says even with zero cache hits the transfer ceiling is
~26 tok/s, and realistic routing skew makes the real number far better.
That model is currently BLOCKED on architecture support (hybrid SSM layers,
fused QKV, shared experts) — see Roadmap.

## How it works

- **GGUF parser** (`crates/gguf-rs`) — from-scratch Rust parser, verified
  byte-identical against `llama-gguf` on a 17.7 GB model. 13 tests.
- **Vulkan backend** (`crates/vk-backend`) — raw `ash` compute. Kernels are
  llama.cpp's Vulkan shaders vendored and driven from Rust, plus custom
  shaders for the pieces ggml does not have (fused decode attention, top-k
  routing, MoE reduce, strided matvec). Whole tokens are recorded into one
  command buffer; one fence per token.
- **Expert residency** (`crates/model`) — non-expert weights (~1 GB) pinned
  in VRAM; experts live in system memory (GTT) by default. Popularity-based
  pinning copies the hottest experts contiguously into per-layer VRAM
  buffers; online repinning swaps residents every N tokens against a
  live hit histogram. A reserved KV arena is never touched by streaming.

Why streaming beats `--n-cpu-moe` style offloading: compute never leaves
the GPU. A cold expert miss is a ~1-3 MB PCIe gather (measured 25-28 GB/s
on this box), not a CPU matmul. The 0%-hit-rate ceiling for the 30B is
still ~27 tok/s on measured bandwidth — any cache locality makes it better.

## Usage

```bash
cargo build --release

# greedy decode with default placement (experts in system memory)
./target/release/examples/generate_moe \
    /path/to/model.gguf "your prompt"

# 6 GiB expert budget pinned in VRAM, with online repinning every 64 tokens
MOE_EXPERTS_VRAM_MB=6144 MOE_REPIN_INTERVAL=64 \
    ./target/release/examples/generate_moe model.gguf "..."

# telemetry: hot/cold hit rates, PCIe traffic estimate, per-layer heatmap
MOE_HUD=1 ./target/release/examples/generate_moe model.gguf "..."

# check whether any GGUF can run on this engine
./target/release/examples/onboard /path/to/model.gguf
```

Verification: `scripts/verify.sh` (add `SLOW=1` for the long-generation
tier). Builds, runs unit + GPU kernel tests, and replays a 64-token golden
at three placement strategies.

## Design notes worth knowing

- **Sparse bindings lose.** Vulkan sparse buffers (per-64 KiB hot/cold
  binding) cost ~2x on RADV regardless of placement. Dense contiguous
  copies win everywhere we tried them.
- **Popularity histograms generalize poorly.** A histogram traced on one
  prompt covers ~91% of hits on itself but only ~53% on a different prompt.
  That finding is what motivated online repinning.
- **GTT reads are stall-bound, not bandwidth-bound.** A compute shader
  gathering contiguous data from system memory runs at ~28 GB/s — but the
  expert matvec reading quant blocks in place achieves only ~5.8 GiB/s.
  The fix (M7b, in progress) is to gather cold slabs into a small VRAM
  scratch arena first, then compute from VRAM.
- **Next-token prefetch does not work.** Token-to-token expert overlap is
  only ~0.51 on average — too low to bet on.

## Roadmap

- **M7b** — cold-path staging: gather cold expert slabs GTT -> VRAM scratch
  before the matvec. Kill probe passed (25-28 GB/s gather); implementation
  next. Expected to close most of the 24 -> 33 tok/s gap at a 6 GiB budget.
- **M8a/b/c** — architecture generalization for the 80B target: metadata
  dispatch table, fused-QKV split at load, shared experts, router variants.
- **M9** — hybrid SSM/linear-attention layers (Qwen3-Next uses them in
  36 of its 48 layers). The big remaining research item.

## Requirements

- Linux, Vulkan 1.3 GPU with shaderInt16 + 8/16-bit storage features
  (RADV and a 7900 XTX are the validated combination)
- glslc (for custom shaders; vendored ggml shaders ship as .spv)
- ~30 GB RAM for the 30B-class model; ~60 GB for the 80B

MIT. Kernels are llama.cpp's (MIT), vendored.
