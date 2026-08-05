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
(q4_K/q5_K/q6_K/q8_0), greedy decode, popularity pinning, online expert
repinning, telemetry, and a dynamic VRAM budget that sizes itself to the
device.

```
Qwen3-Coder-30B-A3B (Q4_K_XL, 17.5 GB weights) on RX 7900 XTX:
  default, zero config ................... 28.26 tok/s   (auto budget)
  all experts in VRAM (18.6 GB) .......... 33.4  tok/s
  6 GiB budget + hot-expert pinning ...... 24.0  tok/s   (91% cache hits)
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
- **Dynamic VRAM budget** — at load the engine queries the device's
  DEVICE_LOCAL heap and greedily fills it: `heap − headroom − pinned
  weights − KV at max context − streaming arena`. Nothing is hard-coded to
  a 24 GB card; a 8 GB or 48 GB GPU sizes itself automatically.
- **Expert residency** (`crates/model`) — non-expert weights (~1 GB) pinned
  in VRAM; experts that don't fit live in system memory (GTT).
  Popularity-based pinning copies the hottest experts contiguously into
  per-layer VRAM buffers; online repinning swaps residents every N tokens
  against a live hit histogram. A reserved KV arena is never touched by
  streaming.
- **M7b-A scratch arenas** — when experts live in system memory, a compute
  gather copies the token's cold expert slabs GTT → VRAM scratch before the
  matvec reads them (measured 25-28 GB/s gather vs ~5.8 GiB/s for
  in-place reads). Auto-enabled for pure system-memory placement;
  `MOE_ARENA=1/0` overrides.

Why streaming beats `--n-cpu-moe` style offloading: compute never leaves
the GPU. A cold expert miss is a ~2 MB PCIe gather, not a CPU matmul. The
0%-hit-rate ceiling for the 30B is still ~27 tok/s on measured bandwidth —
any cache locality makes it better.

## How to use

### Requirements

- Linux with a Vulkan 1.3 GPU (RADV + RX 7900 XTX is the validated combo;
  the engine queries your device's VRAM and adapts)
- `glslc` (for the custom shaders; vendored ggml kernels ship as .spv)
- Rust stable (2024 edition)
- RAM: at least the model file size (GGUF is read fully)

### Build

```bash
git clone https://github.com/aDirtyScoundrel/styx.git moe-stream
cd moe-stream
cargo build --release
```

### Run

```bash
./target/release/examples/generate_moe <model.gguf> <n_tokens> <token ids...>
```

Example — 64 tokens, prompt given as token ids:

```bash
./target/release/examples/generate_moe Qwen3-Coder-30B-A3B.gguf 64 151644 872
```

That's it. With no configuration the engine auto-budgets VRAM (leaving
~1 GB for the desktop/GUI) and picks the fastest placement for your card.
You'll see one line like:

```
auto expert budget: 21768 MiB (device VRAM 24.0 GiB - 1024 MiB headroom
  - 0.95 GiB pinned - 0.75 GiB KV - 22 MiB arena)
```

### Tuning knobs (all optional)

| Env var | Effect |
|---|---|
| `MOE_HEADROOM_MB=N` | VRAM left free for GUI/other apps (default 1024). Lower it to squeeze more experts in; raise it if your desktop stutters. |
| `MOE_EXPERTS_VRAM_MB=N` | Override the auto budget with an explicit MiB figure (0 = all experts in system memory). |
| `MOE_EXPERTS_VRAM=1` | Force ALL experts into VRAM (fails if they don't fit). |
| `MOE_EXPERT_HIST=hist.csv` | Popularity pinning: CSV of `layer,expert,hits` from a trace; hottest slabs get VRAM first within the budget. |
| `MOE_REPIN_INTERVAL=N` | Online repinning every N tokens (0 = off). Best on long generations; break-even ~512 tokens. Pair with `MOE_REPIN_MAX=2`. |
| `MOE_ARENA=0/1` | Force the M7b-A gather arenas off/on (default: auto, on only for pure system-memory placement). |
| `MOE_HUD=1` | Telemetry: hot/cold hit rates, estimated PCIe traffic, per-layer cold heatmap. ~0.5% overhead. |

Example — 6 GiB pinning with telemetry:

```bash
MOE_EXPERT_HIST=hist.csv MOE_EXPERTS_VRAM_MB=6144 MOE_HUD=1 \
    ./target/release/examples/generate_moe model.gguf 128 <tokens...>
```

### Check whether a model can run

```bash
./target/release/examples/onboard /path/to/model.gguf
```

Prints READY or BLOCKED with the missing features named. Currently only
`qwen3moe` models are READY.

### Verification

```bash
scripts/verify.sh          # fast tier: build, tests, goldens (3 placements)
SLOW=1 scripts/verify.sh   # adds the 600-tok paged-KV growth run
```

Requires a Vulkan GPU and the reference GGUF (override with
`MOE_VERIFY_GGUF`).

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
  The M7b-A arena gathers first, then computes from VRAM. (Honest result:
  +0.7% at pure-GTT; under pinning the gather barrier serializes traffic
  that in-place reads overlap, so it's auto-disabled there. True overlap
  needs M7b-B.)
- **Next-token prefetch does not work.** Token-to-token expert overlap is
  only ~0.51 on average — too low to bet on.
- **Push-constant sizes are silent.** A pipeline declared with fewer push
  bytes than the shader reads gives uninitialized values, not an error.
  This bit us once (top-k renorm); verify.sh goldens catch the class now.

## Roadmap

- **M7b-B** — hide the gather behind compute: family-1 async queue +
  timeline semaphores so cold-slab transfers overlap layer compute. The
  remaining lever for pinned placements.
- **M8a/b/c** — architecture generalization for the 80B target: metadata
  dispatch table, fused-QKV split at load, shared experts, router variants.
- **M9** — hybrid SSM/linear-attention layers (Qwen3-Next uses them in
  36 of its 48 layers). The big remaining research item.

MIT. Kernels are llama.cpp's (MIT), vendored.
