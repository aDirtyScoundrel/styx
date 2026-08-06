//! Qwen3-MoE (qwen3moe) forward pass, GPU-resident single-token decode.
//!
//! Same graph executor as the dense path (see lib.rs) plus:
//!   - mixed-quant matvec (q4_K/q5_K/q6_K/q8_0) selected per tensor
//!   - router: f32 matvec -> topk_softmax (custom shader, renormalized)
//!   - expert FFN via mul_mat_vec_id_* (vendored, ABI-tested): gate/up
//!     read the same normed x (ne11=1), down reads per-slot rows (ne11=k)
//!   - weighted reduce of the k expert outputs back into the residual.
//!
//! Expert tensors are placed in HOST (GTT) memory when `MOE_EXPERTS_HOST=1`
//! or when the model would not fit in VRAM; attention/router/head stay in
//! VRAM. The GPU reads experts over PCIe on demand — only ~3B of the 30B
//! parameters are touched per token, so the working set is small.

use crate::{AttnPush, HParams, WHOLE, as_bytes, as_bytes_of};
use gguf_rs::Gguf;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use vk_backend::ops::{BinaryPush, GluPush, MatVecIdPush, MatVecPush, RopePush, UnaryPush};
use vk_backend::{Batch, Buffer, Gpu, Pipeline, ash_vk};

const QK_K: usize = 256;

fn type_name(t: u32) -> &'static str {
    match t {
        8 => "q8_0",
        12 => "q4_k",
        13 => "q5_k",
        14 => "q6_k",
        _ => panic!("unsupported ggml type {t}"),
    }
}

fn quant_k(t: u32) -> usize {
    match t {
        8 => 32,
        12 | 13 | 14 => QK_K,
        _ => panic!("unsupported ggml type {t}"),
    }
}

struct GpuMat {
    buf: Buffer,
    nrows: usize,
    ncols: usize,
    ggml_type: u32,
}

struct ExpMat {
    buf: Buffer,
    nrows: usize, // per expert
    ncols: usize,
    ggml_type: u32,
    slab: usize, // bytes per expert slab
    // Dense hot buffer: hottest experts' slabs copied contiguously into VRAM
    // (plus one trailing zero slab), ids remapped by the router shader.
    hot: Option<Buffer>,
}

struct Layer {
    attn_norm: Buffer,
    q_norm: Option<Buffer>,
    k_norm: Option<Buffer>,
    // attn projection biases (qwen2moe): f32 vectors added post-matvec
    q_bias: Option<Buffer>,
    k_bias: Option<Buffer>,
    v_bias: Option<Buffer>,
    ffn_norm: Buffer,
    wq: GpuMat,
    wk: GpuMat,
    wv: GpuMat,
    wo: GpuMat,
    router: GpuMat, // f32 (n_expert, n_embd)
    gate_exps: ExpMat,
    up_exps: ExpMat,
    down_exps: ExpMat,
    // shared expert branch (qwen2moe): dense mats, always resident in VRAM,
    // plus the scalar sigmoid gate row (ffn_gate_inp_shexp, n_embd f32).
    shexp: Option<Shexp>,
    remap: Buffer,          // n_expert u32: expert -> dense hot idx, or SKIP (u32::MAX)
    hot_mask: Vec<bool>,    // host copy: expert -> pinned in VRAM?
    remap_vec: Vec<u32>,    // host copy of remap
    slot_experts: Vec<u32>, // hot slot -> expert currently resident
    hits: Vec<u64>,         // online per-expert hit counters (repinning)
    kcache: Buffer,
    vtcache: Buffer,
}

struct Shexp {
    gate_inp: Buffer, // f32 n_embd row: scalar gate logit
    gate: GpuMat,
    up: GpuMat,
    down: GpuMat,
}

pub struct MoeHParams {
    pub base: HParams,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
}

/// How Q/K are normalized after their projections.
#[derive(Clone, Copy, PartialEq)]
pub enum QkNorm {
    /// RMS per attention head over head_dim (qwen3moe).
    PerHead,
    /// One RMS over the whole q/k vector (olmoe).
    FullVec,
    /// No qk norm; projections may carry biases instead (qwen2moe).
    None,
}

/// Per-architecture wiring resolved from `general.architecture`.
pub struct ArchCfg {
    pub name: &'static str,
    pub qk_norm: QkNorm,
    /// attn_{q,k,v}.bias tensors present and added after the projections.
    pub attn_bias: bool,
    /// Renormalize the top-k softmax weights (norm_topk_prob).
    pub norm_topk: bool,
    /// Always-on shared expert branch (ffn_*_shexp + sigmoid gate).
    pub shexp: bool,
}

impl ArchCfg {
    fn resolve(arch: &str) -> Result<ArchCfg, String> {
        Ok(match arch {
            "qwen3moe" => ArchCfg {
                name: "qwen3moe",
                qk_norm: QkNorm::PerHead,
                attn_bias: false,
                norm_topk: true,
                shexp: false,
            },
            "olmoe" => ArchCfg {
                name: "olmoe",
                qk_norm: QkNorm::FullVec,
                attn_bias: false,
                norm_topk: false,
                shexp: false,
            },
            "qwen2moe" => ArchCfg {
                name: "qwen2moe",
                qk_norm: QkNorm::None,
                attn_bias: true,
                norm_topk: false,
                shexp: true,
            },
            other => return Err(format!("unsupported architecture '{other}'")),
        })
    }
}

pub struct MoeModel {
    pub hp: MoeHParams,
    cfg: ArchCfg,
    gpu: Gpu,
    // quant matvec pipelines keyed by (ggml_type, num_rows)
    mmv: HashMap<(u32, u32), Pipeline>,
    mmv_id: HashMap<u32, Pipeline>,
    mmv_f32: Pipeline,
    p_rms: Pipeline,
    p_add: Pipeline,
    p_cpy: Pipeline,
    p_attn: Pipeline,
    p_glu: Pipeline,
    p_rope: Pipeline,
    p_topk: Pipeline,
    p_reduce: Pipeline,
    p_gather: Pipeline,
    layers: Vec<Layer>,
    output_norm: Buffer,
    head: GpuMat,
    embd_raw: Vec<u8>,
    #[allow(dead_code)]
    embd_type: u32,
    // activations
    bx: Buffer,
    bnorm: Buffer,
    bq: Buffer,
    bk: Buffer,
    bv: Buffer,
    battn: Buffer,
    bproj: Buffer,
    brouter: Buffer,  // n_expert logits
    bweights: Buffer, // k f32
    bids: Buffer,     // n_layers x 64B rows; k i32 ids per layer at offset l*64
    bgate: Buffer,    // k * n_ff_exp
    bup: Buffer,      // k * n_ff_exp
    bdown: Buffer,    // k * n_embd
    blogits: Buffer,
    bpos: Buffer,
    bdummy: Buffer,
    /// M7b-A scratch arenas: k contiguous slabs each, DEVICE_LOCAL VRAM.
    /// Cold experts are gathered here from GTT each layer before the cold
    /// matvec reads them (25-28 GB/s gather vs ~5.8 GiB/s in-place reads).
    /// None when no expert tensor lives in GTT (all-VRAM placement).
    barena_g: Option<Buffer>,
    barena_u: Option<Buffer>,
    barena_d: Option<Buffer>,
    batch: Option<Batch>,
    pub n_ctx_max: usize,
    n_past: usize,
    /// Paged KV: tokens currently backed per layer cache. Grows by
    /// KV_PAGE_TOKENS (device copy-on-grow) up to n_ctx_max.
    kv_cap: usize,
    /// Bytes of expert weights (gate+up+down) per (layer,expert) slab —
    /// the PCIe cost of one cold expert hit.
    pub slab_bytes: u64,
    /// Online repinning: every N tokens swap the coldest resident experts
    /// for the hottest non-resident ones (MOE_REPIN_INTERVAL; 0/unset off).
    repin_interval: usize,
    /// M7b-A arena path (MOE_ARENA=1 force on, =0 force off, default auto:
    /// on only when no layer has a hot buffer — pure-GTT placement. With
    /// hot/cold pinning the arena's post-gather barrier serializes the
    /// PCIe gather that would otherwise overlap hot compute, so the
    /// in-place cold path is faster there; overlap needs M7b-B).
    arena_enabled: bool,
    /// M7b-B async prefetch. MOE_PREFETCH=0 off, =1 force on, unset auto
    /// (on iff arena_enabled AND a second compute family exists).
    prefetch: bool,
    /// Async gather batch (family async_qfam) + timeline semaphore.
    /// prefetch_sem value v: all gathers for token v are complete.
    async_batch: Option<Batch>,
    prefetch_sem: Option<ash_vk::Semaphore>,
    sem_counter: u64,
    /// Per-layer prefetch prediction ids (k u32 each), uploaded to
    /// bpf_ids before the async gather reads them. Predicted from the
    /// previous token's routed experts.
    bpf_ids: Buffer,
    /// Per-layer pmap (n_expert u32: expert -> arena slot or SKIP),
            /// built host-side from the prediction, uploaded each token.
            bpmaps: Buffer,
            /// Double-buffered pmap: we upload to the inactive one while the
            /// active one is read by the other queue's topk. Swapped after the
            /// next token's topk consumes it. Avoids in-place HOST_COHERENT
            /// update while the GPU may be reading (RADV cache coherency bug).
            bpmaps_alt: Buffer,
            /// Host mirror of the current pmap (n_layers x n_expert u32).
            pmap_host: Vec<u32>,
    /// Host mirror of the prediction ids (n_layers x 64 u32, 256B stride).
    pf_ids_host: Vec<u32>,
    /// Timeline value the NEXT forward_token's batch must wait on for the
    /// in-flight prefetch (0 = none).
    prefetch_wait_val: u64,
    /// Arena geometry: per FFN tensor (gate,up,down) the MAX slab across
    /// layers in bytes (arena slots pad to this) and in uvec4 count
    /// (gather dst stride). Mixed-quant layers vary in bytes; the arena
    /// always pads to the max so slot math is uniform. The arena matvec's
    /// batch_stride_a is derived per layer from bytes + that layer's type.
    arena_slab_bytes: [u64; 3],
    arena_slab_v: [usize; 3],
    /// Max slab swaps per layer per repin event (MOE_REPIN_MAX, default 2).
    repin_max: usize,
    /// Total expert slabs swapped in by repinning so far.
    pub repin_swaps: u64,
}

/// Paged-KV page size in tokens. One page = KV_PAGE_TOKENS*kv_dim*4 bytes
/// per cache (k and v^T separately) per layer.
pub const KV_PAGE_TOKENS: usize = 512;

/// Scalar dequant of one q4_K block (ggml layout) — for host embedding rows.
fn dequant_q4k(block: &[u8], out: &mut [f32]) {
    let f16 =
        |o: usize| half::f16::from_bits(u16::from_le_bytes([block[o], block[o + 1]])).to_f32();
    let (d, dmin) = (f16(0), f16(2));
    let scales = &block[4..16];
    let qs = &block[16..144];
    let get_scale_min = |j: usize| -> (f32, f32) {
        if j < 4 {
            ((scales[j] & 63) as f32, (scales[j + 4] & 63) as f32)
        } else {
            (
                ((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)) as f32,
                ((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)) as f32,
            )
        }
    };
    let (mut ql, mut o) = (0, 0);
    for j in 0..(QK_K / 64) {
        let (sc1, m1) = get_scale_min(2 * j);
        let (sc2, m2) = get_scale_min(2 * j + 1);
        for l in 0..32 {
            out[o + l] = d * sc1 * (qs[ql + l] & 0x0F) as f32 - dmin * m1;
            out[o + 32 + l] = d * sc2 * (qs[ql + l] >> 4) as f32 - dmin * m2;
        }
        o += 64;
        ql += 32;
    }
}

impl MoeModel {
    pub fn load(path: &Path) -> Result<MoeModel, String> {
        let g = Gguf::open(path).map_err(|e| e.to_string())?;
        let get_u32 = |k: &str| -> u32 {
            match g.metadata.get(k) {
                Some(gguf_rs::Value::U32(v)) => *v,
                other => panic!("missing/unexpected {k}: {other:?}"),
            }
        };
        let get_f32 = |k: &str| -> f32 {
            match g.metadata.get(k) {
                Some(gguf_rs::Value::F32(v)) => *v,
                other => panic!("missing/unexpected {k}: {other:?}"),
            }
        };
        let arch = match g.metadata.get("general.architecture") {
            Some(gguf_rs::Value::String(s)) => s.clone(),
            other => return Err(format!("missing general.architecture: {other:?}")),
        };
        let cfg = ArchCfg::resolve(&arch)?;
        let pfx = cfg.name;
        let n_layers = get_u32(&format!("{pfx}.block_count")) as usize;
        let n_embd = get_u32(&format!("{pfx}.embedding_length")) as usize;
        let n_heads = get_u32(&format!("{pfx}.attention.head_count")) as usize;
        let n_kv_heads = get_u32(&format!("{pfx}.attention.head_count_kv")) as usize;
        // key_length is absent when head_dim == n_embd / n_heads (olmoe, qwen2moe)
        let head_dim = match g.metadata.get(&format!("{pfx}.attention.key_length")) {
            Some(gguf_rs::Value::U32(v)) => *v as usize,
            _ => n_embd / n_heads,
        };
        let n_expert = get_u32(&format!("{pfx}.expert_count")) as usize;
        let n_expert_used = get_u32(&format!("{pfx}.expert_used_count")) as usize;
        // olmoe/qwen2moe: no expert_feed_forward_length key; expert FF width
        // comes from the tensor shape (ffn_gate_exps dims = [n_embd, n_ff, n_e]).
        let n_ff_exp = match g.metadata.get(&format!("{pfx}.expert_feed_forward_length")) {
            Some(gguf_rs::Value::U32(v)) => *v as usize,
            _ => g
                .tensors
                .iter()
                .find(|t| t.name == "blk.0.ffn_gate_exps.weight")
                .map(|t| t.dims[1] as usize)
                .ok_or("no expert_feed_forward_length key and no ffn_gate_exps tensor")?,
        };
        let rms_eps = get_f32(&format!("{pfx}.attention.layer_norm_rms_epsilon"));
        let rope_base = get_f32(&format!("{pfx}.rope.freq_base"));
        assert!(n_expert_used <= 16, "topk shader caps k at 16");

        let tmap: HashMap<String, _> = g.tensors.iter().map(|t| (t.name.clone(), t)).collect();
        let n_vocab = tmap["token_embd.weight"].dims[1] as usize;
        let raw = std::fs::read(path).map_err(|e| e.to_string())?;
        let data = &raw[g.data_start as usize..];
        let bytes = |name: &str| -> &[u8] {
            let t = &tmap[name];
            &data[t.offset as usize..(t.offset + t.size_bytes) as usize]
        };
        let f32vec = |name: &str| -> Vec<f32> {
            let t = &tmap[name];
            assert_eq!(t.ggml_type, 0, "{name} not f32");
            bytes(name)
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };

        let gpu = Gpu::new()?;
        let spv = |name: &str| -> PathBuf {
            format!(
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../vendor/llama.cpp/build-shaders/ggml/src/ggml-vulkan",
                    "/vulkan-shaders.spv/{}.spv"
                ),
                name
            )
            .into()
        };
        let ours = |name: &str| -> PathBuf {
            format!(
                concat!(env!("CARGO_MANIFEST_DIR"), "/../../shaders/{}.spv"),
                name
            )
            .into()
        };

        // Pipelines for every quant type present.
        let mut types: Vec<u32> = g
            .tensors
            .iter()
            .filter(|t| t.ggml_type != 0)
            .map(|t| t.ggml_type)
            .collect();
        types.sort();
        types.dedup();
        let mmv_pc = std::mem::size_of::<MatVecPush>() as u32;
        let id_pc = std::mem::size_of::<MatVecIdPush>() as u32;
        let mut mmv = HashMap::new();
        let mut mmv_id = HashMap::new();
        for &t in &types {
            let n = type_name(t);
            for rows in [1u32, 2, 4] {
                mmv.insert(
                    (t, rows),
                    gpu.create_pipeline(
                        &spv(&format!("mul_mat_vec_{n}_f32_f32")),
                        5,
                        mmv_pc,
                        &[(0, 64), (1, rows), (2, 1)],
                    )?,
                );
            }
            let p = spv(&format!("mul_mat_vec_id_{n}_f32_f32"));
            if p.exists() {
                mmv_id.insert(
                    t,
                    gpu.create_pipeline(&p, 6, id_pc, &[(0, 32), (1, 1), (2, 1)])?,
                );
            }
        }
        let mmv_f32 = gpu.create_pipeline(&ours("matvec_strided_f32"), 3, 8, &[])?;
        let p_add = gpu.create_pipeline(&ours("add_flat_f32"), 3, 4, &[])?;
        let bin_pc = std::mem::size_of::<BinaryPush>() as u32;
        let p_rms = gpu.create_pipeline(&spv("rms_norm_f32"), 3, bin_pc, &[(1, 1)])?;
        let p_cpy = gpu.create_pipeline(
            &spv("cpy_f32_f32"),
            2,
            std::mem::size_of::<UnaryPush>() as u32,
            &[],
        )?;
        let p_attn = gpu.create_pipeline(&ours("attn_decode_f32"), 4, 20, &[])?;
        let p_glu = gpu.create_pipeline(
            &spv("swiglu_f32"),
            3,
            std::mem::size_of::<GluPush>() as u32,
            &[],
        )?;
        let p_rope = gpu.create_pipeline(
            &spv("rope_neox_f32"),
            5,
            std::mem::size_of::<RopePush>() as u32,
            &[],
        )?;
        let p_topk = gpu.create_pipeline(&ours("topk_softmax_f32"), 5, 12, &[])?;
        let p_reduce = gpu.create_pipeline(&ours("moe_reduce_f32"), 3, 8, &[])?;
        let p_gather = gpu.create_pipeline(&ours("gather_slabs_f32"), 7, 36, &[])?;

        // Expert placement policy:
        //   default              -> AUTO budget: greedily fill device VRAM
        //                           (heap - headroom - pinned - KV - arena)
        //   MOE_EXPERTS_VRAM_MB=N -> explicit budget in MiB (override auto)
        //   MOE_EXPERTS_VRAM=1   -> all experts in VRAM (errors if too big)
        //   + MOE_EXPERT_HIST=csv -> popularity pinning: hottest (layer,expert)
        //     slabs are COPIED into per-layer dense VRAM buffers (hot buffer),
        //     router ids remapped in-shader; cold experts stay in the GTT
        //     buffer. Budget = MOE_EXPERTS_VRAM_MB or auto. (Sparse lost 2x.)
        //   MOE_HEADROOM_MB=N    -> VRAM left for GUI/other apps (default 1024)
        let all_vram = std::env::var("MOE_EXPERTS_VRAM").is_ok();
        // Max context for KV reservation accounting (matches the load-time
        // arena below; both must stay in sync).
        let n_ctx_max = 4096usize;
        let explicit_mb: Option<u64> = std::env::var("MOE_EXPERTS_VRAM_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok());
        // Reservations for the auto budget (bytes):
        //   pinned tier = every non-_exps. tensor (norms, attn, router, head)
        let pinned_bytes: u64 = g
            .tensors
            .iter()
            .filter(|t| !t.name.contains("_exps."))
            .map(|t| t.size_bytes)
            .sum();
        //   KV at full context: 2 caches x n_layers x n_ctx x kv_dim f32
        let kv_max_bytes = (n_layers * 2 * n_ctx_max * n_kv_heads * head_dim * 4) as u64;
        //   Arena (M7b-B): n_layers x k slots, each padded to the per-tensor
        //   MAX slab across layers (allocated iff any expert spills to GTT;
        //   reserved here because auto usually spills).
        let max_slab_sum: u64 = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
            .iter()
            .map(|s| {
                (0..n_layers)
                    .map(|l| {
                        let t = &tmap[&format!("blk.{l}.{s}.weight")];
                        t.size_bytes / n_expert as u64
                    })
                    .max()
                    .unwrap_or(0)
            })
            .sum();
        let arena_bytes = n_layers as u64 * n_expert_used as u64 * max_slab_sum;
        let headroom_mb: u64 = std::env::var("MOE_HEADROOM_MB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let device_vram = gpu.vram_bytes();
        let auto_budget = device_vram
            .saturating_sub(headroom_mb * 1024 * 1024 + pinned_bytes + kv_max_bytes + arena_bytes);
        let vram_budget: u64 = if all_vram {
            u64::MAX
        } else if let Some(mb) = explicit_mb {
            mb * 1024 * 1024
        } else {
            auto_budget
        };
        if explicit_mb.is_none() && !all_vram {
            eprintln!(
                "auto expert budget: {} MiB (device VRAM {:.1} GiB - {} MiB headroom \
                 - {:.2} GiB pinned - {:.2} GiB KV - {:.0} MiB arena)",
                auto_budget >> 20,
                device_vram as f64 / (1 << 30) as f64,
                headroom_mb,
                pinned_bytes as f64 / (1 << 30) as f64,
                kv_max_bytes as f64 / (1 << 30) as f64,
                arena_bytes as f64 / (1 << 20) as f64,
            );
        }
        let vram_left = std::cell::Cell::new(vram_budget);
        let vram_used = std::cell::Cell::new(0u64);
        // Set true when any expert tensor lands in GTT (host memory) —
        // then the M7b-A gather arena is allocated and the cold matvec
        // reads from VRAM scratch instead of PCIe.
        let experts_in_gtt = std::cell::Cell::new(false);

        // Popularity pinning: MOE_EXPERT_HIST=<csv "layer,expert,hits">.
        // Globally rank (layer,expert) by hits, pin hottest experts' slabs
        // (gate+up+down) into ReBAR VRAM until the MB budget is used, rest GTT.
        // Requires MOE_EXPERTS_VRAM_MB (budget) alongside.
        let hot_sets: Option<Vec<std::collections::HashSet<u32>>> =
            std::env::var("MOE_EXPERT_HIST").ok().map(|p| {
                let txt = std::fs::read_to_string(&p).expect("read MOE_EXPERT_HIST");
                let mut entries: Vec<(u64, usize, u32)> = txt
                    .lines()
                    .skip(1)
                    .filter_map(|l| {
                        let mut it = l.split(',');
                        let layer: usize = it.next()?.parse().ok()?;
                        let expert: u32 = it.next()?.parse().ok()?;
                        let hits: u64 = it.next()?.parse().ok()?;
                        Some((hits, layer, expert))
                    })
                    .collect();
                entries.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                // Per-expert byte cost across gate+up+down for one layer.
                let cost: u64 = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
                    .iter()
                    .map(|s| {
                        let t = &tmap[&format!("blk.0.{s}.weight")];
                        t.size_bytes / n_expert as u64
                    })
                    .sum();
                let mut sets = vec![std::collections::HashSet::new(); n_layers];
                let mut left = vram_budget;
                for (_hits, l, e) in entries {
                    if left < cost {
                        break;
                    }
                    left -= cost;
                    sets[l].insert(e);
                }
                let pinned: usize = sets.iter().map(|s| s.len()).sum();
                eprintln!("popularity pinning: {pinned} (layer,expert) slabs hot");
                sets
            });

        let upload_mat = |name: &str| -> Result<GpuMat, String> {
            let t = &tmap[name];
            let b = gpu.create_buffer(t.size_bytes, true)?;
            gpu.upload(&b, bytes(name))?;
            Ok(GpuMat {
                buf: b,
                nrows: t.dims[1] as usize,
                ncols: t.dims[0] as usize,
                ggml_type: t.ggml_type,
            })
        };
        // hot_order[layer] = sorted hot expert ids; dense hot index = position.
        let hot_order: Option<Vec<Vec<u32>>> = hot_sets.as_ref().map(|sets| {
            sets.iter()
                .map(|s| {
                    let mut v: Vec<u32> = s.iter().copied().collect();
                    v.sort_unstable();
                    v
                })
                .collect()
        });
        let upload_exp = |name: &str, layer: usize| -> Result<ExpMat, String> {
            let t = &tmap[name];
            let (buf, hot) = if let Some(order) = &hot_order {
                // Dense reorder: copy hot slabs contiguously into a VRAM
                // buffer; the full tensor stays in GTT for the cold path.
                let slab = (t.size_bytes / n_expert as u64) as usize;
                let src = bytes(name);
                let ord = &order[layer];
                let hot_buf = if ord.is_empty() {
                    None
                } else {
                    let mut hb = Vec::with_capacity(ord.len() * slab);
                    for &e in ord {
                        hb.extend_from_slice(&src[e as usize * slab..(e as usize + 1) * slab]);
                    }
                    let b = gpu.create_buffer(hb.len() as u64, true)?;
                    gpu.upload(&b, &hb)?;
                    vram_used.set(vram_used.get() + hb.len() as u64);
                    Some(b)
                };
                let hb = (gpu.create_buffer_host_shared(t.size_bytes)?, hot_buf);
                experts_in_gtt.set(true);
                hb
            } else if vram_left.get() >= t.size_bytes {
                vram_left.set(vram_left.get() - t.size_bytes);
                vram_used.set(vram_used.get() + t.size_bytes);
                (gpu.create_buffer(t.size_bytes, true)?, None)
            } else {
                // Shared (CONCURRENT) so the async prefetch gather on the
                // second queue family can read it; falls back to a plain
                // host buffer when the device has only one compute family.
                let hb = (gpu.create_buffer_host_shared(t.size_bytes)?, None);
                experts_in_gtt.set(true);
                hb
            };
            gpu.upload(&buf, bytes(name))?;
            Ok(ExpMat {
                buf,
                ncols: t.dims[0] as usize,
                nrows: t.dims[1] as usize,
                ggml_type: t.ggml_type,
                slab: (t.size_bytes / n_expert as u64) as usize,
                hot,
            })
        };
        let upload_f32_mat = |name: &str| -> Result<GpuMat, String> {
            let t = &tmap[name];
            assert_eq!(t.ggml_type, 0);
            let b = gpu.create_buffer(t.size_bytes, true)?;
            gpu.upload(&b, bytes(name))?;
            Ok(GpuMat {
                buf: b,
                nrows: t.dims[1] as usize,
                ncols: t.dims[0] as usize,
                ggml_type: 0,
            })
        };
        let upload_f32 = |name: &str| -> Result<Buffer, String> {
            let v = f32vec(name);
            let b = gpu.create_buffer((v.len() * 4) as u64, true)?;
            gpu.upload(&b, as_bytes(&v))?;
            Ok(b)
        };

        let kv_dim = n_kv_heads * head_dim;
        let mut layers = Vec::new();
        for l in 0..n_layers {
            let n = |s: &str| format!("blk.{l}.{s}.weight");
            // remap[e] = dense hot index or SKIP (0xFFFFFFFF)
            let mut remap_vec = vec![u32::MAX; n_expert];
            if let Some(order) = &hot_order {
                for (i, &e) in order[l].iter().enumerate() {
                    remap_vec[e as usize] = i as u32;
                }
            }
            let remap = gpu.create_buffer((n_expert * 4) as u64, true)?;
            gpu.upload(&remap, as_bytes(&remap_vec))?;
            let hot_mask: Vec<bool> = remap_vec.iter().map(|&r| r != u32::MAX).collect();
            let slot_experts: Vec<u32> =
                hot_order.as_ref().map(|o| o[l].clone()).unwrap_or_default();
            let opt_f32 = |s: &str| -> Result<Option<Buffer>, String> {
                let name = format!("blk.{l}.{s}");
                if tmap.contains_key(&name) {
                    let t = &tmap[&name];
                    assert_eq!(t.ggml_type, 0, "{name} not f32");
                    let b = gpu.create_buffer(t.size_bytes, true)?;
                    gpu.upload(&b, bytes(&name))?;
                    Ok(Some(b))
                } else {
                    Ok(None)
                }
            };
            let q_norm = opt_f32("attn_q_norm.weight")?;
            let k_norm = opt_f32("attn_k_norm.weight")?;
            match cfg.qk_norm {
                QkNorm::None => assert!(q_norm.is_none(), "unexpected attn_q_norm"),
                _ => assert!(q_norm.is_some() && k_norm.is_some(), "missing qk norm"),
            }
            let q_bias = opt_f32("attn_q.bias")?;
            let k_bias = opt_f32("attn_k.bias")?;
            let v_bias = opt_f32("attn_v.bias")?;
            assert_eq!(cfg.attn_bias, q_bias.is_some(), "attn bias mismatch");
            let shexp = if cfg.shexp {
                Some(Shexp {
                    gate_inp: upload_f32(&n("ffn_gate_inp_shexp"))?,
                    gate: upload_mat(&n("ffn_gate_shexp"))?,
                    up: upload_mat(&n("ffn_up_shexp"))?,
                    down: upload_mat(&n("ffn_down_shexp"))?,
                })
            } else {
                None
            };
            layers.push(Layer {
                attn_norm: upload_f32(&n("attn_norm"))?,
                q_norm,
                k_norm,
                q_bias,
                k_bias,
                v_bias,
                ffn_norm: upload_f32(&n("ffn_norm"))?,
                wq: upload_mat(&n("attn_q"))?,
                wk: upload_mat(&n("attn_k"))?,
                wv: upload_mat(&n("attn_v"))?,
                wo: upload_mat(&n("attn_output"))?,
                router: upload_f32_mat(&n("ffn_gate_inp"))?,
                gate_exps: upload_exp(&n("ffn_gate_exps"), l)?,
                up_exps: upload_exp(&n("ffn_up_exps"), l)?,
                down_exps: upload_exp(&n("ffn_down_exps"), l)?,
                shexp,
                remap,
                hot_mask,
                remap_vec,
                slot_experts,
                hits: vec![0u64; n_expert],
                // Paged KV: start with one page, grown on demand in
                // forward_token (copy-on-grow at page boundaries).
                kcache: gpu.create_buffer((KV_PAGE_TOKENS * kv_dim * 4) as u64, true)?,
                vtcache: gpu.create_buffer((KV_PAGE_TOKENS * kv_dim * 4) as u64, true)?,
            });
            if l % 8 == 0 {
                eprintln!("loaded layer {l}/{n_layers}");
            }
        }
        let head = upload_mat("output.weight")?;
        eprintln!(
            "expert placement: {:.2} GiB in VRAM, budget {}",
            vram_used.get() as f64 / (1 << 30) as f64,
            if vram_budget == u64::MAX {
                "ALL".into()
            } else {
                format!("{} MiB", vram_budget >> 20)
            }
        );
        let output_norm = upload_f32("output_norm.weight")?;
        let embd_raw = bytes("token_embd.weight").to_vec();
        let embd_type = tmap["token_embd.weight"].ggml_type;
        assert!(
            matches!(embd_type, 0 | 1 | 8 | 12),
            "token_embd ggml_type {embd_type} not host-dequantable (f32/f16/q8_0/q4_K)"
        );

        let q_dim = n_heads * head_dim;
        let k = n_expert_used;
        let mk = |n: usize| gpu.create_buffer((n * 4) as u64, true);
        let force_gtt = std::env::var("MOE_FORCE_GTT").is_ok();
        let no_pmap_reupload = std::env::var("MOE_NO_PMAP_REUPLOAD").is_ok();
        let bx = mk(n_embd)?;
        let bnorm = mk(n_embd)?;
        let bq = mk(q_dim)?;
        let bk = mk(kv_dim)?;
        let bv = mk(kv_dim)?;
        let battn = mk(q_dim)?;
        let bproj = mk(n_embd)?;
        let brouter = mk(n_expert)?;
        let bweights = mk(k)?;
        // M7b-B: 4 rows x 64 i32 = 256B per layer
        // [orig ids | hot ids | arena ids | gtt ids], 256B stride (matches
        // the topk writes at l*256 and last_expert_ids' download size).
        let bids = if force_gtt {
            gpu.create_buffer_host((n_layers * 256) as u64)?
        } else {
            mk(n_layers * 256)?
        };
        let bgate = mk(k * n_ff_exp)?;
        let bup = mk(k * n_ff_exp)?;
        let bdown = mk(k * n_embd)?;
        let blogits = mk(n_vocab)?;
        let bpos = mk(1)?;
        let bdummy = mk(1)?;

        // Scratch arenas (only when experts live in GTT). M7b-B: sized
        // n_layers x k slots (layer-major) so an async prefetch can fill
        // the whole token's predicted experts between tokens. Slot
        // (layer, s) = layer * k + s. Mixed-quant models vary slab size
        // per layer, so size from the MAX slab across all layers.
        let (barena_g, barena_u, barena_d) = if experts_in_gtt.get() {
            let max_slab = |prefix: &str| -> u64 {
                (0..n_layers)
                    .map(|l| {
                        let t = &tmap[&format!("blk.{l}.{prefix}_exps.weight")];
                        t.size_bytes / n_expert as u64
                    })
                    .max()
                    .unwrap()
            };
            let ab = |prefix: &str| -> Result<Buffer, String> {
                let slab = max_slab(prefix);
                assert!(slab % 16 == 0, "{prefix}_exps slab not 16B-aligned");
                let bytes = n_layers as u64 * k as u64 * slab;
                vram_used.set(vram_used.get() + bytes);
                // CONCURRENT: the async gather queue writes these while
                // the main queue reads them (ordered by the timeline sem).
                gpu.create_buffer_shared(bytes, false)
            };
            (
                Some(ab("ffn_gate")?),
                Some(ab("ffn_up")?),
                Some(ab("ffn_down")?),
            )
        } else {
            (None, None, None)
        };

        // Auto arena: on only for pure-GTT placement (no hot buffers).
        // With hot/cold pinning the in-batch gather barrier serializes
        // PCIe traffic that would otherwise overlap hot compute — the
        // pinned regime uses M7b-B async prefetch instead.
        let arena_enabled = match std::env::var("MOE_ARENA").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => !layers.iter().any(|l| l.gate_exps.hot.is_some()),
        };

        // M7b-B async prefetch: on iff there is a GTT expert tier AND a
        // second compute family to run the gather on. MOE_PREFETCH forces.
        let (async_batch, prefetch_sem) = if experts_in_gtt.get() && gpu.async_qfam.is_some() {
            (
                Some(gpu.create_async_batch(1024, 8192)?),
                Some(gpu.create_timeline_semaphore()?),
            )
        } else {
            (None, None)
        };
        // M7b-B NOTE (2026-08-06): prefetch was auto-enabled here whenever a
        // second compute family existed, but the async-gather path has an
        // unresolved correctness bug (topk writes to cold layers vanish — see
        // Obsidian vault "M7b-B Session Handoff.md"). Correctness first:
        // default OFF; MOE_PREFETCH=1 opts in for debugging/benchmarking.
        let prefetch = match std::env::var("MOE_PREFETCH").as_deref() {
            Ok("1") => {
                assert!(
                    async_batch.is_some(),
                    "MOE_PREFETCH=1 requires experts in GTT + a second compute family"
                );
                true
            }
            _ => false,
        };
        // pmaps always cover all layers (topk binds a per-layer offset);
        // all-SKIP content means "nothing prefetched" — every cold expert
        // falls to the in-place GTT row. Prediction ids only needed when
        // prefetch actually runs.
        let bpmaps = if force_gtt {
            gpu.create_buffer_host(n_layers as u64 * n_expert as u64 * 4)?
        } else {
            gpu.create_buffer(n_layers as u64 * n_expert as u64 * 4, true)?
        };
        gpu.upload_staged(&bpmaps, &vec![0xFFu8; n_layers * n_expert * 4])?;
        // Double-buffered pmap: inactive copy for upload, swapped after next token's topk.
        let bpmaps_alt = if force_gtt {
            gpu.create_buffer_host(n_layers as u64 * n_expert as u64 * 4)?
        } else {
            gpu.create_buffer(n_layers as u64 * n_expert as u64 * 4, true)?
        };
        gpu.upload_staged(&bpmaps_alt, &vec![0xFFu8; n_layers * n_expert * 4])?;
        let bpf_ids = if prefetch {
            // prediction ids: 16 i32 slots per layer (64B row, 256B stride).
            // CONCURRENT: read by the family-1 async gather queue.
            if force_gtt {
                gpu.create_buffer_host_shared(n_layers as u64 * 256)?
            } else {
                gpu.create_buffer_shared(n_layers as u64 * 256, true)?
            }
        } else {
            gpu.create_buffer(1, true)?
        };

        // Arena geometry: max slab per FFN tensor across layers.
        let arena_geom = |prefix: &str| -> (u64, usize) {
            let mut max_b = 0u64;
            let mut max_v = 0usize;
            for l in 0..n_layers {
                let t = &tmap[&format!("blk.{l}.{prefix}_exps.weight")];
                let b = t.size_bytes / n_expert as u64;
                max_b = max_b.max(b);
                max_v = max_v.max((b / 16) as usize);
            }
            (max_b, max_v)
        };
        let (gb, gv) = arena_geom("ffn_gate");
        let (ub, uv) = arena_geom("ffn_up");
        let (db, dv) = arena_geom("ffn_down");

        let batch = Some(gpu.create_batch(8192, 49152)?);

        Ok(MoeModel {
            hp: MoeHParams {
                base: HParams {
                    n_layers,
                    n_embd,
                    n_ff: n_ff_exp,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    rms_eps,
                    rope_base,
                    n_vocab,
                },
                n_expert,
                n_expert_used,
                n_ff_exp,
            },
            cfg,
            gpu,
            mmv,
            mmv_id,
            mmv_f32,
            p_rms,
            p_add,
            p_cpy,
            p_attn,
            p_glu,
            p_rope,
            p_topk,
            p_reduce,
            p_gather,
            layers,
            output_norm,
            head,
            embd_raw,
            embd_type,
            bx,
            bnorm,
            bq,
            bk,
            bv,
            battn,
            bproj,
            brouter,
            bweights,
            bids,
            bgate,
            bup,
            bdown,
            blogits,
            bpos,
            bdummy,
            barena_g,
            barena_u,
            barena_d,
            batch,
            n_ctx_max,
            n_past: 0,
            kv_cap: KV_PAGE_TOKENS,
            repin_interval: std::env::var("MOE_REPIN_INTERVAL")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            repin_max: std::env::var("MOE_REPIN_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2),
            repin_swaps: 0,
            arena_enabled,
            prefetch,
            async_batch,
            prefetch_sem,
            sem_counter: 0,
            bpf_ids,
            bpmaps,
            bpmaps_alt,
            pmap_host: vec![u32::MAX; n_layers * n_expert],
            // bpf_ids uses 256B (64 u32) stride per layer to match the
            // gather's per-layer row binding.
            pf_ids_host: vec![u32::MAX; n_layers * 64],
            prefetch_wait_val: 0,
            arena_slab_bytes: [gb, ub, db],
            arena_slab_v: [gv, uv, dv],
            slab_bytes: ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
                .iter()
                .map(|s| {
                    let t = &tmap[&format!("blk.0.{s}.weight")];
                    t.size_bytes / n_expert as u64
                })
                .sum(),
        })
    }

    pub fn reset(&mut self) {
        self.n_past = 0;
    }

    /// Online repinning: fold the last token's routed experts into per-layer
    /// hit counters; every `repin_interval` tokens, for each layer swap
    /// resident experts out for non-resident ones with strictly more hits
    /// (cold GTT slab -> hot VRAM slot via mapped copy, remap table updated).
    /// Requires the previous submit to have completed (bids readback and the
    /// hot buffers are idle between forward_token calls).
    fn maybe_repin(&mut self) {
        if self.repin_interval == 0 {
            return;
        }
        let ids = self.last_expert_ids();
        for (l, row) in ids.iter().enumerate() {
            for &e in row {
                self.layers[l].hits[e as usize] += 1;
            }
        }
        if self.n_past % self.repin_interval != 0 {
            return;
        }
        for ly in &mut self.layers {
            let n_slots = ly.slot_experts.len();
            if n_slots == 0 {
                continue;
            }
            // residents ranked coldest-first, outsiders hottest-first
            let mut res: Vec<usize> = (0..n_slots).collect();
            res.sort_unstable_by_key(|&s| ly.hits[ly.slot_experts[s] as usize]);
            let mut out: Vec<u32> = (0..ly.hot_mask.len() as u32)
                .filter(|&e| !ly.hot_mask[e as usize])
                .collect();
            out.sort_unstable_by_key(|&e| std::cmp::Reverse(ly.hits[e as usize]));
            let mut dirty = false;
            // Throttle: at most MOE_REPIN_MAX swaps per layer per event, and
            // require a 2x hit margin (hysteresis) so borderline experts
            // don't ping-pong — each swap costs a ~15 MiB PCIe copy.
            for (&slot, &newcomer) in res.iter().zip(out.iter()).take(self.repin_max) {
                let old = ly.slot_experts[slot];
                if ly.hits[newcomer as usize] < 2 * ly.hits[old as usize].max(1) {
                    break; // ranked lists: no further profitable swap
                }
                for m in [&ly.gate_exps, &ly.up_exps, &ly.down_exps] {
                    let hot = m.hot.as_ref().unwrap();
                    self.gpu
                        .copy_region(
                            &m.buf,
                            newcomer as u64 * m.slab as u64,
                            hot,
                            slot as u64 * m.slab as u64,
                            m.slab as u64,
                        )
                        .unwrap();
                }
                ly.remap_vec[old as usize] = u32::MAX;
                ly.remap_vec[newcomer as usize] = slot as u32;
                ly.hot_mask[old as usize] = false;
                ly.hot_mask[newcomer as usize] = true;
                ly.slot_experts[slot] = newcomer;
                self.repin_swaps += 1;
                dirty = true;
            }
            if dirty {
                self.gpu.upload(&ly.remap, as_bytes(&ly.remap_vec)).unwrap();
            }
        }
    }

    /// Telemetry for the last forwarded token: per-layer (hot_hits, cold_hits)
    /// against the current pinning, derived from the original-id row of bids.
    pub fn last_hot_cold(&self) -> Vec<(u32, u32)> {
        self.last_expert_ids()
            .iter()
            .zip(&self.layers)
            .map(|(ids, ly)| {
                let hot = ids.iter().filter(|&&e| ly.hot_mask[e as usize]).count() as u32;
                (hot, ids.len() as u32 - hot)
            })
            .collect()
    }

    fn rec_matvec_b(&self, batch: &mut Batch, w: &GpuMat, x: &Buffer, y: &Buffer, barrier: bool) {
        let qk = quant_k(w.ggml_type);
        let num_rows = if w.nrows % 4 == 0 && w.nrows / 4 > 4096 {
            4u32
        } else {
            2
        };
        let pipe = &self.mmv[&(w.ggml_type, num_rows)];
        let groups = (w.nrows as u32).div_ceil(num_rows);
        assert!(groups <= 65535, "workgroup overflow: {groups}");
        let mut push = MatVecPush::simple(w.ncols as u32, w.nrows as u32);
        push.stride_a = (w.ncols / qk) as u32;
        batch
            .dispatch_ranges_barrier(
                &self.gpu,
                pipe,
                &[
                    (&w.buf, 0, WHOLE),
                    (x, 0, WHOLE),
                    (y, 0, WHOLE),
                    (&self.bdummy, 0, WHOLE),
                    (&self.bdummy, 0, WHOLE),
                ],
                push.as_bytes(),
                (groups, 1, 1),
                barrier,
            )
            .unwrap();
    }

    /// Expert matvec: for each of the k selected experts (ids buffer),
    /// y[slot] = W[expert] * x[slot % ne11]. `wbuf` is the weight buffer to
    /// read (full GTT tensor, dense hot VRAM buffer, or the arena);
    /// `ids_off` selects which id row drives the dispatch (slots with SKIP
    /// ids early-out). `stride_a_elems` overrides batch_stride_a for arena
    /// reads (arena slots pad to the max slab across layers, so the stride
    /// is the arena slab in elements, not this layer's own nrows*ncols).
    #[allow(clippy::too_many_arguments)]
    fn rec_matvec_id_buf(
        &self,
        batch: &mut Batch,
        w: &ExpMat,
        wbuf: &Buffer,
        x: &Buffer,
        y: &Buffer,
        ne11: u32,
        barrier: bool,
        ids_off: u64,
        stride_a_elems: Option<u32>,
    ) {
        let k = self.hp.n_expert_used as u32;
        let pipe = &self.mmv_id[&w.ggml_type];
        let push = MatVecIdPush {
            ncols: w.ncols as u32,
            stride_a: (w.ncols / quant_k(w.ggml_type)) as u32,
            stride_b: w.ncols as u32,
            stride_d: w.nrows as u32,
            batch_stride_a: stride_a_elems.unwrap_or((w.nrows * w.ncols) as u32),
            batch_stride_b: w.ncols as u32,
            batch_stride_d: k * w.nrows as u32,
            fusion_flags: 0,
            nei0: k,
            ne11,
            expert_i1: 0,
            nbi1: 1,
        };
        batch
            .dispatch_ranges_barrier(
                &self.gpu,
                pipe,
                &[
                    (wbuf, 0, WHOLE),
                    (x, 0, WHOLE),
                    (y, 0, WHOLE),
                    (&self.bdummy, 0, WHOLE),
                    (&self.bdummy, 0, WHOLE),
                    (&self.bids, ids_off, WHOLE),
                ],
                push.as_bytes(),
                (w.nrows as u32, k, 1),
                barrier,
            )
            .unwrap();
    }

    /// Bytes per quant block (QUANT_K elements): q8_0=34, q4_K=144,
    /// q5_K=176, q6_K=210. Derives arena strides from byte sizes.
    fn block_bytes(t: u32) -> usize {
        match t {
            8 => 34,
            12 => 144,
            13 => 176,
            14 => 210,
            _ => panic!("unsupported ggml type {t}"),
        }
    }

    /// Arena batch_stride_a in elements for tensor index ti (0=gate,
    /// 1=up, 2=down): every arena slot pads to arena_slab_bytes, so the
    /// stride in THIS layer's quant blocks is ceil(slab/block_bytes),
    /// times QUANT_K elements. ceil: max slab may not divide this layer's
    /// block size (mixed quant).
    fn arena_stride_elems(&self, ti: usize, w: &ExpMat) -> u32 {
        let bb = Self::block_bytes(w.ggml_type);
        let blocks = (self.arena_slab_bytes[ti] as usize).div_ceil(bb);
        (blocks * quant_k(w.ggml_type)) as u32
    }

    fn arena_buf(&self, ti: usize) -> &Buffer {
        match ti {
            0 => self.barena_g.as_ref().unwrap(),
            1 => self.barena_u.as_ref().unwrap(),
            _ => self.barena_d.as_ref().unwrap(),
        }
    }

    /// Expert matvec with the three-way cold split (M7b-B). The topk
    /// shader partitions the k routed experts into disjoint rows:
    ///   row 1 (base+64):  hot   -> dense VRAM hot buffer (pinned regime)
    ///   row 2 (base+128): arena -> prefetched arena slot (global id
    ///                              layer*k+slot, so a_offset lands in this
    ///                              layer's arena region)
    ///   row 3 (base+192): gtt   -> in-place GTT tensor (misses fall here)
    /// SKIP early-out makes the dispatches write disjoint output slots —
    /// no barriers between them. With prefetch off, pmap is all-SKIP so
    /// the arena row is empty and every cold expert hits the GTT row.
    fn rec_matvec_id(
        &self,
        batch: &mut Batch,
        w: &ExpMat,
        x: &Buffer,
        y: &Buffer,
        ne11: u32,
        barrier: bool,
        layer: usize,
        ti: usize,
    ) {
        let base = (layer * 256) as u64;
        let arena_ok = self.barena_g.is_some();
        match (&w.hot, arena_ok) {
            (Some(hot), true) => {
                self.rec_matvec_id_buf(batch, w, hot, x, y, ne11, false, base + 64, None);
                self.rec_matvec_id_buf(
                    batch,
                    w,
                    self.arena_buf(ti),
                    x,
                    y,
                    ne11,
                    false,
                    base + 128,
                    Some(self.arena_stride_elems(ti, w)),
                );
                self.rec_matvec_id_buf(batch, w, &w.buf, x, y, ne11, barrier, base + 192, None);
            }
            (Some(hot), false) => {
                self.rec_matvec_id_buf(batch, w, hot, x, y, ne11, false, base + 64, None);
                self.rec_matvec_id_buf(batch, w, &w.buf, x, y, ne11, barrier, base + 192, None);
            }
            (None, true) => {
                self.rec_matvec_id_buf(
                    batch,
                    w,
                    self.arena_buf(ti),
                    x,
                    y,
                    ne11,
                    false,
                    base + 128,
                    Some(self.arena_stride_elems(ti, w)),
                );
                self.rec_matvec_id_buf(batch, w, &w.buf, x, y, ne11, barrier, base + 192, None);
            }
            (None, false) => {
                self.rec_matvec_id_buf(batch, w, &w.buf, x, y, ne11, barrier, base, None)
            }
        }
    }

    /// Record the M7b-B async gather for one layer: predicted experts
    /// (bpf_ids) -> arena slot (layer*k+s). Reads the prediction row, not
    /// bids. dst stride = arena slab (max across layers), src stride =
    /// this layer's own slab. Called into the ASYNC batch between tokens.
    fn rec_gather(
        &self,
        batch: &mut Batch,
        ly: &Layer,
        layer: usize,
        arena_g: &Buffer,
        arena_u: &Buffer,
        arena_d: &Buffer,
    ) {
        let canary = std::env::var("MOE_PF_CANARY").is_ok();
        let push = [
            self.arena_slab_v[0] as u32,
            self.arena_slab_v[1] as u32,
            self.arena_slab_v[2] as u32,
            (ly.gate_exps.slab / 16) as u32,
            (ly.up_exps.slab / 16) as u32,
            (ly.down_exps.slab / 16) as u32,
            layer as u32,
            self.hp.n_expert_used as u32,
            if canary { 1u32 } else { 0u32 },
        ];
        batch
            .dispatch_ranges(
                &self.gpu,
                &self.p_gather,
                &[
                    (&ly.gate_exps.buf, 0, WHOLE),
                    (&ly.up_exps.buf, 0, WHOLE),
                    (&ly.down_exps.buf, 0, WHOLE),
                    (arena_g, 0, WHOLE),
                    (arena_u, 0, WHOLE),
                    (arena_d, 0, WHOLE),
                    (&self.bpf_ids, (layer * 256) as u64, WHOLE),
                ],
                as_bytes_of(&push),
                (self.hp.n_expert_used as u32, 1, 1),
            )
            .unwrap();
    }

    fn rec_rms(
        &self,
        batch: &mut Batch,
        src: &Buffer,
        weight: &Buffer,
        dst: &Buffer,
        ncols: u32,
        nrows: u32,
    ) {
        let mut push = BinaryPush::contig2d((ncols, nrows), (ncols, 1), (ncols, nrows));
        push.param1 = self.hp.base.rms_eps;
        batch
            .dispatch_ranges(
                &self.gpu,
                &self.p_rms,
                &[(src, 0, WHOLE), (weight, 0, WHOLE), (dst, 0, WHOLE)],
                push.as_bytes(),
                (nrows, 1, 1),
            )
            .unwrap();
    }

    fn rec_add(&self, batch: &mut Batch, dst: &Buffer, src: &Buffer, n: u32) {
        batch
            .dispatch_ranges(
                &self.gpu,
                &self.p_add,
                &[(dst, 0, WHOLE), (src, 0, WHOLE), (dst, 0, WHOLE)],
                as_bytes(&[n]),
                (n.div_ceil(256), 1, 1),
            )
            .unwrap();
    }

    pub fn forward_token(&mut self, token: u32) -> Vec<f32> {
        // Online repinning: previous token's bids are final and the GPU is
        // idle between forward_token calls — safe to rewrite hot slabs/remap.
        if self.n_past > 0 {
            self.maybe_repin();
        }
        let hp = &self.hp;
        let (n_embd, n_heads, n_kv, hd) = (
            hp.base.n_embd,
            hp.base.n_heads,
            hp.base.n_kv_heads,
            hp.base.head_dim,
        );
        let (n_ff, k) = (hp.n_ff_exp, hp.n_expert_used);
        let n_ctx = self.n_ctx_max;
        let kv_dim = n_kv * hd;
        let gqa = n_heads / n_kv;
        let pos = self.n_past;
        assert!(pos < n_ctx, "KV arena full ({n_ctx})");
        let n_t = pos + 1;
        let scale = 1.0 / (hd as f32).sqrt();

        // Paged KV copy-on-grow: when the next write crosses the current
        // capacity, reallocate every layer's caches one page larger and
        // migrate contents (host-visible; once per KV_PAGE_TOKENS tokens).
        if pos >= self.kv_cap {
            let new_cap = (self.kv_cap + KV_PAGE_TOKENS).min(n_ctx);
            let old_bytes = self.kv_cap * kv_dim * 4;
            let mut tmp = vec![0u8; old_bytes];
            for ly in &mut self.layers {
                for cache in [&mut ly.kcache, &mut ly.vtcache] {
                    let new_buf = self
                        .gpu
                        .create_buffer((new_cap * kv_dim * 4) as u64, true)
                        .unwrap();
                    self.gpu.download(cache, &mut tmp).unwrap();
                    self.gpu.upload(&new_buf, &tmp).unwrap();
                    let old = std::mem::replace(cache, new_buf);
                    self.gpu.destroy_buffer(old);
                }
            }
            self.kv_cap = new_cap;
        }

        // embedding row (q4_K) on host
        let mut x = vec![0f32; n_embd];
        let bpr = n_embd / QK_K;
        for b in 0..bpr {
            let off = (token as usize * bpr + b) * 144;
            dequant_q4k(
                &self.embd_raw[off..off + 144],
                &mut x[b * QK_K..(b + 1) * QK_K],
            );
        }
        self.gpu.upload(&self.bx, as_bytes(&x)).unwrap();
        self.gpu
            .upload(&self.bpos, as_bytes(&[pos as i32]))
            .unwrap();

        let mut batch = self.batch.take().unwrap();
        batch.begin(&self.gpu).unwrap();

        for l in 0..self.hp.base.n_layers {
            let ly = &self.layers[l];
            // --- attention (identical structure to dense path) ---
            self.rec_rms(
                &mut batch,
                &self.bx,
                &ly.attn_norm,
                &self.bnorm,
                n_embd as u32,
                1,
            );
            self.rec_matvec_b(&mut batch, &ly.wq, &self.bnorm, &self.bq, false);
            self.rec_matvec_b(&mut batch, &ly.wk, &self.bnorm, &self.bk, false);
            self.rec_matvec_b(&mut batch, &ly.wv, &self.bnorm, &self.bv, true);
            match self.cfg.qk_norm {
                QkNorm::PerHead => {
                    self.rec_rms(
                        &mut batch,
                        &self.bq,
                        ly.q_norm.as_ref().unwrap(),
                        &self.bq,
                        hd as u32,
                        n_heads as u32,
                    );
                    self.rec_rms(
                        &mut batch,
                        &self.bk,
                        ly.k_norm.as_ref().unwrap(),
                        &self.bk,
                        hd as u32,
                        n_kv as u32,
                    );
                }
                QkNorm::FullVec => {
                    self.rec_rms(
                        &mut batch,
                        &self.bq,
                        ly.q_norm.as_ref().unwrap(),
                        &self.bq,
                        (n_heads * hd) as u32,
                        1,
                    );
                    self.rec_rms(
                        &mut batch,
                        &self.bk,
                        ly.k_norm.as_ref().unwrap(),
                        &self.bk,
                        kv_dim as u32,
                        1,
                    );
                }
                QkNorm::None => {}
            }
            for (buf, nh) in [(&self.bq, n_heads), (&self.bk, n_kv)] {
                let push = RopePush::neox(hd as u32, nh as u32, 1, self.hp.base.rope_base);
                batch
                    .dispatch_ranges(
                        &self.gpu,
                        &self.p_rope,
                        &[
                            (buf, 0, WHOLE),
                            (&self.bpos, 0, WHOLE),
                            (&self.bdummy, 0, WHOLE),
                            (buf, 0, WHOLE),
                            (&self.bdummy, 0, WHOLE),
                        ],
                        push.as_bytes(),
                        (nh as u32, (hd as u32 / 2).div_ceil(256), 1),
                    )
                    .unwrap();
            }
            for (src, cache) in [(&self.bk, &ly.kcache), (&self.bv, &ly.vtcache)] {
                let cp = UnaryPush::contig_copy(kv_dim as u32);
                batch
                    .dispatch_ranges(
                        &self.gpu,
                        &self.p_cpy,
                        &[(src, 0, WHOLE), (cache, (pos * kv_dim * 4) as u64, WHOLE)],
                        cp.as_bytes(),
                        ((kv_dim as u32).div_ceil(512), 1, 1),
                    )
                    .unwrap();
            }
            let ap = AttnPush {
                n_t: n_t as u32,
                hd: hd as u32,
                kv_dim: kv_dim as u32,
                gqa: gqa as u32,
                scale,
            };
            batch
                .dispatch_ranges(
                    &self.gpu,
                    &self.p_attn,
                    &[
                        (&self.bq, 0, WHOLE),
                        (&ly.kcache, 0, WHOLE),
                        (&ly.vtcache, 0, WHOLE),
                        (&self.battn, 0, WHOLE),
                    ],
                    as_bytes_of(&ap),
                    (n_heads as u32, 1, 1),
                )
                .unwrap();
            self.rec_matvec_b(&mut batch, &ly.wo, &self.battn, &self.bproj, true);
            self.rec_add(&mut batch, &self.bx, &self.bproj, n_embd as u32);

            // --- MoE FFN ---
            self.rec_rms(
                &mut batch,
                &self.bx,
                &ly.ffn_norm,
                &self.bnorm,
                n_embd as u32,
                1,
            );
            // router logits: f32 matvec (n_expert rows)
            batch
                .dispatch_ranges(
                    &self.gpu,
                    &self.mmv_f32,
                    &[
                        (&ly.router.buf, 0, WHOLE),
                        (&self.bnorm, 0, WHOLE),
                        (&self.brouter, 0, WHOLE),
                    ],
                    as_bytes(&[ly.router.ncols as u32, ly.router.ncols as u32]),
                    (ly.router.nrows as u32, 1, 1),
                )
                .unwrap();
            // top-k softmax with renorm + hot/arena/gtt id split
            batch
                .dispatch_ranges(
                    &self.gpu,
                    &self.p_topk,
                    &[
                        (&self.brouter, 0, WHOLE),
                        (&self.bweights, 0, WHOLE),
                        (&self.bids, (l * 256) as u64, 256),
                        (&ly.remap, 0, WHOLE),
                        (&self.bpmaps, (l * hp.n_expert * 4) as u64, (hp.n_expert * 4) as u64),
                    ],
                    as_bytes(&[
                        hp.n_expert as u32,
                        k as u32,
                        if self.cfg.norm_topk { 1u32 } else { 0u32 },
                    ]),
                    (1, 1, 1),
                )
                .unwrap();
            // M7b-A: gather cold slabs GTT -> VRAM arenas (no-op when all
            // experts are VRAM-resident or MOE_ARENA=0). Barrier included,
            // so the cold matvecs below read coherent arena data.
            // expert gate/up (same x for all slots), swiglu, down (per-slot x).
            // Cold experts read the prefetched arena (pmap hits) or fall
            // back to in-place GTT (pmap misses) — see rec_matvec_id.
            self.rec_matvec_id(&mut batch, &ly.gate_exps, &self.bnorm, &self.bgate, 1, false, l, 0);
            self.rec_matvec_id(&mut batch, &ly.up_exps, &self.bnorm, &self.bup, 1, true, l, 1);
            let glu = GluPush::split(n_ff as u32, k as u32);
            batch
                .dispatch_ranges(
                    &self.gpu,
                    &self.p_glu,
                    &[
                        (&self.bgate, 0, WHOLE),
                        (&self.bup, 0, WHOLE),
                        (&self.bgate, 0, WHOLE),
                    ],
                    glu.as_bytes(),
                    (((n_ff * k) as u32).div_ceil(512), 1, 1),
                )
                .unwrap();
            self.rec_matvec_id(
                &mut batch,
                &ly.down_exps,
                &self.bgate,
                &self.bdown,
                k as u32,
                true,
                l,
                2,
            );
            // weighted reduce into residual
            batch
                .dispatch_ranges(
                    &self.gpu,
                    &self.p_reduce,
                    &[
                        (&self.bdown, 0, WHOLE),
                        (&self.bweights, 0, WHOLE),
                        (&self.bx, 0, WHOLE),
                    ],
                    as_bytes(&[n_embd as u32, k as u32]),
                    ((n_embd as u32).div_ceil(256), 1, 1),
                )
                .unwrap();
        }

        self.rec_rms(
            &mut batch,
            &self.bx,
            &self.output_norm,
            &self.bnorm,
            n_embd as u32,
            1,
        );
        self.rec_matvec_b(&mut batch, &self.head, &self.bnorm, &self.blogits, true);

        // PRE-SUBMIT snapshot: bids is written by topk (recorded above). If
        // valid here but corrupt after the fence, the GPU execution of this
        // batch (or the async prefetch) is the culprit.
        if std::env::var("MOE_PF_DBG").is_ok() {
            let nl = self.hp.base.n_layers;
            let mut raw = vec![0u8; nl * 256];
            self.gpu.download(&self.bids, &mut raw).unwrap();
            let bad = (0..nl)
                .filter(|&l| u32::from_le_bytes([raw[l * 256], raw[l * 256 + 1], raw[l * 256 + 2], raw[l * 256 + 3]]) == u32::MAX)
                .count();
            eprintln!("pf-dbg pre-submit corrupt_layers={bad}");
        }

        // M7b-B: if a prefetch is in flight, this batch waits on its
        // timeline value so the arena reads see gathered data.
        if self.prefetch_wait_val > 0 {
            let sem = self.prefetch_sem.unwrap();
            let val = self.prefetch_wait_val;
            batch.submit_wait_sem(&self.gpu, sem, val).unwrap();
            self.prefetch_wait_val = 0;
        } else {
            batch.submit(&self.gpu).unwrap();
        }
        if std::env::var("MOE_PF_DBG").is_ok() {
            // post-fence bids snapshot: corrupted here => GPU-side cause
            let n_layers = self.hp.base.n_layers;
            let mut raw = vec![0u8; n_layers * 256];
            self.gpu.download(&self.bids, &mut raw).unwrap();
            let row0_ff = |l: usize| {
                u32::from_le_bytes([
                    raw[l * 256],
                    raw[l * 256 + 1],
                    raw[l * 256 + 2],
                    raw[l * 256 + 3],
                ]) == u32::MAX
            };
            let bad = (0..n_layers).filter(|&l| row0_ff(l)).count();
            eprintln!("pf-dbg post-submit corrupt_layers={bad}");
            if bad > 0 {
                for l in [4usize, 5, 47] {
                    let row: Vec<u32> = (0..64)
                        .map(|i| {
                            let o = l * 256 + i * 4;
                            u32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]])
                        })
                        .collect();
                    eprintln!("pf-dbg layer {l} full256={row:?}");
                }
            }
        }
        self.batch = Some(batch);
        self.n_past += 1;

        let mut logits = vec![0f32; self.hp.base.n_vocab];
        self.gpu
            .download(&self.blogits, unsafe {
                std::slice::from_raw_parts_mut(logits.as_mut_ptr() as *mut u8, logits.len() * 4)
            })
            .unwrap();
        // M7b-B: now that this token's routing is final (fence waited),
        // prefetch its experts for the NEXT token while the host idles.
        self.prepare_prefetch();
        if std::env::var("MOE_PF_DBG").is_ok() {
            // post-PREFETCH snapshot: bids right after the async gather was
            // submitted (still in flight unless MOE_PF_SYNC). Corrupt here
            // => the gather itself clobbers bids.
            let nl = self.hp.base.n_layers;
            let mut raw = vec![0u8; nl * 256];
            self.gpu.download(&self.bids, &mut raw).unwrap();
            let bad = (0..nl)
                .filter(|&l| {
                    u32::from_le_bytes([
                        raw[l * 256],
                        raw[l * 256 + 1],
                        raw[l * 256 + 2],
                        raw[l * 256 + 3],
                    ]) == u32::MAX
                })
                .count();
            eprintln!("pf-dbg post-prefetch corrupt_layers={bad}");
        }
        logits
    }

    /// M7b-B prefetch state machine. Called after each token's fence:
    /// predicts the next token's experts (the token just routed — overlap
    /// mean 0.51), builds per-layer pmaps + prediction ids, records the
    /// async gathers on the family-1 queue and signals the timeline
    /// semaphore that the next token's batch will wait on.
    ///
    /// Correctness: predictions are only ever a speed optimization. A
    /// prefetched-but-unrouted expert is never read (SKIP in the arena
    /// row); a routed-but-unprefetched expert falls to the in-place GTT
    /// row. Output is identical to no-prefetch.
    fn prepare_prefetch(&mut self) {
        if !self.prefetch {
            return;
        }
        let n_layers = self.hp.base.n_layers;
        let k = self.hp.n_expert_used;
        // Predict = this token's routed experts. Skip experts that are
        // hot (they never touch the arena) and dedupe per layer.
        let ids = self.last_expert_ids();
        if std::env::var("MOE_PF_DBG").is_ok() {
            for l in 0..n_layers {
                eprintln!("pf-dbg layer {} row0={:?}", l, &ids[l]);
            }
        }
        for l in 0..n_layers {
            let ly = &self.layers[l];
            // reset this layer's pmap to all-SKIP
            for e in 0..self.hp.n_expert {
                self.pmap_host[l * self.hp.n_expert + e] = u32::MAX;
            }
            // MOE_PF_CANARY_PMAP: stamp a distinctive value into the pmap
            // host vector; if it appears in bids, the leak is proven.
            if std::env::var("MOE_PF_CANARY_PMAP").is_ok() {
                self.pmap_host[l * self.hp.n_expert] = 0x0BADF00D;
            }
            let mut n_pf = 0usize;
            for s in 0..k {
                let e = ids[l][s] as usize;
                if e >= self.hp.n_expert {
                    eprintln!(
                        "pf-dbg layer {} slot {} id={:#x} row={:?}",
                        l, s, ids[l][s], &ids[l]
                    );
                    panic!("SKIP in bids row 0");
                }
                if ly.hot_mask[e] {
                    continue; // hot experts read the dense hot buffer
                }
                if self.pmap_host[l * self.hp.n_expert + e] != u32::MAX {
                    continue; // duplicate slot this layer
                }
                let slot = l * k + n_pf; // GLOBAL arena slot
                self.pmap_host[l * self.hp.n_expert + e] = slot as u32;
                self.pf_ids_host[l * 64 + n_pf] = e as u32;
                n_pf += 1;
            }
            // pad remaining prediction slots with SKIP
            for s in n_pf..k {
                self.pf_ids_host[l * 64 + s] = u32::MAX;
            }
        }
        // MOE_PF_FORCE_SKIP: overwrite every prediction id with SKIP so the
        // gather shader dispatches + reads ids[] but performs ZERO arena
        // writes (early-out). Isolates "dispatch/descriptor binding" from
        // "arena write" as the corruption source.
        if std::env::var("MOE_PF_FORCE_SKIP").is_ok() {
            for v in self.pf_ids_host.iter_mut() {
                *v = u32::MAX;
            }
        }
        // upload pmaps + prediction ids (host-visible, cheap memcpy)
        // MOE_PF_NO_UPLOAD: skip both uploads (isolation probe).
        // MOE_PF_UPLOAD=pmaps|pfids: upload only that one.
        let up = std::env::var("MOE_PF_UPLOAD").unwrap_or_default();
        let no_up = std::env::var("MOE_PF_NO_UPLOAD").is_ok();
        // upload pmap to the INACTIVE buffer, then swap so next token's topk sees it
        // MOE_PF_NO_SWAP: upload directly to bpmaps (no double-buffer swap) to
        // test whether the swap is implicated in bids corruption.
        let no_swap = std::env::var("MOE_PF_NO_SWAP").is_ok();
        let pmap_dst = if no_swap { &mut self.bpmaps } else { &mut self.bpmaps_alt };
        // MOE_PF_HOST_UP: use host-memcpy upload() instead of GPU-copy
        // upload_staged() for the pmap, to isolate whether the GPU transfer
        // command is the corruption vector.
        let host_up = std::env::var("MOE_PF_HOST_UP").is_ok();
        if !no_up && up != "pfids" {
            let src = unsafe {
                std::slice::from_raw_parts(
                    self.pmap_host.as_ptr() as *const u8,
                    self.pmap_host.len() * 4,
                )
            };
            if host_up {
                self.gpu.upload(pmap_dst, src).unwrap();
            } else {
                self.gpu.upload_staged(pmap_dst, src).unwrap();
            }
            // swap active/inactive — next token's topk binds the new active pmap
            if !no_swap {
                std::mem::swap(&mut self.bpmaps, &mut self.bpmaps_alt);
            }
            if std::env::var("MOE_PF_DBG").is_ok() {
                let va = self.gpu.device_address(&self.bpmaps);
                let va_alt = self.gpu.device_address(&self.bpmaps_alt);
                eprintln!("pf-dbg pmap swap: active bpmaps VA={va:#x}, inactive VA={va_alt:#x}");
                // Download both pmaps to verify content
                let nl = self.hp.base.n_layers;
                let ne = self.hp.n_expert;
                let mut raw_active = vec![0u8; nl * ne * 4];
                let mut raw_inactive = vec![0u8; nl * ne * 4];
                self.gpu.download(&self.bpmaps, &mut raw_active).unwrap();
                self.gpu.download(&self.bpmaps_alt, &mut raw_inactive).unwrap();
                let first_active = u32::from_le_bytes([raw_active[0], raw_active[1], raw_active[2], raw_active[3]]);
                let first_inactive = u32::from_le_bytes([raw_inactive[0], raw_inactive[1], raw_inactive[2], raw_inactive[3]]);
                eprintln!(
                                    "pf-dbg pmap swap content: active[0]={first_active:#x}, inactive[0]={first_inactive:#x} (0xFF={:#x})",
                                    0xFFFFFFFFu32
                                );
            }
        }
        if !no_up && up != "pmaps" {
            self.gpu
                .upload_staged(&self.bpf_ids, unsafe {
                    std::slice::from_raw_parts(
                        self.pf_ids_host.as_ptr() as *const u8,
                        self.pf_ids_host.len() * 4,
                    )
                })
                .unwrap();
        }

        // record async gathers for every layer into one command buffer
        self.sem_counter += 1;
        let wait_val = self.sem_counter;
        if std::env::var("MOE_ADDR_DBG").is_ok() {
            let va = |b: &Buffer| (self.gpu.device_address(b), b.size);
            let (bs, bl) = va(&self.bids);
            let (gs, gl) = va(&self.barena_g.as_ref().unwrap());
            let (us, ul) = va(&self.barena_u.as_ref().unwrap());
            let (ds, dl) = va(&self.barena_d.as_ref().unwrap());
            let (ps, pl) = va(&self.bpmaps);
            let (fs, fl) = va(&self.bpf_ids);
            let (pas, pal) = va(&self.bpmaps_alt);
            let rng = |a: u64, b: u64| format!("{a:#x}..{:#x}", a + b);
            eprintln!(
                "va-dbg bids {} | pmaps {} | pmapsALT {} | pfids {} | arenaG {} | arenaU {} | arenaD {}",
                rng(bs, bl), rng(ps, pl), rng(pas, pal), rng(fs, fl), rng(gs, gl), rng(us, ul), rng(ds, dl)
            );
            let overlaps = |s1: u64, l1: u64, s2: u64, l2: u64| {
                s1 < s2 + l2 && s2 < s1 + l1
            };
            for (name, s, l) in [
                ("arenaG", gs, gl), ("arenaU", us, ul), ("arenaD", ds, dl),
                ("pmaps", ps, pl), ("pmapsALT", pas, pal), ("pfids", fs, fl),
            ] {
                if overlaps(bs, bl, s, l) {
                    eprintln!("VA OVERLAP: {name} {:#x}+{:#x} vs bids {:#x}+{:#x}", s, l, bs, bl);
                }
            }
            // Physical-page aliasing check: export each small buffer's
            // device memory as a DMA-buf fd; same inode = same pages.
            for (name, b) in [
                ("bids", &self.bids),
                ("bpmaps", &self.bpmaps),
                ("bpmaps_alt", &self.bpmaps_alt),
                ("bpf_ids", &self.bpf_ids),
                ("arenaG", self.barena_g.as_ref().unwrap()),
                ("arenaU", self.barena_u.as_ref().unwrap()),
                ("arenaD", self.barena_d.as_ref().unwrap()),
            ] {
                eprintln!(
                    "mem-dbg {name}: buf={:#x} mem={:#x} size={}",
                    self.gpu.raw_buffer_handle(b),
                    self.gpu.raw_mem_handle(b),
                    b.size
                );
                match self.gpu.memory_fd_info(b) {
                    Ok((fd, ino, sz)) => {
                        eprintln!("fd-dbg {name}: fd={fd} inode={ino} size={sz}");
                    }
                    Err(e) => eprintln!("fd-dbg {name}: ERR {e}"),
                }
            }
        }
        {
            let mut abatch = self.async_batch.take().unwrap();
            let (ag, au, ad) = (
                self.barena_g.take().unwrap(),
                self.barena_u.take().unwrap(),
                self.barena_d.take().unwrap(),
            );
            abatch.begin(&self.gpu).unwrap();
            // MOE_PF_NO_GATHER: still begin+submit (exercising the async
            // machinery and semaphore) but record zero gather dispatches —
            // isolates "async submit" from "gather writes" as the cause.
            if std::env::var("MOE_PF_NO_GATHER").is_err() {
                // MOE_PF_ONLY_LAYER=N: record a gather for only one layer
                // (probe whether corruption scales with gather count).
                let only: Option<usize> = std::env::var("MOE_PF_ONLY_LAYER")
                    .ok()
                    .and_then(|s| s.parse().ok());
                for l in 0..n_layers {
                    if let Some(o) = only {
                        if l != o {
                            continue;
                        }
                    }
                    let ly = &self.layers[l];
                    self.rec_gather(&mut abatch, ly, l, &ag, &au, &ad);
                }
            }
            let sem = self.prefetch_sem.unwrap();
            abatch.submit_async(&self.gpu, sem, wait_val).unwrap();
            // MOE_PF_SYNC: block until the gather's fence signals — fully
            // serializes the gather with the next token's host uploads and
            // main-queue work. Clean result here => concurrency race.
            if std::env::var("MOE_PF_SYNC").is_ok() {
                abatch.wait_idle(&self.gpu).unwrap();
            }
            self.async_batch = Some(abatch);
            self.barena_g = Some(ag);
            self.barena_u = Some(au);
            self.barena_d = Some(ad);
        }
        self.prefetch_wait_val = wait_val;
    }

    /// Selected expert ids from the LAST forward_token: n_layers rows of k ids.
    /// Reads back bids (host-visible); call after forward_token.
    pub fn last_expert_ids(&self) -> Vec<Vec<u32>> {
        let n_layers = self.hp.base.n_layers;
        let k = self.hp.n_expert_used;
        let mut raw = vec![0u8; n_layers * 256];
        self.gpu.download(&self.bids, &mut raw).unwrap();
        (0..n_layers)
            .map(|l| {
                (0..k)
                    .map(|i| {
                        let o = l * 256 + i * 4;
                        u32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]])
                    })
                    .collect()
            })
            .collect()
    }
}

impl Drop for MoeModel {
    fn drop(&mut self) {
        if let Some(b) = self.batch.take() {
            self.gpu.destroy_batch(b);
        }
    }
}
