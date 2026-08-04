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

use crate::{as_bytes, as_bytes_of, AttnPush, HParams, WHOLE};
use gguf_rs::Gguf;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use vk_backend::ops::{BinaryPush, GluPush, MatVecIdPush, MatVecPush, RopePush, UnaryPush};
use vk_backend::{Batch, Buffer, Gpu, Pipeline};

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
    // Dense hot buffer: hottest experts' slabs copied contiguously into VRAM
    // (plus one trailing zero slab), ids remapped by the router shader.
    hot: Option<Buffer>,
}

struct Layer {
    attn_norm: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    ffn_norm: Buffer,
    wq: GpuMat,
    wk: GpuMat,
    wv: GpuMat,
    wo: GpuMat,
    router: GpuMat, // f32 (n_expert, n_embd)
    gate_exps: ExpMat,
    up_exps: ExpMat,
    down_exps: ExpMat,
    remap: Buffer, // n_expert u32: expert -> dense hot idx, or SKIP (u32::MAX)
    kcache: Buffer,
    vtcache: Buffer,
}

pub struct MoeHParams {
    pub base: HParams,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
}

pub struct MoeModel {
    pub hp: MoeHParams,
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
    layers: Vec<Layer>,
    output_norm: Buffer,
    head: GpuMat,
    embd_raw: Vec<u8>,
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
    batch: Option<Batch>,
    pub n_ctx_max: usize,
    n_past: usize,
}

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
        let pfx = "qwen3moe";
        let n_layers = get_u32(&format!("{pfx}.block_count")) as usize;
        let n_embd = get_u32(&format!("{pfx}.embedding_length")) as usize;
        let n_heads = get_u32(&format!("{pfx}.attention.head_count")) as usize;
        let n_kv_heads = get_u32(&format!("{pfx}.attention.head_count_kv")) as usize;
        let head_dim = get_u32(&format!("{pfx}.attention.key_length")) as usize;
        let n_expert = get_u32(&format!("{pfx}.expert_count")) as usize;
        let n_expert_used = get_u32(&format!("{pfx}.expert_used_count")) as usize;
        let n_ff_exp = get_u32(&format!("{pfx}.expert_feed_forward_length")) as usize;
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
        let p_topk = gpu.create_pipeline(&ours("topk_softmax_f32"), 4, 8, &[])?;
        let p_reduce = gpu.create_pipeline(&ours("moe_reduce_f32"), 3, 8, &[])?;

        // Expert placement policy:
        //   default              -> all experts in GTT (host visible, PCIe reads)
        //   MOE_EXPERTS_VRAM=1   -> all experts in VRAM
        //   MOE_EXPERTS_VRAM_MB=N -> first N MiB of expert tensors in VRAM, rest GTT
        //   + MOE_EXPERT_HIST=csv -> popularity pinning: hottest (layer,expert)
        //     slabs are COPIED into per-layer dense VRAM buffers (hot buffer),
        //     router ids remapped in-shader; cold experts stay in the GTT
        //     buffer. Budget = MOE_EXPERTS_VRAM_MB. (Sparse binding lost 2x.)
        let all_vram = std::env::var("MOE_EXPERTS_VRAM").is_ok();
        let vram_budget: u64 = if all_vram {
            u64::MAX
        } else {
            std::env::var("MOE_EXPERTS_VRAM_MB")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .map(|mb| mb * 1024 * 1024)
                .unwrap_or(0)
        };
        let vram_left = std::cell::Cell::new(vram_budget);
        let vram_used = std::cell::Cell::new(0u64);

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
                (gpu.create_buffer_host(t.size_bytes)?, hot_buf)
            } else if vram_left.get() >= t.size_bytes {
                vram_left.set(vram_left.get() - t.size_bytes);
                vram_used.set(vram_used.get() + t.size_bytes);
                (gpu.create_buffer(t.size_bytes, true)?, None)
            } else {
                (gpu.create_buffer_host(t.size_bytes)?, None)
            };
            gpu.upload(&buf, bytes(name))?;
            Ok(ExpMat {
                buf,
                ncols: t.dims[0] as usize,
                nrows: t.dims[1] as usize,
                ggml_type: t.ggml_type,
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

        let n_ctx_max = 4096usize;
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
            layers.push(Layer {
                attn_norm: upload_f32(&n("attn_norm"))?,
                q_norm: upload_f32(&n("attn_q_norm"))?,
                k_norm: upload_f32(&n("attn_k_norm"))?,
                ffn_norm: upload_f32(&n("ffn_norm"))?,
                wq: upload_mat(&n("attn_q"))?,
                wk: upload_mat(&n("attn_k"))?,
                wv: upload_mat(&n("attn_v"))?,
                wo: upload_mat(&n("attn_output"))?,
                router: upload_f32_mat(&n("ffn_gate_inp"))?,
                gate_exps: upload_exp(&n("ffn_gate_exps"), l)?,
                up_exps: upload_exp(&n("ffn_up_exps"), l)?,
                down_exps: upload_exp(&n("ffn_down_exps"), l)?,
                remap,
                kcache: gpu.create_buffer((n_ctx_max * kv_dim * 4) as u64, true)?,
                vtcache: gpu.create_buffer((n_ctx_max * kv_dim * 4) as u64, true)?,
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
        assert_eq!(embd_type, 12, "expected q4_K token_embd");

        let q_dim = n_heads * head_dim;
        let k = n_expert_used;
        let mk = |n: usize| gpu.create_buffer((n * 4) as u64, true);
        let bx = mk(n_embd)?;
        let bnorm = mk(n_embd)?;
        let bq = mk(q_dim)?;
        let bk = mk(kv_dim)?;
        let bv = mk(kv_dim)?;
        let battn = mk(q_dim)?;
        let bproj = mk(n_embd)?;
        let brouter = mk(n_expert)?;
        let bweights = mk(k)?;
        // 3 rows x 64B per layer: [orig ids | hot ids | cold ids]
        let bids = mk(n_layers * 48)?;
        let bgate = mk(k * n_ff_exp)?;
        let bup = mk(k * n_ff_exp)?;
        let bdown = mk(k * n_embd)?;
        let blogits = mk(n_vocab)?;
        let bpos = mk(1)?;
        let bdummy = mk(1)?;

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
            batch,
            n_ctx_max,
            n_past: 0,
        })
    }

    pub fn reset(&mut self) {
        self.n_past = 0;
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
    /// read (full GTT tensor or dense hot VRAM buffer); `ids_off` selects
    /// which id row drives the dispatch (slots with SKIP ids early-out).
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
    ) {
        let k = self.hp.n_expert_used as u32;
        let pipe = &self.mmv_id[&w.ggml_type];
        let push = MatVecIdPush {
            ncols: w.ncols as u32,
            stride_a: (w.ncols / quant_k(w.ggml_type)) as u32,
            stride_b: w.ncols as u32,
            stride_d: w.nrows as u32,
            batch_stride_a: (w.nrows * w.ncols) as u32,
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

    /// Expert matvec with hot/cold split: when the layer has a dense hot
    /// buffer, dispatch twice (hot ids into the VRAM copy, cold ids into
    /// the GTT tensor) — disjoint slots, so no barrier between the pair.
    fn rec_matvec_id(
        &self,
        batch: &mut Batch,
        w: &ExpMat,
        x: &Buffer,
        y: &Buffer,
        ne11: u32,
        barrier: bool,
        layer: usize,
    ) {
        let base = (layer * 192) as u64;
        match &w.hot {
            Some(hot) => {
                self.rec_matvec_id_buf(batch, w, hot, x, y, ne11, false, base + 64);
                self.rec_matvec_id_buf(batch, w, &w.buf, x, y, ne11, barrier, base + 128);
            }
            None => self.rec_matvec_id_buf(batch, w, &w.buf, x, y, ne11, barrier, base),
        }
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
            self.rec_rms(
                &mut batch,
                &self.bq,
                &ly.q_norm,
                &self.bq,
                hd as u32,
                n_heads as u32,
            );
            self.rec_rms(
                &mut batch,
                &self.bk,
                &ly.k_norm,
                &self.bk,
                hd as u32,
                n_kv as u32,
            );
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
            // top-k softmax with renorm + hot/cold id split via remap table
            batch
                .dispatch_ranges(
                    &self.gpu,
                    &self.p_topk,
                    &[
                        (&self.brouter, 0, WHOLE),
                        (&self.bweights, 0, WHOLE),
                        (&self.bids, (l * 192) as u64, WHOLE),
                        (&ly.remap, 0, WHOLE),
                    ],
                    as_bytes(&[hp.n_expert as u32, k as u32]),
                    (1, 1, 1),
                )
                .unwrap();
            // expert gate/up (same x for all slots), swiglu, down (per-slot x)
            self.rec_matvec_id(
                &mut batch,
                &ly.gate_exps,
                &self.bnorm,
                &self.bgate,
                1,
                false,
                l,
            );
            self.rec_matvec_id(&mut batch, &ly.up_exps, &self.bnorm, &self.bup, 1, true, l);
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

        batch.submit(&self.gpu).unwrap();
        self.batch = Some(batch);
        self.n_past += 1;

        let mut logits = vec![0f32; self.hp.base.n_vocab];
        self.gpu
            .download(&self.blogits, unsafe {
                std::slice::from_raw_parts_mut(logits.as_mut_ptr() as *mut u8, logits.len() * 4)
            })
            .unwrap();
        logits
    }

    /// Selected expert ids from the LAST forward_token: n_layers rows of k ids.
    /// Reads back bids (host-visible); call after forward_token.
    pub fn last_expert_ids(&self) -> Vec<Vec<u32>> {
        let n_layers = self.hp.base.n_layers;
        let k = self.hp.n_expert_used;
        let mut raw = vec![0u8; n_layers * 192];
        self.gpu.download(&self.bids, &mut raw).unwrap();
        (0..n_layers)
            .map(|l| {
                (0..k)
                    .map(|i| {
                        let o = l * 192 + i * 4;
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
