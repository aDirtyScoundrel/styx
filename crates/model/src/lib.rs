//! Qwen3-dense forward pass, GPU-resident (M3 graph executor).
//!
//! Every op per token is recorded into one command buffer (vk_backend::Batch)
//! and submitted with a single fence: rms norms (fused weight multiply),
//! q8_0 matvecs, neox rope, attention scores/softmax/values via strided f32
//! matvecs against GPU KV caches, swiglu, residual adds, lm_head. The only
//! per-token host traffic is the embedding row upload, a position upload,
//! and the final logits download.
//!
//! KV layout: K cache is [t][kv_dim] (row per token); V cache is stored
//! TRANSPOSED as [kv_dim][n_ctx_max] so the value reduction is a matvec with
//! row stride n_ctx_max. Per-head sub-views are bound as descriptor
//! sub-ranges — all offsets are multiples of 64 B.

use gguf_rs::Gguf;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use vk_backend::ops::{BinaryPush, GluPush, MatVecPush, RopePush, SoftMaxPush, UnaryPush};
use vk_backend::{Batch, Buffer, Gpu, Pipeline};

pub mod moe;
pub use moe::MoeModel;

const Q8_BLOCK: usize = 34;
const Q8_K: usize = 32;

pub struct HParams {
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
    pub n_vocab: usize,
}

struct GpuMat {
    buf: Buffer,
    nrows: usize,
    ncols: usize,
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
    w_gate: GpuMat,
    w_up: GpuMat,
    w_down: GpuMat,
    kcache: Buffer,  // [n_ctx_max][kv_dim] f32
    vtcache: Buffer, // [n_ctx_max][kv_dim] f32 (row-major, same layout as kcache)
}

pub struct Model {
    pub hp: HParams,
    gpu: Gpu,
    // pipelines
    mmv1: Pipeline, // q8_0 matvec NUM_ROWS=1
    mmv4: Pipeline, // q8_0 matvec NUM_ROWS=4 (lm_head)
    #[allow(dead_code)] // kept: pre-fused-attention path, used by tests/debug
    mmv_f32: Pipeline, // f32 matvec (attention)
    p_rms: Pipeline, // rms_norm_f32, do_multiply=true
    p_add: Pipeline, // add_f32_f32_f32, norepeat
    p_cpy: Pipeline, // cpy_f32_f32 (strided KV writes)
    #[allow(dead_code)]
    p_soft: Pipeline, // soft_max_f32
    p_attn: Pipeline, // fused decode attention (custom)
    p_glu: Pipeline, // swiglu_f32 split
    p_rope: Pipeline, // rope_neox_f32
    layers: Vec<Layer>,
    output_norm: Buffer,
    embd_raw: Vec<u8>, // q8_0 token_embd, host-side for row gather
    embd_gpu: GpuMat,  // tied lm_head
    // GPU activations (f32)
    bx: Buffer,    // hidden state (n_embd)
    bnorm: Buffer, // normed hidden (n_embd)
    bq: Buffer,    // q (q_dim)
    bk: Buffer,    // k (kv_dim)
    bv: Buffer,    // v (kv_dim)
    battn: Buffer, // attention output (q_dim)
    bproj: Buffer, // projection scratch (n_embd)
    bgate: Buffer, // ffn gate (n_ff)
    bup: Buffer,   // ffn up (n_ff)
    #[allow(dead_code)]
    bscores: Buffer, // [n_heads][n_ctx_max] raw scores
    #[allow(dead_code)]
    bprobs: Buffer, // [n_heads][n_ctx_max] softmaxed
    blogits: Buffer, // (n_vocab)
    bpos: Buffer,  // i32[1] current position (rope)
    bdummy: Buffer, // dummy for unused bindings
    batch: Option<Batch>,
    pub n_ctx_max: usize,
    n_past: usize,
}

fn dequant_q8_0_row(raw: &[u8], row: usize, ncols: usize, out: &mut [f32]) {
    let bpr = ncols / Q8_K;
    for b in 0..bpr {
        let off = (row * bpr + b) * Q8_BLOCK;
        let blk = &raw[off..off + Q8_BLOCK];
        let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        for i in 0..Q8_K {
            out[b * Q8_K + i] = d * (blk[2 + i] as i8) as f32;
        }
    }
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn as_bytes_of<T>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}

#[repr(C)]
pub(crate) struct AttnPush {
    pub n_t: u32,
    pub hd: u32,
    pub kv_dim: u32,
    pub gqa: u32,
    pub scale: f32,
}

pub(crate) const WHOLE: u64 = u64::MAX; // vk::WHOLE_SIZE

impl Model {
    pub fn load(path: &Path) -> Result<Model, String> {
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
        let tmap: HashMap<&str, &gguf_rs::TensorInfo> =
            g.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        let embd_info = tmap["token_embd.weight"];
        let hp = HParams {
            n_layers: get_u32("qwen3.block_count") as usize,
            n_embd: get_u32("qwen3.embedding_length") as usize,
            n_ff: get_u32("qwen3.feed_forward_length") as usize,
            n_heads: get_u32("qwen3.attention.head_count") as usize,
            n_kv_heads: get_u32("qwen3.attention.head_count_kv") as usize,
            head_dim: get_u32("qwen3.attention.key_length") as usize,
            rms_eps: get_f32("qwen3.attention.layer_norm_rms_epsilon"),
            rope_base: get_f32("qwen3.rope.freq_base"),
            n_vocab: embd_info.dims[1] as usize,
        };

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
        let mmv_pc = std::mem::size_of::<MatVecPush>() as u32;
        let mmv1 = gpu.create_pipeline(
            &spv("mul_mat_vec_q8_0_f32_f32"),
            5,
            mmv_pc,
            &[(0, 64), (1, 2), (2, 1)],
        )?;
        let mmv4 = gpu.create_pipeline(
            &spv("mul_mat_vec_q8_0_f32_f32"),
            5,
            mmv_pc,
            &[(0, 64), (1, 4), (2, 1)],
        )?;
        let strided_spv: PathBuf = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../shaders/matvec_strided_f32.spv"
        )
        .into();
        let mmv_f32 = gpu.create_pipeline(&strided_spv, 3, 8, &[])?;
        let add_spv: PathBuf = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../shaders/add_flat_f32.spv"
        )
        .into();
        let p_add = gpu.create_pipeline(&add_spv, 3, 4, &[])?;
        let bin_pc = std::mem::size_of::<BinaryPush>() as u32;
        let p_rms = gpu.create_pipeline(&spv("rms_norm_f32"), 3, bin_pc, &[(1, 1)])?;
        let p_cpy = gpu.create_pipeline(
            &spv("cpy_f32_f32"),
            2,
            std::mem::size_of::<UnaryPush>() as u32,
            &[],
        )?;
        let p_soft = gpu.create_pipeline(
            &spv("soft_max_f32"),
            4,
            std::mem::size_of::<SoftMaxPush>() as u32,
            &[(0, 128)],
        )?;
        let attn_spv: PathBuf = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../shaders/attn_decode_f32.spv"
        )
        .into();
        let p_attn = gpu.create_pipeline(&attn_spv, 4, 20, &[])?;
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

        let upload_mat = |name: &str| -> Result<GpuMat, String> {
            let t = &tmap[name];
            assert_eq!(t.ggml_type, 8, "{name} not q8_0 (type {})", t.ggml_type);
            let b = gpu.create_buffer(t.size_bytes, true)?;
            gpu.upload(&b, bytes(name))?;
            Ok(GpuMat {
                buf: b,
                nrows: t.dims[1] as usize,
                ncols: t.dims[0] as usize,
            })
        };
        let upload_f32 = |name: &str| -> Result<Buffer, String> {
            let v = f32vec(name);
            let b = gpu.create_buffer((v.len() * 4) as u64, true)?;
            gpu.upload(&b, as_bytes(&v))?;
            Ok(b)
        };

        let n_ctx_max = 4096usize;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let mut layers = Vec::new();
        for l in 0..hp.n_layers {
            let n = |s: &str| format!("blk.{l}.{s}.weight");
            layers.push(Layer {
                attn_norm: upload_f32(&n("attn_norm"))?,
                q_norm: upload_f32(&n("attn_q_norm"))?,
                k_norm: upload_f32(&n("attn_k_norm"))?,
                ffn_norm: upload_f32(&n("ffn_norm"))?,
                wq: upload_mat(&n("attn_q"))?,
                wk: upload_mat(&n("attn_k"))?,
                wv: upload_mat(&n("attn_v"))?,
                wo: upload_mat(&n("attn_output"))?,
                w_gate: upload_mat(&n("ffn_gate"))?,
                w_up: upload_mat(&n("ffn_up"))?,
                w_down: upload_mat(&n("ffn_down"))?,
                kcache: gpu.create_buffer((n_ctx_max * kv_dim * 4) as u64, true)?,
                vtcache: gpu.create_buffer((kv_dim * n_ctx_max * 4) as u64, true)?,
            });
        }
        let embd_gpu = upload_mat("token_embd.weight")?;
        let output_norm = upload_f32("output_norm.weight")?;
        let embd_raw = bytes("token_embd.weight").to_vec();

        let q_dim = hp.n_heads * hp.head_dim;
        let mk = |n: usize| gpu.create_buffer((n * 4) as u64, true);
        let bx = mk(hp.n_embd)?;
        let bnorm = mk(hp.n_embd)?;
        let bq = mk(q_dim)?;
        let bk = mk(kv_dim)?;
        let bv = mk(kv_dim)?;
        let battn = mk(q_dim)?;
        let bproj = mk(hp.n_embd)?;
        let bgate = mk(hp.n_ff)?;
        let bup = mk(hp.n_ff)?;
        let bscores = mk(hp.n_heads * n_ctx_max)?;
        let bprobs = mk(hp.n_heads * n_ctx_max)?;
        let blogits = mk(hp.n_vocab)?;
        let bpos = mk(1)?;
        let bdummy = mk(1)?;

        // ~66 dispatches/layer * 28 layers + head/tail ≈ 1900 sets, ≤5 buffers each.
        let batch = Some(gpu.create_batch(4096, 20480)?);

        Ok(Model {
            hp,
            gpu,
            mmv1,
            mmv4,
            mmv_f32,
            p_rms,
            p_add,
            p_cpy,
            p_soft,
            p_attn,
            p_glu,
            p_rope,
            layers,
            output_norm,
            embd_raw,
            embd_gpu,
            bx,
            bnorm,
            bq,
            bk,
            bv,
            battn,
            bproj,
            bgate,
            bup,
            bscores,
            bprobs,
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

    /// Record y = W · x (W q8_0). Whole-buffer bindings.
    fn rec_matvec(&self, batch: &mut Batch, w: &GpuMat, x: &Buffer, y: &Buffer) {
        self.rec_matvec_b(batch, w, x, y, true)
    }

    /// Like rec_matvec but with control over the trailing barrier (false when
    /// the next dispatch does not read `y`).
    fn rec_matvec_b(&self, batch: &mut Batch, w: &GpuMat, x: &Buffer, y: &Buffer, barrier: bool) {
        let (pipe, num_rows) = if w.nrows % 4 == 0 && w.nrows / 4 > 4096 {
            (&self.mmv4, 4usize)
        } else {
            (&self.mmv1, 2)
        };
        let groups = (w.nrows / num_rows) as u32;
        assert!(groups <= 65535, "workgroup overflow: {groups}");
        let mut push = MatVecPush::simple(w.ncols as u32, w.nrows as u32);
        push.stride_a = (w.ncols / Q8_K) as u32;
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

    /// Record rms_norm with fused weight multiply over `nrows` rows of `ncols`.
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
        push.param1 = self.hp.rms_eps;
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

    /// Record dst += src (element count n).
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

    /// Feed one token at position = current cache length. Returns logits.
    pub fn forward_token(&mut self, token: u32) -> Vec<f32> {
        let hp = &self.hp;
        let (n_embd, n_heads, n_kv, hd) = (hp.n_embd, hp.n_heads, hp.n_kv_heads, hp.head_dim);
        let (n_ff, n_ctx) = (hp.n_ff, self.n_ctx_max);
        let kv_dim = n_kv * hd;
        let gqa = n_heads / n_kv;
        let pos = self.n_past;
        assert!(pos < n_ctx, "KV arena full ({n_ctx})");
        let n_t = pos + 1;
        let scale = 1.0 / (hd as f32).sqrt();

        // Host->GPU: embedding row + rope position.
        let mut x = vec![0f32; n_embd];
        dequant_q8_0_row(&self.embd_raw, token as usize, n_embd, &mut x);
        self.gpu.upload(&self.bx, as_bytes(&x)).unwrap();
        self.gpu
            .upload(&self.bpos, as_bytes(&[pos as i32]))
            .unwrap();

        let mut batch = self.batch.take().unwrap();
        batch.begin(&self.gpu).unwrap();

        let dbg = std::env::var("MODEL_DEBUG").is_ok() && pos == 0;
        // In debug mode, flush the batch and print a buffer sum after a stage.
        macro_rules! probe {
            ($b:expr, $n:expr, $name:expr) => {
                if dbg {
                    batch.submit(&self.gpu).unwrap();
                    let mut v = vec![0f32; $n];
                    self.gpu
                        .download($b, unsafe {
                            std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, $n * 4)
                        })
                        .unwrap();
                    eprintln!("{} sum = {:.6}", $name, v.iter().sum::<f32>());
                    batch.begin(&self.gpu).unwrap();
                }
            };
        }

        for l in 0..self.hp.n_layers {
            let ly = &self.layers[l];
            // --- attention ---
            self.rec_rms(
                &mut batch,
                &self.bx,
                &ly.attn_norm,
                &self.bnorm,
                n_embd as u32,
                1,
            );
            if l == 0 {
                probe!(&self.bnorm, n_embd, "attn_norm-0");
            }
            self.rec_matvec_b(&mut batch, &ly.wq, &self.bnorm, &self.bq, false);
            self.rec_matvec_b(&mut batch, &ly.wk, &self.bnorm, &self.bk, false);
            self.rec_matvec(&mut batch, &ly.wv, &self.bnorm, &self.bv);
            if l == 0 {
                probe!(&self.bq, n_heads * hd, "Qcur-0");
                probe!(&self.bk, kv_dim, "Kcur-0");
                probe!(&self.bv, kv_dim, "Vcur-0");
            }
            // per-head q/k rms norm (qwen3), in place
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
            // rope q and k in place
            for (buf, nh) in [(&self.bq, n_heads), (&self.bk, n_kv)] {
                let push = RopePush::neox(hd as u32, nh as u32, 1, self.hp.rope_base);
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
            if l == 0 {
                probe!(&self.bq, n_heads * hd, "Q(norm+rope)-0");
                probe!(&self.bk, kv_dim, "K(norm+rope)-0");
            }
            // write k row into kcache at byte offset pos*kv_dim*4 (64B-aligned)
            let cp = UnaryPush::strided_copy(kv_dim as u32, 1, 0);
            batch
                .dispatch_ranges(
                    &self.gpu,
                    &self.p_cpy,
                    &[
                        (&self.bk, 0, WHOLE),
                        (&ly.kcache, (pos * kv_dim * 4) as u64, WHOLE),
                    ],
                    cp.as_bytes(),
                    ((kv_dim as u32).div_ceil(512), 1, 1),
                )
                .unwrap();
            // copy v row into vtcache (row-major, like kcache)
            let cpv = UnaryPush::contig_copy(kv_dim as u32);
            batch
                .dispatch_ranges(
                    &self.gpu,
                    &self.p_cpy,
                    &[
                        (&self.bv, 0, WHOLE),
                        (&ly.vtcache, (pos * kv_dim * 4) as u64, WHOLE),
                    ],
                    cpv.as_bytes(),
                    ((kv_dim as u32).div_ceil(512), 1, 1),
                )
                .unwrap();
            // --- fused decode attention: one dispatch, one workgroup per head ---
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
            self.rec_matvec(&mut batch, &ly.wo, &self.battn, &self.bproj);
            if l == 0 {
                probe!(&self.battn, n_heads * hd, "attn_out-0");
                probe!(&self.bproj, n_embd, "attn_proj-0");
                probe!(&self.bx, n_embd, "x_before_add-0");
            }
            self.rec_add(&mut batch, &self.bx, &self.bproj, n_embd as u32);
            if l == 0 {
                probe!(&self.bx, n_embd, "ffn_inp-0");
            }
            // --- ffn ---
            self.rec_rms(
                &mut batch,
                &self.bx,
                &ly.ffn_norm,
                &self.bnorm,
                n_embd as u32,
                1,
            );
            self.rec_matvec_b(&mut batch, &ly.w_gate, &self.bnorm, &self.bgate, false);
            self.rec_matvec(&mut batch, &ly.w_up, &self.bnorm, &self.bup);
            if l == 0 {
                probe!(&self.bnorm, n_embd, "ffn_norm-0");
                probe!(&self.bgate, n_ff, "ffn_gate-0");
                probe!(&self.bup, n_ff, "ffn_up-0");
            }
            let glu = GluPush::split(n_ff as u32, 1);
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
                    ((n_ff as u32).div_ceil(512), 1, 1),
                )
                .unwrap();
            self.rec_matvec(&mut batch, &ly.w_down, &self.bgate, &self.bproj);
            if l == 0 {
                probe!(&self.bgate, n_ff, "ffn_swiglu-0");
                probe!(&self.bproj, n_embd, "ffn_out-0");
            }
            self.rec_add(&mut batch, &self.bx, &self.bproj, n_embd as u32);
            probe!(&self.bx, n_embd, format!("l_out-{l}"));
        }

        // final norm + tied lm_head
        self.rec_rms(
            &mut batch,
            &self.bx,
            &self.output_norm,
            &self.bnorm,
            n_embd as u32,
            1,
        );
        self.rec_matvec(&mut batch, &self.embd_gpu, &self.bnorm, &self.blogits);

        batch.submit(&self.gpu).unwrap();
        self.batch = Some(batch);
        self.n_past += 1;

        if std::env::var("MODEL_DEBUG").is_ok() && pos == 0 {
            let dump = |b: &Buffer, n: usize, name: &str| {
                let mut v = vec![0f32; n];
                self.gpu
                    .download(b, unsafe {
                        std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4)
                    })
                    .unwrap();
                eprintln!("{name} sum = {:.6}", v.iter().sum::<f32>());
            };
            dump(&self.bx, n_embd, "x(final)");
            dump(&self.bnorm, n_embd, "norm(final)");
            dump(&self.bq, n_heads * hd, "q(last layer)");
            dump(&self.bk, kv_dim, "k(last layer)");
            dump(&self.battn, n_heads * hd, "attn(last layer)");
        }

        let mut logits = vec![0f32; self.hp.n_vocab];
        self.gpu
            .download(&self.blogits, unsafe {
                std::slice::from_raw_parts_mut(logits.as_mut_ptr() as *mut u8, logits.len() * 4)
            })
            .unwrap();
        logits
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if let Some(b) = self.batch.take() {
            self.gpu.destroy_batch(b);
        }
    }
}
