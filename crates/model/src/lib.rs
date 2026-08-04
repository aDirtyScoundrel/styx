//! Minimal Qwen3-dense forward pass for M1 logit-match.
//!
//! GPU: every quantized (q8_0) matvec — attn q/k/v/o, ffn gate/up/down,
//! and the tied lm_head. CPU: rms norms, rope (neox), attention over the
//! KV cache, swiglu glue. f32 activations throughout.

use gguf_rs::Gguf;
use std::collections::HashMap;
use std::path::Path;
use vk_backend::ops::MatVecPush;
use vk_backend::{Buffer, Gpu, Pipeline};

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
    attn_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    wq: GpuMat,
    wk: GpuMat,
    wv: GpuMat,
    wo: GpuMat,
    w_gate: GpuMat,
    w_up: GpuMat,
    w_down: GpuMat,
}

pub struct Model {
    pub hp: HParams,
    gpu: Gpu,
    mmv1: Pipeline, // NUM_ROWS=1
    mmv4: Pipeline, // NUM_ROWS=4 (lm_head: n_vocab rows > 65535 workgroups)
    layers: Vec<Layer>,
    output_norm: Vec<f32>,
    embd_raw: Vec<u8>, // q8_0 token_embd, kept host-side for row gather
    embd_gpu: GpuMat,  // same tensor on GPU as tied lm_head
    // scratch buffers reused across dispatches
    sx: Buffer, // input vector (max n_ff)
    sy: Buffer, // output vector (max n_vocab)
    sf: Buffer, // dummy fuse
    // KV reserve arena: preallocated at load for n_ctx_max tokens.
    // [layer] -> flat [t * kv_dim ..] storage; n_past counts filled tokens.
    kcache: Vec<Vec<f32>>,
    vcache: Vec<Vec<f32>>,
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

fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (mean + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * s * w[i];
    }
}

fn rope_neox(x: &mut [f32], n_heads: usize, head_dim: usize, pos: i32, base: f32) {
    let half = head_dim / 2;
    let theta_scale = (base as f64).powf(-2.0 / head_dim as f64);
    for h in 0..n_heads {
        let o = h * head_dim;
        for i in 0..half {
            let theta = pos as f64 * theta_scale.powi(i as i32);
            let (s, c) = theta.sin_cos();
            let x0 = x[o + i] as f64;
            let x1 = x[o + i + half] as f64;
            x[o + i] = (x0 * c - x1 * s) as f32;
            x[o + i + half] = (x0 * s + x1 * c) as f32;
        }
    }
}

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
        let make_pipe = |num_rows: u32| {
            gpu.create_pipeline(
                Path::new(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../vendor/llama.cpp/build-shaders/ggml/src/ggml-vulkan",
                    "/vulkan-shaders.spv/mul_mat_vec_q8_0_f32_f32.spv"
                )),
                5,
                std::mem::size_of::<MatVecPush>() as u32,
                &[(0, 32), (1, num_rows), (2, 1)],
            )
        };
        let mmv1 = make_pipe(1)?;
        let mmv4 = make_pipe(4)?;

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

        let mut layers = Vec::new();
        for l in 0..hp.n_layers {
            let n = |s: &str| format!("blk.{l}.{s}.weight");
            layers.push(Layer {
                attn_norm: f32vec(&n("attn_norm")),
                q_norm: f32vec(&n("attn_q_norm")),
                k_norm: f32vec(&n("attn_k_norm")),
                ffn_norm: f32vec(&n("ffn_norm")),
                wq: upload_mat(&n("attn_q"))?,
                wk: upload_mat(&n("attn_k"))?,
                wv: upload_mat(&n("attn_v"))?,
                wo: upload_mat(&n("attn_output"))?,
                w_gate: upload_mat(&n("ffn_gate"))?,
                w_up: upload_mat(&n("ffn_up"))?,
                w_down: upload_mat(&n("ffn_down"))?,
            });
        }
        let embd_gpu = upload_mat("token_embd.weight")?;
        let output_norm = f32vec("output_norm.weight");
        let embd_raw = bytes("token_embd.weight").to_vec();

        let max_in = hp.n_ff.max(hp.n_embd) * 4;
        let max_out = hp.n_vocab.max(hp.n_ff) * 4;
        let sx = gpu.create_buffer(max_in as u64, true)?;
        let sy = gpu.create_buffer(max_out as u64, true)?;
        let sf = gpu.create_buffer(4, true)?;

        let n_layers = hp.n_layers;
        let n_ctx_max = 4096usize;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        Ok(Model {
            hp,
            gpu,
            mmv1,
            mmv4,
            layers,
            output_norm,
            embd_raw,
            embd_gpu,
            sx,
            sy,
            sf,
            kcache: vec![vec![0f32; n_ctx_max * kv_dim]; n_layers],
            vcache: vec![vec![0f32; n_ctx_max * kv_dim]; n_layers],
            n_ctx_max,
            n_past: 0,
        })
    }

    /// y = W · x on GPU (W q8_0, x/y f32).
    fn matvec(&self, w: &GpuMat, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), w.ncols);
        assert_eq!(y.len(), w.nrows);
        let (pipe, num_rows) = if w.nrows % 4 == 0 && w.nrows / 4 > 4096 {
            (&self.mmv4, 4)
        } else {
            (&self.mmv1, 1)
        };
        let groups = (w.nrows / num_rows) as u32;
        assert!(groups <= 65535, "workgroup overflow: {groups}");
        let mut push = MatVecPush::simple(w.ncols as u32, w.nrows as u32);
        push.stride_a = (w.ncols / Q8_K) as u32;
        self.gpu
            .upload(&self.sx, unsafe {
                std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4)
            })
            .unwrap();
        self.gpu
            .dispatch_sync(
                pipe,
                &[&w.buf, &self.sx, &self.sy, &self.sf, &self.sf],
                push.as_bytes(),
                (groups, 1, 1),
            )
            .unwrap();
        self.gpu
            .download(&self.sy, unsafe {
                std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut u8, y.len() * 4)
            })
            .unwrap();
    }

    pub fn reset(&mut self) {
        self.n_past = 0;
    }

    /// Feed one token at position = current cache length. Returns logits.
    pub fn forward_token(&mut self, token: u32) -> Vec<f32> {
        let hp = &self.hp;
        let (n_embd, n_heads, n_kv, hd) = (hp.n_embd, hp.n_heads, hp.n_kv_heads, hp.head_dim);
        let kv_dim = n_kv * hd;
        let q_dim = n_heads * hd;
        let gqa = n_heads / n_kv;
        let pos = self.n_past;
        assert!(pos < self.n_ctx_max, "KV arena full ({})", self.n_ctx_max);

        let mut x = vec![0f32; n_embd];
        dequant_q8_0_row(&self.embd_raw, token as usize, n_embd, &mut x);
        let dbg = std::env::var("MODEL_DEBUG").is_ok() && pos == 0;
        let sum = |v: &[f32]| v.iter().sum::<f32>();
        if dbg {
            eprintln!("embd sum = {:.6}", sum(&x));
        }

        let mut norm = vec![0f32; n_embd];
        let mut q = vec![0f32; q_dim];
        let mut k = vec![0f32; kv_dim];
        let mut v = vec![0f32; kv_dim];
        let mut attn_out = vec![0f32; q_dim];
        let mut proj = vec![0f32; n_embd];
        let mut gate = vec![0f32; self.hp.n_ff];
        let mut up = vec![0f32; self.hp.n_ff];

        for l in 0..self.hp.n_layers {
            let ly = &self.layers[l];
            // --- attention ---
            rms_norm(&x, &ly.attn_norm, self.hp.rms_eps, &mut norm);
            if dbg && l == 0 {
                eprintln!("attn_norm-0 sum = {:.6}", sum(&norm));
            }
            self.matvec(&ly.wq, &norm, &mut q);
            self.matvec(&ly.wk, &norm, &mut k);
            self.matvec(&ly.wv, &norm, &mut v);
            if dbg && l == 0 {
                eprintln!("Qcur-0 sum = {:.6}", sum(&q));
            }
            // per-head q/k rms norm (qwen3), then rope
            let mut tmp = vec![0f32; hd];
            for h in 0..n_heads {
                tmp.copy_from_slice(&q[h * hd..(h + 1) * hd]);
                rms_norm(
                    &tmp,
                    &ly.q_norm,
                    self.hp.rms_eps,
                    &mut q[h * hd..(h + 1) * hd],
                );
            }
            for h in 0..n_kv {
                tmp.copy_from_slice(&k[h * hd..(h + 1) * hd]);
                rms_norm(
                    &tmp,
                    &ly.k_norm,
                    self.hp.rms_eps,
                    &mut k[h * hd..(h + 1) * hd],
                );
            }
            rope_neox(&mut q, n_heads, hd, pos as i32, self.hp.rope_base);
            rope_neox(&mut k, n_kv, hd, pos as i32, self.hp.rope_base);
            if dbg && l == 0 {
                eprintln!("Qcur_normed+rope-0 sum = {:.6}", sum(&q));
            }
            self.kcache[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
            self.vcache[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);
            let n_t = pos + 1;
            let scale = 1.0 / (hd as f32).sqrt();
            for h in 0..n_heads {
                let kvh = h / gqa;
                let qh = &q[h * hd..(h + 1) * hd];
                let mut scores = vec![0f32; n_t];
                for t in 0..n_t {
                    let kt = &self.kcache[l][t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                    scores[t] = qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                let m = scores.iter().fold(f32::MIN, |a, &b| a.max(b));
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                let out = &mut attn_out[h * hd..(h + 1) * hd];
                out.fill(0.0);
                for t in 0..n_t {
                    let w = scores[t] / sum;
                    let vt = &self.vcache[l][t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                    for i in 0..hd {
                        out[i] += w * vt[i];
                    }
                }
            }
            self.matvec(&ly.wo, &attn_out, &mut proj);
            for i in 0..n_embd {
                x[i] += proj[i];
            }
            // --- ffn ---
            rms_norm(&x, &ly.ffn_norm, self.hp.rms_eps, &mut norm);
            self.matvec(&ly.w_gate, &norm, &mut gate);
            self.matvec(&ly.w_up, &norm, &mut up);
            for i in 0..self.hp.n_ff {
                gate[i] = gate[i] / (1.0 + (-gate[i]).exp()) * up[i];
            }
            self.matvec(&ly.w_down, &gate, &mut proj);
            for i in 0..n_embd {
                x[i] += proj[i];
            }
        }

        rms_norm(&x.clone(), &self.output_norm, self.hp.rms_eps, &mut x);
        self.n_past += 1;
        let mut logits = vec![0f32; self.hp.n_vocab];
        self.matvec(&self.embd_gpu, &x, &mut logits);
        if dbg {
            eprintln!(
                "result_output sum = {:.4}, first = {:.4} {:.4} {:.4}",
                sum(&logits),
                logits[0],
                logits[1],
                logits[2]
            );
        }
        logits
    }
}
