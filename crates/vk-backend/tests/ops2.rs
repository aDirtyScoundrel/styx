//! GPU-vs-CPU validation for the M1 op set beyond matvec/rms_norm:
//! get_rows_q8_0, add, swiglu, soft_max, rope_neox, mul_mat_vec_q8_0.
//! Gated behind MOE_GPU_TESTS=1.

use vk_backend::Gpu;
use vk_backend::ops::*;

const SPV_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../vendor/llama.cpp/build-shaders/ggml/src/ggml-vulkan/vulkan-shaders.spv"
);

fn gpu_or_skip() -> Option<Gpu> {
    if std::env::var("MOE_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("set MOE_GPU_TESTS=1 to run GPU tests");
        return None;
    }
    Some(Gpu::new().expect("vulkan init"))
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn as_bytes_mut<T: Copy>(v: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}

fn spv(name: &str) -> std::path::PathBuf {
    format!("{SPV_DIR}/{name}.spv").into()
}

// ---------------- add_f32_f32_f32 ----------------

#[test]
fn add_f32_matches_cpu() {
    let Some(gpu) = gpu_or_skip() else { return };
    let n0 = 2048u32;
    let n1 = 3u32;
    let n = (n0 * n1) as usize;
    let mut rng = Lcg(7);
    let a: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
    let expect: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let ba = gpu.create_buffer((n * 4) as u64, true).unwrap();
    let bb = gpu.create_buffer((n * 4) as u64, true).unwrap();
    let bd = gpu.create_buffer((n * 4) as u64, true).unwrap();
    let bp = gpu.create_buffer(4, true).unwrap(); // partial_sums, unused (param3=0)
    gpu.upload(&ba, as_bytes(&a)).unwrap();
    gpu.upload(&bb, as_bytes(&b)).unwrap();

    let push = BinaryPush::contig2d((n0, n1), (n0, n1), (n0, n1));
    // add.comp: 256 threads, 2 iters stepping +256 — group g covers
    // [256g, 256g+512) with idempotent overlap; ceil(n/256) groups covers all.
    let pipe = gpu
        .create_pipeline(
            &spv("add_f32_f32_f32"),
            4,
            std::mem::size_of::<BinaryPush>() as u32,
            &[(0, 1)], // norepeat = true (same shapes)
        )
        .unwrap();
    gpu.dispatch_sync(
        &pipe,
        &[&ba, &bb, &bd, &bp],
        push.as_bytes(),
        ((n as u32).div_ceil(256), 1, 1),
    )
    .unwrap();

    let mut got = vec![0f32; n];
    gpu.download(&bd, as_bytes_mut(&mut got)).unwrap();
    let max_err = got
        .iter()
        .zip(&expect)
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max);
    eprintln!("add max abs err: {max_err:e}");
    assert!(max_err == 0.0);

    gpu.destroy_pipeline(pipe);
    for x in [ba, bb, bd, bp] {
        gpu.destroy_buffer(x);
    }
}

// ---------------- get_rows_q8_0_f32 ----------------
// q8_0 block: f16 d + 32 x i8 = 34 bytes / 32 elements.

const Q8_BLOCK: usize = 34;
const Q8_K: usize = 32;

fn dequant_q8_0(block: &[u8], out: &mut [f32]) {
    let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    for (i, o) in out.iter_mut().enumerate() {
        *o = d * (block[2 + i] as i8) as f32;
    }
}

#[test]
fn get_rows_q8_0_matches_cpu() {
    let Some(gpu) = gpu_or_skip() else { return };
    let ncols = 128usize; // 4 q8_0 blocks per row
    let nrows_src = 64usize;
    let blocks_per_row = ncols / Q8_K;
    let ids: [i32; 5] = [3, 60, 0, 33, 3];

    let mut rng = Lcg(99);
    let mut a = vec![0u8; nrows_src * blocks_per_row * Q8_BLOCK];
    for v in a.iter_mut() {
        *v = (rng.next() >> 16) as u8;
    }
    for blk in a.chunks_exact_mut(Q8_BLOCK) {
        blk[1] = (blk[1] & 0x83) | 0x30; // keep d finite/small
    }

    let mut expect = vec![0f32; ids.len() * ncols];
    for (r, &id) in ids.iter().enumerate() {
        for b in 0..blocks_per_row {
            let off = (id as usize * blocks_per_row + b) * Q8_BLOCK;
            dequant_q8_0(
                &a[off..off + Q8_BLOCK],
                &mut expect[r * ncols + b * Q8_K..r * ncols + (b + 1) * Q8_K],
            );
        }
    }

    let ba = gpu.create_buffer(a.len() as u64, true).unwrap();
    let bb = gpu.create_buffer((ids.len() * 4) as u64, true).unwrap();
    let bd = gpu
        .create_buffer((ids.len() * ncols * 4) as u64, true)
        .unwrap();
    gpu.upload(&ba, &a).unwrap();
    gpu.upload(&bb, as_bytes(&ids)).unwrap();

    // generic_binary_head: src0 = quant matrix (ne00=ncols, nb01 in BLOCKS),
    // src1 = ids (ne10 = n ids), dst strides in elements.
    let push = BinaryPush {
        ne: (ids.len() * ncols) as u32,
        src0: [
            ncols as u32,
            nrows_src as u32,
            1,
            1,
            1,
            blocks_per_row as u32,
            (nrows_src * blocks_per_row) as u32,
            (nrows_src * blocks_per_row) as u32,
        ],
        src1: [
            ids.len() as u32,
            1,
            1,
            1,
            1,
            ids.len() as u32,
            ids.len() as u32,
            ids.len() as u32,
        ],
        dst: [
            ncols as u32,
            ids.len() as u32,
            1,
            1,
            1,
            ncols as u32,
            (ids.len() * ncols) as u32,
            (ids.len() * ncols) as u32,
        ],
        misalign_offsets: 0,
        param1: 0.0,
        param2: 0.0,
        param3: 0,
    };

    let pipe = gpu
        .create_pipeline(
            &spv("get_rows_q8_0_f32"),
            3,
            std::mem::size_of::<BinaryPush>() as u32,
            &[(0, 0)], // norepeat=false
        )
        .unwrap();
    // shader consumes 2 elements per x-thread; y walks ids, z walks batch.
    gpu.dispatch_sync(
        &pipe,
        &[&ba, &bb, &bd],
        push.as_bytes(),
        ((ncols as u32 / 2).div_ceil(512), ids.len() as u32, 1),
    )
    .unwrap();

    let mut got = vec![0f32; ids.len() * ncols];
    gpu.download(&bd, as_bytes_mut(&mut got)).unwrap();
    let max_err = got
        .iter()
        .zip(&expect)
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max);
    eprintln!("get_rows_q8_0 max abs err: {max_err:e}");
    assert!(max_err < 1e-6, "get_rows mismatch: {max_err}");
    // duplicate id must produce identical rows
    assert_eq!(&got[..ncols], &got[4 * ncols..5 * ncols]);

    gpu.destroy_pipeline(pipe);
    for x in [ba, bb, bd] {
        gpu.destroy_buffer(x);
    }
}

// ---------------- mul_mat_vec_q8_0_f32_f32 ----------------

#[test]
fn mul_mat_vec_q8_0_matches_cpu() {
    let Some(gpu) = gpu_or_skip() else { return };
    let ncols = 1024usize;
    let nrows = 128usize;
    let blocks_per_row = ncols / Q8_K;

    let mut rng = Lcg(0x51ce);
    let mut a = vec![0u8; nrows * blocks_per_row * Q8_BLOCK];
    for v in a.iter_mut() {
        *v = (rng.next() >> 16) as u8;
    }
    for blk in a.chunks_exact_mut(Q8_BLOCK) {
        blk[1] = (blk[1] & 0x83) | 0x30;
    }
    let x: Vec<f32> = (0..ncols).map(|_| rng.f32()).collect();

    let mut expect = vec![0f32; nrows];
    let mut row = vec![0f32; ncols];
    for r in 0..nrows {
        for b in 0..blocks_per_row {
            let off = (r * blocks_per_row + b) * Q8_BLOCK;
            dequant_q8_0(&a[off..off + Q8_BLOCK], &mut row[b * Q8_K..(b + 1) * Q8_K]);
        }
        expect[r] = row.iter().zip(&x).map(|(w, v)| w * v).sum();
    }

    let ba = gpu.create_buffer(a.len() as u64, true).unwrap();
    let bb = gpu.create_buffer((ncols * 4) as u64, true).unwrap();
    let bd = gpu.create_buffer((nrows * 4) as u64, true).unwrap();
    let bf0 = gpu.create_buffer(4, true).unwrap();
    let bf1 = gpu.create_buffer(4, true).unwrap();
    gpu.upload(&ba, &a).unwrap();
    gpu.upload(&bb, as_bytes(&x)).unwrap();

    let mut push = MatVecPush::simple(ncols as u32, nrows as u32);
    push.stride_a = blocks_per_row as u32; // quant paths use block strides
    let pipe = gpu
        .create_pipeline(
            &spv("mul_mat_vec_q8_0_f32_f32"),
            5,
            std::mem::size_of::<MatVecPush>() as u32,
            &[(0, 32), (1, 1), (2, 1)],
        )
        .unwrap();
    gpu.dispatch_sync(
        &pipe,
        &[&ba, &bb, &bd, &bf0, &bf1],
        push.as_bytes(),
        (nrows as u32, 1, 1),
    )
    .unwrap();

    let mut got = vec![0f32; nrows];
    gpu.download(&bd, as_bytes_mut(&mut got)).unwrap();
    let max_rel = got
        .iter()
        .zip(&expect)
        .map(|(g, e)| (g - e).abs() / e.abs().max(1e-3))
        .fold(0f32, f32::max);
    eprintln!("q8_0 matvec max rel err: {max_rel:e}");
    assert!(max_rel < 1e-3, "q8_0 matvec mismatch: {max_rel}");

    gpu.destroy_pipeline(pipe);
    for b in [ba, bb, bd, bf0, bf1] {
        gpu.destroy_buffer(b);
    }
}

// ---------------- swiglu_f32 (split mode) ----------------

#[test]
fn swiglu_split_matches_cpu() {
    let Some(gpu) = gpu_or_skip() else { return };
    let n0 = 3072u32;
    let n1 = 2u32;
    let n = (n0 * n1) as usize;
    let mut rng = Lcg(0x5119);
    let gate: Vec<f32> = (0..n).map(|_| rng.f32() * 4.0).collect();
    let up: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
    let expect: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(a, b)| a / (1.0 + (-a).exp()) * b)
        .collect();

    let ba = gpu.create_buffer((n * 4) as u64, true).unwrap();
    let bb = gpu.create_buffer((n * 4) as u64, true).unwrap();
    let bd = gpu.create_buffer((n * 4) as u64, true).unwrap();
    gpu.upload(&ba, as_bytes(&gate)).unwrap();
    gpu.upload(&bb, as_bytes(&up)).unwrap();

    let push = GluPush::split(n0, n1);
    let pipe = gpu
        .create_pipeline(
            &spv("swiglu_f32"),
            3,
            std::mem::size_of::<GluPush>() as u32,
            &[],
        )
        .unwrap();
    gpu.dispatch_sync(
        &pipe,
        &[&ba, &bb, &bd],
        push.as_bytes(),
        ((n as u32).div_ceil(512), 1, 1),
    )
    .unwrap();

    let mut got = vec![0f32; n];
    gpu.download(&bd, as_bytes_mut(&mut got)).unwrap();
    let max_err = got
        .iter()
        .zip(&expect)
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max);
    eprintln!("swiglu max abs err: {max_err:e}");
    assert!(max_err < 1e-5, "swiglu mismatch: {max_err}");

    gpu.destroy_pipeline(pipe);
    for x in [ba, bb, bd] {
        gpu.destroy_buffer(x);
    }
}

// ---------------- soft_max_f32 ----------------

#[test]
fn soft_max_matches_cpu() {
    let Some(gpu) = gpu_or_skip() else { return };
    let ncols = 700usize; // deliberately not a multiple of block size
    let nrows = 8usize;
    let scale = 0.125f32;
    let mut rng = Lcg(0x50f7);
    let x: Vec<f32> = (0..ncols * nrows).map(|_| rng.f32() * 10.0).collect();

    let mut expect = vec![0f32; ncols * nrows];
    for r in 0..nrows {
        let row = &x[r * ncols..(r + 1) * ncols];
        let m = row.iter().fold(f32::MIN, |a, &b| a.max(b * scale));
        let exps: Vec<f32> = row.iter().map(|v| (v * scale - m).exp()).collect();
        let s: f32 = exps.iter().sum();
        for c in 0..ncols {
            expect[r * ncols + c] = exps[c] / s;
        }
    }

    let ba = gpu.create_buffer((x.len() * 4) as u64, true).unwrap();
    let bb = gpu.create_buffer(4, true).unwrap(); // mask, unused (KY=0)
    let bc = gpu.create_buffer(4, true).unwrap(); // sinks, unused
    let bd = gpu.create_buffer((x.len() * 4) as u64, true).unwrap();
    gpu.upload(&ba, as_bytes(&x)).unwrap();

    let push = SoftMaxPush::rows(ncols as u32, nrows as u32, scale);
    let pipe = gpu
        .create_pipeline(
            &spv("soft_max_f32"),
            4,
            std::mem::size_of::<SoftMaxPush>() as u32,
            &[(0, 128)], // BLOCK_SIZE
        )
        .unwrap();
    gpu.dispatch_sync(
        &pipe,
        &[&ba, &bb, &bc, &bd],
        push.as_bytes(),
        (nrows as u32, 1, 1),
    )
    .unwrap();

    let mut got = vec![0f32; x.len()];
    gpu.download(&bd, as_bytes_mut(&mut got)).unwrap();
    let max_err = got
        .iter()
        .zip(&expect)
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max);
    eprintln!("soft_max max abs err: {max_err:e}");
    assert!(max_err < 1e-6, "soft_max mismatch: {max_err}");

    gpu.destroy_pipeline(pipe);
    for x in [ba, bb, bc, bd] {
        gpu.destroy_buffer(x);
    }
}

// ---------------- rope_neox_f32 ----------------

#[test]
fn rope_neox_matches_cpu() {
    let Some(gpu) = gpu_or_skip() else { return };
    let head_dim = 128usize;
    let n_heads = 4usize;
    let n_tokens = 3usize;
    let freq_base = 1_000_000f32; // qwen3 rope base
    let positions: [i32; 3] = [0, 5, 17];

    let n = head_dim * n_heads * n_tokens;
    let mut rng = Lcg(0x40e0);
    let x: Vec<f32> = (0..n).map(|_| rng.f32()).collect();

    // CPU reference (NEOX pairing: x[i], x[i + half])
    let mut expect = x.clone();
    let half_dim = head_dim / 2;
    let theta_scale = (freq_base as f64).powf(-2.0 / head_dim as f64);
    for t in 0..n_tokens {
        for h in 0..n_heads {
            let base = t * n_heads * head_dim + h * head_dim;
            for i in 0..half_dim {
                let theta = positions[t] as f64 * theta_scale.powi(i as i32);
                let (sin_t, cos_t) = theta.sin_cos();
                let x0 = x[base + i] as f64;
                let x1 = x[base + i + half_dim] as f64;
                expect[base + i] = (x0 * cos_t - x1 * sin_t) as f32;
                expect[base + i + half_dim] = (x0 * sin_t + x1 * cos_t) as f32;
            }
        }
    }

    let ba = gpu.create_buffer((n * 4) as u64, true).unwrap();
    let bpos = gpu.create_buffer((n_tokens * 4) as u64, true).unwrap();
    let bff = gpu.create_buffer(4, true).unwrap(); // freq factors, unused
    let bd = gpu.create_buffer((n * 4) as u64, true).unwrap();
    let bi = gpu.create_buffer(8, true).unwrap(); // set_rows indices, unused
    gpu.upload(&ba, as_bytes(&x)).unwrap();
    gpu.upload(&bpos, as_bytes(&positions)).unwrap();

    let push = RopePush::neox(head_dim as u32, n_heads as u32, n_tokens as u32, freq_base);
    let pipe = gpu
        .create_pipeline(
            &spv("rope_neox_f32"),
            5,
            std::mem::size_of::<RopePush>() as u32,
            &[],
        )
        .unwrap();
    // local size (1, 256, 1); x walks rows (head-rows), y walks dim/2.
    gpu.dispatch_sync(
        &pipe,
        &[&ba, &bpos, &bff, &bd, &bi],
        push.as_bytes(),
        (
            (n_heads * n_tokens) as u32,
            (half_dim as u32).div_ceil(256),
            1,
        ),
    )
    .unwrap();

    let mut got = vec![0f32; n];
    gpu.download(&bd, as_bytes_mut(&mut got)).unwrap();
    let max_err = got
        .iter()
        .zip(&expect)
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max);
    eprintln!("rope_neox max abs err: {max_err:e}");
    assert!(max_err < 1e-4, "rope mismatch: {max_err}");
    // position 0 must be identity
    assert_eq!(&got[..head_dim * n_heads], &x[..head_dim * n_heads]);

    gpu.destroy_pipeline(pipe);
    for b in [ba, bpos, bff, bd, bi] {
        gpu.destroy_buffer(b);
    }
}
