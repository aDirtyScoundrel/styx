//! GPU-vs-CPU validation for the two M1 kernels, using the vendored
//! ggml-vulkan SPIR-V. Requires a Vulkan device; tests are ignored unless
//! MOE_GPU_TESTS=1.

use vk_backend::Gpu;

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

// ---------------- rms_norm_f32 ----------------
// generic_binary_head push constants: ne, ne00..nb03, ne10..nb13,
// ne20..nb23 (25 u32), misalign_offsets, param1 f32, param2 f32, param3 i32.

#[repr(C)]
#[derive(Clone, Copy)]
struct RmsPush {
    ne: u32,
    a: [u32; 8], // ne00..ne03, nb00..nb03 (element strides)
    b: [u32; 8],
    d: [u32; 8],
    misalign: u32,
    param1: f32,
    param2: f32,
    param3: i32,
}

#[test]
fn rms_norm_f32_matches_cpu() {
    let Some(gpu) = gpu_or_skip() else { return };
    eprintln!("device: {}", gpu.device_name);

    let ncols = 2048usize; // one Qwen3 hidden row
    let nrows = 4usize;
    let eps = 1e-6f32;

    let mut x = vec![0f32; ncols * nrows];
    let mut seed = 0x12345678u32;
    for v in x.iter_mut() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
    }
    let w = vec![1.0f32; ncols]; // unused (do_multiply=false) but bound

    // CPU reference
    let mut expect = vec![0f32; ncols * nrows];
    for r in 0..nrows {
        let row = &x[r * ncols..(r + 1) * ncols];
        let mean = row.iter().map(|v| v * v).sum::<f32>() / ncols as f32;
        let scale = 1.0 / (mean + eps).sqrt();
        for c in 0..ncols {
            expect[r * ncols + c] = row[c] * scale;
        }
    }

    let ba = gpu.create_buffer((x.len() * 4) as u64, true).unwrap();
    let bb = gpu.create_buffer((w.len() * 4) as u64, true).unwrap();
    let bd = gpu.create_buffer((x.len() * 4) as u64, true).unwrap();
    gpu.upload(&ba, as_bytes(&x)).unwrap();
    gpu.upload(&bb, as_bytes(&w)).unwrap();

    let dims = |n0: u32, n1: u32| [n0, n1, 1, 1, 1, n0, n0 * n1, n0 * n1];
    let push = RmsPush {
        ne: (ncols * nrows) as u32,
        a: dims(ncols as u32, nrows as u32),
        b: dims(ncols as u32, 1),
        d: dims(ncols as u32, nrows as u32),
        misalign: 0,
        param1: eps,
        param2: 0.0,
        param3: 0,
    };

    let pipe = gpu
        .create_pipeline(
            format!("{SPV_DIR}/rms_norm_f32.spv").as_ref(),
            3,
            std::mem::size_of::<RmsPush>() as u32,
            &[(1, 0)], // do_multiply = false
        )
        .unwrap();

    gpu.dispatch_sync(
        &pipe,
        &[&ba, &bb, &bd],
        as_bytes(&[push]),
        (nrows as u32, 1, 1),
    )
    .unwrap();

    let mut got = vec![0f32; ncols * nrows];
    gpu.download(&bd, unsafe {
        std::slice::from_raw_parts_mut(got.as_mut_ptr() as *mut u8, got.len() * 4)
    })
    .unwrap();

    let max_err = got
        .iter()
        .zip(&expect)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("rms_norm max abs err: {max_err:e}");
    assert!(max_err < 1e-5, "rms_norm mismatch: {max_err}");

    gpu.destroy_pipeline(pipe);
    gpu.destroy_buffer(ba);
    gpu.destroy_buffer(bb);
    gpu.destroy_buffer(bd);
}

// ---------------- mul_mat_vec_id_q4_k_f32_f32 (expert indirection) ----------------
// MUL_MAT_ID push constants (mul_mat_vec_base.glsl): ncols, stride_a,
// stride_b, stride_d, batch strides, fusion_flags, then nei0, ne11,
// expert_i1, nbi1. Binding 5 = ids buffer (i32). Offsets per get_offsets():
//   expert_id = data_ids[WorkGroupID.y + expert_i1*nbi1]
//   a_offset  = expert_id * (batch_stride_a / QUANT_K)   [in blocks]
//   b_offset  = (WorkGroupID.y % ne11) * stride_b + expert_i1 * batch_stride_b
//   d_offset  = WorkGroupID.y * stride_d + expert_i1 * batch_stride_d

#[repr(C)]
#[derive(Clone, Copy)]
struct MmvIdPush {
    ncols: u32,
    stride_a: u32,
    stride_b: u32,
    stride_d: u32,
    batch_stride_a: u32,
    batch_stride_b: u32,
    batch_stride_d: u32,
    fusion_flags: u32,
    nei0: u32,
    ne11: u32,
    expert_i1: u32,
    nbi1: u32,
}

#[test]
fn mul_mat_vec_id_q4_k_routes_experts_correctly() {
    let Some(gpu) = gpu_or_skip() else { return };

    let ncols = 512usize;
    let nrows = 32usize; // expert FFN rows
    let n_experts = 8usize;
    let top_k = 4usize; // experts selected for this token
    let blocks_per_row = ncols / QK_K;
    let expert_bytes = nrows * blocks_per_row * Q4K_BLOCK_BYTES;

    // n_experts stacked q4_K matrices with distinct random content.
    let mut a = vec![0u8; n_experts * expert_bytes];
    let mut seed = 0xcafef00du32;
    for v in a.iter_mut() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (seed >> 16) as u8;
    }
    for blk in a.chunks_exact_mut(Q4K_BLOCK_BYTES) {
        blk[1] = (blk[1] & 0x83) | 0x30;
        blk[3] = (blk[3] & 0x83) | 0x30;
    }

    let mut x = vec![0f32; ncols];
    for v in x.iter_mut() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
    }

    // Non-trivial routing: out-of-order, non-contiguous expert ids.
    let ids: [i32; 4] = [6, 1, 3, 0];
    assert_eq!(ids.len(), top_k);

    // CPU reference: for each selected expert, dequant its matrix, dot rows.
    let mut expect = vec![0f32; top_k * nrows];
    let mut row_f = vec![0f32; ncols];
    for (slot, &eid) in ids.iter().enumerate() {
        let ebase = eid as usize * expert_bytes;
        for r in 0..nrows {
            for b in 0..blocks_per_row {
                let off = ebase + (r * blocks_per_row + b) * Q4K_BLOCK_BYTES;
                dequant_q4k(
                    &a[off..off + Q4K_BLOCK_BYTES],
                    &mut row_f[b * QK_K..(b + 1) * QK_K],
                );
            }
            expect[slot * nrows + r] = row_f.iter().zip(&x).map(|(w, v)| w * v).sum();
        }
    }

    let ba = gpu.create_buffer(a.len() as u64, true).unwrap();
    let bb = gpu.create_buffer((x.len() * 4) as u64, true).unwrap();
    let bd = gpu.create_buffer((top_k * nrows * 4) as u64, true).unwrap();
    let bf0 = gpu.create_buffer(4, true).unwrap();
    let bf1 = gpu.create_buffer(4, true).unwrap();
    let bids = gpu.create_buffer((ids.len() * 4) as u64, true).unwrap();
    gpu.upload(&ba, &a).unwrap();
    gpu.upload(&bb, as_bytes(&x)).unwrap();
    gpu.upload(&bids, as_bytes(&ids)).unwrap();

    let push = MmvIdPush {
        ncols: ncols as u32,
        stride_a: (ncols / QK_K) as u32,
        stride_b: ncols as u32,
        stride_d: nrows as u32,
        batch_stride_a: (nrows * ncols) as u32, // elements per expert
        batch_stride_b: ncols as u32,
        batch_stride_d: (top_k * nrows) as u32,
        fusion_flags: 0,
        nei0: top_k as u32,
        ne11: 1, // one token: every selected expert reads the same x
        expert_i1: 0,
        nbi1: 1,
    };

    let pipe = gpu
        .create_pipeline(
            format!("{SPV_DIR}/mul_mat_vec_id_q4_k_f32_f32.spv").as_ref(),
            6,
            std::mem::size_of::<MmvIdPush>() as u32,
            &[(0, 32), (1, 1), (2, 1)],
        )
        .unwrap();

    // groups: x walks rows, y walks selected experts (expert_i0).
    gpu.dispatch_sync(
        &pipe,
        &[&ba, &bb, &bd, &bf0, &bf1, &bids],
        as_bytes(&[push]),
        (nrows as u32, top_k as u32, 1),
    )
    .unwrap();

    let mut got = vec![0f32; top_k * nrows];
    gpu.download(&bd, unsafe {
        std::slice::from_raw_parts_mut(got.as_mut_ptr() as *mut u8, got.len() * 4)
    })
    .unwrap();

    let mut max_rel = 0f32;
    for (g, e) in got.iter().zip(&expect) {
        max_rel = max_rel.max((g - e).abs() / e.abs().max(1e-3));
    }
    eprintln!("id matvec (experts {ids:?}) max rel err: {max_rel:e}");
    assert!(max_rel < 2e-2, "expert-routed matvec mismatch: {max_rel}");

    // Sanity: swapping the routing table must change the output, proving the
    // ids buffer is actually being read (not just expert 0 four times).
    let ids2: [i32; 4] = [0, 6, 1, 3];
    gpu.upload(&bids, as_bytes(&ids2)).unwrap();
    gpu.dispatch_sync(
        &pipe,
        &[&ba, &bb, &bd, &bf0, &bf1, &bids],
        as_bytes(&[push]),
        (nrows as u32, top_k as u32, 1),
    )
    .unwrap();
    let mut got2 = vec![0f32; top_k * nrows];
    gpu.download(&bd, unsafe {
        std::slice::from_raw_parts_mut(got2.as_mut_ptr() as *mut u8, got2.len() * 4)
    })
    .unwrap();
    // slot 0 now expert 0 -> must equal old slot 3; slot 1 now expert 6 -> old slot 0.
    let close = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() / x.abs().max(1e-3) < 1e-4)
    };
    assert!(close(&got2[..nrows], &got[3 * nrows..4 * nrows]));
    assert!(close(&got2[nrows..2 * nrows], &got[..nrows]));

    gpu.destroy_pipeline(pipe);
    for b in [ba, bb, bd, bf0, bf1, bids] {
        gpu.destroy_buffer(b);
    }
}

const QK_K: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144; // 2xf16 + 12 scales + 128 qs

fn f16_to_f32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

/// Scalar dequant of one q4_K block, ggml layout.
fn dequant_q4k(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), Q4K_BLOCK_BYTES);
    assert_eq!(out.len(), QK_K);
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
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

    // Blocks of 64 values: two 32-value halves share a qs byte (lo/hi nibble).
    let mut ql_off = 0;
    let mut o = 0;
    for j in 0..(QK_K / 64) {
        let (sc1, m1) = get_scale_min(2 * j);
        let (sc2, m2) = get_scale_min(2 * j + 1);
        let d1 = d * sc1;
        let d2 = d * sc2;
        let mm1 = dmin * m1;
        let mm2 = dmin * m2;
        for l in 0..32 {
            out[o + l] = d1 * (qs[ql_off + l] & 0x0F) as f32 - mm1;
        }
        for l in 0..32 {
            out[o + 32 + l] = d2 * (qs[ql_off + l] >> 4) as f32 - mm2;
        }
        o += 64;
        ql_off += 32;
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MmvPush {
    ncols: u32,
    stride_a: u32,
    stride_b: u32,
    stride_d: u32,
    batch_stride_a: u32,
    batch_stride_b: u32,
    batch_stride_d: u32,
    fusion_flags: u32,
    base_work_group_y: u32,
    ne02: u32,
    ne12: u32,
    broadcast2: u32,
    broadcast3: u32,
}

#[test]
fn mul_mat_vec_q4_k_matches_cpu_dequant() {
    let Some(gpu) = gpu_or_skip() else { return };

    let ncols = 512usize; // 2 superblocks per row
    let nrows = 64usize;
    let blocks_per_row = ncols / QK_K;

    // Random q4_K matrix bytes (any bit pattern is a valid block).
    let mut a = vec![0u8; nrows * blocks_per_row * Q4K_BLOCK_BYTES];
    let mut seed = 0xdeadbeefu32;
    for v in a.iter_mut() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (seed >> 16) as u8;
    }
    // Keep f16 d/dmin finite and small: overwrite exponents.
    for blk in a.chunks_exact_mut(Q4K_BLOCK_BYTES) {
        blk[1] = (blk[1] & 0x83) | 0x30; // d   ~ [2^-3, 2^-2) range-ish
        blk[3] = (blk[3] & 0x83) | 0x30; // dmin
    }

    let mut x = vec![0f32; ncols];
    for v in x.iter_mut() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
    }

    // CPU reference: dequant each row, dot with x.
    let mut expect = vec![0f32; nrows];
    let mut row_f = vec![0f32; ncols];
    for r in 0..nrows {
        for b in 0..blocks_per_row {
            let off = (r * blocks_per_row + b) * Q4K_BLOCK_BYTES;
            dequant_q4k(
                &a[off..off + Q4K_BLOCK_BYTES],
                &mut row_f[b * QK_K..(b + 1) * QK_K],
            );
        }
        expect[r] = row_f.iter().zip(&x).map(|(w, v)| w * v).sum();
    }

    let ba = gpu.create_buffer(a.len() as u64, true).unwrap();
    let bb = gpu.create_buffer((x.len() * 4) as u64, true).unwrap();
    let bd = gpu.create_buffer((nrows * 4) as u64, true).unwrap();
    // fuse buffers are bound but unused (fusion_flags = 0)
    let bf0 = gpu.create_buffer(4, true).unwrap();
    let bf1 = gpu.create_buffer(4, true).unwrap();
    gpu.upload(&ba, &a).unwrap();
    gpu.upload(&bb, as_bytes(&x)).unwrap();

    let push = MmvPush {
        ncols: ncols as u32,
        stride_a: (ncols / QK_K) as u32, // unused by q4_k path but set sanely
        stride_b: ncols as u32,
        stride_d: nrows as u32,
        batch_stride_a: (nrows * ncols) as u32,
        batch_stride_b: ncols as u32,
        batch_stride_d: nrows as u32,
        fusion_flags: 0,
        base_work_group_y: 0,
        ne02: 1,
        ne12: 1,
        broadcast2: 1,
        broadcast3: 1,
    };

    // spec: BLOCK_SIZE=32, NUM_ROWS=1, NUM_COLS=1 -> one workgroup per row
    let pipe = gpu
        .create_pipeline(
            format!("{SPV_DIR}/mul_mat_vec_q4_k_f32_f32.spv").as_ref(),
            5,
            std::mem::size_of::<MmvPush>() as u32,
            &[(0, 32), (1, 1), (2, 1)],
        )
        .unwrap();

    gpu.dispatch_sync(
        &pipe,
        &[&ba, &bb, &bd, &bf0, &bf1],
        as_bytes(&[push]),
        (nrows as u32, 1, 1),
    )
    .unwrap();

    let mut got = vec![0f32; nrows];
    gpu.download(&bd, unsafe {
        std::slice::from_raw_parts_mut(got.as_mut_ptr() as *mut u8, got.len() * 4)
    })
    .unwrap();

    let mut max_rel = 0f32;
    for (g, e) in got.iter().zip(&expect) {
        let rel = (g - e).abs() / e.abs().max(1e-3);
        max_rel = max_rel.max(rel);
    }
    eprintln!("q4_k matvec max rel err: {max_rel:e}");
    eprintln!("first outputs gpu={:?} cpu={:?}", &got[..4], &expect[..4]);
    assert!(max_rel < 2e-2, "q4_k matvec mismatch: {max_rel}");

    gpu.destroy_pipeline(pipe);
    for b in [ba, bb, bd, bf0, bf1] {
        gpu.destroy_buffer(b);
    }
}
