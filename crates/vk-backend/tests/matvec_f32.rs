//! mul_mat_vec_f32_f32 ABI validation (needed for GPU attention scores/values).
//! Gated behind MOE_GPU_TESTS=1.

use vk_backend::ops::MatVecPush;
use vk_backend::Gpu;

const SPV_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../vendor/llama.cpp/build-shaders/ggml/src/ggml-vulkan/vulkan-shaders.spv"
);

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn as_bytes_mut<T: Copy>(v: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

struct Lcg(u32);
impl Lcg {
    fn f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}

#[test]
fn mul_mat_vec_f32_matches_cpu() {
    if std::env::var("MOE_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("set MOE_GPU_TESTS=1 to run GPU tests");
        return;
    }
    let gpu = Gpu::new().expect("vulkan init");
    let ncols = 128usize; // head_dim-sized
    let nrows = 300usize; // n_past-sized (scores) — not power of two
    let mut rng = Lcg(0xf32f);
    let a: Vec<f32> = (0..nrows * ncols).map(|_| rng.f32()).collect();
    let x: Vec<f32> = (0..ncols).map(|_| rng.f32()).collect();
    let expect: Vec<f32> = (0..nrows)
        .map(|r| {
            a[r * ncols..(r + 1) * ncols]
                .iter()
                .zip(&x)
                .map(|(w, v)| w * v)
                .sum()
        })
        .collect();

    let ba = gpu.create_buffer((a.len() * 4) as u64, true).unwrap();
    let bb = gpu.create_buffer((ncols * 4) as u64, true).unwrap();
    let bd = gpu.create_buffer((nrows * 4) as u64, true).unwrap();
    let bf0 = gpu.create_buffer(4, true).unwrap();
    let bf1 = gpu.create_buffer(4, true).unwrap();
    gpu.upload(&ba, as_bytes(&a)).unwrap();
    gpu.upload(&bb, as_bytes(&x)).unwrap();

    // f32 path: stride_a stays in ELEMENTS (no quant blocks).
    let push = MatVecPush::simple(ncols as u32, nrows as u32);
    let spv: std::path::PathBuf = format!("{SPV_DIR}/mul_mat_vec_f32_f32_f32.spv").into();
    let pipe = gpu
        .create_pipeline(
            &spv,
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
    eprintln!("f32 matvec max rel err: {max_rel:e}");
    assert!(max_rel < 1e-4, "f32 matvec mismatch: {max_rel}");

    gpu.destroy_pipeline(pipe);
    for b in [ba, bb, bd, bf0, bf1] {
        gpu.destroy_buffer(b);
    }
}
