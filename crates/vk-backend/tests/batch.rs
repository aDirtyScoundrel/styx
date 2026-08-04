//! Batch recorder validation: many chained dispatches, one submit/fence.
//! Gated behind MOE_GPU_TESTS=1.

use vk_backend::ops::BinaryPush;
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

/// Chain 64 adds: acc += a, alternating destination buffers so every
/// dispatch reads the previous dispatch's output (exercises the barrier).
#[test]
fn batch_chained_adds_match_cpu() {
    if std::env::var("MOE_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("set MOE_GPU_TESTS=1 to run GPU tests");
        return;
    }
    let gpu = Gpu::new().expect("vulkan init");

    let n0 = 1024u32;
    let n = n0 as usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.25 + 0.5).collect();
    let zero = vec![0f32; n];

    let ba = gpu.create_buffer((n * 4) as u64, true).unwrap();
    let b0 = gpu.create_buffer((n * 4) as u64, true).unwrap();
    let b1 = gpu.create_buffer((n * 4) as u64, true).unwrap();
    let bp = gpu.create_buffer(4, true).unwrap();
    gpu.upload(&ba, as_bytes(&a)).unwrap();
    gpu.upload(&b0, as_bytes(&zero)).unwrap();

    let spv_path: std::path::PathBuf = format!("{SPV_DIR}/add_f32_f32_f32.spv").into();
    let pipe = gpu
        .create_pipeline(
            &spv_path,
            4,
            std::mem::size_of::<BinaryPush>() as u32,
            &[(0, 1)],
        )
        .unwrap();

    let push = BinaryPush::contig2d((n0, 1), (n0, 1), (n0, 1));
    let iters = 64usize;
    let mut batch = gpu
        .create_batch(iters as u32 + 8, 4 * (iters as u32 + 8))
        .unwrap();

    let t0 = std::time::Instant::now();
    batch.begin(&gpu).unwrap();
    let bufs = [&b0, &b1];
    for i in 0..iters {
        let src = bufs[i % 2];
        let dst = bufs[(i + 1) % 2];
        batch
            .dispatch(
                &gpu,
                &pipe,
                &[src, &ba, dst, &bp],
                push.as_bytes(),
                ((n as u32).div_ceil(256), 1, 1),
            )
            .unwrap();
    }
    batch.submit(&gpu).unwrap();
    let dt = t0.elapsed();
    eprintln!(
        "batch: {iters} dispatches in {dt:?} ({:.1} us/dispatch)",
        dt.as_secs_f64() * 1e6 / iters as f64
    );

    let final_buf = bufs[iters % 2];
    let mut got = vec![0f32; n];
    gpu.download(final_buf, as_bytes_mut(&mut got)).unwrap();
    let expect: Vec<f32> = a.iter().map(|v| v * iters as f32).collect();
    let max_err = got
        .iter()
        .zip(&expect)
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max);
    eprintln!("chained add max abs err: {max_err:e}");
    assert!(max_err < 1e-3, "chained add mismatch: {max_err}");

    // Batch must be reusable: second graph after submit.
    batch.begin(&gpu).unwrap();
    batch
        .dispatch(
            &gpu,
            &pipe,
            &[final_buf, &ba, bufs[(iters + 1) % 2], &bp],
            push.as_bytes(),
            ((n as u32).div_ceil(256), 1, 1),
        )
        .unwrap();
    batch.submit(&gpu).unwrap();
    let mut got2 = vec![0f32; n];
    gpu.download(bufs[(iters + 1) % 2], as_bytes_mut(&mut got2))
        .unwrap();
    let max_err2 = got2
        .iter()
        .zip(a.iter().map(|v| v * (iters + 1) as f32))
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max);
    assert!(max_err2 < 1e-3, "batch reuse mismatch: {max_err2}");

    gpu.destroy_batch(batch);
    gpu.destroy_pipeline(pipe);
    for b in [ba, b0, b1, bp] {
        gpu.destroy_buffer(b);
    }
}
