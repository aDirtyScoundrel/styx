//! M2 greedy generation loop with tok/s timing.
//! Usage: generate <gguf> <n_predict> <tok0> [tok1 ...]

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: generate <gguf> <n_predict> <tok...>");
    let n_predict: usize = args.next().unwrap().parse().unwrap();
    let prompt: Vec<u32> = args.map(|a| a.parse().unwrap()).collect();
    assert!(!prompt.is_empty());

    let t_load = Instant::now();
    let mut m = model::Model::load(std::path::Path::new(&path)).unwrap();
    eprintln!(
        "loaded in {:.2}s: {} layers, n_ctx_max {}",
        t_load.elapsed().as_secs_f64(),
        m.hp.n_layers,
        m.n_ctx_max
    );

    let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap() as u32;

    let t_prefill = Instant::now();
    let mut logits = Vec::new();
    for &t in &prompt {
        logits = m.forward_token(t);
    }
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    let t_decode = Instant::now();
    let mut out = Vec::with_capacity(n_predict);
    let mut next = argmax(&logits);
    out.push(next);
    for _ in 1..n_predict {
        logits = m.forward_token(next);
        next = argmax(&logits);
        out.push(next);
    }
    let decode_s = t_decode.elapsed().as_secs_f64();

    println!(
        "{}",
        out.iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    eprintln!(
        "prefill: {} tok in {:.3}s ({:.2} tok/s) | decode: {} tok in {:.3}s ({:.2} tok/s)",
        prompt.len(),
        prefill_s,
        prompt.len() as f64 / prefill_s,
        out.len(),
        decode_s,
        out.len() as f64 / decode_s
    );
}
