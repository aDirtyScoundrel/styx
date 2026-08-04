//! Routing autocorrelation probe: how much does token t+1 reuse the experts
//! routed at token t (per layer)? High overlap => previous-token prefetch of
//! cold slabs can work; low overlap => only async overlap helps.
//! Usage: route_autocorr <gguf> <n_predict> <tok...>
fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: route_autocorr <gguf> <n> <tok..>");
    let n_predict: usize = args.next().unwrap().parse().unwrap();
    let prompt: Vec<u32> = args.map(|a| a.parse().unwrap()).collect();
    let mut m = model::MoeModel::load(std::path::Path::new(&path)).unwrap();
    let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap() as u32;

    let n_layers = m.hp.base.n_layers;
    let k = m.hp.n_expert_used;
    let mut prev: Vec<Vec<u32>> = Vec::new();
    let mut overlap = vec![0u64; n_layers]; // ids shared with previous token
    let mut steps = 0u64;

    let mut logits = Vec::new();
    for &t in &prompt {
        logits = m.forward_token(t);
    }
    let mut next = argmax(&logits);
    for _ in 0..n_predict {
        logits = m.forward_token(next);
        let ids = m.last_expert_ids();
        if !prev.is_empty() {
            for l in 0..n_layers {
                overlap[l] += ids[l].iter().filter(|e| prev[l].contains(e)).count() as u64;
            }
            steps += 1;
        }
        prev = ids;
        next = argmax(&logits);
    }
    let per: Vec<f64> = overlap
        .iter()
        .map(|&o| o as f64 / (steps * k as u64) as f64)
        .collect();
    let mean = per.iter().sum::<f64>() / n_layers as f64;
    println!(
        "token-to-token expert overlap: mean {:.3} (min {:.3}, max {:.3}) over {} steps, k={}",
        mean,
        per.iter().cloned().fold(f64::MAX, f64::min),
        per.iter().cloned().fold(0.0, f64::max),
        steps,
        k
    );
    // horizon-4: overlap with union of previous 4 tokens' experts would need
    // more state; single-step is the decisive number for next-token prefetch.
}
