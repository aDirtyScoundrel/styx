//! Per-(layer, expert) routing histogram over a greedy decode run.
//! Usage: moe_route_hist <gguf> <n_predict> <tok0> [tok1 ...]
//! Emits CSV to stdout: layer,expert,hits  (plus summary to stderr).

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: moe_route_hist <gguf> <n> <tok...>");
    let n_predict: usize = args.next().unwrap().parse().unwrap();
    let prompt: Vec<u32> = args.map(|a| a.parse().unwrap()).collect();
    assert!(!prompt.is_empty());

    let mut m = model::MoeModel::load(std::path::Path::new(&path)).unwrap();
    let n_layers = m.hp.base.n_layers;
    let n_expert = m.hp.n_expert;
    let mut hist = vec![vec![0u64; n_expert]; n_layers];
    let mut record = |m: &model::MoeModel, hist: &mut Vec<Vec<u64>>| {
        for (l, ids) in m.last_expert_ids().iter().enumerate() {
            for &e in ids {
                hist[l][e as usize] += 1;
            }
        }
    };

    let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap() as u32;
    let mut logits = Vec::new();
    for &t in &prompt {
        logits = m.forward_token(t);
        record(&m, &mut hist);
    }
    let mut next = argmax(&logits);
    for _ in 1..n_predict {
        logits = m.forward_token(next);
        record(&m, &mut hist);
        next = argmax(&logits);
    }

    println!("layer,expert,hits");
    for l in 0..n_layers {
        for e in 0..n_expert {
            println!("{l},{e},{}", hist[l][e]);
        }
    }
    // skew summary: what fraction of hits do the top-25% experts carry per layer?
    let mut fr_sum = 0.0;
    for l in 0..n_layers {
        let mut v = hist[l].clone();
        v.sort_unstable_by(|a, b| b.cmp(a));
        let total: u64 = v.iter().sum();
        let top: u64 = v[..n_expert / 4].iter().sum();
        fr_sum += top as f64 / total.max(1) as f64;
    }
    eprintln!(
        "avg fraction of hits carried by top-25% experts: {:.3} (uniform would be 0.250)",
        fr_sum / n_layers as f64
    );
}
