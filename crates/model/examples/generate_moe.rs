//! M5 greedy generation for Qwen3-MoE ggufs.
//! Usage: generate_moe <gguf> <n_predict> <tok0> [tok1 ...]
//! MOE_HUD=1 adds routing telemetry: hot/cold hit rates, estimated PCIe
//! traffic from cold expert reads, and a per-layer cold-hit heatmap.

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: generate_moe <gguf> <n_predict> <tok...>");
    let n_predict: usize = args.next().unwrap().parse().unwrap();
    let prompt: Vec<u32> = args.map(|a| a.parse().unwrap()).collect();
    assert!(!prompt.is_empty());
    let hud = std::env::var("MOE_HUD").is_ok();

    let t_load = Instant::now();
    let mut m = model::MoeModel::load(std::path::Path::new(&path)).unwrap();
    eprintln!(
        "loaded in {:.2}s: {} layers, {} experts (top-{}), n_ctx_max {}",
        t_load.elapsed().as_secs_f64(),
        m.hp.base.n_layers,
        m.hp.n_expert,
        m.hp.n_expert_used,
        m.n_ctx_max
    );

    let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap() as u32;

    let n_layers = m.hp.base.n_layers;
    let mut hot_total = 0u64;
    let mut cold_total = 0u64;
    let mut cold_by_layer = vec![0u64; n_layers];
    let hud_tick = |m: &model::MoeModel,
                    hot_total: &mut u64,
                    cold_total: &mut u64,
                    cold_by_layer: &mut [u64]| {
        for (l, (h, c)) in m.last_hot_cold().iter().enumerate() {
            *hot_total += *h as u64;
            *cold_total += *c as u64;
            cold_by_layer[l] += *c as u64;
        }
    };

    let t_prefill = Instant::now();
    let mut logits = Vec::new();
    for &t in &prompt {
        logits = m.forward_token(t);
        if hud {
            hud_tick(&m, &mut hot_total, &mut cold_total, &mut cold_by_layer);
        }
    }
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    let t_decode = Instant::now();
    let mut out = Vec::with_capacity(n_predict);
    let mut next = argmax(&logits);
    out.push(next);
    for _ in 1..n_predict {
        logits = m.forward_token(next);
        if hud {
            hud_tick(&m, &mut hot_total, &mut cold_total, &mut cold_by_layer);
        }
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

    if hud {
        let n_tok = (prompt.len() + out.len() - 1) as u64;
        let total = hot_total + cold_total;
        let streamed = cold_total * m.slab_bytes;
        eprintln!(
            "hud: hot {:.1}% ({hot_total}) | cold {:.1}% ({cold_total}) | cold PCIe ~{:.2} GiB total, ~{:.1} MiB/tok",
            100.0 * hot_total as f64 / total.max(1) as f64,
            100.0 * cold_total as f64 / total.max(1) as f64,
            streamed as f64 / (1u64 << 30) as f64,
            streamed as f64 / n_tok.max(1) as f64 / (1u64 << 20) as f64,
        );
        // per-layer cold heatmap: one glyph per layer, . = all hot, 9 = all cold
        let k = m.hp.n_expert_used as u64;
        let glyphs: String = cold_by_layer
            .iter()
            .map(|&c| {
                let f = c as f64 / (n_tok * k).max(1) as f64;
                if f == 0.0 {
                    '.'
                } else {
                    char::from_digit(((f * 10.0) as u32).min(9), 10).unwrap()
                }
            })
            .collect();
        eprintln!("hud: cold/layer [{glyphs}] (.=0%, 9=90%+ of k={k} slots)");
        if m.repin_swaps > 0 {
            eprintln!(
                "hud: repin {} slab swaps (~{:.2} GiB copied)",
                m.repin_swaps,
                (m.repin_swaps * m.slab_bytes) as f64 / (1u64 << 30) as f64
            );
        }
    }
}
