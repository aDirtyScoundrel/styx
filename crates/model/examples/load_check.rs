//! M9a load check: load a gguf through MoeModel::load and report the
//! architecture layout without generating. Usage: load_check <gguf>
//! Verifies hybrid layer typing, ssm tensor loading, and recurrent-state
//! allocation for qwen3next; prints a per-layer-kind summary.
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("usage: load_check <gguf>");
    let t = Instant::now();
    let m = model::MoeModel::load(std::path::Path::new(&path)).unwrap();
    let s = m.hybrid_summary();
    println!("loaded in {:.2}s", t.elapsed().as_secs_f64());
    println!(
        "arch={} layers={} experts={} top-{} hybrid={} interval={}",
        s.arch, s.n_layers, s.n_expert, s.n_expert_used, s.hybrid, s.full_attn_interval
    );
    println!(
        "layer kinds: {} full-attention, {} recurrent (linear)",
        s.n_attn, s.n_recr
    );
    if s.hybrid {
        println!(
            "ssm: conv_dim={} value_dim={} n_k={} n_v={} head_v={}",
            s.conv_dim, s.value_dim, s.n_k_heads, s.n_v_heads, s.head_v_dim
        );
        println!(
            "recurrent state per layer: conv_state={} B, ssm_state={} B",
            s.conv_state_bytes, s.ssm_state_bytes
        );
        println!("full-attn layers: {:?}", s.attn_layers);
    }
    println!("M9a LOAD CHECK: OK");
}
