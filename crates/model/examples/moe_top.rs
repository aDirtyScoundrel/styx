//! Print top-5 logits at each step for MoE model (divergence diagnosis).
//! Usage: moe_top <gguf> <n_steps> <tok...>
fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap();
    let n_steps: usize = args.next().unwrap().parse().unwrap();
    let prompt: Vec<u32> = args.map(|a| a.parse().unwrap()).collect();
    let mut m = model::MoeModel::load(std::path::Path::new(&path)).unwrap();
    let mut logits = Vec::new();
    for &t in &prompt {
        logits = m.forward_token(t);
    }
    let mut next = 0u32;
    for step in 0..n_steps {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let top: Vec<String> = idx[..5]
            .iter()
            .map(|&i| format!("{i}:{:.4}", logits[i]))
            .collect();
        next = idx[0] as u32;
        println!("step {step}: {}", top.join(" "));
        logits = m.forward_token(next);
    }
    let _ = next;
}
