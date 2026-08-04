//! Run a few tokens through the model, print top logits.
//! Usage: logits <gguf> <tok0> [tok1 ...]

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: logits <gguf> <tok...>");
    let toks: Vec<u32> = args.map(|a| a.parse().unwrap()).collect();
    assert!(!toks.is_empty());

    let mut m = model::Model::load(std::path::Path::new(&path)).unwrap();
    eprintln!(
        "loaded: {} layers, n_embd {}, n_vocab {}",
        m.hp.n_layers, m.hp.n_embd, m.hp.n_vocab
    );
    let mut logits = Vec::new();
    for &t in &toks {
        logits = m.forward_token(t);
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    for &i in idx.iter().take(10) {
        println!("{i} {:.6}", logits[i]);
    }
}
