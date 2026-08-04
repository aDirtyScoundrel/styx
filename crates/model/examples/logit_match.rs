//! M1 gate: logit-match vs llama.cpp reference dump.
//! Usage: logit_match <gguf> <ref.bin> <tok...>

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap();
    let refbin = args.next().unwrap();
    let toks: Vec<u32> = args.map(|a| a.parse().unwrap()).collect();

    let raw = std::fs::read(&refbin).unwrap();
    let reference: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let mut m = model::Model::load(std::path::Path::new(&path)).unwrap();
    assert_eq!(reference.len(), m.hp.n_vocab);
    let mut logits = Vec::new();
    for &t in &toks {
        logits = m.forward_token(t);
    }

    let mut max_abs = 0f32;
    let mut worst = 0usize;
    for i in 0..logits.len() {
        let d = (logits[i] - reference[i]).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
    }
    let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
    let (am_ours, am_ref) = (argmax(&logits), argmax(&reference));

    // top-10 overlap
    let topk = |v: &[f32]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
        idx[..10].to_vec()
    };
    let (ta, tb) = (topk(&logits), topk(&reference));
    let overlap = ta.iter().filter(|i| tb.contains(i)).count();

    println!(
        "max_abs_diff = {max_abs:.6} at token {worst} (ours {:.4}, ref {:.4})",
        logits[worst], reference[worst]
    );
    println!(
        "argmax ours = {am_ours}, ref = {am_ref}, match = {}",
        am_ours == am_ref
    );
    println!("top10 overlap = {overlap}/10");
    println!("ours  top5: {:?}", &ta[..5]);
    println!("ref   top5: {:?}", &tb[..5]);
    assert_eq!(am_ours, am_ref, "argmax mismatch");
    assert!(max_abs < 0.35, "logit divergence too large: {max_abs}");
    println!("LOGIT MATCH: PASS");
}
