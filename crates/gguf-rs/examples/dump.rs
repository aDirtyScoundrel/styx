use gguf_rs::Gguf;
use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump <model.gguf>");
    let g = Gguf::open(Path::new(&path)).expect("parse failed");
    println!(
        "version {} tensors {} data_start {}",
        g.version,
        g.tensors.len(),
        g.data_start
    );
    let total: u64 = g.tensors.iter().map(|t| t.size_bytes).sum();
    println!("total tensor bytes {}", total);
    for t in g.tensors.iter().take(5) {
        println!(
            "{} dims{:?} type{} off{} size{}",
            t.name, t.dims, t.ggml_type, t.offset, t.size_bytes
        );
    }
    // expert tensors summary
    let exp: Vec<_> = g
        .tensors
        .iter()
        .filter(|t| t.name.contains("_exps."))
        .collect();
    let exp_bytes: u64 = exp.iter().map(|t| t.size_bytes).sum();
    println!(
        "expert tensors: {} totaling {:.2} GB",
        exp.len(),
        exp_bytes as f64 / 1e9
    );
}
