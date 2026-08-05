use gguf_rs::Gguf;
use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("usage: names <model.gguf>");
    let g = Gguf::open(Path::new(&path)).expect("parse failed");
    println!("=== METADATA ===");
    for (k, v) in &g.metadata {
        if k.contains("architecture")
            || k.contains("block_count")
            || k.contains("expert")
            || k.contains("head")
            || k.contains("embedding_length")
            || k.contains("context")
            || k.contains("vocab")
            || k.contains("moe")
            || k.contains("attention")
            || k.contains("layer")
        {
            println!("{k} = {v:?}");
        }
    }
    println!("\n=== TENSOR NAMES (all {}) ===", g.tensors.len());
    for t in &g.tensors {
        println!("{} dims{:?} type{} size{}", t.name, t.dims, t.ggml_type, t.size_bytes);
    }
}
