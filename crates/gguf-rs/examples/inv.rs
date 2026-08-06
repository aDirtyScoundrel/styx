use gguf_rs::Gguf;
use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("usage: inv <model.gguf>");
    let g = Gguf::open(Path::new(&path)).expect("parse failed");
    println!("version {} tensors {} data_start {}", g.version, g.tensors.len(), g.data_start);
    println!("--- metadata ---");
    let mut keys: Vec<&String> = g.metadata.keys().collect();
    keys.sort();
    for k in keys {
        let v = &g.metadata[k];
        let s = match v {
            gguf_rs::Value::U32(x) => format!("u32 {x}"),
            gguf_rs::Value::I32(x) => format!("i32 {x}"),
            gguf_rs::Value::F32(x) => format!("f32 {x}"),
            gguf_rs::Value::U64(x) => format!("u64 {x}"),
            gguf_rs::Value::Bool(x) => format!("bool {x}"),
            gguf_rs::Value::String(x) => format!("str {x}"),
            gguf_rs::Value::Array(a) => format!("arr[{}] {:?}", a.len(), &a[..a.len().min(8)]),
            other => format!("{other:?}"),
        };
        println!("{k} = {s}");
    }
    println!("--- tensors ---");
    for t in &g.tensors {
        println!("{} dims{:?} type{} size{}", t.name, t.dims, t.ggml_type, t.size_bytes);
    }
}
