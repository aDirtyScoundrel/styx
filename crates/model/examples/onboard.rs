//! Onboarding scanner: can moe-stream run this GGUF? If not, what's missing?
//!
//! Usage: onboard <model.gguf>
//!
//! Reads only metadata + tensor names (no weight data), checks them against
//! the engine's capability table, and prints a feature-by-feature report
//! with a final verdict:
//!   READY      — load it with generate_moe today
//!   BLOCKED    — lists each unsupported feature and the milestone that
//!                would unblock it
//! Exit code: 0 ready, 2 blocked, 1 parse error.
//!
//! When adding support for a new architecture, extend CAPS below — the
//! scanner is the single source of truth for what the engine claims to run.

use std::collections::BTreeSet;

/// Engine capability table. Update alongside the loader.
const SUPPORTED_ARCHS: &[&str] = &["qwen3moe", "qwen3next"];
/// ggml_type ids the expert matvec pipelines cover (q4_K=12 q5_K=13 q6_K=14 q8_0=8).
const SUPPORTED_EXPERT_QUANTS: &[u32] = &[8, 12, 13, 14];
/// token_embd types the host dequant covers (q4_K only, see dequant_q4k).
const SUPPORTED_EMBD_QUANTS: &[u32] = &[12];
const MAX_EXPERTS: usize = 512;
const MAX_TOPK: usize = 16;

/// Tensor-name fragments the engine does NOT implement, with the milestone
/// that would unblock each. Checked as substrings of per-block tensor names.
const UNSUPPORTED_TENSORS: &[(&str, &str, &str)] = &[
    (
        "exp_probs_b",
        "router expert bias (DeepSeek-style scoring)",
        "M8b",
    ),
    (
        "attn_kv_a_mqa",
        "MLA latent attention",
        "out of scope (MLA)",
    ),
    ("attn_k_b", "MLA latent attention", "out of scope (MLA)"),
    ("attn_v_b", "MLA latent attention", "out of scope (MLA)"),
    (
        "ssm_",
        "hybrid SSM / linear-attention layers",
        "M9b (hybrid forward; M9a loads these)",
    ),
    ("shortconv", "short-conv layers", "M9 (hybrid)"),
    ("linear_attn", "linear attention layers", "M9 (hybrid)"),
    (
        "attn_qkv",
        "fused QKV projection",
        "M9b (hybrid forward; M9a loads these)",
    ),
    (
        "attn_gate",
        "gated attention output",
        "M9b (hybrid forward; M9a loads these)",
    ),
    (
        "ffn_gate_up_exps",
        "fused gate+up expert tensor",
        "M8b (split at load)",
    ),
    (".scale", "per-tensor QAT scales", "M8b"),
    (
        "post_ffw_norm",
        "extra pre/post FFN norms (gemma-style)",
        "M8b",
    ),
];

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: onboard <model.gguf>");
    let g = match gguf_rs::Gguf::open(std::path::Path::new(&path)) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("PARSE ERROR: {e}");
            std::process::exit(1);
        }
    };
    let meta_str = |k: &str| match g.metadata.get(k) {
        Some(gguf_rs::Value::String(s)) => Some(s.clone()),
        _ => None,
    };
    let meta_u32 = |k: &str| match g.metadata.get(k) {
        Some(gguf_rs::Value::U32(v)) => Some(*v),
        Some(gguf_rs::Value::U64(v)) => Some(*v as u32),
        Some(gguf_rs::Value::I32(v)) => Some(*v as u32),
        _ => None,
    };

    let arch = meta_str("general.architecture").unwrap_or_else(|| "?".into());
    let name = meta_str("general.name").unwrap_or_default();
    println!("model: {name}");
    println!("arch:  {arch}");

    let mut blockers: Vec<String> = Vec::new();
    let mut notes: Vec<String> = Vec::new();

    // 1. architecture
    if SUPPORTED_ARCHS.contains(&arch.as_str()) {
        println!("[ok] architecture supported");
    } else {
        blockers.push(format!(
            "architecture '{arch}' not in supported set {SUPPORTED_ARCHS:?} -> M8a metadata dispatch"
        ));
    }

    // 2. MoE shape from <arch>.* keys
    let n_expert = meta_u32(&format!("{arch}.expert_count")).unwrap_or(0) as usize;
    let n_used = meta_u32(&format!("{arch}.expert_used_count")).unwrap_or(0) as usize;
    if n_expert == 0 {
        blockers.push("no <arch>.expert_count — not a MoE gguf (or unknown key layout)".into());
    } else {
        println!("[ok] experts: {n_expert}, top-k {n_used}");
        if n_expert > MAX_EXPERTS {
            blockers.push(format!(
                "expert_count {n_expert} > shader cap {MAX_EXPERTS}"
            ));
        }
        if n_used > MAX_TOPK {
            blockers.push(format!(
                "expert_used_count {n_used} > shader cap {MAX_TOPK}"
            ));
        }
    }

    // 3. tensor scan: unsupported structures + expert/embd quants
    let mut hit: BTreeSet<(&str, &str)> = BTreeSet::new();
    let mut expert_types: BTreeSet<u32> = BTreeSet::new();
    let mut embd_type = None;
    for t in &g.tensors {
        for (frag, what, ms) in UNSUPPORTED_TENSORS {
            if t.name.contains(frag) {
                hit.insert((*what, *ms));
            }
        }
        if t.name.contains("_exps") {
            expert_types.insert(t.ggml_type);
        }
        if t.name == "token_embd.weight" {
            embd_type = Some(t.ggml_type);
        }
    }
    for (what, ms) in &hit {
        blockers.push(format!("{what} -> {ms}"));
    }
    for ty in &expert_types {
        if SUPPORTED_EXPERT_QUANTS.contains(ty) {
            notes.push(format!("expert quant ggml_type {ty}: ok"));
        } else {
            blockers.push(format!(
                "expert quant ggml_type {ty} has no mmv_id pipeline -> M8b quant expansion"
            ));
        }
    }
    match embd_type {
        Some(ty) if SUPPORTED_EMBD_QUANTS.contains(&ty) => {
            notes.push(format!("token_embd ggml_type {ty}: ok"))
        }
        Some(ty) => blockers.push(format!(
            "token_embd ggml_type {ty} not host-dequantable (q4_K only) -> M8a embd dequant"
        )),
        None => blockers.push("token_embd.weight missing".into()),
    }

    // 4. routing semantics (only meaningful for supported archs)
    if arch == "qwen3moe" {
        notes.push("router: softmax->topk + norm_topk_prob (native)".into());
    }

    for n in &notes {
        println!("[ok] {n}");
    }
    if blockers.is_empty() {
        let sz: u64 = g
            .tensors
            .iter()
            .filter(|t| t.name.contains("_exps"))
            .map(|t| t.size_bytes)
            .sum();
        println!(
            "\nVERDICT: READY — expert weights {:.2} GiB; suggested start:",
            sz as f64 / (1u64 << 30) as f64
        );
        println!("  MOE_EXPERTS_VRAM_MB=6144 MOE_REPIN_INTERVAL=64 generate_moe {path} <n> <toks>");
        std::process::exit(0);
    }
    println!(
        "\nVERDICT: BLOCKED — {} feature(s) missing:",
        blockers.len()
    );
    for b in &blockers {
        println!("  - {b}");
    }
    std::process::exit(2);
}
