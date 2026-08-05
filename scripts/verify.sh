#!/usr/bin/env bash
# Canonical verification suite for moe-stream.
#
#   scripts/verify.sh          fast tier: build, unit tests, GPU kernel
#                              tests, golden-token regression at 3 expert
#                              placements, HUD arithmetic check
#   SLOW=1 scripts/verify.sh   adds 600-tok paged-KV growth run
#
# Requirements: Vulkan GPU, glslc, and the reference gguf (see $MODEL).
# GPU-dependent stages skip with a notice when the model/GPU is absent.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MOE_VERIFY_GGUF:-/home/tyler/LMStudioModels/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf}
HIST=tests/golden/route_hist_30b.csv
GOLD=tests/golden/qwen3-coder-30b_greedy64_151644-872.txt
GEN=./target/release/examples/generate_moe
PROMPT="151644 872"   # tokens used by every golden artifact
PASS=0; FAIL=0
ok()  { echo "  ok: $1"; PASS=$((PASS+1)); }
die() { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

echo "[1/5] build (release, all targets)"
cargo build --release --workspace --examples 2>&1 | grep -E '^error' \
    && { die "build"; exit 1; } || ok "cargo build --release"

echo "[2/5] host unit tests"
if cargo test --release -q 2>&1 | grep -E 'test result.*[1-9][0-9]* failed'; then
    die "host unit tests"
else
    ok "cargo test (host)"
fi

echo "[3/5] GPU kernel tests (MOE_GPU_TESTS=1)"
if vulkaninfo --summary >/dev/null 2>&1 || [ -e /dev/dri/renderD128 ]; then
    if MOE_GPU_TESTS=1 cargo test --release -q -p vk-backend 2>&1 \
        | grep -E 'test result.*[1-9][0-9]* failed'; then
        die "GPU kernel tests"
    else
        ok "vk-backend kernel tests"
    fi
else
    echo "  skip: no Vulkan device visible"
fi

if [ ! -f "$MODEL" ]; then
    echo "[4/5][5/5] skip: reference gguf not found: $MODEL"
    echo "  (set MOE_VERIFY_GGUF to run golden-token stages)"
else
    echo "[3.5/5] onboarding scanner"
    ONBOARD=./target/release/examples/onboard
    "$ONBOARD" "$MODEL" >/dev/null 2>&1 && ok "onboard: reference model READY" \
        || die "onboard: reference model not READY"
    "$ONBOARD" /dev/null >/dev/null 2>&1 && die "onboard: accepted non-gguf" \
        || ok "onboard: rejects non-gguf"

    echo "[4/5] golden-token regression (64-tok greedy, 3 placements)"
    run() { env "$@" "$GEN" "$MODEL" 64 $PROMPT 2>/dev/null; }
    GOLD_TOKS=$(cat "$GOLD")
    for place in "GTT-only:MOE_EXPERTS_VRAM_MB=0" \
                 "hot/cold-6GiB:MOE_EXPERT_HIST=$HIST MOE_EXPERTS_VRAM_MB=6144" \
                 "all-VRAM:MOE_EXPERTS_VRAM=1"; do
        name=${place%%:*}; envs=${place#*:}
        got=$(run $envs)
        [ "$got" = "$GOLD_TOKS" ] && ok "golden match: $name" \
                                  || die "golden MISMATCH: $name"
    done

    echo "[5/5] HUD arithmetic (hot+cold == tok*layers*k)"
    ERR=$(mktemp)
    got=$(env MOE_HUD=1 MOE_EXPERT_HIST="$HIST" MOE_EXPERTS_VRAM_MB=6144 \
          "$GEN" "$MODEL" 32 $PROMPT 2>"$ERR")
    TOT=$(grep -oE '\([0-9]+\)' "$ERR" | tr -d '()' | paste -sd+ | bc)
    # 2 prefill + 32 decode - 1 unforwarded = 33 tok; 48 layers; k=8
    [ "${TOT:-0}" = "12672" ] && ok "HUD totals ($TOT)" || die "HUD totals ($TOT != 12672)"
    grep -qE '^hud: cold/layer \[[.0-9]{48}\]' "$ERR" && ok "HUD heatmap (48 glyphs)" \
                                                      || die "HUD heatmap"
    rm -f "$ERR"

    if [ "${SLOW:-0}" = "1" ]; then
        echo "[slow] paged-KV growth: 600-tok greedy (crosses 512/1024)"
        long=$("$GEN" "$MODEL" 600 $PROMPT 2>/dev/null)
        # first 64 must extend the golden prefix; count must be 600
        [ "$(echo $long | cut -d' ' -f1-64)" = "$GOLD_TOKS" ] \
            && ok "600-tok prefix matches golden" || die "600-tok prefix"
        [ "$(echo $long | wc -w)" = "600" ] && ok "600 tokens emitted" || die "token count"
    fi
fi

echo
echo "verify: $PASS passed, $FAIL failed"
[ "$FAIL" = "0" ]
