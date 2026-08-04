#!/usr/bin/env python3
"""Parse a llama-eval-callback -v prefill dump of ffn_moe_topk-<L> tensors.

Each topk tensor is {8, n_tokens}: the printer shows per-token rows (first/last
rows for big batches) with the middle 2 of 8 ids elided -> ~6 ids per token.

Outputs:
  1. per-layer expert-usage skew across prompt tokens
  2. token working set: distinct (layer, expert) pairs used by windows of
     k consecutive tokens -> the residency working-set curve
  3. LRU hit-rate sim over the token sequence (proxy for decode locality)

CAVEAT: prefill routing, not decode routing. Treat as an approximation of
temporal locality; M3 will measure decode routing exactly inside our engine.
"""
import re, sys
from collections import Counter, OrderedDict

N_EXPERTS, TOPK = 128, 8
hdr = re.compile(r"ffn_moe_topk-(\d+) = \(i32\)\s+VIEW")

# layer -> list of token rows (each row = list of expert ids)
per_layer_rows = {}
layer = None
for line in open(sys.argv[1] if len(sys.argv) > 1 else "topk_big.log", errors="replace"):
    m = hdr.search(line)
    if m:
        layer = int(m.group(1))
        per_layer_rows.setdefault(layer, [])
        continue
    if layer is None:
        continue
    if "ffn_moe" in line or "common_debug" in line:
        layer = None
        continue
    nums = re.findall(r"(\d+)\.0000", line)
    if nums:
        ids = [int(x) for x in nums]
        if 4 <= len(ids) <= TOPK and all(0 <= i < N_EXPERTS for i in ids):
            per_layer_rows[layer].append(ids)

layers = sorted(per_layer_rows)
n_tok = min(len(v) for v in per_layer_rows.values())
print(f"layers: {len(layers)}, tokens with routing rows: {n_tok}")

# 1. per-layer skew
for k in (16, 24, 32, 48, 64):
    fr = []
    for l in layers:
        c = Counter(e for row in per_layer_rows[l] for e in row)
        t = sum(c.values())
        fr.append(sum(n for _, n in c.most_common(k)) / t)
    print(f"per-layer hottest {k:>2}/128 experts cover {100*sum(fr)/len(fr):5.1f}% of activations (avg)")

# 2. working set: distinct experts per layer touched by k consecutive tokens
print("\nworking set (avg distinct experts PER LAYER for k consecutive tokens):")
for k in (1, 2, 4, 8, 16, min(32, n_tok)):
    tot = []
    for l in layers:
        rows = per_layer_rows[l][:n_tok]
        ws = [len({e for r in rows[i:i+k] for e in r}) for i in range(0, max(1, len(rows)-k+1), k)]
        tot.append(sum(ws)/len(ws))
    avg = sum(tot)/len(tot)
    print(f"  k={k:>2}: {avg:6.1f} experts/layer  -> whole model ~{avg*len(layers):6.0f} slices")

# 3. LRU sim across token sequence (interleaved layer order, like decode)
def lru_sim(slots):
    cache, hits, miss = OrderedDict(), 0, 0
    for t in range(n_tok):
        for l in layers:
            for e in per_layer_rows[l][t]:
                key = (l, e)
                if key in cache:
                    cache.move_to_end(key); hits += 1
                else:
                    miss += 1; cache[key] = 1
                    if len(cache) > slots: cache.popitem(last=False)
    return hits, miss

total = len(layers) * N_EXPERTS
print(f"\nLRU sim over token sequence ({total} total expert slices), incl. cold misses:")
for frac in (0.25, 0.375, 0.5, 0.625, 0.75, 0.875):
    h, m = lru_sim(int(total*frac))
    # steady-state estimate: ignore first-quarter warmup by rerunning and
    # reporting misses in second half
    print(f"  pool {frac*100:4.1f}%: hit {100*h/(h+m):5.1f}%  ({m} misses / {h+m} accesses over {n_tok} tokens)")
