#!/usr/bin/env python3
"""Parse llama-eval-callback -v dumps of ffn_moe_topk-<layer> tensors into an
expert-activation histogram, then simulate LRU cache hit rates per pool size.

Log block format:
common_debug_cb_eval:           ffn_moe_topk-17 = (i32) VIEW(...) = {8, N, 1, 1}
    [
        [
            [ 62.0000, 21.0000, ..., 47.0000 ],   (one row per token)
"""
import re, sys, glob
from collections import Counter, OrderedDict

N_EXPERTS = 128
TOPK = 8

hdr = re.compile(r"ffn_moe_topk-(\d+) ")
row = re.compile(r"\[\s*([\d\.,\s\-]+?)\s*\]\s*,?\s*$")

# events: list of (layer, [expert ids]) in execution order
events = []
for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "topk_run*.log")):
    layer = None
    for line in open(path, errors="replace"):
        m = hdr.search(line)
        if m:
            layer = int(m.group(1))
            continue
        if layer is None:
            continue
        # rows look like: [ 62.0000, 21.0000, 75.0000, ..., 126.0000, 34.0000, 47.0000 ],
        # the printer elides middle elements, so we get 6 of the 8 top-k ids.
        if "[" in line and "." in line:
            nums = re.findall(r"(\d+)\.0000", line)
            ids = [int(x) for x in nums]
            if 4 <= len(ids) <= TOPK and all(0 <= i < N_EXPERTS for i in ids):
                events.append((layer, ids))

layers = sorted({l for l, _ in events})
n_layers = len(layers)
tokens = len(events) // max(n_layers, 1)
print(f"parsed {len(events)} routing events, {n_layers} layers, ~{tokens} token-steps")

# global + per-layer histograms
glob_c = Counter()
per_layer = {l: Counter() for l in layers}
for l, ids in events:
    for e in ids:
        glob_c[e] += 1
        per_layer[l][e] += 1

total = sum(glob_c.values())
top = glob_c.most_common()
def cum_frac(k):
    return sum(c for _, c in top[:k]) / total
print(f"\nglobal expert skew (all layers pooled, {N_EXPERTS} ids):")
for k in (8, 16, 32, 48, 64, 96):
    print(f"  hottest {k:>3}/128 ids capture {cum_frac(k)*100:5.1f}% of activations")

# per-layer skew: average coverage of each layer's hottest-k experts
for k in (16, 32, 48, 64):
    fr = []
    for l in layers:
        t = sum(per_layer[l].values())
        fr.append(sum(c for _, c in per_layer[l].most_common(k)) / t)
    print(f"per-layer: hottest {k:>2} experts/layer cover {100*sum(fr)/len(fr):5.1f}% (avg)")

# LRU simulation: cache key = (layer, expert), slots shared across layers
def lru_sim(slots):
    cache = OrderedDict()
    hits = misses = 0
    for l, ids in events:
        for e in ids:
            k = (l, e)
            if k in cache:
                cache.move_to_end(k)
                hits += 1
            else:
                misses += 1
                cache[k] = True
                if len(cache) > slots:
                    cache.popitem(last=False)
    return hits / (hits + misses)

total_experts = n_layers * N_EXPERTS
print(f"\nLRU hit-rate sim (shared pool, {total_experts} total expert slices):")
for frac in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75):
    slots = int(total_experts * frac)
    hr = lru_sim(slots)
    print(f"  pool = {slots:>5} slots ({frac*100:4.1f}% of experts): hit rate {hr*100:5.1f}%")
