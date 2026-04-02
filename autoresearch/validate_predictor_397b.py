#!/usr/bin/env python3
"""
validate_predictor_397b.py — Offline routing predictor validation.

Parses routing traces collected with --collect-routing, simulates candidate
predictors, and applies the 50% incremental hit rate gate.

Usage:
  python3 autoresearch/validate_predictor_397b.py /tmp/routing_p1.bin [more.bin ...]

Gate: incremental hit rate > 50% over temporal prediction required before any
integration work begins. Script makes NO changes to the binary.

Record format (16,408 bytes each):
  int32_t layer_idx     (4 bytes)
  int32_t K             (4 bytes)
  float[4096] hidden    (16,384 bytes)
  int32_t[4] experts    (16 bytes)  — K=4 always
"""

import sys
import struct
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
HIDDEN_DIM = 4096
K          = 4
NUM_LAYERS = 60
NUM_EXPERTS = 512
RECORD_SIZE = 4 + 4 + HIDDEN_DIM * 4 + K * 4   # 16,408 bytes
GATE        = 0.50   # >50% incremental hit rate required


# ── Parsing ───────────────────────────────────────────────────────────────────
def parse_trace(path):
    """
    Parse a routing trace binary. Returns list of dicts:
      {'layer': int, 'K': int, 'hidden': float32[4096], 'experts': int[4]}
    Records are in layer-major order: layers 0..59 for token 0, then 0..59 for token 1, etc.
    """
    data = Path(path).read_bytes()
    n_rec = len(data) // RECORD_SIZE
    if len(data) % RECORD_SIZE:
        print(f"  WARNING: {len(data) % RECORD_SIZE} trailing bytes ignored")

    records = []
    off = 0
    for _ in range(n_rec):
        layer_idx, k_val = struct.unpack_from('<ii', data, off)
        off += 8
        hidden = np.frombuffer(data, dtype=np.float32, count=HIDDEN_DIM, offset=off).copy()
        off += HIDDEN_DIM * 4
        experts = list(struct.unpack_from('<4i', data, off))
        off += K * 4
        records.append({'layer': layer_idx, 'K': k_val, 'hidden': hidden, 'experts': experts})

    print(f"  Parsed {n_rec:,} records  ({len(data)/1e6:.1f} MB)  path={path}")
    return records


def group_by_token(records):
    """
    Reshape flat records into token list.
    tokens[t][l] = {'experts': [...], 'hidden': array}
    Assumes records arrive in layer-sequential order within each token.
    """
    n_tok = len(records) // NUM_LAYERS
    remainder = len(records) % NUM_LAYERS
    if remainder:
        print(f"  WARNING: {remainder} extra records dropped (not a full token)")

    tokens = []
    for t in range(n_tok):
        tok = {}
        for i in range(NUM_LAYERS):
            r = records[t * NUM_LAYERS + i]
            l = r['layer']
            tok[l] = {'experts': r['experts'], 'hidden': r['hidden']}
        tokens.append(tok)

    print(f"  Grouped into {n_tok} tokens × {NUM_LAYERS} layers")
    return tokens


# ── Hit counting ──────────────────────────────────────────────────────────────
def hits(predicted, actual):
    """Count how many predicted experts appear in actual."""
    return len(set(predicted) & set(actual))


# ── Predictors ────────────────────────────────────────────────────────────────
def eval_temporal(tokens):
    """
    Temporal predictor — predict same K experts as previous token at same layer.
    This is always active in the binary (g_pred_enabled=1). Baseline.
    Returns (hit_rate, total_hits, total_possible).
    """
    total_h = total_p = 0
    for t in range(1, len(tokens)):
        for l in range(NUM_LAYERS):
            if l not in tokens[t] or l not in tokens[t-1]:
                continue
            total_h += hits(tokens[t-1][l]['experts'], tokens[t][l]['experts'])
            total_p += K
    rate = total_h / total_p if total_p else 0.0
    return rate, total_h, total_p


def eval_freq_lru(tokens, window):
    """
    Frequency-LRU predictor — predict top-K most frequently seen experts
    per layer over the last `window` tokens.
    Implementable in C: per-layer frequency array updated by sliding window.
    """
    total_h = total_p = 0
    for t in range(1, len(tokens)):
        for l in range(NUM_LAYERS):
            if l not in tokens[t]:
                continue
            counter = Counter()
            for prev_t in range(max(0, t - window), t):
                if l in tokens[prev_t]:
                    for e in tokens[prev_t][l]['experts']:
                        counter[e] += 1
            if not counter:
                continue
            predicted = [e for e, _ in counter.most_common(K)]
            total_h += hits(predicted, tokens[t][l]['experts'])
            total_p += K
    rate = total_h / total_p if total_p else 0.0
    return rate, total_h, total_p


def eval_ngram(tokens):
    """
    N-gram predictor — given previous token's experts at layer l, predict
    current token's experts via co-occurrence statistics.

    Oracle version: trained and evaluated on the same trace.
    This gives an upper bound — real deployment would build the table online
    (causal: only use data available at inference time).

    Falls back to temporal prediction when the prev-key is unseen.
    """
    # Build co-occurrence table: cooccur[layer][prev_key][expert] += count
    cooccur = defaultdict(lambda: defaultdict(Counter))
    for t in range(1, len(tokens)):
        for l in range(NUM_LAYERS):
            if l not in tokens[t] or l not in tokens[t-1]:
                continue
            prev_key = tuple(sorted(tokens[t-1][l]['experts']))
            for e in tokens[t][l]['experts']:
                cooccur[l][prev_key][e] += 1

    total_h = total_p = 0
    for t in range(1, len(tokens)):
        for l in range(NUM_LAYERS):
            if l not in tokens[t] or l not in tokens[t-1]:
                continue
            prev_key = tuple(sorted(tokens[t-1][l]['experts']))
            if prev_key in cooccur[l]:
                predicted = [e for e, _ in cooccur[l][prev_key].most_common(K)]
            else:
                predicted = tokens[t-1][l]['experts']  # temporal fallback
            total_h += hits(predicted, tokens[t][l]['experts'])
            total_p += K
    rate = total_h / total_p if total_p else 0.0
    return rate, total_h, total_p


def eval_ngram_causal(tokens):
    """
    Causal N-gram predictor — same as eval_ngram but ONLY uses tokens 0..t-1
    to predict token t.  This is what can actually be implemented at inference
    time: the lookup table is built online during generation and consulted
    before each token's expert dispatch.

    This is the directly-implementable version of the oracle N-gram.
    Falls back to temporal prediction on unseen prev-key.
    """
    cooccur = defaultdict(lambda: defaultdict(Counter))  # layer → prev_key → next_experts

    total_h = total_p = 0
    for t in range(1, len(tokens)):
        # Predict token t using table built from 0..t-1
        for l in range(NUM_LAYERS):
            if l not in tokens[t] or l not in tokens[t-1]:
                continue
            prev_key = tuple(sorted(tokens[t-1][l]['experts']))
            if prev_key in cooccur[l]:
                predicted = [e for e, _ in cooccur[l][prev_key].most_common(K)]
            else:
                predicted = tokens[t-1][l]['experts']  # temporal fallback
            total_h += hits(predicted, tokens[t][l]['experts'])
            total_p += K

        # Update table with token t's ground truth (causal: after prediction)
        for l in range(NUM_LAYERS):
            if l not in tokens[t] or l not in tokens[t-1]:
                continue
            prev_key = tuple(sorted(tokens[t-1][l]['experts']))
            for e in tokens[t][l]['experts']:
                cooccur[l][prev_key][e] += 1

    rate = total_h / total_p if total_p else 0.0
    return rate, total_h, total_p


def eval_hidden_cosine(tokens, n_neighbors=8):
    """
    Hidden-state cosine similarity predictor — for token t at layer l,
    find the `n_neighbors` most similar past hidden states and use their
    expert choices (majority vote) as prediction.

    The router itself uses the hidden state, so this is a theoretical upper
    bound on what any hidden-state-based predictor can achieve.

    NOTE: Requires computing attention output BEFORE dispatching to experts,
    which IS available at inference time (h_t arrives before top-K routing).
    However, this requires an in-memory embedding database and O(t) cosine
    similarity per step — not directly implementable without approximation.
    Reported here as an oracle ceiling.
    """
    total_h = total_p = 0

    for l in range(NUM_LAYERS):
        # Collect all (hidden, experts) for this layer
        layer_tokens = [(t, tokens[t][l]) for t in range(len(tokens)) if l in tokens[t]]
        if len(layer_tokens) < 2:
            continue

        hiddens = np.stack([d['hidden'] for _, d in layer_tokens])  # (T, 4096)
        norms = np.linalg.norm(hiddens, axis=1, keepdims=True) + 1e-8
        hiddens_normed = hiddens / norms  # (T, 4096) normalized

        # For each token t > 0, find n_neighbors most similar past tokens
        for i in range(1, len(layer_tokens)):
            t_idx = layer_tokens[i][0]
            actual = layer_tokens[i][1]['experts']

            # Cosine similarity to all previous tokens at this layer
            past_normed = hiddens_normed[:i]          # (i, 4096)
            sims = past_normed @ hiddens_normed[i]    # (i,)

            n_k = min(n_neighbors, i)
            top_idx = np.argpartition(sims, -n_k)[-n_k:]

            # Aggregate experts from top neighbors by frequency
            counter = Counter()
            for idx in top_idx:
                for e in layer_tokens[idx][1]['experts']:
                    counter[e] += 1
            predicted = [e for e, _ in counter.most_common(K)]

            total_h += hits(predicted, actual)
            total_p += K

    rate = total_h / total_p if total_p else 0.0
    return rate, total_h, total_p


# ── Gate computation ──────────────────────────────────────────────────────────
def incremental(t_hits, t_possible, cand_hits):
    """
    Incremental hit rate = new_hits / temporal_misses
    = (cand_hits - t_hits) / (t_possible - t_hits)
    Measures how much of temporal's misses the candidate recovers.
    """
    misses = t_possible - t_hits
    if misses == 0:
        return 1.0
    return max(0.0, cand_hits - t_hits) / misses


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else ['/tmp/routing_p1.bin']

    print("=" * 72)
    print("Flash-MoE 397B  Routing Predictor Validation")
    print(f"Gate: incremental hit rate > {GATE*100:.0f}% over temporal prediction")
    print("NO binary modifications — analysis only")
    print("=" * 72)

    all_records = []
    for f in files:
        print(f"\nLoading {f} ...")
        all_records.extend(parse_trace(f))

    print(f"\nTotal records: {len(all_records):,}")
    tokens = group_by_token(all_records)
    n_eval = len(tokens) - 1
    print(f"Evaluation samples: {n_eval} token transitions × {NUM_LAYERS} layers "
          f"= {n_eval * NUM_LAYERS:,} predictions\n")

    # ── Temporal (baseline always active in binary) ───────────────────────────
    print("─" * 72)
    print("Temporal prediction (baseline, always active in binary)")
    t_rate, t_hits, t_pos = eval_temporal(tokens)
    print(f"  Hit rate:  {t_rate*100:.2f}%  ({t_hits:,}/{t_pos:,})")

    results = []

    # ── Frequency-LRU ─────────────────────────────────────────────────────────
    for window in [4, 8, 16, 32]:
        print(f"\n─ Frequency-LRU  window={window} " + "─" * (50 - len(str(window))))
        rate, h, p = eval_freq_lru(tokens, window)
        incr = incremental(t_hits, t_pos, h)
        results.append((f"Freq-LRU(w={window})", rate, incr, False))
        print(f"  Hit rate:       {rate*100:.2f}%  ({h:,}/{p:,})")
        print(f"  Incremental:    {incr*100:.2f}%  "
              f"{'PASS ✓' if incr > GATE else 'fail ✗'}")

    # ── N-gram causal (implementable) ────────────────────────────────────────
    print("\n─ N-gram causal (online table, temporal fallback on cold start) ──────")
    ngc_rate, ngc_h, ngc_p = eval_ngram_causal(tokens)
    ngc_incr = incremental(t_hits, t_pos, ngc_h)
    results.append(("N-gram causal", ngc_rate, ngc_incr, False))
    print(f"  Hit rate:       {ngc_rate*100:.2f}%  ({ngc_h:,}/{ngc_p:,})")
    print(f"  Incremental:    {ngc_incr*100:.2f}%  "
          f"{'PASS ✓' if ngc_incr > GATE else 'fail ✗'}")
    print(f"  Note: implementable in C — hash table built online during generation.")

    # ── N-gram (oracle) ───────────────────────────────────────────────────────
    print("\n─ N-gram oracle (trained+evaluated on same trace, upper bound) ──────")
    ng_rate, ng_h, ng_p = eval_ngram(tokens)
    ng_incr = incremental(t_hits, t_pos, ng_h)
    results.append(("N-gram oracle", ng_rate, ng_incr, True))
    print(f"  Hit rate:       {ng_rate*100:.2f}%  ({ng_h:,}/{ng_p:,})")
    print(f"  Incremental:    {ng_incr*100:.2f}%  "
          f"{'PASS ✓' if ng_incr > GATE else 'fail ✗'}")
    print(f"  Note: oracle upper bound — uses all data including future tokens.")

    # ── Hidden cosine (oracle upper bound) ────────────────────────────────────
    print("\n─ Hidden-cosine KNN (oracle upper bound, not directly implementable) ─")
    print("  Computing cosine similarity across hidden states ...", flush=True)
    hc_rate, hc_h, hc_p = eval_hidden_cosine(tokens, n_neighbors=8)
    hc_incr = incremental(t_hits, t_pos, hc_h)
    results.append(("Hidden-cosine KNN (oracle)", hc_rate, hc_incr, True))
    print(f"  Hit rate:       {hc_rate*100:.2f}%  ({hc_h:,}/{hc_p:,})")
    print(f"  Incremental:    {hc_incr*100:.2f}%  "
          f"{'PASS ✓' if hc_incr > GATE else 'fail ✗'}")
    print(f"  Note: oracle — requires O(t) cosine search per step; needs ANN approximation.")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Predictor':<32} {'Hit rate':>10} {'Incremental':>13} {'Gate >50%':>10}")
    print("─" * 72)
    print(f"{'Temporal (baseline, in binary)':<32} {t_rate*100:>9.2f}%  {'(baseline)':>12}  {'—':>10}")
    for name, rate, incr, oracle in results:
        label = "PASS ✓" if incr > GATE else "fail ✗"
        oracle_tag = " [oracle]" if oracle else ""
        print(f"{name+oracle_tag:<32} {rate*100:>9.2f}%  {incr*100:>12.2f}%  {label:>10}")

    print("\n" + "=" * 72)
    passing = [(n, i) for n, r, i, _ in results if i > GATE]
    non_oracle_passing = [(n, i) for n, r, i, is_oracle in results if i > GATE and not is_oracle]

    if non_oracle_passing:
        best_n, best_i = max(non_oracle_passing, key=lambda x: x[1])
        print(f"GATE STATUS: PASS")
        print(f"Best implementable predictor: {best_n}  incremental={best_i*100:.2f}%")
        print(f"Next step: request user approval before any binary integration.")
    elif passing:
        oracle_n, oracle_i = max(passing, key=lambda x: x[1])
        best_impl = max([(n, i) for n, r, i, _ in results if not _], key=lambda x: x[1], default=("none", 0))
        print(f"GATE STATUS: FAIL (no implementable predictor passes)")
        print(f"Best implementable: {best_impl[0]}  incremental={best_impl[1]*100:.2f}%")
        print(f"Oracle ceiling: {oracle_n}  incremental={oracle_i*100:.2f}%")
        print(f"Gap suggests implementable predictors cannot reach 50% with this data volume.")
        print(f"Recommendation: collect more diverse routing traces before concluding.")
    else:
        best_n, best_i = max(results, key=lambda x: x[2]) if results else ("none", 0, 0, False)[:2]
        print(f"GATE STATUS: FAIL — even oracle predictors below 50% gate")
        print(f"Interpretation: 397B routing is sufficiently non-repetitive that")
        print(f"  simple predictors cannot recover >50% of temporal misses.")
        print(f"  This is expected for diverse prompts; collect more traces or")
        print(f"  investigate context-window-specific routing patterns.")
    print("=" * 72)

    return 0 if non_oracle_passing else 1


if __name__ == '__main__':
    sys.exit(main())
