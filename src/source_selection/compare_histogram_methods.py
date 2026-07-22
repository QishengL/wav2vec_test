"""
Compare unigram / TF-IDF unigram / bigram histogram methods for Unit-Proxy.
Uses layer6 features, K=500.
"""
import os, sys, pickle, json
import numpy as np
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import jensenshannon

OUT_DIR = '/mnt/storage/qisheng/github/wav2vec_test/results/pseudo_units'
CACHE_DIR = f'{OUT_DIR}/cache_layer06'
K = 500

CANDIDATES = ['ar','ba','ca','cs','en','eo','fr','hu','it','lt','lv','nl','ro','ru','sw','ta','tr','tt','ug']

# ── Load frames ──
lang_files = sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('_frames.npy')])
langs = [f.replace('_frames.npy','') for f in lang_files]
print(f"Loading {len(langs)} languages from layer6...", flush=True)

frames_per_lang = {}
all_frames_list = []
lang_splits = {}
offset = 0
for lang in langs:
    f = np.load(f'{CACHE_DIR}/{lang}_frames.npy').astype(np.float32)
    frames_per_lang[lang] = f
    all_frames_list.append(f)
    lang_splits[lang] = (offset, offset + len(f))
    offset += len(f)

all_frames = np.concatenate(all_frames_list, axis=0)
del all_frames_list
print(f"Total: {len(langs)} langs, {len(all_frames)} frames", flush=True)

# ── Train K-means ──
kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=8192, verbose=0)
kmeans.fit(all_frames)
with open(f'{OUT_DIR}/kmeans_layer06_K{K}.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
print("K-means trained and saved", flush=True)

# ── Predict labels per language ──
all_labels = {}  # lang -> 1D array of labels
bigram_counts = {}  # lang -> Counter of (u_i, u_{i+1})
unigram_counts = {}  # lang -> Counter of cluster

for lang, (start, end) in lang_splits.items():
    labels = kmeans.predict(all_frames[start:end])
    all_labels[lang] = labels
    
    # Unigram
    unigram_counts[lang] = Counter(labels)
    
    # Bigram (sequential pairs)
    bg = Counter()
    # Process per-utterance: we need utterance boundaries
    # We don't have per-utterance boundaries, so we use per-frame sequential pairs
    # The frames are concatenated per utterance, but utterance boundaries aren't saved
    # For bigram across utterance boundaries: it's a minor approximation
    for i in range(len(labels) - 1):
        bg[(labels[i], labels[i+1])] += 1
    bigram_counts[lang] = bg
    
    if langs.index(lang) % 10 == 0:
        print(f"  {lang} labeled ({len(labels)} frames)", flush=True)

print("All labels computed", flush=True)

# ── Build histograms ──

# 1. Unigram
unigram_hists = {}
for lang in langs:
    total = sum(unigram_counts[lang].values())
    hist = np.zeros(K, dtype=np.float32)
    for c, cnt in unigram_counts[lang].items():
        hist[c] = cnt
    hist /= (total + 1e-10)
    unigram_hists[lang] = hist

# 2. TF-IDF weighted unigram
# IDF: idf(k) = log(N / (1 + n_langs_where_cluster_exists))
N = len(langs)
cluster_presence = np.zeros(K)
for lang in langs:
    nonzero = np.where(unigram_hists[lang] > 1e-10)[0]
    cluster_presence[nonzero] += 1
# Use smoothed IDF that guarantees positive weights
# Inverse cluster frequency: idf(k) = log(1 + N / (1 + n_langs_where_k_appears))
# Ensures: idf > 0 for all clusters, max down-weight for universal clusters
idf = np.log(1 + N / (1 + cluster_presence))

tfidf_hists = {}
for lang in langs:
    weighted = unigram_hists[lang] * idf
    weighted /= (weighted.sum() + 1e-10)
    tfidf_hists[lang] = weighted

# 3. Bigram
# Build global bigram vocabulary (top 20000 bigrams by total frequency)
global_bg = Counter()
for lang in langs:
    global_bg.update(bigram_counts[lang])
top_bigrams = [bg for bg, _ in global_bg.most_common(20000)]
bg_to_idx = {bg: i for i, bg in enumerate(top_bigrams)}
N_BG = len(top_bigrams)
print(f"Bigram vocabulary: {N_BG} types (from {len(global_bg)} total)", flush=True)

bigram_hists = {}
for lang in langs:
    hist = np.zeros(N_BG, dtype=np.float32)
    total = sum(bigram_counts[lang].values())
    for bg, cnt in bigram_counts[lang].items():
        if bg in bg_to_idx:
            hist[bg_to_idx[bg]] = cnt
    hist /= (total + 1e-10)
    bigram_hists[lang] = hist

# ── Compute JS matrices ──
results = {}
for method_name, hists in [('unigram', unigram_hists), ('tfidf', tfidf_hists), ('bigram', bigram_hists)]:
    n = len(langs)
    js = np.zeros((n, n))
    for i, li in enumerate(langs):
        hi = hists[li]
        for j, lj in enumerate(langs):
            hj = hists[lj]
            js[i, j] = 1.0 - float(jensenshannon(hi, hj))
    
    # Stats
    triu = js[np.triu_indices(n, k=1)]
    row = {
        'K': K,
        'method': method_name,
        'js_mean': float(js.mean()),
        'js_std': float(js.std()),
        'js_min': float(triu.min()),
        'js_max': float(triu.max()),
        'js_range': float(triu.max() - triu.min()),
        'js_p5': float(np.percentile(triu, 5)),
        'js_p95': float(np.percentile(triu, 95)),
        'hist_dim': len(list(hists.values())[0])
    }
    
    # Per-language entropy
    entropies = []
    for lang in langs:
        h = hists[lang]
        ent = -np.sum(h * np.log(h + 1e-10))
        max_ent = np.log(len(h))
        entropies.append(ent / max_ent if max_ent > 0 else 0)
    row['mean_entropy_norm'] = float(np.mean(entropies))
    
    # Save JS matrix
    np.save(f'{OUT_DIR}/js_sim_matrix_layer06_K{K}_{method_name}.npy', js)
    
    results[method_name] = row
    print(f"\n{method_name.upper()}:", flush=True)
    print(f"  Dim: {row['hist_dim']}", flush=True)
    print(f"  JS μ={row['js_mean']:.4f} σ={row['js_std']:.4f} range={row['js_range']:.4f}", flush=True)
    print(f"  JS p5={row['js_p5']:.4f} p95={row['js_p95']:.4f}", flush=True)
    print(f"  Entropy (norm)={row['mean_entropy_norm']:.4f}", flush=True)

# ── Comparison table ──
print(f"\n{'='*70}", flush=True)
print("COMPARISON SUMMARY", flush=True)
print(f"{'='*70}", flush=True)
print(f"{'Method':<12} {'Dim':>6} {'JS_μ':>8} {'JS_σ':>8} {'JS_range':>10} {'JS_p5':>8} {'JS_p95':>8} {'Entropy':>8}", flush=True)
print("-" * 70, flush=True)
for method in ['unigram', 'tfidf', 'bigram']:
    r = results[method]
    print(f"{method:<12} {r['hist_dim']:>6} {r['js_mean']:>8.4f} {r['js_std']:>8.4f} "
          f"{r['js_range']:>10.4f} {r['js_p5']:>8.4f} {r['js_p95']:>8.4f} "
          f"{r['mean_entropy_norm']:>8.4f}", flush=True)

# ── Unit-Proxy ranking for CANDIDATES only ──
# Check: for each candidate, which other candidate is most similar (by each method)
print(f"\n{'='*70}", flush=True)
print("CANDIDATE TOP-1 SELF-CHECK (cross-method agreement)", flush=True)
print(f"{'='*70}", flush=True)
for method_name, hists in [('unigram', unigram_hists), ('tfidf', tfidf_hists), ('bigram', bigram_hists)]:
    print(f"\n{method_name.upper()}:", flush=True)
    for c in CANDIDATES[:5]:
        scores = [(c2, 1 - float(jensenshannon(hists[c], hists[c2]))) for c2 in CANDIDATES if c2 != c]
        scores.sort(key=lambda x: -x[1])
        print(f"  {c} → {scores[0][0]}({scores[0][1]:.4f})", flush=True)

with open(f'{OUT_DIR}/histogram_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_DIR}/histogram_comparison.json", flush=True)
print("Done!", flush=True)
