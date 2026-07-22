"""
Diagnostic grid: compare K × Layer combinations for Unit-Proxy.
Records: dead clusters, entropy, JS stats, ranking, stability.
"""
import os, sys, pickle, json
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import jensenshannon

OUT_DIR = '/mnt/storage/qisheng/github/wav2vec_test/results/pseudo_units'

LAYERS = {'layer06': 6, 'layer12': 12, 'layer24': 24}
K_VALS = [50, 100, 200, 500]

CANDIDATES = ['ar','ba','ca','cs','en','eo','fr','hu','it','lt','lv','nl','ro','ru','sw','ta','tr','tt','ug']
TARGETS = ['sq','ltg','ur','cy','gn','tn','am','az','mt','af','da','ky','tk','kk','sk','id']

def load_language_frames(layer_name):
    cache_dir = f'{OUT_DIR}/cache_{layer_name}'
    langs = sorted([f.replace('_frames.npy','') for f in os.listdir(cache_dir) if f.endswith('_frames.npy')])
    frames_concat = []
    lang_splits = {}
    total = 0
    for lang in langs:
        f = np.load(f'{cache_dir}/{lang}_frames.npy').astype(np.float32)
        frames_concat.append(f)
        lang_splits[lang] = (total, total + len(f))
        total += len(f)
    return langs, np.concatenate(frames_concat, axis=0), lang_splits

results = {}

for layer_name, layer_idx in LAYERS.items():
    print(f"\n{'='*70}", flush=True)
    print(f"Layer {layer_idx} ({layer_name})", flush=True)
    print(f"{'='*70}", flush=True)
    
    langs, all_frames, lang_splits = load_language_frames(layer_name)
    print(f"Loaded {len(langs)} languages, {len(all_frames)} frames", flush=True)
    
    for K in K_VALS:
        print(f"\n  --- K={K} ---", flush=True)
        suffix = f'{layer_name}_K{K}'
        
        # Train K-means
        kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=8192, verbose=0)
        kmeans.fit(all_frames)
        
        # Compute histograms
        histograms = {}
        dead_clusters = 0
        total_assignments = np.zeros(K)
        for lang, (start, end) in lang_splits.items():
            labels = kmeans.predict(all_frames[start:end])
            total_assignments += np.bincount(labels, minlength=K)
            hist = np.bincount(labels, minlength=K).astype(np.float32)
            hist /= (hist.sum() + 1e-10)
            histograms[lang] = hist
        
        dead_clusters = int((total_assignments == 0).sum())
        
        # Per-language entropy
        entropies = {}
        for lang in langs:
            h = histograms[lang]
            ent = -np.sum(h * np.log(h + 1e-10))
            entropies[lang] = float(ent / np.log(K))  # normalized
        
        # JS similarity matrix
        n = len(langs)
        js = np.zeros((n, n))
        for i, li in enumerate(langs):
            hi = histograms[li]
            for j, lj in enumerate(langs):
                hj = histograms[lj]
                js[i, j] = 1.0 - float(jensenshannon(hi, hj))
        
        js_mean = float(js.mean())
        js_std = float(js.std())
        js_min = float(js.min())
        js_max = float(js[np.triu_indices(n, k=1)].max())
        
        # Unit-Proxy ranking for targets
        ranking = {}
        for tgt_code in TARGETS:
            # Load target histogram from last-layer unigram (or try to compute)
            tgt_path = f'{OUT_DIR}/{tgt_code}_lang_hist_{suffix}.npy'
            # If not available, skip
            scores = []
            for cand in CANDIDATES:
                if cand in histograms:
                    cand_hist = histograms[cand]
                    # Use a placeholder for now - target histograms need to be computed separately
                    pass
            ranking[tgt_code] = scores
        
        row = {
            'layer': layer_idx,
            'K': K,
            'n_languages': len(langs),
            'n_frames': len(all_frames),
            'dead_clusters': dead_clusters,
            'dead_ratio': float(dead_clusters / K),
            'mean_entropy_norm': float(np.mean(list(entropies.values()))),
            'min_entropy_norm': float(min(entropies.values())),
            'max_entropy_norm': float(max(entropies.values())),
            'js_mean': js_mean,
            'js_std': js_std,
            'js_min': js_min,
            'js_max': js_max,
            'js_range': js_max - js_min,
            'js_dynamic_range': (js_max - js_min) / js_std if js_std > 0 else 0,
        }
        
        key = f'layer{layer_idx}_K{K}'
        results[key] = row
        
        print(f"    Dead clusters: {dead_clusters}/{K} ({dead_clusters/K*100:.1f}%)", flush=True)
        print(f"    Mean entropy (norm): {row['mean_entropy_norm']:.4f}", flush=True)
        print(f"    JS mean={js_mean:.4f} std={js_std:.4f} range={row['js_range']:.4f}", flush=True)
        print(f"    JS min={js_min:.4f} max={js_max:.4f}", flush=True)

# Summary table
print(f"\n\n{'='*90}", flush=True)
print("SUMMARY: Layer × K diagnostic", flush=True)
print(f"{'='*90}", flush=True)
header = f"{'Config':<20} {'Dead%':>8} {'Entropy':>8} {'JS_μ':>8} {'JS_σ':>8} {'JS_range':>10} {'JS_min':>8} {'JS_max':>8}"
print(header, flush=True)
print("-" * len(header), flush=True)
for key in sorted(results.keys()):
    r = results[key]
    print(f"{key:<20} {r['dead_ratio']*100:>7.1f}% {r['mean_entropy_norm']:>8.4f} "
          f"{r['js_mean']:>8.4f} {r['js_std']:>8.4f} {r['js_range']:>10.4f} "
          f"{r['js_min']:>8.4f} {r['js_max']:>8.4f}", flush=True)

# Save results
with open(f'{OUT_DIR}/diagnostic_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_DIR}/diagnostic_results.json", flush=True)
print("Done!", flush=True)
