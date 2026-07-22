"""
Diagnostic grid for HuBERT: K × Layer comparison.
Same logic as diagnostic_grid.py but reads from cache_hubert_* dirs.
"""
import os, sys, pickle, json
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import jensenshannon

OUT_DIR = '/mnt/storage/qisheng/github/wav2vec_test/results/pseudo_units'
CACHE_PREFIX = 'cache_hubert_layer'
LAYER_NAMES = ['layer06', 'layer12', 'layer24']
LAYER_IDS = {'layer06': 6, 'layer12': 12, 'layer24': 24}
K_VALS = [50, 100, 200, 500]

def load_frames(layer_name):
    cache_dir = f'{OUT_DIR}/{CACHE_PREFIX}{layer_name.replace("layer","")}'
    if not os.path.exists(cache_dir):
        # Try long name
        cache_dir = f'{OUT_DIR}/{CACHE_PREFIX}{layer_name}'
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

for layer_name in LAYER_NAMES:
    layer_idx = LAYER_IDS[layer_name]
    print(f"\n{'='*70}", flush=True)
    print(f"HuBERT Layer {layer_idx}", flush=True)
    print(f"{'='*70}", flush=True)
    
    langs, all_frames, lang_splits = load_frames(layer_name)
    print(f"Loaded {len(langs)} langs, {len(all_frames)} frames", flush=True)
    
    for K in K_VALS:
        print(f"\n  K={K}", flush=True)
        kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=8192, verbose=0)
        kmeans.fit(all_frames)
        
        total_assignments = np.zeros(K)
        for lang, (start, end) in lang_splits.items():
            labels = kmeans.predict(all_frames[start:end])
            total_assignments += np.bincount(labels, minlength=K)
        
        dead = int((total_assignments == 0).sum())
        
        # Entropy
        entropies = []
        for lang, (start, end) in lang_splits.items():
            labels = kmeans.predict(all_frames[start:end])
            hist = np.bincount(labels, minlength=K).astype(np.float32)
            hist /= (hist.sum() + 1e-10)
            ent = -np.sum(hist * np.log(hist + 1e-10))
            entropies.append(ent / np.log(K))
        
        # JS matrix
        n = len(langs)
        js = np.zeros((n, n))
        for i, li in enumerate(langs):
            si, ei = lang_splits[li]
            hi = np.bincount(kmeans.predict(all_frames[si:ei]), minlength=K).astype(np.float32)
            hi /= (hi.sum() + 1e-10)
            for j, lj in enumerate(langs):
                sj, ej = lang_splits[lj]
                hj = np.bincount(kmeans.predict(all_frames[sj:ej]), minlength=K).astype(np.float32)
                hj /= (hj.sum() + 1e-10)
                js[i, j] = 1.0 - float(jensenshannon(hi, hj))
        
        key = f'hubert_layer{layer_idx}_K{K}'
        results[key] = {
            'model': 'hubert', 'layer': layer_idx, 'K': K,
            'dead_pct': float(dead / K * 100),
            'mean_entropy': float(np.mean(entropies)),
            'js_mean': float(js.mean()), 'js_std': float(js.std()),
            'js_range': float(js.max() - js.min()),
            'js_min': float(js.min()), 'js_max': float(js.max()),
        }
        print(f"    dead={dead}/{K}, entropy={np.mean(entropies):.4f}", flush=True)
        print(f"    JS μ={js.mean():.4f} σ={js.std():.4f} range={js.max()-js.min():.4f}", flush=True)

# Summary
print(f"\n{'='*90}")
print("HUBERT DIAGNOSTIC SUMMARY")
print(f"{'='*90}")
h = f"{'Config':<22} {'Dead%':>8} {'Entropy':>8} {'JS_μ':>8} {'JS_σ':>8} {'JS_range':>10} {'JS_min':>8} {'JS_max':>8}"
print(h)
print("-" * len(h))
for k in sorted(results.keys()):
    r = results[k]
    print(f"{k:<22} {r['dead_pct']:>7.1f}% {r['mean_entropy']:>8.4f} "
          f"{r['js_mean']:>8.4f} {r['js_std']:>8.4f} {r['js_range']:>10.4f} "
          f"{r['js_min']:>8.4f} {r['js_max']:>8.4f}", flush=True)

with open(f'{OUT_DIR}/hubert_diagnostic_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved!", flush=True)
