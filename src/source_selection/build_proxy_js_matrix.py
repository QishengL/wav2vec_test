"""
Generate JS similarity matrix matching the contrastive model's language order.
Uses layer6, K=500, unigram histograms.
Saves to replace the existing js_sim_matrix.npy used by proxy training.
"""
import os, sys, pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import jensenshannon

OUT_DIR = '/mnt/storage/qisheng/github/wav2vec_test/results/pseudo_units'
CACHE_DIR = f'{OUT_DIR}/cache_layer06'
K = 500

# Must match the order in proxy_scratch.py's lan_list
LAN_ORDER = ['ar', 'ba', 'eu', 'be', 'bn', 'ca', 'yue', 'cs', 'nl', 'en', 'eo', 
             'fa', 'fr', 'ka', 'de', 'hu', 'it', 'ja', 'lv', 'lt', 'pl', 'pt',
             'ro', 'ru', 'uk', 'es', 'sw', 'ta', 'th', 'tt', 'tr', 'ug', 'ur', 
             'uz', 'cy', 'zh-CN']

# Load K-means
with open(f'{OUT_DIR}/kmeans_layer06_K{K}.pkl', 'rb') as f:
    kmeans = pickle.load(f)
print("K-means loaded", flush=True)

# Load cached frames and predict labels, compute histograms
histograms = {}
for lang in LAN_ORDER:
    path = f'{CACHE_DIR}/{lang}_frames.npy'
    if not os.path.exists(path):
        print(f"  [SKIP] {lang} not found", flush=True)
        continue
    frames = np.load(path).astype(np.float32)
    labels = kmeans.predict(frames)
    hist = np.bincount(labels, minlength=K).astype(np.float32)
    hist /= (hist.sum() + 1e-10)
    histograms[lang] = hist
    print(f"  {lang}: {len(frames)} frames", flush=True)

# Build JS matrix in LAN_ORDER
n = len(LAN_ORDER)
js_matrix = np.zeros((n, n))
for i, li in enumerate(LAN_ORDER):
    hi = histograms.get(li)
    if hi is None:
        continue
    for j, lj in enumerate(LAN_ORDER):
        hj = histograms.get(lj)
        if hj is None:
            continue
        js_matrix[i, j] = 1.0 - float(jensenshannon(hi, hj))

print(f"\nJS matrix: {n}×{n}, mean={js_matrix.mean():.4f}, std={js_matrix.std():.4f}", flush=True)

# Save to proxy path (overwrite)
out_path = '/mnt/storage/qisheng/github/wav2vec_test/results/pseudo_units/js_sim_matrix.npy'
np.save(out_path, js_matrix)
print(f"Saved to {out_path}", flush=True)

# Also save a backup with layer info
backup = f'{OUT_DIR}/js_sim_matrix_layer06_K500_unigram.npy'
np.save(backup, js_matrix)
print(f"Backup: {backup}", flush=True)

# Save languages.txt for reference
with open(f'{OUT_DIR}/languages.txt', 'w') as f:
    for l in LAN_ORDER:
        f.write(f'{l}\n')
print("languages.txt updated", flush=True)

# Quick check: lv similarity to everything
lv_idx = LAN_ORDER.index('lv')
print(f"\nlv similarities:")
for i, lang in enumerate(LAN_ORDER):
    print(f"  {lang}: {js_matrix[lv_idx][i]:.4f}")
print("Done!")