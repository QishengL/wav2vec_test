"""
Load cached frame features, run K-means with specified K, compute
histograms and JS similarity matrix.
"""
import os, sys, pickle, argparse
import numpy as np

OUT_DIR = '/mnt/storage/qisheng/github/wav2vec_test/results/pseudo_units'
CACHE_DIR = f'{OUT_DIR}/frame_cache'

parser = argparse.ArgumentParser()
parser.add_argument('--k_vals', type=str, default='50,100,200',
                    help='Comma-separated K values')
args = parser.parse_args()
k_vals = [int(k) for k in args.k_vals.split(',')]

from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import jensenshannon

# Load all cached frames
languages = sorted(os.listdir(CACHE_DIR))
languages = [l.replace('_frames.npy', '') for l in languages if l.endswith('_frames.npy')]
print(f"Loaded {len(languages)} languages from cache", flush=True)

all_frames_concat = []
lang_splits = {}
total = 0
for lang in languages:
    frames = np.load(f'{CACHE_DIR}/{lang}_frames.npy').astype(np.float32)
    all_frames_concat.append(frames)
    lang_splits[lang] = (total, total + len(frames))
    total += len(frames)
    
all_frames = np.concatenate(all_frames_concat, axis=0)
print(f"Total frames: {len(all_frames)}", flush=True)

for K in k_vals:
    print(f"\n{'='*60}", flush=True)
    print(f"K = {K}", flush=True)
    print(f"{'='*60}", flush=True)
    
    suffix = f'K{K}'
    
    # Train K-means
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=4096, verbose=0)
    kmeans.fit(all_frames)
    print(f"K-means trained", flush=True)
    
    # Assign labels & compute histograms per language
    for lang, (start, end) in lang_splits.items():
        labels = kmeans.predict(all_frames[start:end])
        hist = np.bincount(labels, minlength=K).astype(np.float32)
        hist /= (hist.sum() + 1e-10)
        np.save(f'{OUT_DIR}/{lang}_lang_hist_{suffix}.npy', hist)
    
    print("Histograms computed", flush=True)
    
    # JS similarity matrix
    n_lang = len(languages)
    js_matrix = np.zeros((n_lang, n_lang))
    for i, li in enumerate(languages):
        hi = np.load(f'{OUT_DIR}/{li}_lang_hist_{suffix}.npy')
        for j, lj in enumerate(languages):
            hj = np.load(f'{OUT_DIR}/{lj}_lang_hist_{suffix}.npy')
            js_matrix[i, j] = 1.0 - float(jensenshannon(hi, hj))
    
    np.save(f'{OUT_DIR}/js_sim_matrix_{suffix}.npy', js_matrix)
    
    # Save kmeans model
    with open(f'{OUT_DIR}/kmeans_K{K}.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    
    print(f"JS matrix: {n_lang}×{n_lang}, mean={js_matrix.mean():.4f}", flush=True)
    
    # Check entropy
    for i, lang in enumerate(languages):
        hist = np.load(f'{OUT_DIR}/{lang}_lang_hist_{suffix}.npy')
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        norm = entropy / np.log(K)
        if norm > 0.95:
            print(f"  WARNING: {lang} uniformity={norm:.3f} (>0.95!)", flush=True)
    
print("\nAll done!", flush=True)
