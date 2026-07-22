"""
Build 3-gram JS matrix for SupConWithProxyLoss.
Uses XLSR-53 layer6, K=500, trigram distributions.
"""
import os, sys, pickle
import numpy as np
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import jensenshannon

os.environ["HF_HOME"] = "/mnt/storage/ldl_linguistics/hf_home"
CACHE_DATASET = "/mnt/storage/ldl_linguistics/datasets"
OUT_DIR = '/mnt/storage/qisheng/github/wav2vec_test/results/pseudo_units'
CACHE_DIR = f'{OUT_DIR}/cache_layer06'
K = 500

# Must match contrastive model's lan_list
LAN_ORDER = ['ar', 'ba', 'eu', 'be', 'bn', 'ca', 'yue', 'cs', 'nl', 'en', 'eo', 
             'fa', 'fr', 'ka', 'de', 'hu', 'it', 'ja', 'lv', 'lt', 'pl', 'pt',
             'ro', 'ru', 'uk', 'es', 'sw', 'ta', 'th', 'tt', 'tr', 'ug', 'ur', 
             'uz', 'cy', 'zh-CN']

import torch
from transformers import Wav2Vec2Model, AutoFeatureExtractor
from datasets import load_dataset, Audio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", cache_dir=CACHE_DATASET).to(device).eval()
fe = AutoFeatureExtractor.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", cache_dir=CACHE_DATASET)
print("Model loaded", flush=True)

ALREADY_SAVED = {'ar','be','bg','bn','cs','cy','da','de','el','es','et','fa',
    'fi','hi','hu','it','ja','ka','ko','lt','lv','mk','ml','mn',
    'mr','nl','pl','pt','ro','ru','sk','sl','sr','sw','ta','te',
    'th','tr','uk','ur','vi','en','fr'}

# Load K-means
kmeans_path = f'{OUT_DIR}/kmeans_layer06_K{K}.pkl'
if os.path.exists(kmeans_path):
    with open(kmeans_path, 'rb') as f:
        kmeans = pickle.load(f)
else:
    print("Training K-means...", flush=True)
    all_frames_list = []
    for lang in LAN_ORDER:
        f = np.load(f'{CACHE_DIR}/{lang}_frames.npy').astype(np.float32)
        all_frames_list.append(f)
    all_frames = np.concatenate(all_frames_list, axis=0)
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=8192)
    kmeans.fit(all_frames)
    with open(kmeans_path, 'wb') as f:
        pickle.dump(kmeans, f)
print("K-means ready", flush=True)

# Extract per-utterance sequences
def get_sequences(lang, n_samples=200):
    ds_name = "fixie-ai/common_voice_17_0" if lang in ALREADY_SAVED else "fsicoli/common_voice_22_0"
    try:
        ds = load_dataset(ds_name, lang, split='train', trust_remote_code=True, cache_dir=CACHE_DATASET)
    except:
        ds = load_dataset("fixie-ai/common_voice_17_0", lang, split='train',
                          trust_remote_code=True, cache_dir=CACHE_DATASET)
    if len(ds) > n_samples:
        ds = ds.shuffle(seed=42).select(range(n_samples))
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))
    
    seqs = []
    for ex in ds:
        audio = ex['audio']
        inputs = fe(audio['array'], sampling_rate=16000, return_tensors='pt')
        inp = inputs['input_values'].to(device)
        with torch.no_grad():
            outputs = model(inp, output_hidden_states=True)
            frames = outputs.hidden_states[6][0].cpu().numpy()
        labels = kmeans.predict(frames)
        seqs.append(labels.tolist())
    return seqs

# Compute 3-gram distributions for all 36 languages
print("\nComputing 3-gram distributions...", flush=True)
N = 3
all_seqs = {}
for lang in LAN_ORDER:
    seqs = get_sequences(lang, n_samples=200)
    all_seqs[lang] = seqs
    print(f"  {lang}: {len(seqs)} utts", flush=True)

# Global trigram vocabulary
global_counter = Counter()
for lang in LAN_ORDER:
    for seq in all_seqs[lang]:
        for i in range(len(seq) - N + 1):
            global_counter[tuple(seq[i:i+N])] += 1

# Cap vocabulary (top 50000)
top_ngrams = [ng for ng, _ in global_counter.most_common(50000)]
ng_to_idx = {ng: i for i, ng in enumerate(top_ngrams)}
vocab_size = len(top_ngrams)
print(f"Trigram vocabulary: {vocab_size} types", flush=True)

# Build distributions
dists = {}
for lang in LAN_ORDER:
    counter = Counter()
    for seq in all_seqs[lang]:
        for i in range(len(seq) - N + 1):
            counter[tuple(seq[i:i+N])] += 1
    hist = np.zeros(vocab_size, dtype=np.float32)
    for ng, cnt in counter.items():
        if ng in ng_to_idx:
            hist[ng_to_idx[ng]] = cnt
    total = hist.sum()
    if total > 0:
        hist /= total
    else:
        hist[:] = 1.0 / vocab_size
    dists[lang] = hist

# JS matrix in LAN_ORDER
n = len(LAN_ORDER)
js_matrix = np.zeros((n, n))
for i, li in enumerate(LAN_ORDER):
    hi = dists[li]
    for j, lj in enumerate(LAN_ORDER):
        hj = dists[lj]
        js_matrix[i, j] = 1.0 - float(jensenshannon(hi, hj))

print(f"\nJS matrix: {n}×{n}, mean={js_matrix.mean():.4f}, std={js_matrix.std():.4f}", flush=True)

# Save
out_path = f'{OUT_DIR}/js_sim_matrix.npy'
np.save(out_path, js_matrix)
print(f"Saved: {out_path}", flush=True)

# Also backup
backup = f'{OUT_DIR}/js_sim_matrix_layer06_K500_3gram.npy'
np.save(backup, js_matrix)
print(f"Backup: {backup}", flush=True)

# languages.txt
with open(f'{OUT_DIR}/languages.txt', 'w') as f:
    for l in LAN_ORDER:
        f.write(f'{l}\n')

print("Done!", flush=True)
