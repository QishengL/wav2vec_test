"""
Full Unit-Proxy comparison for 16 target languages:
  - layer6, K=500
  - 3 methods: unigram / TF-IDF / bigram
Records rankings and stability.
"""
import os, sys, pickle, json
import numpy as np
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import jensenshannon

os.environ["HF_HOME"] = "/mnt/storage/ldl_linguistics/hf_home"
CACHE_DIR_DATASET = "/mnt/storage/ldl_linguistics/datasets"
OUT_DIR = '/mnt/storage/qisheng/github/wav2vec_test/results/pseudo_units'
CACHE_DIR = f'{OUT_DIR}/cache_layer06'
K = 500

CANDIDATES = ['ar','ba','ca','cs','en','eo','fr','hu','it','lt','lv','nl','ro','ru','sw','ta','tr','tt','ug']
TARGETS = ['sq','ltg','ur','cy','gn','tn','am','az','mt','af','da','ky','tk','kk','sk','id']
N_SAMPLES = 100

# ── 1. Load pre-computed candidate histograms from layer6_K500 ──
# We need to re-compute since compare_histogram_methods didn't save them
print("Loading cached frame features...", flush=True)
lang_files = sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('_frames.npy')])
langs = [f.replace('_frames.npy','') for f in lang_files]
all_frames_list = []
lang_splits = {}
offset = 0
for lang in langs:
    f = np.load(f'{CACHE_DIR}/{lang}_frames.npy').astype(np.float32)
    all_frames_list.append(f)
    lang_splits[lang] = (offset, offset + len(f))
    offset += len(f)

all_frames = np.concatenate(all_frames_list, axis=0)
del all_frames_list
print(f"Loaded {len(langs)} languages, {len(all_frames)} frames", flush=True)

# Load K-means (or train if not saved)
kmeans_path = f'{OUT_DIR}/kmeans_layer06_K{K}.pkl'
if os.path.exists(kmeans_path):
    with open(kmeans_path, 'rb') as f:
        kmeans = pickle.load(f)
    print("K-means loaded from disk", flush=True)
else:
    print("Training K-means K=500...", flush=True)
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=8192, verbose=0)
    kmeans.fit(all_frames)
    with open(kmeans_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print("K-means trained and saved", flush=True)

# ── 2. Compute candidate histograms for all 3 methods ──
print("Computing candidate histograms...", flush=True)
cand_unigram = {}
cand_tfidf = {}
cand_bigram = {}

# Unigram + bigram per candidate
all_bigram_global = Counter()
unigram_cts = {}

for lang in langs:
    start, end = lang_splits[lang]
    labels = kmeans.predict(all_frames[start:end])
    
    ct = Counter(labels)
    unigram_cts[lang] = ct
    
    bg = Counter()
    for i in range(len(labels) - 1):
        bg[(labels[i], labels[i+1])] += 1
    all_bigram_global.update(bg)
    
    if langs.index(lang) % 10 == 0:
        print(f"  {lang} done", flush=True)

# Unigram histograms
for lang in langs:
    ct = unigram_cts[lang]
    total = sum(ct.values())
    hist = np.zeros(K, dtype=np.float32)
    for c, cnt in ct.items():
        hist[c] = cnt
    hist /= (total + 1e-10)
    cand_unigram[lang] = hist

# TF-IDF weights
N = len(langs)
cluster_presence = np.zeros(K)
for lang in langs:
    nonzero = np.where(cand_unigram[lang] > 1e-10)[0]
    cluster_presence[nonzero] += 1
idf = np.log(1 + N / (1 + cluster_presence))

for lang in langs:
    weighted = cand_unigram[lang] * idf
    weighted /= (weighted.sum() + 1e-10)
    cand_tfidf[lang] = weighted

# Bigram vocabulary (top 20000)
top_bigrams = [bg for bg, _ in all_bigram_global.most_common(20000)]
bg_to_idx = {bg: i for i, bg in enumerate(top_bigrams)}
N_BG = len(top_bigrams)

for lang in langs:
    start, end = lang_splits[lang]
    labels = kmeans.predict(all_frames[start:end])
    bg = Counter()
    for i in range(len(labels) - 1):
        bg[(labels[i], labels[i+1])] += 1
    total = sum(bg.values())
    hist = np.zeros(N_BG, dtype=np.float32)
    for b, cnt in bg.items():
        if b in bg_to_idx:
            hist[bg_to_idx[b]] = cnt
    hist /= (total + 1e-10)
    cand_bigram[lang] = hist

print("Candidate histograms ready", flush=True)
print(f"  Unigram dim={K}, Bigram dim={N_BG}", flush=True)

# ── 3. Extract target language features & compute histograms ──
print("\nLoading model for target language features...", flush=True)
import torch
from transformers import Wav2Vec2Model, AutoFeatureExtractor
from datasets import load_dataset, Audio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", cache_dir=CACHE_DIR_DATASET).to(device).eval()
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", cache_dir=CACHE_DIR_DATASET)
print("Model loaded", flush=True)

ALREADY_SAVED = ['ar', 'be', 'bg', 'bn', 'cs', 'cy', 'da', 'de', 'el', 'es', 'et', 'fa',
                 'fi', 'hi', 'hu', 'it', 'ja', 'ka', 'ko', 'lt', 'lv', 'mk', 'ml', 'mn',
                 'mr', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'ta', 'te',
                 'th', 'tr', 'uk', 'ur', 'vi','en','fr']

TARGET_HIST_DIR = f'{OUT_DIR}/target_hists_layer06'
os.makedirs(TARGET_HIST_DIR, exist_ok=True)

target_hist_cache = {}  # method -> target -> histogram

for method_name in ['unigram', 'tfidf', 'bigram']:
    target_hist_cache[method_name] = {}

for tgt in TARGETS:
    # Check if cached
    cache_path = f'{TARGET_HIST_DIR}/{tgt}_histograms.npz'
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        for method_name in ['unigram', 'tfidf', 'bigram']:
            if method_name in data:
                target_hist_cache[method_name][tgt] = data[method_name]
        print(f"  [{tgt}] loaded from cache", flush=True)
        continue

    ds_name = "fixie-ai/common_voice_17_0" if tgt in ALREADY_SAVED else "fsicoli/common_voice_22_0"
    try:
        ds = load_dataset(ds_name, tgt, split='test',
                          trust_remote_code=True, cache_dir=CACHE_DIR_DATASET)
    except:
        try:
            ds = load_dataset('fixie-ai/common_voice_17_0', tgt, split='test',
                              trust_remote_code=True, cache_dir=CACHE_DIR_DATASET)
        except:
            print(f"  [{tgt}] SKIP (no dataset)", flush=True)
            continue

    if len(ds) > N_SAMPLES:
        ds = ds.shuffle(seed=42).select(range(N_SAMPLES))
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))

    all_frames_tgt = []
    for ex in ds:
        audio = ex['audio']
        inputs = feature_extractor(audio['array'], sampling_rate=16000, return_tensors='pt')
        inp = inputs['input_values'].to(device)
        with torch.no_grad():
            outputs = model(inp, output_hidden_states=True)
            hs = outputs.hidden_states[6][0].cpu().numpy()  # layer6
        all_frames_tgt.append(hs)

    if not all_frames_tgt:
        print(f"  [{tgt}] SKIP (no frames)", flush=True)
        continue
    all_frames_tgt = np.concatenate(all_frames_tgt, axis=0).astype(np.float32)

    # Predict labels
    labels = kmeans.predict(all_frames_tgt)
    ct = Counter(labels)
    total = len(labels)

    # Unigram
    unigram_hist = np.zeros(K, dtype=np.float32)
    for c, cnt in ct.items():
        unigram_hist[c] = cnt
    unigram_hist /= (total + 1e-10)
    target_hist_cache['unigram'][tgt] = unigram_hist

    # TF-IDF
    tfidf_hist = unigram_hist * idf
    tfidf_hist /= (tfidf_hist.sum() + 1e-10)
    target_hist_cache['tfidf'][tgt] = tfidf_hist

    # Bigram
    bg = Counter()
    for i in range(len(labels) - 1):
        bg[(labels[i], labels[i+1])] += 1
    total_bg = sum(bg.values())
    bigram_hist = np.zeros(N_BG, dtype=np.float32)
    for b, cnt in bg.items():
        if b in bg_to_idx:
            bigram_hist[bg_to_idx[b]] = cnt
    bigram_hist /= (total_bg + 1e-10)
    target_hist_cache['bigram'][tgt] = bigram_hist

    # Save cache
    save_dict = {
        'unigram': unigram_hist,
        'tfidf': tfidf_hist,
        'bigram': bigram_hist,
    }
    np.savez_compressed(cache_path, **save_dict)
    print(f"  [{tgt}] {len(ds)} utts, {total} frames → saved", flush=True)
    del ds

print(f"\nTarget languages processed: {[t for t in TARGETS if t in target_hist_cache['unigram']]}", flush=True)

# ── 4. Rank candidates for each target, each method ──
print(f"\n{'='*80}", flush=True)
print("UNIT-PROXY RANKINGS (layer6, K=500)", flush=True)
print(f"{'='*80}", flush=True)

results = {}
for method_name, cand_hists in [('unigram', cand_unigram), ('tfidf', cand_tfidf), ('bigram', cand_bigram)]:
    print(f"\n--- {method_name.upper()} ---", flush=True)
    method_results = {}
    for tgt in TARGETS:
        if tgt not in target_hist_cache[method_name]:
            continue
        t_hist = target_hist_cache[method_name][tgt]
        scores = []
        for cand in CANDIDATES:
            if cand not in cand_hists:
                continue
            sim = 1.0 - float(jensenshannon(t_hist, cand_hists[cand]))
            scores.append((cand, round(sim, 4)))
        scores.sort(key=lambda x: -x[1])
        method_results[tgt] = scores
        print(f"  {tgt} → {scores[0][0]}({scores[0][1]:.4f})", flush=True)
    results[method_name] = method_results

# ── 5. Cross-method agreement ──
print(f"\n{'='*80}", flush=True)
print("CROSS-METHOD AGREEMENT", flush=True)
print(f"{'='*80}", flush=True)

methods = ['unigram', 'tfidf', 'bigram']
for i, m1 in enumerate(methods):
    for m2 in methods[i+1:]:
        match = 0
        total = 0
        for tgt in TARGETS:
            if tgt in results.get(m1, {}) and tgt in results.get(m2, {}):
                r1 = results[m1][tgt][0][0]
                r2 = results[m2][tgt][0][0]
                if r1 == r2:
                    match += 1
                total += 1
        if total > 0:
            print(f"  {m1} vs {m2}: {match}/{total} agree ({match/total*100:.1f}%)", flush=True)

# Save
out_path = f'{OUT_DIR}/unit_proxy_comparison.json'
# Convert to serializable format
serializable = {}
for method_name, method_results in results.items():
    serializable[method_name] = {}
    for tgt, scores in method_results.items():
        serializable[method_name][tgt] = [(c, s) for c, s in scores]
with open(out_path, 'w') as f:
    json.dump(serializable, f, indent=2)
print(f"\nSaved to {out_path}", flush=True)
print("Done!", flush=True)
