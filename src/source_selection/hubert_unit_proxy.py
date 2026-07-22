"""
HuBERT Unit-Proxy: full pipeline for target languages.
- Loads cached HuBERT layer24 features for 36 training languages
- Trains K-means K=500
- Extracts HuBERT layer24 features for 16 target languages
- Computes rankings and Spearman
"""
import os, sys, pickle, json
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

os.environ["HF_HOME"] = "/mnt/storage/ldl_linguistics/hf_home"
CACHE_DATASET = "/mnt/storage/ldl_linguistics/datasets"
OUT_DIR = '/mnt/storage/qisheng/github/wav2vec_test/results/pseudo_units'
CACHE_DIR = f'{OUT_DIR}/cache_hubert_layer24'
K = 500

CANDIDATES = ['ar','ba','ca','cs','en','eo','fr','hu','it','lt','lv','nl','ro','ru','sw','ta','tr','tt','ug']
TARGETS = ['sq','ltg','ur','cy','gn','tn','am','az','mt','af','da','ky','tk','kk','sk','id']

# ── Step 1: Load cached training frames ──
print("Loading HuBERT layer24 cached frames...", flush=True)
train_langs = sorted([f.replace('_frames.npy','') for f in os.listdir(CACHE_DIR) if f.endswith('_frames.npy')])
all_train = []
train_splits = {}
offset = 0
for lang in train_langs:
    f = np.load(f'{CACHE_DIR}/{lang}_frames.npy').astype(np.float32)
    all_train.append(f)
    train_splits[lang] = (offset, offset + len(f))
    offset += len(f)

train_frames = np.concatenate(all_train, axis=0)
del all_train
print(f"Training: {len(train_langs)} langs, {len(train_frames)} frames", flush=True)

# ── Step 2: Train K-means ──
kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=8192, verbose=0)
kmeans.fit(train_frames)
with open(f'{OUT_DIR}/kmeans_hubert_layer24_K{K}.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
print("K-means K=500 trained and saved", flush=True)

# ── Step 3: Training language histograms ──
train_hists = {}
for lang, (start, end) in train_splits.items():
    labels = kmeans.predict(train_frames[start:end])
    hist = np.bincount(labels, minlength=K).astype(np.float32)
    hist /= (hist.sum() + 1e-10)
    train_hists[lang] = hist
    if train_langs.index(lang) % 10 == 0:
        print(f"  [{lang}] histogram computed", flush=True)

# ── Step 4: Extract target features ──
import torch
from transformers import HubertModel, AutoFeatureExtractor
from datasets import load_dataset, Audio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HubertModel.from_pretrained(
    "facebook/hubert-large-ls960-ft", cache_dir=CACHE_DATASET).to(device).eval()
fe = AutoFeatureExtractor.from_pretrained(
    "facebook/hubert-large-ls960-ft", cache_dir=CACHE_DATASET)
print("\nHuBERT model loaded", flush=True)

ALREADY_SAVED = {'ar','be','bg','bn','cs','cy','da','de','el','es','et','fa',
    'fi','hi','hu','it','ja','ka','ko','lt','lv','mk','ml','mn',
    'mr','nl','pl','pt','ro','ru','sk','sl','sr','sw','ta','te',
    'th','tr','uk','ur','vi','en','fr'}

TARGET_HIST_DIR = f'{OUT_DIR}/target_hists_hubert_layer24'
os.makedirs(TARGET_HIST_DIR, exist_ok=True)

target_hists = {}
for tgt in TARGETS:
    cache_path = f'{TARGET_HIST_DIR}/{tgt}_hist.npy'
    if os.path.exists(cache_path):
        target_hists[tgt] = np.load(cache_path)
        print(f"  [{tgt}] loaded from cache", flush=True)
        continue

    ds_name = "fixie-ai/common_voice_17_0" if tgt in ALREADY_SAVED else "fsicoli/common_voice_22_0"
    try:
        ds = load_dataset(ds_name, tgt, split='test', trust_remote_code=True, cache_dir=CACHE_DATASET)
    except:
        print(f"  [{tgt}] SKIP", flush=True)
        continue

    n = min(100, len(ds))
    ds = ds.shuffle(seed=42).select(range(n))
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))

    all_frames = []
    for ex in ds:
        audio = ex['audio']
        inputs = fe(audio['array'], sampling_rate=16000, return_tensors='pt')
        inp = inputs['input_values'].to(device)
        with torch.no_grad():
            outputs = model(inp, output_hidden_states=True)
            frames = outputs.hidden_states[24][0].cpu().numpy()
        all_frames.append(frames)

    if not all_frames:
        continue
    all_frames = np.concatenate(all_frames, axis=0).astype(np.float32)

    labels = kmeans.predict(all_frames)
    hist = np.bincount(labels, minlength=K).astype(np.float32)
    hist /= (hist.sum() + 1e-10)
    target_hists[tgt] = hist
    np.save(cache_path, hist)
    print(f"  [{tgt}] {len(ds)} utts, {len(all_frames)} frames", flush=True)
    del ds

# ── Step 5: Rankings ──
results = {}
for tgt in TARGETS:
    if tgt not in target_hists:
        continue
    t_hist = target_hists[tgt]
    scores = []
    for cand in CANDIDATES:
        if cand not in train_hists:
            continue
        sim = 1.0 - float(jensenshannon(t_hist, train_hists[cand]))
        scores.append((cand, round(sim, 4)))
    scores.sort(key=lambda x: -x[1])
    results[tgt] = scores
    print(f"  HuBERT {tgt} → {scores[0][0]}({scores[0][1]:.4f})", flush=True)

# ── Step 6: Spearman ──
with open('/mnt/storage/qisheng/github/wav2vec_test/results/s2_results.json') as f:
    s2 = json.load(f)

per_lookup = {}
for exp in s2:
    t = exp.get('target_lang','')
    per = exp.get('heldout_test_wer')
    if per is None or 'base' in exp.get('experiment',''): continue
    e = exp.get('experiment','')
    if '_' in e:
        src = e.rsplit('_',1)[-1].replace('53','').replace('base','')
        if '+' not in src and 'multi' not in src and 'direct' not in src and src:
            key = (t, src)
            if key not in per_lookup or per < per_lookup[key]:
                per_lookup[key] = per
extra = [('da','de',0.3734),('cy','th',0.3197),('cy','ca',0.2571),('gn','eu',0.999),
         ('am','ur',0.3006),('sq','pt',0.2439),('ur','bn',0.3825),('ur','sw',0.3895),
         ('gn','ur',0.1870),('mt','ur',0.2924)]
for t,s,p in extra: per_lookup[(t,s)] = p

sims, pers = [], []
for tgt in TARGETS:
    if tgt not in results: continue
    for cand, sim in results[tgt]:
        if (tgt, cand) in per_lookup:
            sims.append(sim)
            pers.append(per_lookup[(tgt, cand)])

r, pv = spearmanr(sims, pers) if len(sims) >= 4 else (0, 1)
sig = '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else 'ns'
print(f"\n{'='*60}")
print(f"HuBERT Unit-Proxy: N={len(sims)}, Spearman r={r:.4f}, p={pv:.4f} {sig}")
print(f"{'='*60}")

# Compare with XLSR-53
print(f"\nComparison:")
print(f"  XLSR-53 layer6_K500 unigram: r=-0.036, p=0.733 ns")
print(f"  HuBERT layer24_K500:         r={r:.4f}, p={pv:.4f} {sig}")

# Save
with open(f'{OUT_DIR}/hubert_unit_proxy_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Compare top-1
print(f"\n{'='*60}")
print("TOP-1: HuBERT vs XLSR-53 vs SupCon")
print(f"{'='*60}")
with open(f'{OUT_DIR}/unit_proxy_comparison.json') as f:
    xlsr = json.load(f)['unigram']
with open('/mnt/storage/qisheng/github/wav2vec_test/results/ablation/contrastive_n=50.json') as f:
    supcon = json.load(f)

print(f"{'Target':<6} {'HuBERT':<12} {'XLSR-53':<12} {'SupCon':<12}")
for t in TARGETS:
    h = results.get(t, [['-']])[0][0]
    x = xlsr.get(t, [['-']])[0][0]
    s = supcon.get(t, [['-']])[0][0]
    print(f"{t:<6} {h:<12} {x:<12} {s:<12}")

print("\nDone!", flush=True)
