"""
Compute Unit-Proxy histograms for target languages using existing K-means model.
Then rank candidates by JS similarity.
"""
import os, sys, pickle, json, numpy as np
from scipy.spatial.distance import jensenshannon

os.environ["HF_HOME"] = "/mnt/storage/ldl_linguistics/hf_home"
CACHE_DIR = "/mnt/storage/ldl_linguistics/datasets"

PU_DIR = '/mnt/storage/qisheng/github/wav2vec_test/results/pseudo_units'
KMEANS_PATH = f'{PU_DIR}/kmeans_K200.pkl'
JS_MATRIX_PATH = f'{PU_DIR}/js_sim_matrix.npy'
LANG_LIST_PATH = f'{PU_DIR}/languages.txt'

TARGETS = ['sq','ltg','ur','cy','gn','tn','am','az',
           'mt','af','da','ky','tk','kk','sk','id']
CANDIDATES = ['ar','ba','ca','cs','en','eo','fr','hu','it','lt','lv',
              'nl','ro','ru','sw','ta','tr','tt','ug']
N_SAMPLES = 100
SEED = 42

# ── Load K-means ──
with open(KMEANS_PATH, 'rb') as f:
    kmeans = pickle.load(f)
K = kmeans.n_clusters
print(f"K-means loaded: K={K}", flush=True)

# ── Load known language histograms ──
with open(LANG_LIST_PATH) as f:
    known_langs = [l.strip() for l in f]
known_hists = {}
for lang in known_langs:
    path = f'{PU_DIR}/{lang}_lang_hist.npy'
    if os.path.exists(path):
        known_hists[lang] = np.load(path)
print(f"Loaded {len(known_hists)} known language histograms", flush=True)

# ── Load XLSR-53 ──
import torch
from transformers import Wav2Vec2Model, AutoFeatureExtractor
from datasets import load_dataset, Audio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", cache_dir=CACHE_DIR).to(device).eval()
fe = AutoFeatureExtractor.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", cache_dir=CACHE_DIR)
print("XLSR-53 loaded", flush=True)

# ── Compute target language histograms ──
target_hists = {}
for tgt in TARGETS:
    try:
        ds = load_dataset('fsicoli/common_voice_22_0', tgt, split='test',
                          trust_remote_code=True, cache_dir=CACHE_DIR)
    except:
        try:
            ds = load_dataset('fixie-ai/common_voice_17_0', tgt, split='test',
                              trust_remote_code=True, cache_dir=CACHE_DIR)
        except:
            print(f"  [{tgt}] SKIP", flush=True)
            continue
    
    if len(ds) > N_SAMPLES:
        ds = ds.shuffle(seed=SEED).select(range(N_SAMPLES))
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))
    
    all_frames = []
    for ex in ds:
        audio = ex['audio']
        inputs = fe(audio['array'], sampling_rate=16000, return_tensors='pt')
        inp = inputs['input_values'].to(device)
        with torch.no_grad():
            hidden = model(inp).last_hidden_state[0].cpu().numpy()
        all_frames.append(hidden)
    
    if not all_frames:
        continue
    all_frames_np = np.concatenate(all_frames, axis=0)
    labels = kmeans.predict(all_frames_np)
    hist = np.bincount(labels, minlength=K).astype(np.float32)
    hist /= (hist.sum() + 1e-10)
    target_hists[tgt] = hist
    print(f"  [{tgt}] {len(ds)} utts, {len(all_frames_np)} frames → histogram saved", flush=True)

# ── Rank candidates for each target ──
results = {}
for tgt, t_hist in target_hists.items():
    scores = []
    for cand, c_hist in known_hists.items():
        if cand not in CANDIDATES:
            continue
        sim = 1.0 - float(jensenshannon(t_hist, c_hist))
        scores.append((cand, round(sim, 4)))
    scores.sort(key=lambda x: -x[1])
    results[tgt] = scores
    print(f"  Unit-Proxy {tgt} → {scores[0][0]}({scores[0][1]:.4f})", flush=True)

# ── Compare with SupCon ──
with open(f'{PU_DIR}/../ablation/contrastive_n=50.json') as f:
    supcon = json.load(f)

print(f"\n{'='*70}")
print(f"COMPARISON: Unit-Proxy vs SupCon")
print(f"{'='*70}")
print(f"{'Target':<6} {'Unit-Proxy':<22} {'SupCon':<22} {'Match'}")
print("-" * 56)
match = 0
for t in TARGETS:
    up = results.get(t)
    sc = supcon.get(t)
    if up and sc:
        m = '✓' if up[0][0] == sc[0][0] else '✗'
        if m == '✓': match += 1
        print(f"{t:<6} {up[0][0]}({up[0][1]:.4f}){'':>10} {sc[0][0]}({sc[0][1]:.4f}){'':>10} {m}")
    elif up:
        print(f"{t:<6} {up[0][0]}({up[0][1]:.4f}){'':>10} {'—':<22} —")
    else:
        print(f"{t:<6} {'—':<22} {'—':<22} —")
print(f"\nAgreement: {match}/16")

# Save
out_path = f'{PU_DIR}/unit_proxy_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out_path}")
print("Done!")
