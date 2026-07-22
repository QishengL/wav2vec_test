"""
Step 1: Extract XLSR-53 frame-level features + MiniBatch K-means
for pseudo acoustic units (36 training languages).
"""
import os, sys, pickle, argparse
import numpy as np

os.environ["HF_HOME"] = "/mnt/storage/ldl_linguistics/hf_home"
CACHE_DIR = "/mnt/storage/ldl_linguistics/datasets"

LAN_LIST = ['ar', 'ba', 'eu', 'be', 'bn', 'ca', 'yue', 'cs', 'nl', 'en', 'eo',
            'fa', 'fr', 'ka', 'de', 'hu', 'it', 'ja', 'lv', 'lt', 'pl', 'pt',
            'ro', 'ru', 'uk', 'es', 'sw', 'ta', 'th', 'tt', 'tr', 'ug', 'ur',
            'uz', 'cy', 'zh-CN']

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=200)
parser.add_argument('--n_samples', type=int, default=200)
args = parser.parse_args()

K = args.k
N_SAMPLES = args.n_samples
OUT_DIR = '/mnt/storage/qisheng/github/wav2vec_test/results/pseudo_units'
os.makedirs(OUT_DIR, exist_ok=True)
print(f"K={K}, n_samples={N_SAMPLES}", flush=True)

import torch
from transformers import Wav2Vec2Model, AutoFeatureExtractor
from datasets import load_dataset, Audio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", cache_dir=CACHE_DIR).to(device).eval()
fe = AutoFeatureExtractor.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", cache_dir=CACHE_DIR)
print("Model loaded", flush=True)

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=4096, verbose=1)

# ── Phase 1: collect frames and train K-means incrementally ──
lang_info = {}
total_frames = 0

ALREADY_SAVED = ['ar', 'be', 'bg', 'bn', 'cs', 'cy', 'da', 'de', 'el', 'es', 'et', 'fa',
                  'fi', 'hi', 'hu', 'it', 'ja', 'ka', 'ko', 'lt', 'lv', 'mk', 'ml', 'mn',
                  'mr', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'ta', 'te',
                  'th', 'tr', 'uk', 'ur', 'vi','en','fr']

for lang in LAN_LIST:
    ds_name = "fixie-ai/common_voice_17_0" if lang in ALREADY_SAVED else "fsicoli/common_voice_22_0"
    try:
        ds = load_dataset(ds_name, lang, split='train',
                          trust_remote_code=True, cache_dir=CACHE_DIR)
    except:
        print(f"  [{lang}] SKIP", flush=True)
        continue
    
    if len(ds) > N_SAMPLES:
        ds = ds.shuffle(seed=42).select(range(N_SAMPLES))
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))
    
    n_frames = 0
    lang_frames = []
    for ex in ds:
        audio = ex['audio']
        inputs = fe(audio['array'], sampling_rate=16000, return_tensors='pt')
        inp = inputs['input_values'].to(device)
        
        with torch.no_grad():
            hidden = model(inp).last_hidden_state[0].cpu().numpy()
        
        n_frames += len(hidden)
        lang_frames.append(hidden)
    
    # Feed all frames from this language to K-means at once
    if lang_frames:
        all_lang_frames = np.concatenate(lang_frames, axis=0)
        kmeans.partial_fit(all_lang_frames)
    
    lang_info[lang] = {'n_utterances': len(ds), 'n_frames': n_frames, 'dataset': ds}
    total_frames += n_frames
    print(f"  [{lang}] {len(ds)} utts, {n_frames} frames (total: {total_frames})", flush=True)

print(f"\nK-means trained on {total_frames} frames", flush=True)

# ── Phase 2: assign cluster labels & compute histograms ──
utterance_histograms = {}

for lang, info in lang_info.items():
    ds = info['dataset']
    utt_hists = []
    
    for ex in ds:
        audio = ex['audio']
        inputs = fe(audio['array'], sampling_rate=16000, return_tensors='pt')
        inp = inputs['input_values'].to(device)
        
        with torch.no_grad():
            hidden = model(inp).last_hidden_state[0].cpu().numpy()
        
        labels = kmeans.predict(hidden)
        hist = np.bincount(labels, minlength=K).astype(np.float32)
        hist /= (hist.sum() + 1e-10)
        utt_hists.append(hist)
    
    utterance_histograms[lang] = np.array(utt_hists)
    lang_hist = np.mean(utt_hists, axis=0)
    
    np.save(f'{OUT_DIR}/{lang}_utt_hists.npy', utt_hists)
    np.save(f'{OUT_DIR}/{lang}_lang_hist.npy', lang_hist)
    print(f"  [{lang}] {len(utt_hists)} utts, lang_hist saved", flush=True)
    del ds  # free memory

# ── Phase 3: JS similarity matrix ──
from scipy.spatial.distance import jensenshannon

languages = sorted([l for l in lang_info])
n_lang = len(languages)
js_matrix = np.zeros((n_lang, n_lang))

for i, li in enumerate(languages):
    hi = np.load(f'{OUT_DIR}/{li}_lang_hist.npy')
    for j, lj in enumerate(languages):
        hj = np.load(f'{OUT_DIR}/{lj}_lang_hist.npy')
        js_matrix[i, j] = 1.0 - float(jensenshannon(hi, hj))

np.save(f'{OUT_DIR}/js_sim_matrix.npy', js_matrix)
with open(f'{OUT_DIR}/languages.txt', 'w') as f:
    for l in languages:
        f.write(f'{l}\n')

with open(f'{OUT_DIR}/kmeans_K{K}.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

print(f"\nJS matrix: {n_lang}×{n_lang}, mean={js_matrix.mean():.4f}, diag={np.diag(js_matrix).mean():.4f}")
pairs = []
for i in range(n_lang):
    for j in range(i+1, n_lang):
        pairs.append((js_matrix[i,j], languages[i], languages[j]))
pairs.sort(key=lambda x: -x[0])
print("Top 10 similar:")
for sim, li, lj in pairs[:10]:
    print(f"  {li} ↔ {lj}: {sim:.4f}")
print("Bottom 10:")
for sim, li, lj in pairs[-10:]:
    print(f"  {li} ↔ {lj}: {sim:.4f}")

print("Done!", flush=True)
