"""
Sequence matching via n-gram frequency distributions.
Uses existing XLSR-53 layer6 K=500 pseudo-unit labels.
Compares unigram / bigram / trigram / 4-gram / 5-gram.
"""
import os, sys, pickle, json
import numpy as np
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

OUT_DIR = '/mnt/storage/qisheng/github/wav2vec_test/results/pseudo_units'
CACHE_DIR = f'{OUT_DIR}/cache_layer06'
K = 500
N_TOP_NGRAMS = 50000  # vocabulary cap for higher n-grams

CANDIDATES = ['ar','ba','ca','cs','en','eo','fr','hu','it','lt','lv','nl','ro','ru','sw','ta','tr','tt','ug']
TARGETS = ['sq','ltg','ur','cy','gn','tn','am','az','mt','af','da','ky','tk','kk','sk','id']

# ── Load frames ──
print("Loading cached frames...", flush=True)
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
print(f"{len(langs)} langs, {len(all_frames)} frames", flush=True)

# ── Load K-means ──
kmeans_path = f'{OUT_DIR}/kmeans_layer06_K{K}.pkl'
if os.path.exists(kmeans_path):
    with open(kmeans_path, 'rb') as f:
        kmeans = pickle.load(f)
    print("K-means loaded", flush=True)
else:
    print("Training K-means...", flush=True)
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=8192)
    kmeans.fit(all_frames)

# ── Per-language sequences ──
print("Predicting labels per utterance...", flush=True)
# We need per-utterance sequences, not per-language aggregate
# The frame cache doesn't preserve utterance boundaries
# We need to infer them from the original dataset loading
# For now: use the language-level frames but split into synthetic utterances
# A better approach: reload datasets briefly to get utterance boundaries

# Load datasets to get per-utterance sequences (do this once per language)
print("Loading datasets for per-utterance sequences...", flush=True)
from datasets import load_dataset, Audio
CACHE_DIR_D = "/mnt/storage/ldl_linguistics/datasets"
ALREADY_SAVED = ['ar','be','bg','bn','cs','cy','da','de','el','es','et','fa',
    'fi','hi','hu','it','ja','ka','ko','lt','lv','mk','ml','mn',
    'mr','nl','pl','pt','ro','ru','sk','sl','sr','sw','ta','te',
    'th','tr','uk','ur','vi','en','fr']

N_SAMPLES = 100  # fewer to keep memory manageable

# Load model once
import torch
from transformers import Wav2Vec2Model, AutoFeatureExtractor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53", cache_dir=CACHE_DIR_D).to(device).eval()
fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53", cache_dir=CACHE_DIR_D)

def get_sequences(lang, n_samples=100):
    """Get per-utterance pseudo-unit sequences for a language."""
    ds_name = "fixie-ai/common_voice_17_0" if lang in ALREADY_SAVED else "fsicoli/common_voice_22_0"
    try:
        ds = load_dataset(ds_name, lang, split='train', trust_remote_code=True, cache_dir=CACHE_DIR_D)
    except:
        try:
            ds = load_dataset("fixie-ai/common_voice_17_0", lang, split='train',
                              trust_remote_code=True, cache_dir=CACHE_DIR_D)
        except:
            print(f"  [{lang}] SKIP (no dataset)", flush=True)
            return None

    if len(ds) > n_samples:
        ds = ds.shuffle(seed=42).select(range(n_samples))
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))

    sequences = []
    for ex in ds:
        audio = ex['audio']
        inputs = fe(audio['array'], sampling_rate=16000, return_tensors='pt')
        inp = inputs['input_values'].to(device)
        with torch.no_grad():
            outputs = model(inp, output_hidden_states=True)
            frames = outputs.hidden_states[6][0].cpu().numpy()
        labels = kmeans.predict(frames)
        sequences.append(labels.tolist())

    print(f"  [{lang}] {len(sequences)} utts, {sum(len(s) for s in sequences)} frames", flush=True)
    return sequences

# Process all languages for sequences first
all_langs = list(dict.fromkeys(TARGETS + langs))
seqs_per_lang = {}
print(f"Extracting sequences for {len(all_langs)} languages...", flush=True)
for lang in all_langs:
    if lang in seqs_per_lang:
        continue
    try:
        seqs = get_sequences(lang, n_samples=100)
        if seqs:
            seqs_per_lang[lang] = seqs
    except Exception as e:
        print(f"  [{lang}] ERROR: {e}", flush=True)

print(f"\n{len(seqs_per_lang)} languages with sequences", flush=True)

# Target sequences (must be before n-gram computation)
for tgt in TARGETS:
    if tgt in seqs_per_lang:
        continue
    seqs = get_sequences(tgt, n_samples=100)
    if seqs:
        seqs_per_lang[tgt] = seqs
        print(f"  {tgt}: {len(seqs)} utts", flush=True)

# ── N-gram computation ──
def extract_ngrams(sequences, n):
    """Extract n-gram counts from a list of sequences."""
    counter = Counter()
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            counter[tuple(seq[i:i+n])] += 1
    return counter

all_ngrams = {}
for n_val in [1, 2, 3, 4, 5]:
    name = f'{n_val}gram'
    print(f"\nComputing {name}...", flush=True)
    
    # Global vocabulary (use all languages with sequences)
    all_langs_with_seqs = list(seqs_per_lang.keys())
    global_counter = Counter()
    for lang in all_langs_with_seqs:
        if lang not in seqs_per_lang:
            continue
        counter = extract_ngrams(seqs_per_lang[lang], n_val)
        global_counter.update(counter)
    
    # Cap vocabulary
    top_ngrams = [ng for ng, _ in global_counter.most_common(N_TOP_NGRAMS if n_val > 1 else K)]
    ng_to_idx = {ng: i for i, ng in enumerate(top_ngrams)}
    vocab_size = len(top_ngrams)
    print(f"  Vocabulary: {vocab_size} types (from {len(global_counter)} total)", flush=True)
    
    # Build distributions
    dists = {}
    for lang in all_langs_with_seqs:
        if lang not in seqs_per_lang:
            continue
        counter = extract_ngrams(seqs_per_lang[lang], n_val)
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
    
    # JS matrix
    n_lang = len(all_langs_with_seqs)
    js = np.zeros((n_lang, n_lang))
    for i, li in enumerate(all_langs_with_seqs):
        if li not in dists:
            continue
        for j, lj in enumerate(all_langs_with_seqs):
            if lj not in dists:
                continue
            js[i, j] = 1.0 - float(jensenshannon(dists[li], dists[lj]))
    
    # Stats
    triu = js[np.triu_indices(n_lang, k=1)]
    valid = triu[~np.isnan(triu)]
    print(f"  JS μ={valid.mean():.4f} σ={valid.std():.4f} range={valid.max()-valid.min():.4f}", flush=True)
    
    all_ngrams[name] = {
        'distributions': dists,
        'js_matrix': js,
        'vocab_size': vocab_size,
        'js_mean': float(valid.mean()),
        'js_std': float(valid.std()),
        'js_range': float(valid.max() - valid.min()),
    }

# ── Unit-Proxy rankings & Spearman ──
# Compute rankings and Spearman for each n-gram setting
with open('/mnt/storage/qisheng/github/wav2vec_test/results/s2_results.json') as f:
    s2 = json.load(f)
per_lookup = {}
for exp in s2:
    t = exp.get('target_lang', '')
    per = exp.get('heldout_test_wer')
    if per is None or 'base' in exp.get('experiment', ''): continue
    e = exp.get('experiment', '')
    if '_' in e:
        src = e.rsplit('_', 1)[-1].replace('53','').replace('base','')
        if '+' not in src and 'multi' not in src and 'direct' not in src and src:
            key = (t, src)
            if key not in per_lookup or per < per_lookup[key]:
                per_lookup[key] = per
extra = [('da','de',0.3734),('cy','th',0.3197),('cy','ca',0.2571),('gn','eu',0.999),
         ('am','ur',0.3006),('sq','pt',0.2439),('ur','bn',0.3825),('ur','sw',0.3895),
         ('gn','ur',0.1870),('mt','ur',0.2924)]
for t,s,p in extra:
    per_lookup[(t,s)] = p

# Need target language sequences too - compute for TARGETS
# Compute rankings and Spearman for each n-gram setting
for name, data in all_ngrams.items():
    dists = data['distributions']
    
    # Rankings
    scores_list, pers_list = [], []
    for tgt in TARGETS:
        if tgt not in dists:
            continue
        t_dist = dists[tgt]
        for cand in CANDIDATES:
            if cand not in dists:
                continue
            sim = 1.0 - float(jensenshannon(t_dist, dists[cand]))
            if (tgt, cand) in per_lookup:
                scores_list.append(sim)
                pers_list.append(per_lookup[(tgt, cand)])
    
    if len(scores_list) >= 4:
        r, p = spearmanr(scores_list, pers_list)
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    else:
        r, p = float('nan'), float('nan')
        sig = 'N/A'
    
    data['spearman_r'] = float(r)
    data['spearman_p'] = float(p)
    data['n_pairs'] = len(scores_list)
    print(f"\n{name}: Spearman r={r:.4f}, p={p:.4f} {sig} (N={len(scores_list)})", flush=True)

# ── Summary table ──
print(f"\n{'='*70}")
print("N-GRAM COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"{'Method':<12} {'Vocab':>8} {'JS_μ':>8} {'JS_σ':>8} {'JS_range':>10} {'Spearman r':>10} {'p-value':>10} {'N':>6}")
print("-" * 72)
for name in ['1gram', '2gram', '3gram', '4gram', '5gram']:
    d = all_ngrams[name]
    sig = ''
    if d['spearman_p'] < 0.001: sig = '***'
    elif d['spearman_p'] < 0.01: sig = '**'
    elif d['spearman_p'] < 0.05: sig = '*'
    else: sig = 'ns'
    print(f"{name:<12} {d['vocab_size']:>8} {d['js_mean']:>8.4f} {d['js_std']:>8.4f} "
          f"{d['js_range']:>10.4f} {d['spearman_r']:>+10.4f} {d['spearman_p']:>8.4f}{sig:>2} {d['n_pairs']:>6}")

# Save
save_data = {name: {
    'vocab_size': d['vocab_size'],
    'js_mean': d['js_mean'], 'js_std': d['js_std'], 'js_range': d['js_range'],
    'spearman_r': d['spearman_r'], 'spearman_p': d['spearman_p'], 'n_pairs': d['n_pairs']
} for name, d in all_ngrams.items()}
with open(f'{OUT_DIR}/ngram_comparison.json', 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nSaved!", flush=True)
