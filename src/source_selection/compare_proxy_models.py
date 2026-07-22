"""
Compare source selection across 3 contrastive models:
  - FEAT128BS16_WAVE (original SupCon)
  - FEAT128BS16_PROXY_SCRATCH (SupCon+Proxy from scratch)
  - FEAT128BS16_PROXY_CONTINUE (SupCon→SupCon+Proxy finetune)

Outputs: top-1, top-3, Spearman correlation for each.
"""
import sys, json, os, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

os.environ["HF_HOME"] = "/mnt/storage/ldl_linguistics/hf_home"
CACHE_DIR = "/mnt/storage/ldl_linguistics/datasets"

# ── Config ──
CKPTS = {
    'SupCon (original)': '/mnt/storage/qisheng/github/wav2vec_contrastive/weights/FEAT128BS16_WAVE/checkpoint_epoch_50.pt',
    'SupCon+Proxy (scratch)': '/mnt/storage/qisheng/github/wav2vec_contrastive/weights/FEAT128BS16_PROXY_SCRATCH/checkpoint_epoch_50.pt',
    'SupCon+Proxy (continue)': '/mnt/storage/qisheng/github/wav2vec_contrastive/weights/FEAT128BS16_PROXY_CONTINUE/checkpoint_epoch_50.pt',
}

# Unit-Proxy Only baseline: directly use JS similarity from pseudo units
JS_MATRIX_PATH = '/mnt/storage/qisheng/github/wav2vec_test/results/pseudo_units/js_sim_matrix.npy'
LANG_LIST_PATH = '/mnt/storage/qisheng/github/wav2vec_test/results/pseudo_units/languages.txt'

TARGETS = ['sq','ltg','ur','cy','gn','tn','am','az',
           'mt','af','da','ky','tk','kk','sk','id']
CANDIDATES = ['ar','ba','ca','cs','en','eo','fr','hu','it','lt','lv',
              'nl','ro','ru','sw','ta','tr','tt','ug']
N_SAMPLES_TARGET = 50  # use 50 test utterances
N_SAMPLES_CANDIDATE = 50  # use 50 train utterances
SEED = 42

import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Config
from datasets import load_dataset
import datasets as hf_ds

sys.path.insert(0, "/mnt/storage/qisheng/github/wav2vec_contrastive/customized")
from model import Wav2Vec2ForContrastiveLearning

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Load JS similarity (Unit-Proxy baseline) ──
js_matrix = np.load(JS_MATRIX_PATH)
with open(LANG_LIST_PATH) as f:
    js_langs = [l.strip() for l in f]
js_lang_to_idx = {l: i for i, l in enumerate(js_langs)}
print(f"JS matrix: {js_matrix.shape}, langs: {len(js_langs)}", flush=True)

# ── Feature extractor ──
fe = AutoFeatureExtractor.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", trust_remote_code=True, cache_dir=CACHE_DIR)

def get_embeddings(model, lang, n=N_SAMPLES_TARGET, split='test'):
    """Extract contrastive embeddings for a language."""
    try:
        ds = load_dataset('fsicoli/common_voice_22_0', lang, split=split,
                          trust_remote_code=True, cache_dir=CACHE_DIR)
    except:
        try:
            ds = load_dataset('fixie-ai/common_voice_17_0', lang, split=split,
                              trust_remote_code=True, cache_dir=CACHE_DIR)
        except:
            return None
    if len(ds) > n:
        ds = ds.shuffle(seed=SEED).select(range(n))
    ds = ds.cast_column('audio', hf_ds.Audio(sampling_rate=16000))
    
    model.eval()
    embs = []
    for ex in ds:
        audio = ex['audio']
        inputs = fe(audio['array'], sampling_rate=16000, return_tensors='pt')
        inp = inputs['input_values'].to(device)
        attn = inputs.get('attention_mask')
        if attn is not None: attn = attn.to(device)
        with torch.no_grad():
            feat = model(inp, attn).squeeze().cpu().numpy()
        feat = feat / (np.linalg.norm(feat) + 1e-10)
        embs.append(feat)
    return np.array(embs) if embs else None

# ── Unit-Proxy baseline: language-level JS similarity ──
def get_unit_proxy_ranking(tgt):
    """Rank candidates by pseudo-unit JS similarity."""
    if tgt not in js_lang_to_idx:
        return None
    tgt_idx = js_lang_to_idx[tgt]
    scores = {}
    for cand in CANDIDATES:
        if cand in js_lang_to_idx:
            cand_idx = js_lang_to_idx[cand]
            sim = js_matrix[tgt_idx, cand_idx]
            scores[cand] = sim
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked

# ── Main comparison ──
results = {}

for model_name, ckpt_path in CKPTS.items():
    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"Checkpoint: {ckpt_path}", flush=True)
    print(f"{'='*60}", flush=True)
    
    if not os.path.exists(ckpt_path):
        print(f"  CHECKPOINT NOT FOUND, SKIPPING", flush=True)
        continue
    
    # Load model
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2ForContrastiveLearning.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53", config=config, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device).eval()
    print(f"  Model loaded", flush=True)
    
    # Candidate embeddings (from train split)
    cand_embs = {}
    for cand in CANDIDATES:
        emb = get_embeddings(model, cand, N_SAMPLES_CANDIDATE, 'train')
        if emb is not None:
            cand_embs[cand] = emb
    print(f"  {len(cand_embs)} candidate embeddings", flush=True)
    
    # Target rankings
    model_results = {}
    for tgt in TARGETS:
        t_embs = get_embeddings(model, tgt, N_SAMPLES_TARGET, 'test')
        if t_embs is None:
            model_results[tgt] = None
            print(f"  {tgt} → FAILED", flush=True)
            continue
        
        scores = []
        for cand, c_emb in cand_embs.items():
            sim = float(np.mean(cosine_similarity(t_embs, c_emb)))
            scores.append((cand, round(sim, 4)))
        scores.sort(key=lambda x: -x[1])
        model_results[tgt] = scores
        print(f"  {tgt} → {scores[0][0]}({scores[0][1]:.4f})", flush=True)
    
    results[model_name] = model_results

# ── Unit-Proxy baseline ──
print(f"\n{'='*60}", flush=True)
print(f"Unit-Proxy Only (JS similarity)", flush=True)
print(f"{'='*60}", flush=True)
proxy_results = {}
for tgt in TARGETS:
    ranked = get_unit_proxy_ranking(tgt)
    if ranked:
        proxy_results[tgt] = ranked
        print(f"  {tgt} → {ranked[0][0]}({ranked[0][1]:.4f})", flush=True)
    else:
        proxy_results[tgt] = None
        print(f"  {tgt} → NO JS DATA", flush=True)
results['Unit-Proxy Only'] = proxy_results

# ── Comparison table ──
print(f"\n\n{'='*120}")
print("COMPARISON: Top-1 Source by Method")
print("=" * 120)

methods_order = ['Unit-Proxy Only'] + list(CKPTS.keys())
# Filter to only existing models
methods_order = [m for m in methods_order if m in results]

# Header
print(f"\n{'Target':<6}", end="")
for m in methods_order:
    short = m[:20]
    print(f" {short:<22}", end="")
print()

for t in TARGETS:
    print(f"{t:<6}", end="")
    for m in methods_order:
        r = results[m].get(t)
        if r:
            src = r[0][0]
            score = r[0][1]
            print(f" {src}({score:.3f}){'':<12}", end="")
        else:
            print(f" {'—':<22}", end="")
    print()

# Agreement with original SupCon
print(f"\n{'='*120}")
print("AGREEMENT with original SupCon top-1")
print("=" * 120)
ref_name = 'SupCon (original)'
if ref_name in results:
    for m in methods_order:
        if m == ref_name:
            continue
        agree = sum(1 for t in TARGETS 
                    if results[m].get(t) and results[ref_name].get(t)
                    and results[m][t][0][0] == results[ref_name][t][0][0])
        print(f"  {m:<30} {agree}/16 agree with {ref_name}")

# Spearman correlation of full rankings
print(f"\n{'='*120}")
print("SPEARMAN RANK CORRELATION with original SupCon")
print("=" * 120)
if ref_name in results:
    for m in methods_order:
        if m == ref_name:
            continue
        cors = []
        for t in TARGETS:
            r1 = results[ref_name].get(t)
            r2 = results[m].get(t)
            if r1 and r2:
                rank1 = {s[0]: i for i, s in enumerate(r1)}
                rank2 = {s[0]: i for i, s in enumerate(r2)}
                common = [c for c in rank1 if c in rank2]
                if len(common) > 2:
                    v1 = [rank1[c] for c in common]
                    v2 = [rank2[c] for c in common]
                    cor, _ = spearmanr(v1, v2)
                    cors.append(cor)
        if cors:
            avg_cor = np.mean(cors)
            print(f"  {m:<30} avg ρ={avg_cor:.4f} ({len(cors)} langs)")

# Save
out_path = '/mnt/storage/qisheng/github/wav2vec_test/results/proxy_comparison.json'
with open(out_path, 'w') as f:
    # Save top-3 for compactness
    save = {}
    for m, r in results.items():
        save[m] = {}
        for t, scores in r.items():
            if scores:
                save[m][t] = [f"{s[0]}({s[1]:.4f})" for s in scores[:3]]
    json.dump(save, f, indent=2)
print(f"\nSaved to {out_path}")
print("Done!")
