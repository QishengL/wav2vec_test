"""
Run a single source selection method with a specific n_samples.
Usage: python3 run_single_ablation.py --method <method> --n_samples <N>
  method: classifier, contrastive, espeak_cvipa
"""
import sys, json, os, math, random
from collections import Counter
import numpy as np

os.environ["HF_HOME"] = "/mnt/storage/ldl_linguistics/hf_home"
CACHE_DIR = "/mnt/storage/ldl_linguistics/datasets"

ALL_TARGETS = ['sq','ltg','ur','cy','gn','tn','am','az',
               'mt','af','da','ky','tk','kk','sk','id']
TARGET_NAMES = {'sq':'Albanian','ltg':'Latgalian','ur':'Urdu','cy':'Welsh',
                'gn':'Guarani','tn':'Tswana','am':'Amharic','az':'Azerbaijani',
                'mt':'Maltese','af':'Afrikaans','da':'Danish',
                'ky':'Kyrgyz','tk':'Turkmen','kk':'Kazakh','sk':'Slovak','id':'Indonesian'}
CANDIDATES = ['ar','ba','ca','cs','en','eo','fr','hu','it','lt','lv',
              'nl','ro','ru','sw','ta','tr','tt','ug']
SEED = 42

# Parse args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--method', required=True, choices=['classifier','contrastive','espeak_cvipa'])
parser.add_argument('--n_samples', type=int, required=True)
args = parser.parse_args()
method = args.method
ns = args.n_samples
label = f"n={ns}"
out_dir = '/mnt/storage/qisheng/github/wav2vec_test/results/ablation'
os.makedirs(out_dir, exist_ok=True)

print(f"Starting: {method}, n_samples={ns}", flush=True)

# ═══════════════════════════════════════
# CLASSIFIER
# ═══════════════════════════════════════
if method == 'classifier':
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Config
    from datasets import load_dataset
    import datasets as hf_ds
    from sklearn.metrics.pairwise import cosine_similarity
    
    sys.path.insert(0, "/mnt/storage/qisheng/github/wav2vec_contrastive/customized")
    from model_classifier import Wav2Vec2ForLanguageClassification
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}", flush=True)
    
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    config.num_labels = 36
    fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53",
                                                trust_remote_code=True, cache_dir=CACHE_DIR)
    model = Wav2Vec2ForLanguageClassification.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53", config=config, ignore_mismatched_sizes=True)
    ckpt = "/mnt/storage/qisheng/github/wav2vec_contrastive/weights/CLASSIFIER_36LANG_200SAMPLE/checkpoint_latest.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device).eval()
    print("  Model loaded", flush=True)
    
    def get_feat(lang, split='train'):
        try:
            ds = load_dataset('fsicoli/common_voice_22_0', lang, split=split,
                              trust_remote_code=True, cache_dir=CACHE_DIR)
        except:
            return None
        if len(ds) > ns:
            ds = ds.shuffle(seed=SEED).select(range(ns))
        ds = ds.cast_column('audio', hf_ds.Audio(sampling_rate=16000))
        embs = []
        for ex in ds:
            try:
                audio = ex['audio']
                inputs = fe(audio['array'], sampling_rate=16000, return_tensors='pt')
                inp = inputs['input_values'].to(device)
                attn = inputs.get('attention_mask')
                if attn is not None: attn = attn.to(device)
                with torch.no_grad():
                    feat = model.get_embedding(inp, attn).squeeze().cpu().numpy()
                embs.append(feat)
            except:
                continue
        if not embs: return None
        return np.mean(embs, axis=0)
    
    # Candidates (always 50 samples for stable reference)
    cand_embs = {}
    for cand in CANDIDATES:
        emb = get_feat(cand, 'train')
        if emb is not None:
            cand_embs[cand] = emb
        else:
            print(f"  [WARN] Candidate {cand} FAILED", flush=True)
    
    results = {}
    for tgt in ALL_TARGETS:
        t_emb = get_feat(tgt, 'test')
        if t_emb is None:
            results[tgt] = None
            print(f"  {tgt} → FAILED", flush=True)
            continue
        scores = [(c, float(cosine_similarity(t_emb.reshape(1,-1), cand_embs[c].reshape(1,-1))[0,0]))
                  for c in CANDIDATES if c in cand_embs]
        scores.sort(key=lambda x: -x[1])
        results[tgt] = [(c, round(s, 4)) for c, s in scores]
        top = results[tgt][0]
        print(f"  {tgt} → {top[0]}({top[1]:.4f})", flush=True)
    
    with open(f'{out_dir}/classifier_{label}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: classifier_{label}.json", flush=True)

# ═══════════════════════════════════════
# CONTRASTIVE
# ═══════════════════════════════════════
elif method == 'contrastive':
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Config
    from datasets import load_dataset
    import datasets as hf_ds
    from sklearn.metrics.pairwise import cosine_similarity
    
    sys.path.insert(0, "/mnt/storage/qisheng/github/wav2vec_contrastive/customized")
    from model import Wav2Vec2ForContrastiveLearning
    from dataset import vectorize_datasets_classificationForTest, AudioClassificationDataCollatorForTest
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}", flush=True)
    
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53",
                                                trust_remote_code=True, cache_dir=CACHE_DIR)
    model = Wav2Vec2ForContrastiveLearning.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53", config=config, ignore_mismatched_sizes=True)
    ckpt = "/mnt/storage/qisheng/github/wav2vec_contrastive/weights/FEAT128BS16_WAVE/checkpoint_epoch_50.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device).eval()
    print("  Model loaded", flush=True)
    
    def get_emb(lang):
        ds_name = "fixie-ai/common_voice_17_0"
        try:
            ds = load_dataset(ds_name, lang, split='test',
                              trust_remote_code=True, cache_dir=CACHE_DIR)
        except:
            ds_name = "fsicoli/common_voice_22_0"
            try:
                ds = load_dataset(ds_name, lang, split='test',
                                  trust_remote_code=True, cache_dir=CACHE_DIR)
            except:
                return None
        if len(ds) > ns:
            ds = ds.shuffle(seed=SEED).select(range(ns))
        
        cds = hf_ds.DatasetDict({"train": ds})
        DATASET_PARAMS = {
            "audio_column": "audio",
            "max_duration_in_seconds": 20.0,
            "min_duration_in_seconds": 0.0,
            "preprocessing_num_workers": 1,
        }
        vec = vectorize_datasets_classificationForTest(cds, tokenizer=None, feature_extractor=fe, **DATASET_PARAMS)
        coll = AudioClassificationDataCollatorForTest(fe)
        from torch.utils.data import DataLoader
        loader = DataLoader(vec['train'], batch_size=1, shuffle=False, collate_fn=coll)
        
        embs = []
        for batch in loader:
            inp = batch['input_values'].to(device)
            attn = batch.get('attention_mask')
            if attn is not None: attn = attn.to(device)
            with torch.no_grad():
                feat = model(inp, attn).squeeze().cpu().numpy()
            feat = feat / np.linalg.norm(feat)
            embs.append(feat)
        if not embs: return None
        return np.array(embs)
    
    known_embs = {}
    for cand in CANDIDATES:
        emb = get_emb(cand)
        if emb is not None:
            known_embs[cand] = emb
        else:
            print(f"  [WARN] Candidate {cand} FAILED", flush=True)
    
    results = {}
    for tgt in ALL_TARGETS:
        t_embs = get_emb(tgt)
        if t_embs is None:
            results[tgt] = None
            print(f"  {tgt} → FAILED", flush=True)
            continue
        scores = []
        for cand, k_embs in known_embs.items():
            sim = float(np.mean(cosine_similarity(t_embs, k_embs)))
            scores.append((cand, round(sim, 4)))
        scores.sort(key=lambda x: -x[1])
        results[tgt] = scores
        top = results[tgt][0]
        print(f"  {tgt} → {top[0]}({top[1]:.4f})", flush=True)
    
    with open(f'{out_dir}/contrastive_{label}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: contrastive_{label}.json", flush=True)

# ═══════════════════════════════════════
# eSPEAK + CV-IPA (share phonemization)
# ═══════════════════════════════════════════
elif method == 'espeak_cvipa':
    from phonemizer.backend import BACKENDS
    from phonemizer.separator import Separator
    from datasets import load_dataset
    
    PHONEMIZER_MAP = {'fr': 'fr-fr', 'en': 'en-us', 'zh': 'cmn', 'yue': 'yue'}
    
    def phonemize(texts, lang_code):
        ph_code = PHONEMIZER_MAP.get(lang_code, lang_code)
        try:
            backend = BACKENDS['espeak'](ph_code, language_switch='remove-flags')
        except:
            return None
        sep = Separator(phone=' ', word='', syllable='')
        results = []
        for i in range(0, len(texts), 50):
            try:
                results.extend(backend.phonemize(texts[i:i+50], separator=sep, strip=True))
            except:
                continue
        return results
    
    def jaccard(a, b):
        if not a or not b: return 0.0
        return len(a & b) / len(a | b)
    
    def js_similarity(ca, cb):
        all_ph = sorted(set(ca.keys()) | set(cb.keys()))
        if not all_ph: return 0.0
        total_a = max(sum(ca.values()), 1)
        total_b = max(sum(cb.values()), 1)
        p = [ca.get(ph, 0) / total_a for ph in all_ph]
        q = [cb.get(ph, 0) / total_b for ph in all_ph]
        m = [(p[i] + q[i]) / 2 for i in range(len(p))]
        kl_pm = sum(p[i] * math.log2(p[i] / m[i]) for i in range(len(p)) if p[i] > 0 and m[i] > 0)
        kl_qm = sum(q[i] * math.log2(q[i] / m[i]) for i in range(len(q)) if q[i] > 0 and m[i] > 0)
        return 1.0 - (kl_pm + kl_qm) / 2
    
    random.seed(SEED)
    phone_sets, phone_counters = {}, {}
    
    for lang in CANDIDATES + ALL_TARGETS:
        try:
            ds = load_dataset('fsicoli/common_voice_22_0', lang, split='test',
                              trust_remote_code=True, cache_dir=CACHE_DIR)
        except:
            print(f"  [{lang}] NO CV TEST DATA", flush=True)
            continue
        sentences = [ex['sentence'] for ex in ds if len(ex.get('sentence', '')) > 20]
        if not sentences:
            print(f"  [{lang}] NO SENTENCES", flush=True)
            continue
        random.seed(SEED)
        if len(sentences) > ns:
            sentences = random.sample(sentences, ns)
        
        phonemes = phonemize(sentences, lang)
        if not phonemes:
            print(f"  [{lang}] PHONEMIZE FAILED", flush=True)
            continue
        pset, pcnt = set(), Counter()
        for ps in phonemes:
            if ps:
                phones = ps.strip().split()
                pset.update(phones)
                pcnt.update(phones)
        phone_sets[lang] = pset
        phone_counters[lang] = pcnt
        print(f"  [{lang}] {len(pset)} phones", flush=True)
    
    # eSpeak Jaccard
    es_results = {}
    for tgt in ALL_TARGETS:
        t_set = phone_sets.get(tgt)
        if not t_set:
            es_results[tgt] = None
            continue
        scores = [(c, round(jaccard(t_set, phone_sets[c]), 4))
                  for c in CANDIDATES if c in phone_sets]
        scores.sort(key=lambda x: -x[1])
        es_results[tgt] = scores
        print(f"  eSpeak {tgt} → {scores[0][0]}({scores[0][1]:.4f})", flush=True)
    
    # CV-IPA JS
    cv_results = {}
    for tgt in ALL_TARGETS:
        t_cnt = phone_counters.get(tgt)
        if not t_cnt:
            cv_results[tgt] = None
            continue
        scores = [(c, round(js_similarity(t_cnt, phone_counters[c]), 4))
                  for c in CANDIDATES if c in phone_counters]
        scores.sort(key=lambda x: -x[1])
        cv_results[tgt] = scores
        print(f"  CV-IPA {tgt} → {scores[0][0]}({scores[0][1]:.4f})", flush=True)
    
    with open(f'{out_dir}/espeak_{label}.json', 'w') as f:
        json.dump(es_results, f, indent=2)
    with open(f'{out_dir}/cvipa_{label}.json', 'w') as f:
        json.dump(cv_results, f, indent=2)
    print(f"Saved: espeak_{label}.json, cvipa_{label}.json", flush=True)

print(f"Done: {method}, n_samples={ns}", flush=True)
