"""
Run source selection for 4 methods (eSpeak, CV-IPA, Classifier, Contrastive)
with varying target data amounts: 2, 5, 10, 20, 50 samples (~10s to 5min).
"""
import csv, json, math, os, random, sys, time
from collections import Counter
import numpy as np

os.environ["HF_HOME"] = "/mnt/storage/ldl_linguistics/hf_home"
CACHE_DIR = "/mnt/storage/ldl_linguistics/datasets"

# ── Target languages (new candidates) ──
ALL_TARGETS = ['sq','ltg','ur','cy','gn','tn','am','az']
TARGET_NAMES = {
    'sq':'Albanian','ltg':'Latgalian','ur':'Urdu','cy':'Welsh',
    'gn':'Guarani','tn':'Tswana','am':'Amharic','az':'Azerbaijani',
}
CANDIDATES = ['ar','ba','ca','cs','en','eo','fr','hu','it','lt','lv',
              'nl','ro','ru','sw','ta','tr','tt','ug']

SAMPLE_COUNTS = [2, 5, 10, 20, 50]
SEED = 42

# ═══════════════════════════════════════════════════════
# PHONEMIZER (shared by eSpeak + CV-IPA)
# ═══════════════════════════════════════════════════════
from phonemizer.backend import BACKENDS
from phonemizer.separator import Separator

PHONEMIZER_MAP = {'fr': 'fr-fr', 'en': 'en-us', 'zh': 'cmn', 'yue': 'yue'}

def phonemize(texts, lang_code):
    ph_code = PHONEMIZER_MAP.get(lang_code, lang_code)
    try:
        backend = BACKENDS['espeak'](ph_code, language_switch='remove-flags')
    except Exception:
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
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def js_similarity(counter_a, counter_b):
    all_ph = sorted(set(counter_a.keys()) | set(counter_b.keys()))
    if not all_ph:
        return 0.0
    total_a = max(sum(counter_a.values()), 1)
    total_b = max(sum(counter_b.values()), 1)
    p = [counter_a.get(ph, 0) / total_a for ph in all_ph]
    q = [counter_b.get(ph, 0) / total_b for ph in all_ph]
    m = [(p[i] + q[i]) / 2 for i in range(len(p))]
    kl_pm = sum(p[i] * math.log2(p[i] / m[i]) for i in range(len(p)) if p[i] > 0 and m[i] > 0)
    kl_qm = sum(q[i] * math.log2(q[i] / m[i]) for i in range(len(q)) if q[i] > 0 and m[i] > 0)
    return 1.0 - (kl_pm + kl_qm) / 2

# ═══════════════════════════════════════════════════════
# METHOD 1 & 2: eSpeak Jaccard + CV-IPA JS
# ═══════════════════════════════════════════════════════
from datasets import load_dataset

def get_cv_test_sentences(lang_code, n_sentences=50):
    """Load n sentences from CV test split."""
    try:
        ds = load_dataset('fsicoli/common_voice_22_0', lang_code, split='test',
                          trust_remote_code=True, cache_dir=CACHE_DIR)
    except:
        return None
    sentences = [ex['sentence'] for ex in ds if len(ex.get('sentence', '')) > 20]
    if not sentences:
        return None
    random.seed(SEED)
    if len(sentences) > n_sentences:
        sentences = random.sample(sentences, n_sentences)
    return sentences

def run_espeak_cvipa(n_samples):
    """Run eSpeak Jaccard + CV-IPA JS for all targets."""
    print(f"\n{'='*60}")
    print(f"eSpeak + CV-IPA (n_samples={n_samples})")
    print(f"{'='*60}")
    
    # Phonemize all languages (candidates + targets)
    phone_sets, phone_counters = {}, {}
    for lang in CANDIDATES + ALL_TARGETS:
        texts = get_cv_test_sentences(lang, n_sentences=n_samples)
        if not texts:
            print(f"  [{lang}] NO CV TEXT")
            continue
        phonemes = phonemize(texts, lang)
        if not phonemes:
            print(f"  [{lang}] PHONEMIZE FAILED")
            continue
        pset = set()
        pcnt = Counter()
        for ps in phonemes:
            if ps:
                phones = ps.strip().split()
                pset.update(phones)
                pcnt.update(phones)
        phone_sets[lang] = pset
        phone_counters[lang] = pcnt
        print(f"  [{lang}] {len(pset)} phones, {sum(pcnt.values())} tokens")
    
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
        print(f"  eSpeak {tgt} → {scores[0][0]}({scores[0][1]:.4f})")
    
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
        print(f"  CV-IPA {tgt} → {scores[0][0]}({scores[0][1]:.4f})")
    
    return es_results, cv_results


# ═══════════════════════════════════════════════════════
# METHOD 3: Classifier Embedding
# ═══════════════════════════════════════════════════════
def run_classifier(n_samples):
    """Run Classifier embedding source selection."""
    print(f"\n{'='*60}")
    print(f"Classifier Embedding (n_samples={n_samples})")
    print(f"{'='*60}")
    
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Config
    sys.path.insert(0, "/mnt/storage/qisheng/github/wav2vec_contrastive/customized")
    from model_classifier import Wav2Vec2ForLanguageClassification
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    config.num_labels = 36
    fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53",
                                                trust_remote_code=True, cache_dir=CACHE_DIR)
    model = Wav2Vec2ForLanguageClassification.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53", config=config, ignore_mismatched_sizes=True)
    ckpt = "/mnt/storage/qisheng/github/wav2vec_contrastive/weights/CLASSIFIER_36LANG_200SAMPLE/checkpoint_latest.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device).eval()
    
    def get_feat(lang, split='train'):
        try:
            ds = load_dataset('fsicoli/common_voice_22_0', lang, split=split,
                              trust_remote_code=True, cache_dir=CACHE_DIR)
        except:
            return None
        if len(ds) > n_samples:
            ds = ds.shuffle(seed=SEED).select(range(n_samples))
        ds = ds.cast_column('audio', __import__('datasets').Audio(sampling_rate=16000))
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
    
    # Collect candidate embeddings (use train split, 50 samples each, cached)
    cand_embs = {}
    for cand in CANDIDATES:
        emb = get_feat(cand, 'train')
        if emb is not None:
            cand_embs[cand] = emb
            print(f"  [{cand}] embedding OK")
        else:
            print(f"  [{cand}] FAILED")
    
    # For each target, using test split with n_samples
    results = {}
    for tgt in ALL_TARGETS:
        t_emb = get_feat(tgt, 'test')
        if t_emb is None:
            print(f"  Clf {tgt} → NO EMBEDDING")
            results[tgt] = None
            continue
        from sklearn.metrics.pairwise import cosine_similarity
        scores = [(c, float(cosine_similarity(t_emb.reshape(1,-1), cand_embs[c].reshape(1,-1))[0,0]))
                  for c in CANDIDATES if c in cand_embs]
        scores.sort(key=lambda x: -x[1])
        results[tgt] = [(c, round(s, 4)) for c, s in scores]
        print(f"  Clf {tgt} → {results[tgt][0][0]}({results[tgt][0][1]:.4f})")
    
    return results


# ═══════════════════════════════════════════════════════
# METHOD 4: Contrastive Embedding
# ═══════════════════════════════════════════════════════
def run_contrastive(n_samples):
    """Run Contrastive embedding source selection."""
    print(f"\n{'='*60}")
    print(f"Contrastive Embedding (n_samples={n_samples})")
    print(f"{'='*60}")
    
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Config
    sys.path.insert(0, "/mnt/storage/qisheng/github/wav2vec_contrastive/customized")
    from model import Wav2Vec2ForContrastiveLearning
    from dataset import vectorize_datasets_classificationForTest, AudioClassificationDataCollatorForTest
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53",
                                                trust_remote_code=True, cache_dir=CACHE_DIR)
    model = Wav2Vec2ForContrastiveLearning.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53", config=config, ignore_mismatched_sizes=True)
    ckpt = "/mnt/storage/qisheng/github/wav2vec_contrastive/weights/FEAT128BS16_WAVE/checkpoint_epoch_50.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device).eval()
    
    def get_emb(lang):
        ds_name = "fixie-ai/common_voice_17_0"
        try:
            ds = load_dataset(ds_name, lang, split='test',
                              trust_remote_code=True, cache_dir=CACHE_DIR)
        except:
            ds_name = "fsicoli/common_voice_22_0"
            ds = load_dataset(ds_name, lang, split='test',
                              trust_remote_code=True, cache_dir=CACHE_DIR)
        if len(ds) > n_samples:
            ds = ds.shuffle(seed=SEED).select(range(n_samples))
        
        import datasets as hf_ds
        cds = hf_ds.DatasetDict({"train": ds})
        vec = vectorize_datasets_classificationForTest(cds, fe)
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
    
    # Known candidate embeddings (always use 50 samples for consistency)
    known_embs = {}
    for cand in CANDIDATES:
        emb = get_emb(cand)
        if emb is not None:
            known_embs[cand] = emb
            print(f"  [{cand}] {len(emb)} embeddings")
        else:
            print(f"  [{cand}] FAILED")
    
    from sklearn.metrics.pairwise import cosine_similarity
    results = {}
    for tgt in ALL_TARGETS:
        t_embs = get_emb(tgt)
        if t_embs is None:
            print(f"  Con {tgt} → NO EMBEDDING")
            results[tgt] = None
            continue
        scores = []
        for cand, k_embs in known_embs.items():
            sim = float(np.mean(cosine_similarity(t_embs, k_embs)))
            scores.append((cand, round(sim, 4)))
        scores.sort(key=lambda x: -x[1])
        results[tgt] = scores
        print(f"  Con {tgt} → {scores[0][0]}({scores[0][1]:.4f})")
    
    return results


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def top3_str(scores):
    if not scores: return "—"
    return ", ".join(f"{s[0]}({s[1]:.4f})" for s in scores[:3])

if __name__ == '__main__':
    all_results = {}  # {method: {n_samples: {target: [scores]}}}
    
    for ns in SAMPLE_COUNTS:
        label = f"n={ns}"
        all_results.setdefault('espeak', {})[label] = {}
        all_results.setdefault('cvipa', {})[label] = {}
        
        es, cv = run_espeak_cvipa(ns)
        for t in ALL_TARGETS:
            all_results['espeak'][label][t] = es.get(t)
            all_results['cvipa'][label][t] = cv.get(t)
    
    # Classifier + Contrastive (GPU methods)
    for ns in SAMPLE_COUNTS:
        label = f"n={ns}"
        all_results.setdefault('classifier', {})[label] = {}
        all_results.setdefault('contrastive', {})[label] = {}
        
        clf = run_classifier(ns)
        for t in ALL_TARGETS:
            all_results['classifier'][label][t] = clf.get(t)
        
        con = run_contrastive(ns)
        for t in ALL_TARGETS:
            all_results['contrastive'][label][t] = con.get(t)
    
    # ── Print comparison table ──
    methods = ['espeak', 'cvipa', 'classifier', 'contrastive']
    method_labels = ['eSpeak', 'CV-IPA', 'Classifier', 'Contrastive']
    ns_labels = [f"n={ns}" for ns in SAMPLE_COUNTS]
    
    print("\n\n" + "=" * 180)
    print("COMPARISON: Source Selection by Data Amount")
    print("=" * 180)
    
    for t in ALL_TARGETS:
        print(f"\n{'='*120}")
        print(f"  Target: {t} ({TARGET_NAMES[t]})")
        print(f"{'='*120}")
        
        header = f"{'Method':<14}"
        for nl in ns_labels:
            header += f" {nl:<38}"
        print(header)
        print("-" * len(header))
        
        for mi, m in enumerate(methods):
            line = f"{method_labels[mi]:<14}"
            for nl in ns_labels:
                scores = all_results[m][nl].get(t)
                if scores:
                    line += f" {scores[0][0]}({scores[0][1]:.4f}){'':>25}"
                else:
                    line += f" {'—':<38}"
            print(line)
        
        # Check if top1 changes
        print(f"  {'':<14}", end="")
        for nl in ns_labels:
            sources = set()
            for m in methods:
                scores = all_results[m][nl].get(t)
                if scores:
                    sources.add(scores[0][0])
            changes = len(sources) > 1
            status = "⚠ varies" if changes else "✓ stable"
            print(f" {status:<38}", end="")
        print()
    
    # ── Summary per method ──
    print("\n\n" + "=" * 120)
    print("SUMMARY: Does top1 change across sample sizes?")
    print("=" * 120)
    for mi, m in enumerate(methods):
        print(f"\n{method_labels[mi]}:")
        for t in ALL_TARGETS:
            sources = []
            for nl in ns_labels:
                scores = all_results[m][nl].get(t)
                if scores:
                    sources.append(f"{scores[0][0]}({nl})")
            if len(set(s.split('(')[0] for s in sources)) > 1:
                print(f"  ✗ {t}: {' → '.join(sources)}")
            else:
                print(f"  ✓ {t}: always {sources[0].split('(')[0]}")
    
    # Save full results
    save = {}
    for m in methods:
        save[m] = {}
        for nl in ns_labels:
            save[m][nl] = {}
            for t in ALL_TARGETS:
                s = all_results[m][nl].get(t)
                save[m][nl][t] = [f"{x[0]}({x[1]:.4f})" for x in s[:3]] if s else None
    
    out_path = '/mnt/storage/qisheng/github/wav2vec_test/results/ablation_samples.json'
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\nFull results saved: {out_path}")
