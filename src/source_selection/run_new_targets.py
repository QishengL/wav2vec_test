"""
|Run all source selection methods for new candidate target languages.
|Methods: Same-Family (heuristic), PHOIBLE Jaccard (inventory),
|         eSpeak Inventory Jaccard, Wikipedia IPA Distribution,
|         CV Test IPA Distribution, Raw Wav2Vec2 Embedding (Classifier)
"""
import csv, json, math, os, re, sys, time, urllib.request
from collections import Counter
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Config, AutoFeatureExtractor
from datasets import load_dataset, Audio, Dataset, concatenate_datasets

# ── Configuration ──
OLD_TARGETS = ['mt', 'af', 'da', 'ky', 'tk', 'kk', 'sk', 'id']
NEW_TARGETS = ['sq', 'ltg', 'ur', 'cy', 'gn', 'tn', 'am', 'he', 'az']
ALL_TARGETS = OLD_TARGETS + NEW_TARGETS

TARGET_NAMES = {
    'mt': 'Maltese', 'af': 'Afrikaans', 'da': 'Danish',
    'ky': 'Kyrgyz', 'tk': 'Turkmen', 'kk': 'Kazakh',
    'sk': 'Slovak', 'id': 'Indonesian',
    'sq': 'Albanian', 'ltg': 'Latgalian',
    'ur': 'Urdu', 'cy': 'Welsh', 'gn': 'Guarani', 'tn': 'Tswana',
    'am': 'Amharic', 'he': 'Hebrew', 'az': 'Azerbaijani',
}

CANDIDATES = ['ar', 'ba', 'ca', 'cs', 'en', 'eo', 'fr', 'hu', 'it',
              'lt', 'lv', 'nl', 'ro', 'ru', 'sw', 'ta', 'tr', 'tt', 'ug']

ALL_LANGS = list(dict.fromkeys(ALL_TARGETS + CANDIDATES))

os.environ['HF_HOME'] = '/mnt/storage/ldl_linguistics/hf_home'
CACHE_DIR = '/mnt/storage/ldl_linguistics/datasets'

# ═══════════════════════════════════════════════════════
# METHOD 1: SAME-FAMILY HEURISTIC
# ═══════════════════════════════════════════════════════
FAMILIES = {
    # Old targets
    'mt': 'Semitic', 'af': 'Germanic', 'da': 'Germanic',
    'ky': 'Turkic', 'tk': 'Turkic', 'kk': 'Turkic',
    'sk': 'Slavic', 'id': 'Austronesian',
    # New targets
    'sq': 'Albanian', 'ltg': 'Baltic',
    'ur': 'Indo-Aryan', 'cy': 'Celtic', 'gn': 'Tupi-Guarani',
    'tn': 'Bantu', 'am': 'Semitic', 'he': 'Semitic', 'az': 'Turkic',
    'ar': 'Semitic', 'ba': 'Turkic', 'ca': 'Romance',
    'cs': 'Slavic', 'en': 'Germanic', 'eo': 'Constructed',
    'fr': 'Romance', 'hu': 'Uralic', 'it': 'Romance',
    'lt': 'Baltic', 'lv': 'Baltic', 'nl': 'Germanic',
    'ro': 'Romance', 'ru': 'Slavic', 'sw': 'Bantu',
    'ta': 'Dravidian', 'tr': 'Turkic', 'tt': 'Turkic', 'ug': 'Turkic',
}

HEURISTIC_CHOICE = {
    'Albanian': 'it',
    'Baltic': 'lv',
    'Constructed': 'eo',
    'Indo-Aryan': 'ta',
    'Celtic': 'it',
    'Tupi-Guarani': 'sw',
    'Bantu': 'sw',
    'Semitic': 'ar',
    'Turkic': 'tr',
    'Austronesian': 'ta',  # Fallback: Tamil, geographically close
}

def same_family():
    print('\n' + '=' * 60)
    print('METHOD 1: SAME-FAMILY HEURISTIC')
    print('=' * 60)
    results = {}
    for tgt in ALL_TARGETS:
        family = FAMILIES.get(tgt, 'Unknown')
        best = HEURISTIC_CHOICE.get(family)
        results[tgt] = best or 'N/A'
        print(f'  {tgt} ({TARGET_NAMES[tgt]:<12}) family={family:<14} → {best}')
    return results


# ═══════════════════════════════════════════════════════
# METHOD 2: PHOIBLE JACCARD + eSpeak fallback
# ═══════════════════════════════════════════════════════
PHOIBLE_PATH = '/tmp/phoible.csv'

def load_phoible():
    if not os.path.exists(PHOIBLE_PATH):
        print('  Downloading PHOIBLE...')
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/phoible/dev/master/data/phoible.csv',
            PHOIBLE_PATH)
    invs = {}
    with open(PHOIBLE_PATH, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            iso = row['ISO6393']
            if row.get('Marginal', '').lower() == 'true':
                continue
            invs.setdefault(iso, set()).add(row['Phoneme'])
    return invs

def get_espeak_phonemes(lang_code):
    """Get phoneme inventory from eSpeak-NG."""
    from phonemizer.backend import BACKENDS
    from phonemizer.separator import Separator
    ph_map = {'en': 'en-us', 'fr': 'fr-fr', 'zh': 'cmn', 'yue': 'yue'}
    ph_code = ph_map.get(lang_code, lang_code)
    try:
        backend = BACKENDS['espeak'](ph_code)
    except Exception:
        return None
    sep = Separator(phone=' ', word='', syllable='')
    # Use a diverse test set to capture phoneme inventory
    test_texts = [
        'hello world this is a test sentence with many different words',
        'the quick brown fox jumps over the lazy dog',
        'please bring us some food and water for the journey',
        # Language-specific test sentences via CV
    ]
    try:
        result = backend.phonemize(test_texts, separator=sep, strip=True)
        phones = set()
        for r in result:
            if r:
                for p in r.strip().split():
                    phones.add(p)
        return phones if phones else None
    except Exception:
        return None

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

ISO1_TO_ISO3 = {
    # Old targets
    'mt': 'mlt', 'af': 'afr', 'da': 'dan',
    'ky': 'kir', 'tk': 'tuk', 'kk': 'kaz',
    'sk': 'slk', 'id': 'ind',
    # New targets
    'sq': 'als', 'ltg': 'ltg', 'ur': 'urd', 'cy': 'cym',
    'gn': 'gug', 'tn': 'tsn', 'am': 'amh', 'he': 'heb', 'az': 'azj',
    'ar': 'arb', 'ba': 'bak', 'ca': 'cat', 'cs': 'ces',
    'en': 'eng', 'eo': 'epo', 'fr': 'fra', 'hu': 'hun',
    'it': 'ita', 'lt': 'lit', 'lv': 'lvs', 'nl': 'nld',
    'ro': 'ron', 'ru': 'rus', 'sw': 'swh', 'ta': 'tam',
    'tr': 'tur', 'tt': 'tat', 'ug': 'uig',
}

def phoible_jaccard_method():
    """PHOIBLE Jaccard only — skip languages not in PHOIBLE."""
    print('\n' + '=' * 60)
    print('METHOD 2a: PHOIBLE JACCARD (database only)')
    print('=' * 60)
    phoible = load_phoible()
    print(f'  Loaded {len(phoible)} inventories from PHOIBLE')

    # Get PHOIBLE inventories
    inventories = {}
    for lang in ALL_LANGS:
        iso3 = ISO1_TO_ISO3.get(lang)
        inv = phoible.get(iso3)
        if inv:
            inventories[lang] = inv
            print(f'  {lang}: {len(inv)} phones (PHOIBLE)')
        else:
            print(f'  {lang}: NOT in PHOIBLE (skipped)')

    results = {}
    for tgt in ALL_TARGETS:
        t_inv = inventories.get(tgt)
        if not t_inv:
            print(f'  {tgt}: NO PHOIBLE DATA → SKIP')
            results[tgt] = None
            continue
        scores = []
        for cand in CANDIDATES:
            c_inv = inventories.get(cand)
            if c_inv:
                scores.append((cand, round(jaccard(t_inv, c_inv), 4)))
        scores.sort(key=lambda x: -x[1])
        results[tgt] = scores
        best = scores[0] if scores else ('N/A', 0)
        print(f'  {tgt} → {best[0]} ({best[1]:.4f})')
    return results


# ═══════════════════════════════════════════════════════
# METHOD 2b: eSPEAK INVENTORY JACCARD (from CV test sentences)
# ═══════════════════════════════════════════════════════
def get_espeak_inventory(lang_code):
    """Get phoneme inventory by phonemizing CV test sentences (50 sentences)."""
    texts = get_cv_test_texts(lang_code, n_sentences=50)
    if not texts:
        return None
    phonemes = phonemize(texts, lang_code)
    if not phonemes:
        return None
    phones = set()
    for ps in phonemes:
        if ps:
            phones.update(ps.strip().split())
    return phones if phones else None


def espeak_inventory_method():
    print('\n' + '=' * 60)
    print('METHOD 2b: eSPEAK INVENTORY JACCARD')
    print('=' * 60)

    inventories = {}
    for lang in ALL_LANGS:
        print(f'  [{lang}] extracting eSpeak inventory...', end=' ', flush=True)
        inv = get_espeak_inventory(lang)
        if inv:
            inventories[lang] = inv
            print(f'{len(inv)} phones')
        else:
            print('FAILED')

    results = {}
    for tgt in ALL_TARGETS:
        t_inv = inventories.get(tgt)
        if not t_inv:
            print(f'  {tgt}: NO eSPEAK INVENTORY')
            results[tgt] = None
            continue
        scores = []
        for cand in CANDIDATES:
            c_inv = inventories.get(cand)
            if c_inv:
                scores.append((cand, round(jaccard(t_inv, c_inv), 4)))
        scores.sort(key=lambda x: -x[1])
        results[tgt] = scores
        best = scores[0] if scores else ('N/A', 0)
        print(f'  {tgt} → {best[0]} ({best[1]:.4f})')
    return results



# ═══════════════════════════════════════════════════════
# METHOD 3: CommonVoice Phoneme Distribution (IPA-Dist)
# ═══════════════════════════════════════════════════════
PHONEMIZER_MAP = {'fr': 'fr-fr', 'en': 'en-us', 'zh': 'cmn', 'yue': 'yue'}

def phonemize(texts, lang_code):
    from phonemizer.backend import BACKENDS
    from phonemizer.separator import Separator
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
        except Exception:
            continue
    return results

def get_cv_texts(lang_code, n_sentences=200):
    """Get sentences from Common Voice dataset instead of Wikipedia."""
    try:
        ds = load_dataset('fsicoli/common_voice_22_0', lang_code, split='train',
                          trust_remote_code=True, cache_dir=CACHE_DIR)
    except Exception:
        try:
            ds = load_dataset('fsicoli/common_voice_22_0', lang_code, split='train+validation',
                              trust_remote_code=True, cache_dir=CACHE_DIR)
        except Exception:
            print(f'  CV failed for {lang_code}')
            return None
    sentences = [ex['sentence'] for ex in ds if len(ex.get('sentence', '')) > 20]
    if len(sentences) > n_sentences:
        import random
        random.seed(42)
        sentences = random.sample(sentences, n_sentences)
    print(f'  {len(sentences)} sentences from CV')
    return sentences

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


def wikipedia_phoneme_method():
    print('\n' + '=' * 60)
    print('METHOD 3: WIKIPEDIA IPA DISTRIBUTION')
    print('=' * 60)

    distributions = {}
    for lang in ALL_LANGS:
        print(f'  [{lang}] fetching Wikipedia...', end=' ', flush=True)
        texts = get_wikipedia_text(lang, n_sentences=80)
        if not texts:
            print('  NO WIKIPEDIA DATA')
            continue
        phonemes = phonemize(texts, lang)
        if not phonemes:
            print('  PHONEMIZE FAILED')
            continue
        dist = Counter()
        for ps in phonemes:
            if ps:
                dist.update(ps.strip().split())
        distributions[lang] = dist
        print(f'  {len(dist)} phones from {len(texts)} sentences')

    results = {}
    for tgt in ALL_TARGETS:
        if tgt not in distributions:
            results[tgt] = None
            print(f'  {tgt}: NO DISTRIBUTION')
            continue
        scores = []
        for cand in CANDIDATES:
            if cand not in distributions:
                continue
            sim = js_similarity(distributions[tgt], distributions[cand])
            scores.append((cand, round(sim, 4)))
        scores.sort(key=lambda x: -x[1])
        results[tgt] = scores
        best = scores[0] if scores else ('N/A', 0)
        print(f'  {tgt} → {best[0]} ({best[1]:.4f})')
    return results


# ═══════════════════════════════════════════════════════
# METHOD 2b+3b: SHARED CV TEST PHONEMIZATION (parallel)
# ═══════════════════════════════════════════════════════
def phonemize_cv_test_lang(lang_code):
    """Load CV test 50 sentences, phonemize, return (phone_set, phone_counter)."""
    try:
        ds = load_dataset('fsicoli/common_voice_22_0', lang_code, split='test',
                          trust_remote_code=True, cache_dir=CACHE_DIR)
    except Exception:
        return lang_code, None, None
    sentences = [ex['sentence'] for ex in ds if len(ex.get('sentence', '')) > 20]
    if len(sentences) > 50:
        import random
        random.seed(42)
        sentences = random.sample(sentences, 50)
    if not sentences:
        return lang_code, None, None
    phonemes = phonemize(sentences, lang_code)
    if not phonemes:
        return lang_code, None, None
    phone_set = set()
    phone_counter = Counter()
    for ps in phonemes:
        if ps:
            phones = ps.strip().split()
            phone_set.update(phones)
            phone_counter.update(phones)
    return lang_code, phone_set, phone_counter


def cv_test_combined_method():
    """Shared phonemization → eSpeak Jaccard + CV-IPA JS."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    n_workers = min(multiprocessing.cpu_count(), 8)
    print('\n' + '=' * 60)
    print(f'CV TEST PHONEMIZATION (shared, {n_workers} workers)')
    print('=' * 60)

    # Parallel phonemization
    phone_sets = {}
    phone_counters = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(phonemize_cv_test_lang, lang): lang for lang in ALL_LANGS}
        for f in as_completed(futures):
            lang, pset, pcnt = f.result()
            if pset is not None:
                phone_sets[lang] = pset
                phone_counters[lang] = pcnt
                print(f'  [{lang}] {len(pset)} phones, {sum(pcnt.values())} tokens')
            else:
                print(f'  [{lang}] FAILED')

    # ── Method 2b: eSpeak Jaccard ──
    print('\n' + '=' * 60)
    print('METHOD 2b: CV TEST PHONEME SET → JACCARD')
    print('=' * 60)
    results_jaccard = {}
    for tgt in ALL_TARGETS:
        t_set = phone_sets.get(tgt)
        if not t_set:
            results_jaccard[tgt] = None
            continue
        scores = [(c, round(jaccard(t_set, phone_sets[c]), 4))
                  for c in CANDIDATES if c in phone_sets]
        scores.sort(key=lambda x: -x[1])
        results_jaccard[tgt] = scores
        best = scores[0] if scores else ('N/A', 0)
        print(f'  {tgt} → {best[0]} ({best[1]:.4f})')

    # ── Method 3b: CV-IPA JS ──
    print('\n' + '=' * 60)
    print('METHOD 3b: CV TEST PHONEME DISTRIBUTION → JS')
    print('=' * 60)
    results_js = {}
    for tgt in ALL_TARGETS:
        t_cnt = phone_counters.get(tgt)
        if not t_cnt:
            results_js[tgt] = None
            continue
        scores = [(c, round(js_similarity(t_cnt, phone_counters[c]), 4))
                  for c in CANDIDATES if c in phone_counters]
        scores.sort(key=lambda x: -x[1])
        results_js[tgt] = scores
        best = scores[0] if scores else ('N/A', 0)
        print(f'  {tgt} → {best[0]} ({best[1]:.4f})')

    return results_jaccard, results_js


# ═══════════════════════════════════════════════════════
# METHOD 4: TRAINED CLASSIFIER EMBEDDING (36-way lang classifier)
# ═══════════════════════════════════════════════════════
CLASSIFIER_DIR = "/mnt/storage/qisheng/github/wav2vec_contrastive"
CLASSIFIER_CKPT = os.path.join(CLASSIFIER_DIR, "weights/CLASSIFIER_36LANG_200SAMPLE/checkpoint_latest.pt")

def get_clf_embedding(model, feature_extractor, lang_code, device='cpu'):
    """Extract embedding using the trained 36-way classifier."""
    from datasets import load_dataset
    ds_name = 'fsicoli/common_voice_22_0'
    try:
        ds = load_dataset(ds_name, lang_code, split='train',
                          trust_remote_code=True, cache_dir=CACHE_DIR)
    except Exception:
        return None
    if len(ds) > 50:
        ds = ds.shuffle(seed=42).select(range(50))
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))

    # Same vectorize approach as extract_classifier_embeddings.py
    from torch.utils.data import DataLoader
    from datasets import Dataset as HFDataset

    embs = []
    for ex in ds:
        try:
            audio = ex['audio']
            inputs = feature_extractor(audio['array'], sampling_rate=16000, return_tensors='pt')
            inp = inputs['input_values'].to(device)
            attn = inputs.get('attention_mask')
            if attn is not None:
                attn = attn.to(device)
            with torch.no_grad():
                feat = model.get_embedding(inp, attn).squeeze().cpu().numpy()
            embs.append(feat)
        except Exception:
            continue
    if not embs:
        return None
    return np.mean(embs, axis=0)


def classifier_embedding_method():
    print('\n' + '=' * 60)
    print('METHOD 4: TRAINED CLASSIFIER EMBEDDING (36-way)')
    print('=' * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  Device: {device}')

    # Load trained classifier model
    sys.path.insert(0, os.path.join(CLASSIFIER_DIR, "customized"))
    from model_classifier import Wav2Vec2ForLanguageClassification
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    config.num_labels = 36
    fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53",
                                                trust_remote_code=True, cache_dir=CACHE_DIR)
    model = Wav2Vec2ForLanguageClassification.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53", config=config, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location=device))
    model = model.to(device).eval()
    print(f'  Loaded classifier from {CLASSIFIER_CKPT}')

    embeddings = {}
    for lang in ALL_LANGS:
        print(f'  [{lang}] ...', end=' ', flush=True)
        emb = get_clf_embedding(model, fe, lang, device)
        if emb is not None:
            embeddings[lang] = emb
            print(f'✓ (dim={emb.shape[0]})')
        else:
            print('✗')

    results = {}
    for tgt in ALL_TARGETS:
        if tgt not in embeddings:
            results[tgt] = None
            print(f'  {tgt}: NO EMBEDDING')
            continue
        scores = [(c, round(float(np.dot(embeddings[tgt], embeddings[c])), 4))
                  for c in CANDIDATES if c in embeddings]
        scores.sort(key=lambda x: -x[1])
        results[tgt] = scores
        best = scores[0] if scores else ('N/A', 0)
        print(f'  {tgt} → {best[0]} ({best[1]:.4f})')
    return results


# ═══════════════════════════════════════════════════════
# Wikipedia fallback (only if CV fails)
# ═══════════════════════════════════════════════════════
def get_wikipedia_text(lang_code, n_sentences=80):
    """Get sentences from Wikipedia for a language."""
    wiki_code = lang_code
    if lang_code == 'yue':
        wiki_code = 'zh-yue'
    elif lang_code == 'zh':
        wiki_code = 'zh'

    url = (f'https://{wiki_code}.wikipedia.org/w/api.php'
           f'?action=query&format=json&generator=random&grnnamespace=0'
           f'&grnlimit=10&prop=extracts&explaintext=1'
           f'&maxlag=5')
    sentences = []
    for attempt in range(10):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (ResearchProject/1.0; mailto:research@example.com)'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            if not pages:
                print(f'  (no pages returned, attempt {attempt+1})', end=' ', flush=True)
                time.sleep(3)
                continue
            for page in pages.values():
                text = page.get('extract', '')
                for s in re.split(r'(?<=[.!?])\s+', text):
                    s = s.strip().replace('\n', ' ')
                    if 20 < len(s) < 500:
                        sentences.append(s)
            if len(sentences) >= n_sentences:
                break
            time.sleep(2)  # Be nice to Wikipedia API
        except urllib.request.HTTPError as e:
            print(f'  HTTP {e.code} (attempt {attempt+1})', end=' ', flush=True)
            time.sleep(5 * (attempt + 1))  # Exponential backoff
        except Exception as e:
            print(f'  {type(e).__name__} (attempt {attempt+1})', end=' ', flush=True)
            time.sleep(3)
    return sentences[:n_sentences] if sentences else None


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+',
                        choices=['fast', 'cv', 'wiki', 'classifier', 'contrastive', 'all'],
                        default=['all'], help='Methods to run')
    args = parser.parse_args()
    run = args.methods
    if 'all' in run:
        run = ['fast', 'cv', 'wiki', 'classifier', 'contrastive']

    all_results = {'candidates': CANDIDATES, 'targets': ALL_TARGETS}
    out_dir = '/mnt/storage/qisheng/github/wav2vec_test/src/source_selection'

    # Job 2: Family + PHOIBLE + Wiki (fast, no GPU)
    if 'fast' in run:
        sf = same_family()
        all_results['same_family'] = sf
        with open(f'{out_dir}/result_family.json', 'w') as f:
            json.dump(sf, f, indent=2)

        ph = phoible_jaccard_method()
        all_results['phoible'] = ph
        with open(f'{out_dir}/result_phoible.json', 'w') as f:
            json.dump(ph, f, indent=2)

    if 'wiki' in run:
        js = wikipedia_phoneme_method()
        all_results['wikipedia_ipa'] = js
        with open(f'{out_dir}/result_wiki.json', 'w') as f:
            json.dump(js, f, indent=2)

    # Job 1: CV test phonemization → eSpeak Jaccard + CV-IPA JS (parallel, no GPU)
    if 'cv' in run:
        es, cvtest = cv_test_combined_method()
        all_results['espeak_inventory'] = es
        all_results['cv_test_distribution'] = cvtest
        with open(f'{out_dir}/result_espeak.json', 'w') as f:
            json.dump(es, f, indent=2)
        with open(f'{out_dir}/result_cvtest.json', 'w') as f:
            json.dump(cvtest, f, indent=2)

    # Method 4: Classifier (GPU)
    if 'classifier' in run:
        ce = classifier_embedding_method()
        all_results['classifier_embedding'] = ce
        with open(f'{out_dir}/result_classifier.json', 'w') as f:
            json.dump(ce, f, indent=2)

    # Method 5: Contrastive (GPU) — loads from prior run
    if 'contrastive' in run:
        try:
            with open(f'{out_dir}/contrastive_source_results.json') as f:
                cr = json.load(f)['results']
        except:
            cr = {}
        all_results['contrastive'] = cr
        # Filter to only NEW_TARGETS
        cr_filtered = {t: cr.get(t) for t in ALL_TARGETS}
        with open(f'{out_dir}/result_contrastive.json', 'w') as f:
            json.dump(cr_filtered, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────
    # Load all results (merge from separate files + contrastive)
    def load_json(path):
        try:
            with open(path) as f:
                return json.load(f)
        except:
            return None

    sf = load_json(f'{out_dir}/result_family.json') or {}
    ph = load_json(f'{out_dir}/result_phoible.json') or {}
    es = load_json(f'{out_dir}/result_espeak.json') or {}
    js = load_json(f'{out_dir}/result_wiki.json') or {}
    cvtest = load_json(f'{out_dir}/result_cvtest.json') or {}
    ce = load_json(f'{out_dir}/result_classifier.json') or {}
    try:
        with open(f'{out_dir}/contrastive_source_results.json') as f:
            cr = json.load(f)['results']
    except:
        cr = {}

    print('\n' + '=' * 170)
    print('SUMMARY: Source Selection for New Target Languages')
    print('=' * 170)
    header = (f"{'Target':<8} {'Name':<12} {'Family':<14} {'PHOIBLE':<10} {'eSpeak':<10} "
              f"{'Wiki-IPA':<10} {'CV-IPA':<10} "
              f"{'Clf#1':<10} {'Clf#2':<10} {'Clf#3':<10} "
              f"{'Con#1':<10} {'Con#2':<10} {'Con#3':<10}")
    print(header)
    print('-' * 170)
    for t in ALL_TARGETS:
        name = TARGET_NAMES[t]
        sf_s = sf.get(t, '—')
        ph_s = ph.get(t)
        ph_str = f'{ph_s[0][0]}({ph_s[0][1]:.3f})' if ph_s and ph_s[0] else '—'
        es_s = es.get(t)
        es_str = f'{es_s[0][0]}({es_s[0][1]:.3f})' if es_s and es_s[0] else '—'
        js_s = js.get(t)
        js_str = f'{js_s[0][0]}({js_s[0][1]:.3f})' if js_s and js_s[0] else '—'
        cv_s = cvtest.get(t)
        cv_str = f'{cv_s[0][0]}({cv_s[0][1]:.3f})' if cv_s and cv_s[0] else '—'

        ce_s = ce.get(t)
        if ce_s:
            ce_1 = f'{ce_s[0][0]}({ce_s[0][1]:.3f})'
            ce_2 = f'{ce_s[1][0]}({ce_s[1][1]:.3f})' if len(ce_s) > 1 else '—'
            ce_3 = f'{ce_s[2][0]}({ce_s[2][1]:.3f})' if len(ce_s) > 2 else '—'
        else:
            ce_1 = ce_2 = ce_3 = '—'

        ct_s = cr.get(t)
        if ct_s:
            ct_1 = f'{ct_s[0][0]}({ct_s[0][1]:.3f})'
            ct_2 = f'{ct_s[1][0]}({ct_s[1][1]:.3f})' if len(ct_s) > 1 else '—'
            ct_3 = f'{ct_s[2][0]}({ct_s[2][1]:.3f})' if len(ct_s) > 2 else '—'
        else:
            ct_1 = ct_2 = ct_3 = '—'

        print(f'{t:<8} {name:<12} {sf_s:<14} {ph_str:<10} {es_str:<10} '
              f'{js_str:<10} {cv_str:<10} '
              f'{ce_1:<10} {ce_2:<10} {ce_3:<10} '
              f'{ct_1:<10} {ct_2:<10} {ct_3:<10}')

    all_results['contrastive'] = cr
    out_path = '/mnt/storage/qisheng/github/wav2vec_test/src/source_selection/new_targets_source_selection.json'

    def clean(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=clean)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
