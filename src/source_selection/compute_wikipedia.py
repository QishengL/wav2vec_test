"""
Wikipedia IPA Distribution similarity (via eSpeak phonemization).
Computes JS divergence between target and candidate phoneme distributions.
"""
import json, os, re, urllib.request, time
from collections import Counter
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from phonemizer.backend import BACKENDS
from phonemizer.separator import Separator

TARGETS = ['sq','ltg','ur','cy','gn','tn','am','az','mt','af','da','ky','tk','kk','sk','id']
CANDIDATES = ['ar','ba','ca','cs','en','eo','fr','hu','it','lt','lv','nl','ro','ru','sw','ta','tr','tt','ug']
ALL = TARGETS + CANDIDATES

espeak_map = {'en': 'en-us', 'fr': 'fr-fr', 'zh': 'cmn', 'yue': 'yue'}
sep = Separator(phone=' ', word='', syllable='')

LANG_PAGE_NAMES = {
    'sq': 'Albanian_language', 'ltg': 'Latgalian_language', 'ur': 'Urdu',
    'cy': 'Welsh_language', 'gn': 'Guarani_language', 'tn': 'Tswana_language',
    'am': 'Amharic', 'az': 'Azerbaijani_language',
    'mt': 'Maltese_language', 'af': 'Afrikaans', 'da': 'Danish_language',
    'ky': 'Kyrgyz_language', 'tk': 'Turkmen_language', 'kk': 'Kazakh_language',
    'sk': 'Slovak_language', 'id': 'Indonesian_language',
    'ar': 'Arabic', 'ba': 'Bashkir_language', 'ca': 'Catalan_language',
    'cs': 'Czech_language', 'en': 'English_language', 'eo': 'Esperanto',
    'fr': 'French_language', 'hu': 'Hungarian_language', 'it': 'Italian_language',
    'lt': 'Lithuanian_language', 'lv': 'Latvian_language', 'nl': 'Dutch_language',
    'ro': 'Romanian_language', 'ru': 'Russian_language', 'sw': 'Swahili_language',
    'ta': 'Tamil_language', 'tr': 'Turkish_language', 'tt': 'Tatar_language',
    'ug': 'Uyghur_language',
}

def get_wikipedia_text(lang_code):
    """Fetch full Wikipedia article text."""
    page = LANG_PAGE_NAMES.get(lang_code, f'{lang_code}_language')
    api_url = "https://en.wikipedia.org/w/api.php"
    params = (f"?action=query&titles={page}&prop=extracts&explaintext"
              f"&format=json&redirects=1")
    time.sleep(3)  # 3s between requests to avoid rate limiting
    try:
        req = urllib.request.Request(api_url + params, headers={'User-Agent': 'Mozilla/5.0 (compatible; Hermes)'})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print("  RATE LIMITED, waiting 60s...", flush=True)
                time.sleep(60)
                return get_wikipedia_text(lang_code)
            raise
        for pid, page in data.get('query', {}).get('pages', {}).items():
            if 'extract' in page and page['extract'].strip():
                return page['extract']
        return ''
    except:
        return ''

def phonemize(text, lang_code):
    ph_code = espeak_map.get(lang_code, lang_code)
    try:
        backend = BACKENDS['espeak'](ph_code)
        result = backend.phonemize([text], separator=sep, strip=True)
        return result[0].strip().split() if result and result[0] else []
    except:
        return []

# Step 1: get Wikipedia text + phonemize for all languages
print("Wikipedia → eSpeak phonemization", flush=True)
all_phonemes = {}
for i, lang in enumerate(ALL):
    # Skip if cached
    cache = f'/tmp/wiki_phones_{lang}.json'
    if os.path.exists(cache):
        with open(cache) as f:
            all_phonemes[lang] = Counter(json.load(f))
        print(f"  [{i+1}/{len(ALL)}] {lang}: loaded from cache ({sum(all_phonemes[lang].values())} phones)", flush=True)
        continue
    
    text = get_wikipedia_text(lang)
    if not text:
        print(f"  [{i+1}/{len(ALL)}] {lang}: no text", flush=True)
        all_phonemes[lang] = Counter()
        continue
    
    phones = phonemize(text, lang)
    c = Counter(phones)
    all_phonemes[lang] = c
    with open(cache, 'w') as f:
        json.dump(dict(c), f)
    print(f"  [{i+1}/{len(ALL)}] {lang}: {len(text)} chars, {len(phones)} phones, {len(c)} unique", flush=True)
    time.sleep(1.5)  # be nice to Wikipedia

# Step 2: build vocabulary
vocab = sorted(set().union(*[set(c.keys()) for c in all_phonemes.values()]))
print(f"\nTotal unique phonemes: {len(vocab)}", flush=True)

# Step 3: build normalized distributions
def to_dist(counter):
    d = np.zeros(len(vocab), dtype=np.float32)
    for ph, cnt in counter.items():
        d[vocab.index(ph)] = cnt
    total = d.sum()
    if total > 0:
        d /= total
    else:
        d[:] = 1.0 / len(vocab)
    return d

dists = {lang: to_dist(all_phonemes[lang]) for lang in ALL}

# Step 4: JS similarity
print("\nJS similarity (target vs candidate):", flush=True)
wiki_results = {}
for tgt in TARGETS:
    td = dists[tgt]
    scores = [(c, round(1 - float(jensenshannon(td, dists[c])), 4)) for c in CANDIDATES]
    scores.sort(key=lambda x: -x[1])
    wiki_results[tgt] = scores
    print(f"  {tgt} → {scores[0][0]}({scores[0][1]:.4f})", flush=True)

with open('results/ablation/wiki_n=50.json', 'w') as f:
    json.dump(wiki_results, f, indent=2)
print("Saved", flush=True)

# Step 5: Spearman
with open('results/s2_results.json') as f:
    s2 = json.load(f)

pairs = []
seen = set()
for exp in s2:
    e = exp.get('experiment', '')
    if 'base' in e or 'direct' in e or '+' in e or 'multi' in e:
        continue
    t, per = exp['target_lang'], exp['heldout_test_wer']
    if per is None: continue
    src = e.rsplit('_', 1)[-1].replace('53','') if '_' in e else ''
    key = (t, src)
    if key not in seen and src:
        seen.add(key)
        pairs.append({'t':t, 's':src, 'per':per})

sims, pers = [], []
for p in pairs:
    if p['t'] not in wiki_results: continue
    for c, sim in wiki_results[p['t']]:
        if c == p['s']:
            sims.append(sim); pers.append(p['per']); break

r, pv = spearmanr(sims, pers) if len(sims) >= 4 else (0, 1)
sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else 'ns'
print(f"\nWikipedia IPA: N={len(sims)}, r={r:.4f}, p={pv:.4f} {sig}")
