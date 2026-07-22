"""
Two new teacher methods:
A. lang2vec typological vectors — cosine similarity
B. Weighted IPA distance — feature-weighted phoneme edit distance

Both compute language×language similarity matrices for 36 training languages.
Then evaluate Spearman correlation with transfer PER.
No target language data used.
"""
import json, os
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

OUT_DIR = '/mnt/storage/qisheng/github/wav2vec_test/results/teacher_comparison'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Language mappings ──
LAN_ORDER = ['ar', 'ba', 'eu', 'be', 'bn', 'ca', 'yue', 'cs', 'nl', 'en', 'eo',
             'fa', 'fr', 'ka', 'de', 'hu', 'it', 'ja', 'lv', 'lt', 'pl', 'pt',
             'ro', 'ru', 'uk', 'es', 'sw', 'ta', 'th', 'tt', 'tr', 'ug', 'ur',
             'uz', 'cy', 'zh-CN']

# ISO 639-3 codes for lang2vec
ISO3 = {
    'ar':'arb','ba':'bak','eu':'eus','be':'bel','bn':'ben','ca':'cat',
    'yue':'yue','cs':'ces','nl':'nld','en':'eng','eo':'epo','fa':'fas',
    'fr':'fra','ka':'kat','de':'deu','hu':'hun','it':'ita','ja':'jpn',
    'lv':'lvs','lt':'lit','pl':'pol','pt':'por','ro':'ron','ru':'rus',
    'uk':'ukr','es':'spa','sw':'swh','ta':'tam','th':'tha','tt':'tat',
    'tr':'tur','ug':'uig','ur':'urd','uz':'uzb','cy':'cym',
}

TARGETS = ['sq','ltg','ur','cy','gn','tn','am','az','mt','af','da','ky','tk','kk','sk','id']

print("=" * 60)
print("A. Typological vectors (lang2vec)")
print("=" * 60)

import lang2vec
import urllib.request, zipfile, os as os2

# Download URIEL features if not cached
URIEL_CACHE = '/tmp/uriel_features'
URIEL_URL = "http://www.cs.cmu.edu/~dmortens/uriel_v0.5.tar.gz"

# lang2vec typically downloads models on first use
# Let it download

from lang2vec.lang2vec import get_features, available_feature_sets

featsets = available_feature_sets()
print(f"Available: {featsets}", flush=True)

# Get all our languages' iso3 codes
our_iso3 = [ISO3.get(l, l) for l in LAN_ORDER if l in ISO3]
print(f"Querying {len(our_iso3)} languages", flush=True)

# Feature sets most relevant for phoneme recognition
#   inventory_phoible_aa — PHOIBLE all-author inventory features
#   phonology_wals      — WALS phonological features (tone, vowel harmony, etc.)
#   geo                 — geographic coordinates
#   fam                 — language family vector
#   inventory_phoible_aa+phonology_wals+geo — concatenation (combines all)
FSETS = [
    'inventory_phoible_aa',
    'phonology_wals',
    'geo',
    'fam',
    'inventory_phoible_aa+phonology_wals',
    'inventory_phoible_aa+phonology_wals+geo',
]
results_a = {}
for fset in FSETS:
    try:
        feat_dict = get_features(our_iso3, fset, header=True, minimal=False)
        langs_list = [k for k in feat_dict if k != 'CODE']
        n = len(langs_list)
        if n < 2:
            print(f"  {fset}: only {n} langs, skipping", flush=True)
            continue
        
        # Convert dict to numpy array, replacing '--' (missing) with NaN
        feat_array = np.array([
            [np.nan if v == '--' else float(v) for v in feat_dict[l]]
            for l in langs_list
        ], dtype=np.float32)
        
        # Drop all-NaN columns (no data for any language)
        good_cols = ~np.all(np.isnan(feat_array), axis=0)
        feat_array = feat_array[:, good_cols]
        
        n_feats = feat_array.shape[1]
        if n_feats == 0:
            print(f"  {fset:<40}: {n} langs, 0 feats (all missing) — SKIP", flush=True)
            continue
        
        # Impute remaining NaN with column mean
        col_means = np.nanmean(feat_array, axis=0, keepdims=True)
        feat_array = np.where(np.isnan(feat_array), col_means, feat_array)
        
        # Compute cosine similarity matrix
        norms = np.linalg.norm(feat_array, axis=1, keepdims=True)
        feat_array_normed = feat_array / np.where(norms > 0, norms, 1.0)
        sim_matrix = feat_array_normed @ feat_array_normed.T
        
        mean_sim = sim_matrix.mean()
        std_sim = sim_matrix.std()
        n_feats = len(feat_dict.get('CODE', []))
        print(f"  {fset:<40}: {n} langs, {n_feats} feats, μ={mean_sim:.4f}, σ={std_sim:.4f}", flush=True)
        
        # Build similarity dict for Spearman
        results_a[fset] = {}
        for i, li in enumerate(langs_list):
            for j, lj in enumerate(langs_list):
                if i >= j: continue
                results_a[fset][(li, lj)] = float(sim_matrix[i, j])
        
    except Exception as e:
        import traceback
        print(f"  {fset}: FAILED - {e}", flush=True)
        traceback.print_exc()

# ── Weighted IPA distance ──
print(f"\n{'='*60}")
print("B. Weighted IPA distance")
print(f"{'='*60}")

# IPA feature matrix: each phoneme → feature vector
# Basic IPA features: place, manner, voicing
IPA_FEATURES = {
    # Plosives
    'p': [0,0,0], 'b': [0,0,1], 't': [1,0,0], 'd': [1,0,1],
    'ʈ': [2,0,0], 'ɖ': [2,0,1], 'c': [3,0,0], 'ɟ': [3,0,1],
    'k': [4,0,0], 'ɡ': [4,0,1], 'q': [5,0,0], 'ɢ': [5,0,1],
    'ʔ': [6,0,0],
    # Nasals
    'm': [0,1,1], 'ɱ': [0.5,1,1], 'n': [1,1,1],
    'ɳ': [2,1,1], 'ɲ': [3,1,1], 'ŋ': [4,1,1], 'ɴ': [5,1,1],
    # Trills
    'ʙ': [0,2,1], 'r': [1,2,1], 'ʀ': [5,2,1],
    # Taps/Flaps
    'ɾ': [1,3,1], 'ɽ': [2,3,1],
    # Fricatives
    'ɸ': [0,4,0], 'β': [0,4,1], 'f': [0.5,4,0], 'v': [0.5,4,1],
    'θ': [1.2,4,0], 'ð': [1.2,4,1],
    's': [1.5,4,0], 'z': [1.5,4,1],
    'ʃ': [1.7,4,0], 'ʒ': [1.7,4,1],
    'ʂ': [2,4,0], 'ʐ': [2,4,1],
    'ç': [3,4,0], 'ʝ': [3,4,1],
    'x': [4,4,0], 'ɣ': [4,4,1],
    'χ': [5,4,0], 'ʁ': [5,4,1],
    'ħ': [5.5,4,0], 'ʕ': [5.5,4,1],
    'h': [6,4,0], 'ɦ': [6,4,1],
    # Lateral fricatives
    'ɬ': [1.5,5,0], 'ɮ': [1.5,5,1],
    # Approximants
    'ʋ': [0.5,6,1], 'ɹ': [1,6,1], 'ɻ': [2,6,1],
    'j': [3,6,1], 'ɰ': [4,6,1],
    # Lateral approximants
    'l': [1,7,1], 'ɭ': [2,7,1], 'ʎ': [3,7,1], 'ʟ': [4,7,1],
    # Vowels
    'i': [7,8,1], 'y': [7,8,1], 'ɪ': [7.5,8,1], 'ʏ': [7.5,8,1],
    'e': [8,8,1], 'ø': [8,8,1], 'ɛ': [8.5,8,1], 'œ': [8.5,8,1],
    'æ': [8.5,8.5,1], 'ɐ': [9,9,1],
    'a': [9,10,1], 'ɶ': [9,10,1],
    'ə': [8.5,9,1],
    'ɨ': [7.5,8.5,1], 'ʉ': [7.5,8.5,1],
    'ɯ': [7,8.5,1], 'u': [7,9,1],
    'ʊ': [7.5,9,1],
    'ɤ': [8,9,1], 'o': [8,9,1],
    'ʌ': [8.5,9,1], 'ɔ': [8.5,9.5,1],
    'ɑ': [9,10,1], 'ɒ': [9,10,1],
}

# Load PHOIBLE data
import csv
PHOIBLE_PATH = '/tmp/phoible.csv'
if not os.path.exists(PHOIBLE_PATH):
    import urllib.request
    print("Downloading PHOIBLE...", flush=True)
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/phoible/dev/master/data/phoible.csv',
        PHOIBLE_PATH)

phoible_invs = {}
with open(PHOIBLE_PATH, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        iso = row['ISO6393']
        if row.get('Marginal','').lower() == 'true': continue
        phoible_invs.setdefault(iso, set()).add(row['Phoneme'])

ISO1_TO_ISO3 = {
    'ar':'arb','ba':'bak','eu':'eus','be':'bel','bn':'ben','ca':'cat',
    'cs':'ces','nl':'nld','en':'eng','eo':'epo','fa':'fas','fr':'fra',
    'ka':'kat','de':'deu','hu':'hun','it':'ita','ja':'jpn','lv':'lvs',
    'lt':'lit','pl':'pol','pt':'por','ro':'ron','ru':'rus',
    'sw':'swh','ta':'tam','th':'tha','tt':'tat','tr':'tur','ug':'uig',
    'ur':'urd','uz':'uzb','cy':'cym','yue':'yue','zh-CN':'cmn',
    'sq':'als','ltg':'ltg','gn':'gug','tn':'tsn','am':'amh','az':'azj',
    'mt':'mlt','af':'afr','da':'dan','ky':'kir','tk':'tuk','kk':'kaz',
    'sk':'slk','id':'ind','he':'heb'
}

def phoneme_to_featvec(phoneme):
    """Convert phoneme string to feature vector (weighted by IPA features)."""
    # Handle multi-character phonemes (affricates, etc.)
    # Try to find the base character
    for c in reversed(phoneme):
        if c in IPA_FEATURES:
            return np.array(IPA_FEATURES[c], dtype=np.float32)
    # Return unknown phoneme as uniform
    return np.array([5, 5, 0.5], dtype=np.float32)

def inventory_distance(inv_a, inv_b):
    """Weighted distance between two phoneme inventories."""
    if not inv_a or not inv_b:
        return 1.0
    total_dist = 0.0
    count = 0
    for ph_a in inv_a:
        fv_a = phoneme_to_featvec(ph_a)
        best_dist = 1.0
        for ph_b in inv_b:
            fv_b = phoneme_to_featvec(ph_b)
            dist = np.linalg.norm(fv_a - fv_b) / np.linalg.norm(np.array([6, 10, 1]))
            best_dist = min(best_dist, dist)
        total_dist += best_dist
        count += 1
    # Symmetrize
    for ph_b in inv_b:
        fv_b = phoneme_to_featvec(ph_b)
        best_dist = 1.0
        for ph_a in inv_a:
            fv_a = phoneme_to_featvec(ph_a)
            dist = np.linalg.norm(fv_a - fv_b) / np.linalg.norm(np.array([6, 10, 1]))
            best_dist = min(best_dist, dist)
        total_dist += best_dist
        count += 1
    return total_dist / count if count > 0 else 1.0

# Compute weighted IPA similarity
print("Computing weighted IPA similarity...", flush=True)
langs_with_phoible = [l for l in LAN_ORDER if l in ISO1_TO_ISO3]
wipa_sim = np.zeros((len(langs_with_phoible), len(langs_with_phoible)))
for i, li in enumerate(langs_with_phoible):
    inv_i = phoible_invs.get(ISO1_TO_ISO3[li], set())
    for j, lj in enumerate(langs_with_phoible):
        inv_j = phoible_invs.get(ISO1_TO_ISO3[lj], set())
        dist = inventory_distance(inv_i, inv_j)
        wipa_sim[i, j] = 1.0 - dist

print(f"  {len(langs_with_phoible)} langs, μ={wipa_sim.mean():.4f}, σ={wipa_sim.std():.4f}", flush=True)

# ── Spearman evaluation ──
print(f"\n{'='*60}")
print("SPEARMAN vs TRANSFER PER")
print(f"{'='*60}")

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

def evaluate_sim_matrix(sim_matrix, langs_list, name):
    sims, pers = [], []
    for (t, s), per in per_lookup.items():
        if t in langs_list and s in langs_list:
            i = langs_list.index(t)
            j = langs_list.index(s)
            sims.append(sim_matrix[i, j])
            pers.append(per)
    if len(sims) >= 4:
        r, pv = spearmanr(sims, pers)
        sig = '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else 'ns'
        print(f"  {name:<20} r={r:>+8.4f}  p={pv:>8.4f} {sig}  N={len(sims)}")
        return {'name': name, 'r': r, 'p': pv, 'n': len(sims)}
    return None

results_all = []

# lang2vec
for fset in results_a:
    pair_dict = results_a[fset]
    sims, pers = [], []
    for (t, s), per in per_lookup.items():
        li = ISO3.get(t, t)
        lj = ISO3.get(s, s)
        key = (li, lj)
        rev_key = (lj, li)
        if key in pair_dict:
            sims.append(pair_dict[key])
            pers.append(per)
        elif rev_key in pair_dict:
            sims.append(pair_dict[rev_key])
            pers.append(per)
    if len(sims) >= 4:
        r, pv = spearmanr(sims, pers)
        sig = '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else 'ns'
        print(f"  lang2vec-{fset:<12} r={r:>+8.4f}  p={pv:>8.4f} {sig}  N={len(sims)}")
        results_all.append({'name': f'lang2vec-{fset}', 'r': r, 'p': pv, 'n': len(sims)})

# Weighted IPA
res = evaluate_sim_matrix(wipa_sim, langs_with_phoible, 'weighted_ipa')
if res: results_all.append(res)

# PHOIBLE Jaccard for comparison
print(f"\n{'='*60}")
print("Comparison with existing methods:")
print(f"{'='*60}")
print(f"  {'SupCon':<20} r={-0.437:>+8.4f}  p=0.0000***  N=94")
print(f"  {'eSpeak':<20} r={-0.347:>+8.4f}  p=0.0019**   N=94")
print(f"  {'PHOIBLE Jaccard':<20} r={-0.222:>+8.4f}  p=0.0319*   N=94")
print(f"  {'3-gram JS':<20} r={-0.251:>+8.4f}  p=0.0146*   N=94")

# Save results
with open(f'{OUT_DIR}/teacher_results.json', 'w') as f:
    json.dump(results_all, f, indent=2)
print(f"\nSaved!", flush=True)
