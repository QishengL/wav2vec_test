"""
Compute PHOIBLE Jaccard and Wikipedia IPA similarity for all targets,
then compute Spearman correlation with S2 PER.
"""
import csv, json, os, re, sys, urllib.request
from collections import Counter

TARGETS = ['sq','ltg','ur','cy','gn','tn','am','az','mt','af','da','ky','tk','kk','sk','id']
CANDIDATES = ['ar','ba','ca','cs','en','eo','fr','hu','it','lt','lv','nl','ro','ru','sw','ta','tr','tt','ug']

# ── ISO mapping ──
ISO1_TO_ISO3 = {
    'mt':'mlt','af':'afr','da':'dan','ky':'kir','tk':'tuk','kk':'kaz',
    'sk':'slk','id':'ind','sq':'als','ltg':'ltg','ur':'urd','cy':'cym',
    'gn':'gug','tn':'tsn','am':'amh','az':'azj',
    'ar':'arb','ba':'bak','ca':'cat','cs':'ces','en':'eng','eo':'epo',
    'fr':'fra','hu':'hun','it':'ita','lt':'lit','lv':'lvs','nl':'nld',
    'ro':'ron','ru':'rus','sw':'swh','ta':'tam','tr':'tur','tt':'tat','ug':'uig',
}

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# ═══════════════════════════════════════
# PHOIBLE
# ═══════════════════════════════════════
PHOIBLE_PATH = '/tmp/phoible.csv'
if not os.path.exists(PHOIBLE_PATH):
    print("Downloading PHOIBLE...", flush=True)
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/phoible/dev/master/data/phoible.csv',
        PHOIBLE_PATH)

print("Loading PHOIBLE...", flush=True)
phoible_invs = {}
with open(PHOIBLE_PATH, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        iso = row['ISO6393']
        if row.get('Marginal','').lower() == 'true':
            continue
        phoible_invs.setdefault(iso, set()).add(row['Phoneme'])

def get_phoible_inv(lang_code):
    return phoible_invs.get(ISO1_TO_ISO3.get(lang_code, lang_code), set())

print("Computing PHOIBLE Jaccard...", flush=True)
phoible_results = {}
for tgt in TARGETS:
    t_inv = get_phoible_inv(tgt)
    scores = []
    for cand in CANDIDATES:
        c_inv = get_phoible_inv(cand)
        sim = jaccard(t_inv, c_inv)
        scores.append((cand, round(sim, 4)))
    scores.sort(key=lambda x: -x[1])
    phoible_results[tgt] = scores

# Save
with open('results/ablation/phoible_n=50.json', 'w') as f:
    json.dump(phoible_results, f, indent=2)
print("PHOIBLE saved", flush=True)

# ═══════════════════════════════════════
# Wikipedia IPA
# ═══════════════════════════════════════
import unicodedata, subprocess

WIKI_URL = "https://en.wikipedia.org/w/api.php?action=query&titles={lang}&prop=extracts&exintro&explaintext&format=json&redirects=1"

print("Computing Wikipedia IPA similarity...", flush=True)
import urllib.request, json as json_lib

def get_wiki_ipa(text):
    """Extract IPA from Wikipedia text."""
    ipa_pattern = re.compile(r'/([^/]+)/')
    matches = ipa_pattern.findall(text)
    phones = set()
    for m in matches:
        clean = m.strip()
        # Remove suprasegmentals and diacritics markers
        for ch in clean:
            if ch in 'ˈˌːʼ̩̆̃ʲʷʰˑˠˤ̥̪̬̝̞̟̠̤̰̘̙ˡⁿˈˌːʼ':
                continue
            phones.add(ch)
    return phones

wiki_results = {}
for lang_code in TARGETS + CANDIDATES:
    iso3 = ISO1_TO_ISO3.get(lang_code, lang_code)
    url = f"https://en.wikipedia.org/w/api.php?action=query&titles={iso3}_language&prop=extracts&exintro&explaintext&format=json&redirects=1"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json_lib.loads(resp.read())
        pages = data.get('query', {}).get('pages', {})
        text = ''
        for pid, page in pages.items():
            if 'extract' in page:
                text = page['extract']
                break
        ipa_set = get_wiki_ipa(text)
        print(f"  {lang_code}: {len(ipa_set)} IPA phones", flush=True)
    except Exception as e:
        print(f"  {lang_code}: ERROR {e}", flush=True)
        ipa_set = set()
    
    path = f'/tmp/phoible_ipa/{lang_code}.json'
    os.makedirs('/tmp/phoible_ipa', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(list(ipa_set), f)

# Compute similarity
for tgt in TARGETS:
    with open(f'/tmp/phoible_ipa/{tgt}.json') as f:
        t_ipa = set(json.load(f))
    scores = []
    for cand in CANDIDATES:
        with open(f'/tmp/phoible_ipa/{cand}.json') as f:
            c_ipa = set(json.load(f))
        sim = jaccard(t_ipa, c_ipa)
        scores.append((cand, round(sim, 4)))
    scores.sort(key=lambda x: -x[1])
    wiki_results[tgt] = scores
    print(f"  {tgt} → {scores[0][0]}({scores[0][1]:.4f})", flush=True)

with open('results/ablation/wiki_n=50.json', 'w') as f:
    json.dump(wiki_results, f, indent=2)
print("Wikipedia IPA saved", flush=True)

# ═══════════════════════════════════════
# Spearman correlation
# ═══════════════════════════════════════
print(f"\n{'='*60}")
print("Spearman: similarity vs S2 PER")
print(f"{'='*60}")

import numpy as np
from scipy.stats import spearmanr

with open('results/s2_results.json') as f:
    s2 = json.load(f)

pairs = []
seen = set()
for exp in s2:
    e = exp.get('experiment', '')
    if 'base' in e or 'direct' in e or '+' in e or 'multi' in e:
        continue
    t = exp['target_lang']
    per = exp['heldout_test_wer']
    if per is None: continue
    src = e.rsplit('_', 1)[-1].replace('53','') if '_' in e else ''
    key = (t, src)
    if key not in seen and src:
        seen.add(key)
        pairs.append({'t':t, 's':src, 'per':per})

for method_name, results in [('phoible', phoible_results), ('wiki', wiki_results)]:
    sims, pers = [], []
    for p in pairs:
        t, s, per = p['t'], p['s'], p['per']
        if t not in results: continue
        for c, sim in results[t]:
            if c == s:
                sims.append(sim)
                pers.append(per)
                break
    if len(sims) >= 4:
        r, pv = spearmanr(sims, pers)
        print(f"  {method_name:<10} N={len(sims):>3}  r={r:>+8.4f}  p={pv:>8.4f}")
    else:
        print(f"  {method_name:<10} N={len(sims):>3}  insufficient data")
