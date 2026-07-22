"""
Source Language Selection — Baselines 2 & 3

Method 2: Phoneme Inventory Overlap → uses PHOIBLE database
Method 3: IPA Distribution Similarity → uses Wikipedia texts + eSpeak phonemization

No audio download needed. Runs in ~2-3 minutes.
"""
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter

# ── Configuration ─────────────────────────────────────────────────

TARGETS = ["mt", "af", "da", "ky", "tk", "kk", "sk", "id"]
TARGET_NAMES = {
    "mt": "Maltese", "af": "Afrikaans", "da": "Danish",
    "ky": "Kyrgyz", "tk": "Turkmen", "kk": "Kazakh",
    "sk": "Slovak", "id": "Indonesian",
}

CANDIDATES = [
    "ar", "ba", "eu", "be", "bn", "ca", "yue", "cs", "nl", "en",
    "eo", "fa", "fr", "ka", "de", "hu", "it", "ja", "lv", "lt",
    "pl", "pt", "ro", "ru", "uk", "es", "sw", "ta", "th", "tt",
    "tr", "ug", "ur", "uz", "cy", "zh",
]

# ISO 639-1 → ISO 639-3 mapping
ISO1_TO_ISO3 = {
    "af": "afr", "ar": "ara", "ba": "bak", "be": "bel", "bn": "ben",
    "ca": "cat", "cs": "ces", "cy": "cym", "da": "dan", "de": "deu",
    "en": "eng", "eo": "epo", "es": "spa", "eu": "eus", "fa": "fas",
    "fr": "fra", "hu": "hun", "id": "ind", "it": "ita", "ja": "jpn",
    "ka": "kat", "kk": "kaz", "ky": "kir", "lt": "lit", "lv": "lav",
    "mt": "mlt", "nl": "nld", "pl": "pol", "pt": "por", "ro": "ron",
    "ru": "rus", "sk": "slk", "sw": "swa", "ta": "tam", "th": "tha",
    "tk": "tuk", "tr": "tur", "tt": "tat", "ug": "uig", "uk": "ukr",
    "ur": "urd", "uz": "uzb", "zh": "cmn", "yue": "yue",
    # Languages not in ISO 639-1 but we have espeak codes
}
# Map the ones phonemizer uses
ISO3_TO_PHONEMIZER = {
    "cmn": "cmn", "yue": "yue", "fra": "fr-fr", "eng": "en-us",
}

# ── Step 1: PHOIBLE Phoneme Inventories ──────────────────────────

def load_phoible(path="/tmp/phoible.csv"):
    """Load phoneme inventories from PHOIBLE. Returns {iso6393: set(phonemes)}."""
    inventories = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iso = row["ISO6393"]
            phoneme = row["Phoneme"]
            marginal = row["Marginal"]
            if marginal and marginal.lower() == "true":
                continue  # skip marginal phonemes
            if iso not in inventories:
                inventories[iso] = set()
            inventories[iso].add(phoneme)
    return inventories


# ── Step 2: Wikipedia Text Samples ────────────────────────────────

def get_wikipedia_text(lang_code, n_sentences=100):
    """Get sample sentences from Wikipedia for a language using the API.
    Returns list of sentence strings."""
    # Map language code to Wikipedia language code
    wiki_code = lang_code
    if lang_code == "yue":
        wiki_code = "zh-yue"
    elif lang_code == "zh":
        wiki_code = "zh"

    # Try to get Wikipedia article extracts
    url = (
        f"https://{wiki_code}.wikipedia.org/w/api.php"
        f"?action=query&format=json&generator=random&grnnamespace=0"
        f"&grnlimit=10&prop=extracts&exintro=1&explaintext=1"
    )
    sentences = []
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HermesResearch/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                text = page.get("extract", "")
                # Split into sentences
                sents = re.split(r'(?<=[.!?])\s+', text)
                for s in sents:
                    s = s.strip()
                    if len(s) > 20 and len(s) < 500:
                        sentences.append(s)
            if len(sentences) >= n_sentences:
                break
        except Exception as e:
            print(f"    Wikipedia attempt {attempt+1} failed: {e}", file=sys.stderr)
            time.sleep(1)

    if len(sentences) > n_sentences:
        sentences = sentences[:n_sentences]

    return sentences if sentences else None


# ── Step 3: Phonemization ─────────────────────────────────────────

PHONEMIZER_MAP = {
    "fr": "fr-fr", "en": "en-us", "zh": "cmn", "yue": "yue",
}


def phonemize(texts, lang_code):
    """Phonemize texts using phonemizer + espeak-ng."""
    from phonemizer.backend import BACKENDS
    from phonemizer.separator import Separator

    ph_code = PHONEMIZER_MAP.get(lang_code, lang_code)
    try:
        backend = BACKENDS["espeak"](ph_code, language_switch="remove-flags")
    except Exception:
        return None

    sep = Separator(phone=" ", word="", syllable="")
    results = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i+50]
        try:
            phonemes = backend.phonemize(batch, separator=sep, strip=True)
            results.extend(phonemes)
        except Exception:
            continue
    return results


# ── Step 4: Similarities ──────────────────────────────────────────

def jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


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
    js = (kl_pm + kl_qm) / 2
    return 1.0 - js


# ── Main ──────────────────────────────────────────────────────────

def main():
    # Load PHOIBLE
    print("Loading PHOIBLE database...")
    phoible = load_phoible()
    print(f"  Loaded {len(phoible)} language inventories from PHOIBLE")

    # Map our codes to PHOIBLE (ISO 639-3)
    def get_phoible_inventory(lang_code):
        iso3 = ISO1_TO_ISO3.get(lang_code, lang_code)
        if iso3 in phoible:
            return phoible[iso3]
        # Try alternate codes
        for k, v in phoible.items():
            if k.startswith(iso3):
                return v
        return None

    # Check coverage
    all_langs = TARGETS + CANDIDATES
    phoible_hits = 0
    for lang in all_langs:
        inv = get_phoible_inventory(lang)
        if inv:
            phoible_hits += 1
    print(f"  PHOIBLE coverage: {phoible_hits}/{len(all_langs)} languages")

    # ── METHOD 2: Phoneme Inventory Overlap (PHOIBLE) ──────────
    print()
    print("=" * 70)
    print("METHOD 2: PHONEME INVENTORY OVERLAP (PHOIBLE Jaccard)")
    print("=" * 70)

    jaccard_results = {}
    for target in TARGETS:
        t_inv = get_phoible_inventory(target)
        if t_inv is None:
            print(f"  {target}: NO PHOIBLE DATA")
            continue
        name = TARGET_NAMES[target]
        print(f"\n--- {target} ({name}) [{len(t_inv)} phones] ---")
        scores = []
        for cand in CANDIDATES:
            if cand == target:
                continue
            c_inv = get_phoible_inventory(cand)
            if c_inv is None:
                continue
            jac = jaccard(t_inv, c_inv)
            scores.append((cand, round(jac, 4)))
        scores.sort(key=lambda x: -x[1])
        jaccard_results[target] = scores
        for rank, (c, s) in enumerate(scores[:5], 1):
            print(f"  {rank}. {c} ({len(get_phoible_inventory(c))} phones): {s:.4f}")

    # ── METHOD 3: IPA Distribution Similarity ──────────────────
    print()
    print("=" * 70)
    print("METHOD 3: IPA DISTRIBUTION SIMILARITY (Wikipedia + eSpeak)")
    print("=" * 70)

    distributions = {}  # lang → Counter

    for lang in all_langs:
        name = TARGET_NAMES.get(lang, lang)
        print(f"  [{lang}] {name} ...", end=" ", flush=True)
        texts = get_wikipedia_text(lang, n_sentences=80)
        if not texts:
            print("NO WIKIPEDIA DATA")
            continue
        phonemes = phonemize(texts, lang)
        if not phonemes:
            print("PHONEMIZER FAILED")
            continue
        dist = Counter()
        for ps in phonemes:
            if ps:
                dist.update(ps.strip().split())
        distributions[lang] = dist
        print(f"{len(dist)} phones, {sum(dist.values())} tokens")

    print()
    js_results = {}
    for target in TARGETS:
        if target not in distributions:
            print(f"  {target}: NO DISTRIBUTION DATA")
            continue
        name = TARGET_NAMES[target]
        t_dist = distributions[target]
        print(f"\n--- {target} ({name}) ---")
        scores = []
        for cand in CANDIDATES:
            if cand == target or cand not in distributions:
                continue
            sim = js_similarity(t_dist, distributions[cand])
            scores.append((cand, round(sim, 4)))
        scores.sort(key=lambda x: -x[1])
        js_results[target] = scores
        for rank, (c, s) in enumerate(scores[:5], 1):
            print(f"  {rank}. {c}: {s:.4f}")

    # ── Summary ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY: TOP SOURCE PER TARGET")
    print("=" * 70)
    print(f"{'Target':<12} {'Jaccard (M2)':<22} {'JS_dist (M3)':<22}")
    print("-" * 56)
    for target in TARGETS:
        name = TARGET_NAMES[target]
        m2_top = jaccard_results.get(target, [("N/A", 0)])[0]
        m3_top = js_results.get(target, [("N/A", 0)])[0]
        print(f"{name:<12} {m2_top[0]:<10} ({m2_top[1]:.4f})  {m3_top[0]:<10} ({m3_top[1]:.4f})")

    # ── Save ──
    out = {
        "method": "PHOIBLE + Wikipedia/eSpeak",
        "method2_jaccard_phoible": {
            t: [(c, s) for c, s in scores]
            for t, scores in jaccard_results.items()
        },
        "method3_js_distribution_wikipedia": {
            t: [(c, s) for c, s in scores]
            for t, scores in js_results.items()
        },
    }
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "source_selection_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    return jaccard_results, js_results


if __name__ == "__main__":
    main()
