"""
Same-Family Source Language Selection (Heuristic)

For each target language, select the most representative / well-known
source language from the same language family among the 36 candidates.

Family mapping based on standard linguistic classification (Ethnologue / Glottolog).
"""

TARGETS = ["mt", "af", "da", "ky", "tk", "kk", "sk", "id"]
TARGET_NAMES = {
    "mt": "Maltese", "af": "Afrikaans", "da": "Danish",
    "ky": "Kyrgyz", "tk": "Turkmen", "kk": "Kazakh",
    "sk": "Slovak", "id": "Indonesian",
}

# Language family classification of all 36 candidate source languages
FAMILIES = {
    # Semitic
    "ar": "Semitic",
    "mt": "Semitic",   # target
    # Germanic
    "en": "Germanic", "de": "Germanic", "nl": "Germanic",
    "af": "Germanic",  # target
    "da": "Germanic",  # target
    # Turkic
    "tr": "Turkic", "ba": "Turkic", "tt": "Turkic",
    "ug": "Turkic", "uz": "Turkic",
    "ky": "Turkic",  # target
    "tk": "Turkic",  # target
    "kk": "Turkic",  # target
    # Slavic
    "ru": "Slavic", "uk": "Slavic", "be": "Slavic",
    "pl": "Slavic", "cs": "Slavic",
    "sk": "Slavic",  # target
    # Malayo-Polynesian (Austronesian)
    "id": "Austronesian",  # target — NO same-family candidate in pool
    # Romance
    "fr": "Romance", "es": "Romance", "pt": "Romance",
    "ca": "Romance", "ro": "Romance", "it": "Romance",
    # Other families (one or two languages each)
    "eu": "Isolate (Basque)",
    "be": "Slavic",  # already listed above
    "bn": "Indo-Aryan",
    "yue": "Sinitic", "zh": "Sinitic",
    "eo": "Constructed",
    "fa": "Iranian",
    "ka": "Kartvelian",
    "hu": "Uralic",
    "ja": "Japonic",
    "lv": "Baltic", "lt": "Baltic",
    "sw": "Bantu",
    "ta": "Dravidian",
    "th": "Tai-Kadai",
    "ur": "Indo-Aryan",
    "cy": "Celtic",
}

# Same-family candidates for each target
SAME_FAMILY_CANDIDATES = {
    "mt": ["ar"],                                                    # Semitic: only Arabic
    "af": ["en", "de", "nl"],                                        # Germanic
    "da": ["en", "de", "nl"],                                        # Germanic
    "ky": ["tr", "ba", "tt", "ug", "uz"],                           # Turkic
    "tk": ["tr", "ba", "tt", "ug", "uz"],                           # Turkic
    "kk": ["tr", "ba", "tt", "ug", "uz"],                           # Turkic
    "sk": ["ru", "uk", "be", "pl", "cs"],                           # Slavic
    "id": [],                                                        # Austronesian: NONE in candidate pool
}

# Heuristic selection: pick the most representative / well-known language per family
# Rationale: largest speaker population, most studied, or most resource-rich
HEURISTIC_CHOICE = {
    "Semitic":       "ar",    # Arabic — 400M+ speakers, most resourced Semitic language
    "Germanic":      "en",    # English — most data, most studied Germanic language
    "Turkic":        "tr",    # Turkish — 80M+ speakers, most resourced Turkic language
    "Slavic":        "ru",    # Russian — 150M+ speakers, most resourced Slavic language
    "Austronesian":  "ta",    # Fallback: Tamil — geographically close, historic trade influence, many Sanskrit loanwords shared with Indonesian
}

# ── Main output ───────────────────────────────────────────────────

def main():
    print("Same-Family Source Language Selection (Heuristic)")
    print("=" * 56)
    print(f"{'Target':<14} {'Family':<16} {'Source':<10} {'Rationale'}")
    print("-" * 56)

    results = {}
    for code in TARGETS:
        name = TARGET_NAMES[code]
        family = FAMILIES.get(code, "Unknown")
        source = HEURISTIC_CHOICE.get(family)

        if source is None:
            rationale = "No same-family candidate in pool"
        elif code == "id":
            rationale = "Fallback: Tamil — geographically close, historic influence"
        elif source == code:
            rationale = "Target is the representative itself (should not happen)"
        else:
            # List same-family alternatives
            alts = SAME_FAMILY_CANDIDATES.get(code, [])
            alts_str = ", ".join(a for a in alts if a != source)
            rationale = f"Most resourced {family} language"
            if alts_str:
                rationale += f" (alternatives: {alts_str})"

        print(f"{name:<14} {family:<16} {source or 'N/A':<10} {rationale}")
        results[code] = {
            "target": name,
            "family": family,
            "source": source,
            "alternatives": SAME_FAMILY_CANDIDATES.get(code, []),
            "rationale": rationale,
        }

    print()
    print("Note: Indonesian has no Austronesian candidate in the pool.")
    print("      Fallback heuristic: Tamil (ta) — geographically close, historic influence.")
    print()
    print("Same-family → target pairs to train:")
    for code in TARGETS:
        src = results[code]["source"]
        if src:
            print(f"  {src} → {code}  ({results[code]['target']})")

    return results


if __name__ == "__main__":
    main()
