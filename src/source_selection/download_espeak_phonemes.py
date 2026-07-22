"""
Download eSpeak-ng phoneme source files and extract complete phoneme inventories.
"""
import os, json, re, urllib.request

GITHUB_API = "https://api.github.com/repos/espeak-ng/espeak-ng/contents/phsource"
RAW_BASE = "https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/phsource"
CACHE_DIR = "/mnt/storage/qisheng/github/wav2vec_test/src/source_selection/espeak_ph_data"
OUT_PATH = "/mnt/storage/qisheng/github/wav2vec_test/src/source_selection/espeak_phoneme_inventories.json"

# Language code → expected ph_ file name suffix
LANG_MAP = {
    'sq': 'albanian', 'ltg': 'latvian', 'ur': 'urdu', 'cy': 'welsh',
    'gn': 'guarani', 'tn': 'setswana', 'am': 'amhari', 'he': 'hebrew', 'az': 'azerbaijani',
    'ar': 'arabic', 'ba': 'bashkir', 'ca': 'catalan', 'cs': 'czech',
    'en': 'english', 'eo': 'esperanto', 'fr': 'french', 'hu': 'hungarian',
    'it': 'italian', 'lt': 'lithuanian', 'lv': 'latvian', 'nl': 'dutch',
    'ro': 'romanian', 'ru': 'russian', 'sw': 'swahili', 'ta': 'tamil',
    'tr': 'turkish', 'tt': 'tatar', 'ug': 'uyghur',
}

os.makedirs(CACHE_DIR, exist_ok=True)

def get_available_files():
    """Get list of ph_* files from GitHub API."""
    req = urllib.request.Request(GITHUB_API, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    return {item['name'] for item in data if item['name'].startswith('ph_')}

def download_ph_file(filename):
    """Download a ph_ file, return its content."""
    cache_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return f.read()
    url = f"{RAW_BASE}/{filename}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as r:
        content = r.read().decode('utf-8')
    with open(cache_path, 'w') as f:
        f.write(content)
    return content

def extract_phonemes(content):
    """Extract phoneme names from a ph_ file.
    Phonemes are defined as: 'phoneme NAME' at the start of a line."""
    phonemes = set()
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('phoneme '):
            name = line.split()[1].strip()
            if name and not name.startswith('//') and not name.startswith('#'):
                phonemes.add(name)
    return phonemes

def main():
    print("Fetching available ph_ files...")
    available = get_available_files()
    print(f"  {len(available)} files available")
    
    inventories = {}
    for code, name in sorted(LANG_MAP.items()):
        fname = f"ph_{name}"
        if fname not in available:
            # Try alternate name
            alt = {'tn': 'tswana', 'ltg': 'latgalian'}
            alt_name = alt.get(code, name)
            fname = f"ph_{alt_name}"
        
        if fname not in available:
            print(f"  ✗ {code} ({name}): file not found")
            continue
        
        print(f"  [{code}] downloading {fname}...", end=' ', flush=True)
        content = download_ph_file(fname)
        phones = extract_phonemes(content)
        if phones:
            inventories[code] = sorted(phones)
            print(f"{len(phones)} phonemes")
        else:
            print("FAILED (no phonemes extracted)")
    
    # Save
    with open(OUT_PATH, 'w') as f:
        json.dump(inventories, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_PATH}")
    print(f"  {len(inventories)}/{len(LANG_MAP)} languages covered")
    
    # Show what we got
    for code in sorted(inventories):
        print(f"  {code}: {len(inventories[code])} phonemes: {', '.join(inventories[code][:10])}...")


if __name__ == '__main__':
    main()
