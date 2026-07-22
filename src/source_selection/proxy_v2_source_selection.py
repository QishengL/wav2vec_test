"""
Proxy-V2 Source Selection using FEAT128BS16_PROXY_SCRATCH_V2 model.
Then compares with original contrastive (SupCon) results.
"""
import os, sys, json, numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2Config
from datasets import load_dataset
import datasets as hf_datasets
from sklearn.metrics.pairwise import cosine_similarity

CONTRASTIVE_DIR = "/mnt/storage/qisheng/github/wav2vec_contrastive"
sys.path.insert(0, os.path.join(CONTRASTIVE_DIR, "customized"))
from model import Wav2Vec2ForContrastiveLearning
from dataset import AudioClassificationDataCollatorForTest, vectorize_datasets_classificationForTest

CACHE_DIR = "/mnt/storage/ldl_linguistics/datasets"
CKPT_PATH = os.path.join(CONTRASTIVE_DIR, "weights/FEAT128BS16_PROXY_SCRATCH_V2/checkpoint_epoch_50.pt")
RESULTS_PATH = "/mnt/storage/qisheng/github/wav2vec_test/results/ablation/proxy_v2_source_results_36.json"

ALREADY_SAVED = {'ar', 'be', 'bg', 'bn', 'cs', 'cy', 'da', 'de', 'el', 'es',
    'et', 'fa', 'fi', 'hi', 'hu', 'it', 'ja', 'ka', 'ko', 'lt', 'lv',
    'mk', 'ml', 'mn', 'mr', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl',
    'sr', 'sw', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi', 'en', 'fr'}

DATASET_PARAMS = {
    "dataset_name": "fsicoli/common_voice_22_0",
    "train_split": "train",
    "test_split": "test",
    "text_column": 'sentence',
    "audio_column": 'audio',
    "max_duration_in_seconds": 20.0,
    "min_duration_in_seconds": 0.0,
    "preprocessing_num_workers": 1,
    "cache_dir": CACHE_DIR,
}

# ── All 36 training languages as candidates ──
ALL_LANGS = ['ar','ba','eu','be','bn','ca','yue','cs','nl','en','eo',
             'fa','fr','ka','de','hu','it','ja','lv','lt','pl','pt',
             'ro','ru','uk','es','sw','ta','th','tt','tr','ug','ur',
             'uz','cy','zh-CN']
CANDIDATES = ALL_LANGS
TARGETS = ['sq','ltg','ur','cy','gn','tn','am','az','mt','af','da','ky','tk','kk','sk','id']

def get_dataset_name(lang):
    return "fixie-ai/common_voice_17_0" if lang in ALREADY_SAVED else "fsicoli/common_voice_22_0"

def load_embeddings(model, feature_extractor, lang, split="train", n_samples=50):
    ds_name = get_dataset_name(lang)
    try:
        ds = load_dataset(ds_name, lang, split=split, trust_remote_code=True, cache_dir=CACHE_DIR)
    except:
        try:
            ds = load_dataset("fixie-ai/common_voice_17_0", lang, split=split, trust_remote_code=True, cache_dir=CACHE_DIR)
        except:
            print(f"  [{lang}] FAILED")
            return None

    n = min(n_samples, len(ds))
    ds = ds.select(range(n))
    combined = hf_datasets.DatasetDict({"train": ds})
    vec = vectorize_datasets_classificationForTest(combined, tokenizer=None, feature_extractor=feature_extractor, **DATASET_PARAMS)
    collator = AudioClassificationDataCollatorForTest(feature_extractor)
    loader = DataLoader(vec['train'], batch_size=1, shuffle=False, collate_fn=collator)

    embed_list = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input_values'].to('cuda')
            masks = batch['attention_mask'].to('cuda')
            features = model(inputs, masks)
            feat = features.cpu().numpy().flatten()
            embed_list.append(feat / (np.linalg.norm(feat) + 1e-10))
    return np.array(embed_list)

def main():
    print("=" * 70)
    print("PROXY-V2 SOURCE SELECTION")
    print("=" * 70)

    # Load model
    print("\n[1/3] Loading proxy_v2 model...")
    model_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53", trust_remote_code=True)
    model = Wav2Vec2ForContrastiveLearning.from_pretrained("facebook/wav2vec2-large-xlsr-53", config=model_config, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model = model.to('cuda').eval()
    print(f"  Loaded: {CKPT_PATH}")

    # Extract source embeddings
    print("\n[2/3] Extracting source embeddings...")
    source_embs = {}
    for src in CANDIDATES:
        emb = load_embeddings(model, feature_extractor, src, split="train", n_samples=50)
        if emb is not None:
            source_embs[src] = emb
            print(f"  [{src}] {len(emb)} samples")

    # Extract target embeddings
    print("\n[3/3] Extracting target embeddings...")
    target_embs = {}
    for tgt in TARGETS:
        emb = load_embeddings(model, feature_extractor, tgt, split="test", n_samples=50)
        if emb is not None:
            target_embs[tgt] = emb
            print(f"  [{tgt}] {len(emb)} samples")

    # Rank
    print("\nComputing similarities...")
    results = {}
    for tgt in TARGETS:
        if tgt not in target_embs:
            continue
        scores = []
        for src, src_emb in source_embs.items():
            if src == tgt:
                continue
            sim = float(np.mean(cosine_similarity(target_embs[tgt], src_emb)))
            scores.append((src, round(sim, 4)))
        scores.sort(key=lambda x: -x[1])
        results[tgt] = scores
        print(f"  {tgt} → {scores[0][0]}({scores[0][1]:.4f})")

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Compare with original contrastive
    print(f"\n{'='*70}")
    print("COMPARISON: Proxy-V2 vs Original Contrastive")
    print(f"{'='*70}")
    print(f"{'Target':<6} {'Proxy-V2':<22} {'SupCon':<22} {'Agree'}")
    print("-" * 56)
    with open('results/ablation/contrastive_n=50.json') as f:
        supcon = json.load(f)
    match = 0
    for t in TARGETS:
        pv2 = results.get(t, [None])[0]
        sc = supcon.get(t, [None])[0]
        if pv2 and sc:
            m = '✓' if pv2[0] == sc[0] else '✗'
            if m == '✓': match += 1
            print(f"{t:<6} {pv2[0]}({pv2[1]:.4f}){'':>10} {sc[0]}({sc[1]:.4f}){'':>10} {m}")
        elif pv2:
            print(f"{t:<6} {pv2[0]}({pv2[1]:.4f}){'':>10} {'—':<22} —")
    print(f"\nAgreement: {match}/{len([t for t in TARGETS if t in results and t in supcon])}")

if __name__ == "__main__":
    main()
