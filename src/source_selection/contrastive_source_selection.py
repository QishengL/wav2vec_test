"""
Contrastive Source Selection

Uses the contrastive learning model (SupConLoss) to find the best source language
for each target language via embedding cosine similarity.

Pipeline:
1. Load contrastive model checkpoint
2. Extract embeddings for all source languages (train set, 50 samples)
3. Extract embeddings for all target languages (test set, up to 50 samples)
4. Compute mean cosine similarity
5. Rank and output top source for each target
"""
import os, sys, json, numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2Config
from datasets import load_dataset
import datasets as hf_datasets
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ──
CONTRASTIVE_DIR = "/mnt/storage/qisheng/github/wav2vec_contrastive"
sys.path.insert(0, os.path.join(CONTRASTIVE_DIR, "customized"))
from model import Wav2Vec2ForContrastiveLearning
from dataset import AudioClassificationDataCollatorForTest, vectorize_datasets_classificationForTest

CACHE_DIR = "/mnt/storage/ldl_linguistics/datasets"
CKPT_PATH = os.path.join(CONTRASTIVE_DIR, "weights/FEAT128BS16_WAVE/checkpoint_epoch_50.pt")
RESULTS_PATH = "/mnt/storage/qisheng/github/wav2vec_test/src/source_selection/contrastive_source_results.json"

# Languages that were already processed with common_voice_17_0 vs 22_0
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

# ── Source candidates (18 languages with Stage 1 trained) ──
SOURCE_CANDIDATES = ['ar', 'ba', 'ca', 'cs', 'en', 'eo', 'fr', 'hu', 'it',
                     'lt', 'lv', 'nl', 'ro', 'ru', 'sw', 'ta', 'tr', 'tt', 'ug']

# ── Target languages (old 8 + new candidates) ──
OLD_TARGETS = ['mt', 'af', 'da', 'ky', 'tk', 'kk', 'sk', 'id']
NEW_TARGETS = ['sq', 'ltg', 'ia', 'ur', 'cy', 'gn', 'tn', 'am', 'he', 'az']
ALL_TARGETS = OLD_TARGETS + NEW_TARGETS

LANG_NAMES = {
    'mt': 'Maltese', 'af': 'Afrikaans', 'da': 'Danish', 'ky': 'Kyrgyz',
    'tk': 'Turkmen', 'kk': 'Kazakh', 'sk': 'Slovak', 'id': 'Indonesian',
    'sq': 'Albanian', 'ltg': 'Latgalian', 'ia': 'Interlingua', 'ur': 'Urdu',
    'cy': 'Welsh', 'gn': 'Guarani', 'tn': 'Tswana', 'am': 'Amharic',
    'he': 'Hebrew', 'az': 'Azerbaijani',
}


def get_dataset_name(lang):
    if lang in ALREADY_SAVED:
        return "fixie-ai/common_voice_17_0"
    return "fsicoli/common_voice_22_0"


def load_embeddings(model, feature_extractor, lang, split="train", n_samples=50):
    """Extract embeddings from a language using the contrastive model."""
    ds_name = get_dataset_name(lang)
    try:
        ds = load_dataset(ds_name, lang, split=split, trust_remote_code=True, cache_dir=CACHE_DIR)
    except Exception as e:
        print(f"  [{lang}] FAILED to load {ds_name}/{split}: {e}")
        return None

    n = min(n_samples, len(ds))
    ds = ds.select(range(n))
    combined = hf_datasets.DatasetDict({"train": ds})

    vec = vectorize_datasets_classificationForTest(
        combined, tokenizer=None, feature_extractor=feature_extractor, **DATASET_PARAMS
    )

    collator = AudioClassificationDataCollatorForTest(feature_extractor)
    loader = DataLoader(vec['train'], batch_size=1, shuffle=False, collate_fn=collator)

    embed_list = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input_values'].to('cuda')
            masks = batch['attention_mask'].to('cuda')
            features = model(inputs, masks)
            feat = features.cpu().numpy().flatten()
            norm_feat = feat / (np.linalg.norm(feat) + 1e-10)
            embed_list.append(norm_feat)

    return np.array(embed_list)


def main():
    print("=" * 70)
    print("CONTRASTIVE SOURCE SELECTION")
    print("=" * 70)

    # ── Load model ──
    print("\n[1/4] Loading contrastive model...")
    model_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53", trust_remote_code=True
    )
    model = Wav2Vec2ForContrastiveLearning.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        config=model_config,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model = model.to('cuda').eval()
    print(f"  Loaded checkpoint: {CKPT_PATH}")

    # ── Extract source embeddings ──
    print("\n[2/4] Extracting source language embeddings...")
    source_embs = {}
    for src in SOURCE_CANDIDATES:
        print(f"  [{src}] extracting...", end=" ", flush=True)
        emb = load_embeddings(model, feature_extractor, src, split="train", n_samples=50)
        if emb is not None:
            source_embs[src] = emb
            print(f"{len(emb)} samples")
        else:
            print("FAILED")

    print(f"  Done. {len(source_embs)} source languages loaded.")

    # ── Extract target embeddings ──
    print("\n[3/4] Extracting target language embeddings...")
    target_embs = {}
    for tgt in ALL_TARGETS:
        print(f"  [{tgt}] ({LANG_NAMES.get(tgt, '?')}) extracting...", end=" ", flush=True)
        emb = load_embeddings(model, feature_extractor, tgt, split="test", n_samples=50)
        if emb is not None:
            target_embs[tgt] = emb
            print(f"{len(emb)} samples")
        else:
            print("FAILED (skipping)")

    # ── Compute similarity and rank ──
    print("\n[4/4] Computing cosine similarity...")
    results = {}
    for tgt in ALL_TARGETS:
        if tgt not in target_embs:
            continue

        tgt_emb = target_embs[tgt]
        scores = []

        for src, src_emb in source_embs.items():
            if src == tgt:
                continue
            sim_matrix = cosine_similarity(tgt_emb, src_emb)
            mean_sim = float(np.mean(sim_matrix))
            scores.append((src, round(mean_sim, 4)))

        scores.sort(key=lambda x: -x[1])
        results[tgt] = scores

        name = LANG_NAMES.get(tgt, tgt)
        print(f"\n  [{tgt}] {name}:")
        for rank, (src, sim) in enumerate(scores[:5], 1):
            print(f"    {rank}. {src} ({LANG_NAMES.get(src, src)}): {sim:.4f}")

        # Suggest the best source
        best_src = scores[0][0]
        print(f"    → Best source: {best_src} ({LANG_NAMES.get(best_src, best_src)})")

    # ── Save results ──
    output = {
        "method": "Contrastive (SupConLoss + cosine similarity)",
        "model": CKPT_PATH,
        "results": results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {RESULTS_PATH}")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("SUMMARY: Contrastive Source Selection")
    print("=" * 70)
    print(f"{'Target':<8} {'Name':<14} {'Best Source':<14}")
    print("-" * 40)
    for tgt in ALL_TARGETS:
        if tgt not in results:
            continue
        best = results[tgt][0][0] if results[tgt] else "—"
        name = LANG_NAMES.get(tgt, tgt)
        print(f"{tgt:<8} {name:<14} {best:<14}")


if __name__ == "__main__":
    main()
