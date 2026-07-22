"""
Classifier Embedding Baseline — Source Language Selection

Uses raw Wav2Vec2 encoder embeddings (NO contrastive training, NO classifier training).
Mean-pool over time → language embedding → cosine similarity for source selection.

This tests: do generic multilingual SSL representations already capture
enough language similarity, making contrastive pretraining unnecessary?
"""
import json
import os
import sys
import numpy as np
import torch
from transformers import Wav2Vec2Model, AutoFeatureExtractor
from datasets import load_dataset

os.environ["HF_HOME"] = "/mnt/storage/ldl_linguistics/hf_home"
CACHE_DIR = "/mnt/storage/ldl_linguistics/datasets"

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

CV_OVERRIDE = {"zh": "zh-CN"}
N_SAMPLES = 100  # audio samples per language


def get_language_embedding(model, feature_extractor, lang_code, device="cpu"):
    """Extract mean-pooled Wav2Vec2 encoder embeddings for a language."""
    cv_code = CV_OVERRIDE.get(lang_code, lang_code)

    try:
        ds = load_dataset(
            "fsicoli/common_voice_22_0", cv_code, split="train",
            trust_remote_code=True, cache_dir=CACHE_DIR,
        )
    except Exception:
        print(f"    Dataset load failed for {lang_code}", file=sys.stderr)
        return None

    if len(ds) > N_SAMPLES:
        ds = ds.shuffle(seed=42).select(range(N_SAMPLES))

    # Resample to 16kHz (Wav2Vec2 expects 16kHz, Common Voice is 48kHz)
    from datasets import Audio
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    embeddings = []
    for ex in ds:
        try:
            audio = ex["audio"]
            inputs = feature_extractor(
                audio["array"], sampling_rate=16000,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                # Mean pool over time dimension
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(emb)
        except Exception:
            continue

    if not embeddings:
        return None

    # Average across all samples → single language embedding
    lang_emb = np.mean(embeddings, axis=0)
    # Normalize to unit vector
    lang_emb = lang_emb / (np.linalg.norm(lang_emb) + 1e-8)
    return lang_emb


def cosine_sim(a, b):
    return float(np.dot(a, b))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading Wav2Vec2 base model...")

    model_name = "facebook/wav2vec2-base"
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name, cache_dir=CACHE_DIR
    )
    model = Wav2Vec2Model.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = model.to(device)
    model.eval()
    print(f"Model loaded. Hidden size: {model.config.hidden_size}")

    # Extract embeddings for all languages
    all_langs = sorted(set(TARGETS + CANDIDATES))
    embeddings = {}

    for lang in all_langs:
        name = TARGET_NAMES.get(lang, lang)
        print(f"[{lang}] {name} ...", end=" ", flush=True)
        sys.stdout.flush()
        emb = get_language_embedding(model, feature_extractor, lang, device)
        if emb is not None:
            embeddings[lang] = emb
            print(f"✓ (dim={emb.shape[0]})")
        else:
            print("✗")

    print(f"\nExtracted embeddings for {len(embeddings)}/{len(all_langs)} languages")

    # Compute cosine similarity for each target → candidate pair
    print()
    print("=" * 60)
    print("CLASSIFIER EMBEDDING (Raw Wav2Vec2 Encoder)")
    print("Cosine Similarity for Source Selection")
    print("=" * 60)

    results = {}
    for target in TARGETS:
        if target not in embeddings:
            print(f"\n{target}: NO EMBEDDING")
            continue
        name = TARGET_NAMES[target]
        t_emb = embeddings[target]
        print(f"\n--- {target} ({name}) ---")
        scores = []
        for cand in CANDIDATES:
            if cand == target or cand not in embeddings:
                continue
            sim = cosine_sim(t_emb, embeddings[cand])
            scores.append((cand, round(sim, 4)))
        scores.sort(key=lambda x: -x[1])
        results[target] = scores
        for rank, (c, s) in enumerate(scores[:5], 1):
            print(f"  {rank}. {c}: {s:.4f}")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY: Raw Wav2Vec2 Embedding → Source Selection")
    print("=" * 60)
    for target in TARGETS:
        name = TARGET_NAMES[target]
        top = results.get(target, [("N/A", 0)])[0]
        print(f"  {name:<12} → {top[0]:<8} ({top[1]:.4f})")

    # Save
    out = {
        "method": "raw_wav2vec2_encoder_embedding",
        "model": model_name,
        "n_samples_per_language": N_SAMPLES,
        "results": {
            t: [(c, s) for c, s in scores]
            for t, scores in results.items()
        },
    }
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "classifier_embedding_results.json",
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
