"""
Extract XLSR-53 frame-level features for all 36 training languages and
cache them, so we can re-cluster with different K without re-running inference.
"""
import os, sys, pickle, argparse
import numpy as np

os.environ["HF_HOME"] = "/mnt/storage/ldl_linguistics/hf_home"
CACHE_DIR = "/mnt/storage/ldl_linguistics/datasets"

LAN_LIST = ['ar', 'ba', 'eu', 'be', 'bn', 'ca', 'yue', 'cs', 'nl', 'en', 'eo',
            'fa', 'fr', 'ka', 'de', 'hu', 'it', 'ja', 'lv', 'lt', 'pl', 'pt',
            'ro', 'ru', 'uk', 'es', 'sw', 'ta', 'th', 'tt', 'tr', 'ug', 'ur',
            'uz', 'cy', 'zh-CN']

OUT_DIR = '/mnt/storage/qisheng/github/wav2vec_test/results/pseudo_units'
CACHE_DIR_FRAMES = f'{OUT_DIR}/frame_cache'
os.makedirs(CACHE_DIR_FRAMES, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES = 200

import torch
from transformers import Wav2Vec2Model, AutoFeatureExtractor
from datasets import load_dataset, Audio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", cache_dir=CACHE_DIR).to(device).eval()
fe = AutoFeatureExtractor.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", cache_dir=CACHE_DIR)
print("Model loaded", flush=True)

ALREADY_SAVED = ['ar', 'be', 'bg', 'bn', 'cs', 'cy', 'da', 'de', 'el', 'es', 'et', 'fa',
                 'fi', 'hi', 'hu', 'it', 'ja', 'ka', 'ko', 'lt', 'lv', 'mk', 'ml', 'mn',
                 'mr', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'ta', 'te',
                 'th', 'tr', 'uk', 'ur', 'vi','en','fr']

lang_info = {}
total_frames = 0

for lang in LAN_LIST:
    cache_path = f'{CACHE_DIR_FRAMES}/{lang}_frames.npy'
    if os.path.exists(cache_path):
        frames = np.load(cache_path)
        lang_info[lang] = {'frames': frames, 'n_frames': len(frames)}
        total_frames += len(frames)
        print(f"  [{lang}] LOADED from cache: {len(frames)} frames", flush=True)
        continue

    ds_name = "fixie-ai/common_voice_17_0" if lang in ALREADY_SAVED else "fsicoli/common_voice_22_0"
    try:
        ds = load_dataset(ds_name, lang, split='train',
                          trust_remote_code=True, cache_dir=CACHE_DIR)
    except:
        print(f"  [{lang}] SKIP", flush=True)
        continue

    if len(ds) > N_SAMPLES:
        ds = ds.shuffle(seed=42).select(range(N_SAMPLES))
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))

    lang_frames = []
    for ex in ds:
        audio = ex['audio']
        inputs = fe(audio['array'], sampling_rate=16000, return_tensors='pt')
        inp = inputs['input_values'].to(device)
        with torch.no_grad():
            hidden = model(inp).last_hidden_state[0].cpu().numpy()
        lang_frames.append(hidden)

    if lang_frames:
        all_frames = np.concatenate(lang_frames, axis=0).astype(np.float16)
        np.save(cache_path, all_frames)
        lang_info[lang] = {'frames': all_frames, 'n_frames': len(all_frames)}
        total_frames += len(all_frames)
        print(f"  [{lang}] {len(ds)} utts, {len(all_frames)} frames SAVED (total: {total_frames})", flush=True)
    del ds

print(f"\nTotal: {len(lang_info)} languages, {total_frames} frames cached", flush=True)
print("All frame features extracted and cached!", flush=True)
