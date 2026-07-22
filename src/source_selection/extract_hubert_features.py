"""
Extract HuBERT frame features from layers 6, 12, 24 in one pass.
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
N_SAMPLES = 200
LAYERS = {'layer06': 6, 'layer12': 12, 'layer24': 24}

import torch
from transformers import HubertModel, AutoFeatureExtractor
from datasets import load_dataset, Audio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HubertModel.from_pretrained(
    "facebook/hubert-large-ls960-ft", cache_dir=CACHE_DIR).to(device).eval()
fe = AutoFeatureExtractor.from_pretrained(
    "facebook/hubert-large-ls960-ft", cache_dir=CACHE_DIR)
print("HuBERT model loaded", flush=True)

ALREADY_SAVED = ['ar', 'be', 'bg', 'bn', 'cs', 'cy', 'da', 'de', 'el', 'es', 'et', 'fa',
                 'fi', 'hi', 'hu', 'it', 'ja', 'ka', 'ko', 'lt', 'lv', 'mk', 'ml', 'mn',
                 'mr', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'ta', 'te',
                 'th', 'tr', 'uk', 'ur', 'vi','en','fr']

for layer_name in LAYERS:
    os.makedirs(f'{OUT_DIR}/cache_hubert_{layer_name}', exist_ok=True)

total_frames = 0
for lang in LAN_LIST:
    all_cached = all(
        os.path.exists(f'{OUT_DIR}/cache_hubert_{ln}/{lang}_frames.npy')
        for ln in LAYERS
    )
    if all_cached:
        n = np.load(f'{OUT_DIR}/cache_hubert_layer24/{lang}_frames.npy', mmap_mode='r').shape[0]
        total_frames += n
        print(f"  [{lang}] SKIP ({n} frames)", flush=True)
        continue

    ds_name = "fixie-ai/common_voice_17_0" if lang in ALREADY_SAVED else "fsicoli/common_voice_22_0"
    try:
        ds = load_dataset(ds_name, lang, split='train',
                          trust_remote_code=True, cache_dir=CACHE_DIR)
    except:
        print(f"  [{lang}] SKIP (dataset)", flush=True)
        continue

    if len(ds) > N_SAMPLES:
        ds = ds.shuffle(seed=42).select(range(N_SAMPLES))
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))

    lang_frames = {ln: [] for ln in LAYERS}
    for ex in ds:
        audio = ex['audio']
        inputs = fe(audio['array'], sampling_rate=16000, return_tensors='pt')
        inp = inputs['input_values'].to(device)
        with torch.no_grad():
            outputs = model(inp, output_hidden_states=True)
            hs = outputs.hidden_states
            for ln, layer_idx in LAYERS.items():
                lang_frames[ln].append(hs[layer_idx][0].cpu().numpy())

    n_frames = 0
    for ln in LAYERS:
        if lang_frames[ln]:
            af = np.concatenate(lang_frames[ln], axis=0).astype(np.float16)
            np.save(f'{OUT_DIR}/cache_hubert_{ln}/{lang}_frames.npy', af)
            n_frames = len(af)
    total_frames += n_frames
    print(f"  [{lang}] {len(ds)} utts, {n_frames} frames (total: {total_frames})", flush=True)
    del ds

print(f"Done! {total_frames} frames per layer", flush=True)
