#!/usr/bin/env python3
"""Eval held-out test for S2 data ablation - using same pipeline as batch_eval.py"""
import os, sys, json, torch
sys.path.insert(0, '/mnt/storage/qisheng/github/wav2vec_test/src')
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from dataset import load_datasets, preprocess_datasets, vectorize_datasets
from collator import DataCollatorCTCWithPadding
from torch.utils.data import DataLoader
import evaluate

BASE = '/mnt/storage/qisheng/github/wav2vec_test'

TARGETS = ['af', 'gn', 'sq', 'id']
SOURCES = {'af': 'nl', 'gn': 'ro', 'sq': 'eo', 'id': 'cs'}
N_VALS = [10, 50, 150]

def find_best_checkpoint(output_dir, lan):
    state_path = os.path.join(output_dir, 'trainer_state.json')
    if not os.path.exists(state_path):
        return None, None
    with open(state_path) as f:
        state = json.load(f)
    wer_key = f'eval_{lan}_wer'
    best_step, best_wer = None, float('inf')
    for e in state['log_history']:
        if wer_key in e and e[wer_key] < best_wer:
            best_wer = e[wer_key]
            best_step = e['step']
    if best_step is None:
        return None, None
    ckpt_path = os.path.join(output_dir, f'checkpoint-{best_step}')
    if not os.path.exists(ckpt_path):
        return None, None
    return ckpt_path, best_wer

def run_eval(weights_dir, lan):
    full_path = os.path.join(BASE, 'weights', weights_dir)
    if not os.path.exists(os.path.join(full_path, 'trainer_state.json')):
        return None
    best_ckpt, best_val_wer = find_best_checkpoint(full_path, lan)
    if best_ckpt is None:
        return None
    print(f'  Best: {os.path.basename(best_ckpt)} (val_wer={best_val_wer:.4f})', flush=True)
    
    model = Wav2Vec2ForCTC.from_pretrained(best_ckpt).to('cuda').eval()
    processor = Wav2Vec2Processor.from_pretrained(best_ckpt)
    
    raw = load_datasets(lan, max_eval_sample=None,
        dataset_name='fsicoli/common_voice_22_0', dataset_config_name=lan,
        train_split='train', test_split='test',
        chars_to_ignore=[',', '?', '.', '!', '-', ';', ':', '"', '\u201c', '%', '\u2018', '\u201d','\ufffd', '(', ')', "'"],
        text_column='sentence', audio_column='audio',
        max_train_samples_per_language=[100], max_eval_samples_per_language=None,
        max_train_samples=None, max_eval_samples=None,
        max_duration_in_seconds=20.0, min_duration_in_seconds=0.0,
        preprocessing_num_workers=1, eval_metrics=['wer'],
        cache_dir='/mnt/storage/ldl_linguistics/datasets')
    raw = preprocess_datasets(raw,
        dataset_name='fsicoli/common_voice_22_0', dataset_config_name=lan,
        chars_to_ignore=[',', '?', '.', '!', '-', ';', ':', '"', '\u201c', '%', '\u2018', '\u201d','\ufffd', '(', ')', "'"],
        text_column='sentence', audio_column='audio',
        max_duration_in_seconds=20.0, min_duration_in_seconds=0.0,
        preprocessing_num_workers=1,
        cache_dir='/mnt/storage/ldl_linguistics/datasets')
    vec = vectorize_datasets(raw, processor.tokenizer, processor.feature_extractor,
        dataset_name='fsicoli/common_voice_22_0', dataset_config_name=lan,
        text_column='sentence', audio_column='audio',
        max_duration_in_seconds=20.0, min_duration_in_seconds=0.0,
        preprocessing_num_workers=1,
        cache_dir='/mnt/storage/ldl_linguistics/datasets')
    
    full_size = len(vec['eval'])
    half = full_size // 2
    test_data = vec['eval'].select(range(half, full_size))
    print(f'  Held-out test: {len(test_data)} samples', flush=True)
    
    collator = DataCollatorCTCWithPadding(processor=processor)
    loader = DataLoader(test_data, batch_size=32, collate_fn=collator)
    wer_metric = evaluate.load('wer')
    preds, refs = [], []
    with torch.no_grad():
        for batch in loader:
            inp = batch['input_values'].to('cuda')
            attn = batch.get('attention_mask')
            if attn is not None: attn = attn.to('cuda')
            labels = batch['labels']
            labels[labels == -100] = processor.tokenizer.pad_token_id
            logits = model(inp, attention_mask=attn).logits
            pred_ids = torch.argmax(logits, dim=-1)
            preds.extend(processor.batch_decode(pred_ids))
            refs.extend(processor.batch_decode(labels, group_tokens=False))
    wer = wer_metric.compute(predictions=preds, references=refs)
    print(f'  Held-out WER = {wer:.4f}', flush=True)
    
    del model; torch.cuda.empty_cache()
    return {'heldout_test_wer': round(wer, 4), 'best_val_wer': round(best_val_wer, 4) if best_val_wer else None,
            'test_samples': len(test_data), 'best_checkpoint': os.path.basename(best_ckpt)}

# Main
results = {}
for tgt in TARGETS:
    src = SOURCES[tgt]
    for n in N_VALS:
        wdir = f'{tgt}_{n}-xlsr-{src}53'
        if not os.path.exists(os.path.join(BASE, 'weights', wdir)):
            print(f'[{tgt} n={n}] NO WEIGHTS', flush=True)
            continue
        print(f'[{tgt} n={n}]', flush=True)
        res = run_eval(wdir, tgt)
        if res:
            results[f'{tgt}_{n}-{src}53'] = {**res, 'target_lang': tgt, 'experiment': f'ablation_{tgt}_{n}-{src}53', 'n_samples': n}
            print(f'  → {res["heldout_test_wer"]:.4f}', flush=True)

# Add n=100 from existing results
with open(f'{BASE}/results/s2_results.json') as f:
    existing = json.load(f)
for tgt in TARGETS:
    src = SOURCES[tgt]
    for r in existing:
        if r['target_lang'] == tgt and src in r['experiment'] and 'heldout_test_wer' in r:
            if 'direct' not in r['experiment'] and r.get('heldout_test_wer', 1) < 0.99:
                results[f'{tgt}_100-{src}53'] = r
                break

with open(f'{BASE}/results/s2_ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nSaved: results/s2_ablation_results.json')
print(f'\n{"Target":<8} {"n=10(1min)":<14} {"n=50(5min)":<14} {"n=100(10min)":<14} {"n=150(15min)":<14}')
print("-" * 54)
for tgt in TARGETS:
    src = SOURCES[tgt]
    line = f"{tgt:<8}"
    for n in [10, 50, 100, 150]:
        k = f'{tgt}_{n}-{src}53'
        r = results.get(k)
        line += f"{r['heldout_test_wer']:<14.4f}" if r and 'heldout_test_wer' in r else f"{'—':<14}"
    print(line)
