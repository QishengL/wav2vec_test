#!/usr/bin/env python3
"""Batch eval_final for all 26_6_16 experiments."""
import os, sys, json, torch
sys.path.insert(0, '/mnt/storage/qisheng/github/wav2vec_test/src')
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from dataset import load_datasets, preprocess_datasets, vectorize_datasets
from collator import DataCollatorCTCWithPadding
from torch.utils.data import DataLoader
import evaluate

BASE = '/mnt/storage/qisheng/github/wav2vec_test'
WEIGHTS = f'{BASE}/weights'
RESULTS = f'{BASE}/results/s2_results.json'

# All experiments to evaluate: (weights_dir, target_lang, experiment_name, is_direct)
# Direct finetune also uses config from 26_6_16
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
        ckpts = sorted([d for d in os.listdir(output_dir) if d.startswith('checkpoint-')],
                       key=lambda x: int(x.split('-')[1]))
        return os.path.join(output_dir, ckpts[-1]) if ckpts else None, None
    ckpt_path = os.path.join(output_dir, f'checkpoint-{best_step}')
    if not os.path.exists(ckpt_path):
        ckpts = sorted([d for d in os.listdir(output_dir) if d.startswith('checkpoint-')],
                       key=lambda x: int(x.split('-')[1]))
        return os.path.join(output_dir, ckpts[-1]) if ckpts else None, best_wer
    return ckpt_path, best_wer

def run_eval(weights_dir, lan, exp_name):
    full_path = os.path.join(WEIGHTS, weights_dir)
    if not os.path.exists(os.path.join(full_path, 'trainer_state.json')):
        return
    best_ckpt, best_val_wer = find_best_checkpoint(full_path, lan)
    if best_ckpt is None:
        print(f'[SKIP] {exp_name}: no checkpoint')
        return
    print(f'[EVAL] {exp_name}: {os.path.basename(best_ckpt)} (val_wer={best_val_wer:.4f})', flush=True)
    
    model = Wav2Vec2ForCTC.from_pretrained(best_ckpt).to('cuda').eval()
    processor = Wav2Vec2Processor.from_pretrained(best_ckpt)
    
    raw = load_datasets(lan, max_eval_sample=None,
        dataset_name='fsicoli/common_voice_22_0', dataset_config_name=lan,
        train_split='train', test_split='test',
        chars_to_ignore=[], text_column='sentence', audio_column='audio',
        max_train_samples_per_language=[100], max_eval_samples_per_language=None,
        max_train_samples=None, max_eval_samples=None,
        max_duration_in_seconds=20.0, min_duration_in_seconds=0.0,
        preprocessing_num_workers=1, eval_metrics=['wer'],
        cache_dir='/mnt/storage/ldl_linguistics/datasets')
    raw = preprocess_datasets(raw,
        dataset_name='fsicoli/common_voice_22_0', dataset_config_name=lan,
        chars_to_ignore=[], text_column='sentence', audio_column='audio',
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
    print(f'  Held-out test WER = {wer:.4f}', flush=True)
    
    del model; torch.cuda.empty_cache()
    
    entry = {
        'experiment': exp_name,
        'target_lang': lan,
        'best_val_wer': round(best_val_wer, 4) if best_val_wer else None,
        'heldout_test_wer': round(wer, 4),
        'test_samples': len(test_data),
        'best_checkpoint': os.path.basename(best_ckpt),
    }
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            existing = json.load(f)
    else:
        existing = []
    existing.append(entry)
    with open(RESULTS, 'w') as f:
        json.dump(existing, f, indent=2)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=int, default=0, help='Group number (1-8)')
    args = parser.parse_args()
    
    if args.group:
        manifest_path = f'{BASE}/slurm_config/eval_group_{args.group}.json'
        with open(manifest_path) as f:
            manifest = json.load(f)
        all_exps = [(m['weights_dir'], m['target_lang'], m['exp_name']) for m in manifest]
        print(f'Group {args.group}: {len(all_exps)} experiments', flush=True)
    else:
        # Auto-detect all (fallback)
        new_targets = ['sq','ltg','ur','cy','gn','tn','am','he','az']
        new_old = {'mt':['cs','ro','tr'],'af':['tr'],'da':['lv','tr'],'ky':['tt'],'tk':['tt'],'id':['ar','eo','it']}
        all_exps = []
        for d in sorted(os.listdir(WEIGHTS)):
            if not os.path.isdir(os.path.join(WEIGHTS, d)):
                continue
            for t in new_targets:
                if d.startswith(f'{t}_100-xlsr-'):
                    suffix = d.replace(f'{t}_100-xlsr-', '')
                    all_exps.append((d, t, f'{t}_{suffix}'))
                    break
            for t, srcs in new_old.items():
                for s in srcs:
                    for m in ['53', 'base', 'direct53', 'directbase']:
                        if d == f'{t}_100-xlsr-{s}{m}':
                            all_exps.append((d, t, f'{t}_{s}{m}'))
        for d in sorted(os.listdir(WEIGHTS)):
            if not os.path.isdir(os.path.join(WEIGHTS, d)):
                continue
            for t in ['mt','af','da','ky','tk','kk','sk','id']:
                for m in ['direct53', 'directbase']:
                    if d == f'{t}_100-xlsr-{m}':
                        all_exps.append((d, t, f'{t}_{m}'))
        print(f'All: {len(all_exps)} experiments', flush=True)
    
    # Load existing results to skip
    existing_names = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for r in json.load(f):
                existing_names.add(r.get('experiment', ''))
    
    for weights_dir, lan, exp_name in all_exps:
        if exp_name in existing_names:
            print(f'[SKIP] {exp_name} (already in results)', flush=True)
            continue
        try:
            run_eval(weights_dir, lan, exp_name)
        except Exception as e:
            print(f'[ERROR] {exp_name}: {e}', flush=True)

if __name__ == '__main__':
    main()
