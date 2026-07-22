"""Generate S2 configs + slurm scripts for new target languages."""
import os

BASE = "/mnt/storage/qisheng/github/wav2vec_test"
PAIRS = {
    'sq':  ['ca','cs','eo','it','lt','lv','nl','ro','tr'],
    'ltg': ['lt','lv','ru','tr'],
    'ur':  ['ar','cs','en','lv','nl','ro','ta','tr'],
    'cy':  ['ar','ba','hu','it','nl','sw','tr','ug'],
    'gn':  ['cs','eo','it','ro','sw'],
    'tn':  ['eo','hu','lt','sw'],
    'am':  ['ar','ba','ca','eo','hu','it','sw','ta'],
    'he':  ['ar','cs','en','eo','fr','nl','ru','sw','tr','ug'],
    'az':  ['ba','eo','lt','tr','ug'],
}
HALF = {'sq':958,'ltg':1765,'ur':2541,'cy':2704,'gn':480,'tn':184,'am':126,'he':196,'az':47}

def gen(tgt, src, mtype):
    ms = {'53': '53', 'base': 'base'}[mtype]
    cdir = '26_6_5' if mtype == '53' else '26_6_5_base'
    
    s1 = f"/mnt/storage/qisheng/github/wav2vec_test/weights/{src}_9000-xlsr-53_general_1e5/checkpoint-{8000 if src=='ug' else 28150}"
    if mtype == 'base':
        s1 = f"/mnt/storage/qisheng/github/wav2vec_test/weights/{src}_9000-xlsr-base_general_1e5/checkpoint-{10000 if src=='ug' else 26000}"
    
    cfg = f"""# config_{ms}.py
wandb_project = "wav2vec_26_6_5"
wandb_run = "{tgt}_100-{src}{ms}"
preprocessing_only = False
resume = False
TRAINING_PARAMS = {{
    "output_dir": "../weights/{tgt}_100-xlsr-{src}{ms}", 
    "overwrite_output_dir":True,        
    "num_train_epochs": 800,                                    
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-5,
    "lr_scheduler_type":"constant",
    "weight_decay":0.0,
    "warmup_steps":0,
    "max_grad_norm": 1.5,
    "save_steps": 800,                                         
    "save_total_limit": 5,
    "gradient_checkpointing": True,                              
    "fp16": True,                                              
    "eval_steps": 800,
    "logging_steps": 50,
    "evaluation_strategy": "steps",
    "load_best_model_at_end": False,
    "dataloader_num_workers": 8,
    "dataloader_pin_memory": True,
    "group_by_length": True,
    "report_to": ["wandb"],
    "remove_unused_columns": False,
    "ignore_data_skip": False,
    "ddp_find_unused_parameters": False,
    "deepspeed": "../ds_config/ds_config_zero2.json",
}}
MODEL_PARAMS = {{
    "output_dir": "../weights/{tgt}_100-xlsr-{src}{ms}", 
    "model_name_or_path" : "{s1}",
    "word_delimiter_token": "|",
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "feat_proj_dropout": 0.1,
    "attention_dropout": 0.1,
    "hidden_dropout": 0.1,
    "final_dropout": 0.1,
    "mask_time_prob": 0.1,
    "mask_time_length": 10,
    "mask_feature_prob": 0.1,
    "mask_feature_length": 10,
    "layerdrop": 0.0,
    "ctc_loss_reduction": "mean",
    "ctc_zero_infinity": False,
    "activation_dropout": 0.1,
    "add_adapter": False,
    "freeze_feature_encoder":True,
    "freeze_enc": True,
    "vocab_dir": "/mnt/storage/qisheng/github/wav2vec_test/weights/general_phoneme",
    "use_phoneme": True,
    "cache_dir":"/mnt/storage/ldl_linguistics/datasets",
}}
DATASET_PARAMS = {{
    "dataset_name" : "fsicoli/common_voice_22_0",
    "dataset_config_name" : "{tgt}",
    "train_split" : "train",
    "test_split" : "test",
    "chars_to_ignore" : [',', '?', '.', '!', '-', ';', ':', '"', '\u201c', '%', '\u2018', '\u201d', '\ufffd', '(', ')', "'"],
    "text_column":'sentence',
    "audio_column":'audio',
    "max_train_samples_per_language":[100],
    "max_eval_samples_per_language":[{HALF[tgt]}],
    "max_train_samples":None,
    "max_eval_samples":None,
    "max_duration_in_seconds":20.0,
    "min_duration_in_seconds":0.0,
    "preprocessing_num_workers":1,
    "eval_metrics" : ["wer"],
    "cache_dir":"/mnt/storage/ldl_linguistics/datasets",
}}
"""
    cpath = os.path.join(BASE, 'src', 'config', 'lora', cdir, f"{tgt}_100-{src}{ms}.py")
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    with open(cpath, 'w') as f:
        f.write(cfg)
    
    sh = f"""#!/bin/bash
#SBATCH --job-name=s2_{tgt}_100-{src}{ms}
#SBATCH --output={BASE}/slurm_config/out/s2_{tgt}_100-{src}{ms}.out
#SBATCH --error={BASE}/slurm_config/out/s2_{tgt}_100-{src}{ms}.err
#SBATCH --time=12:00:00
#SBATCH --mem=24G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

export WANDB_API_KEY=67129c4138cabfa6fe40ff02f228f65339bbba0d
export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

nvidia-smi

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd {BASE}/src;python3 main.py --config ./config/lora/{cdir}/{tgt}_100-{src}{ms}.py"
"""
    spath = os.path.join(BASE, 'slurm_config', 'lora', cdir, f"{tgt}_100-{src}{ms}.sh")
    os.makedirs(os.path.dirname(spath), exist_ok=True)
    with open(spath, 'w') as f:
        f.write(sh)
    
    print(f"  {tgt}_100-{src}{ms}")

total = 0
for tgt in PAIRS:
    for src in PAIRS[tgt]:
        gen(tgt, src, '53')
        gen(tgt, src, 'base')
        total += 2
print(f"Done! {total} experiments generated")
