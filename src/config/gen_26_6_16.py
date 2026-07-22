"""Generate configs + slurm for:
  - Old targets: 22 missing experiments (classifier/contrastive top3)
  - New targets: 122 experiments (all)
All go to 26_6_16/ and 26_6_16_base/
"""
import json, os

BASE = "/mnt/storage/qisheng/github/wav2vec_test"
cdir53 = '26_6_16'
cdir_base = '26_6_16_base'

# All pairs to generate
PAIRS = {
    # Old targets — only missing ones
    'mt':  ['cs','ro','tr'],
    'af':  ['tr'],
    'da':  ['lv','tr'],
    'ky':  ['tt'],
    'tk':  ['tt'],
    'id':  ['ar','eo','it'],
    # New targets — all 61 pairs
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

HALF = {'mt':830,'af':65,'da':1379,'ky':807,'tk':285,'id':1845,
        'sq':958,'ltg':1765,'ur':2541,'cy':2704,'gn':480,'tn':184,'am':126,'he':196,'az':47}

def gen_one(tgt, src, ms, cdir):
    if ms == '53':
        ckpt_num = '8000' if src == 'ug' else '28150'
        s1 = f"/mnt/storage/qisheng/github/wav2vec_test/weights/{src}_9000-xlsr-53_general_1e5/checkpoint-{ckpt_num}"
    else:
        ckpt_num = '10000' if src == 'ug' else '26000'
        s1 = f"/mnt/storage/qisheng/github/wav2vec_test/weights/{src}_9000-xlsr-base_general_1e5/checkpoint-{ckpt_num}"
    
    cfg_name = f"{tgt}_100-{src}{ms}.py"
    slurm_name = f"{tgt}_100-{src}{ms}.sh"
    wdir = f"{tgt}_100-xlsr-{src}{ms}"
    
    cfg_content = f"""# config_{ms}.py
wandb_project = "wav2vec_26_6_16"
wandb_run = "{tgt}_100-{src}{ms}"
preprocessing_only = False
resume = False
TRAINING_PARAMS = {{
    "output_dir": "../weights/{wdir}", 
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
    "output_dir": "../weights/{wdir}", 
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
    "chars_to_ignore" : [',', '?', '.', '!', '-', ';', ':', "'"],
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
    cfg_path = os.path.join(BASE, 'src', 'config', 'lora', cdir, cfg_name)
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, 'w') as f:
        f.write(cfg_content)
    
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name=s2_{tgt}_{src}{ms}
#SBATCH --output={BASE}/slurm_config/out/s2_{tgt}_{src}{ms}.out
#SBATCH --error={BASE}/slurm_config/out/s2_{tgt}_{src}{ms}.err
#SBATCH --time=12:00:00
#SBATCH --mem=24G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

export WANDB_API_KEY=67129c4138cabfa6fe40ff02f228f65339bbba0d
export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

nvidia-smi

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd {BASE}/src;python3 main.py --config ./config/lora/{cdir}/{cfg_name}"
"""
    slurm_path = os.path.join(BASE, 'slurm_config', 'lora', cdir, slurm_name)
    os.makedirs(os.path.dirname(slurm_path), exist_ok=True)
    with open(slurm_path, 'w') as f:
        f.write(slurm_content)

# Generate all
count = 0
for tgt in PAIRS:
    for src in PAIRS[tgt]:
        gen_one(tgt, src, '53', cdir53)
        gen_one(tgt, src, 'base', cdir_base)
        count += 2
        print(f"  {tgt}_100-{src}53 + base")

print(f"\nDone! {count} experiments generated")
print(f"Configs in: src/config/lora/{cdir53}/ and {cdir_base}/")
print(f"Slurm in: slurm_config/lora/{cdir53}/ and {cdir_base}/")
