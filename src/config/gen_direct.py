"""Generate direct finetune experiments (no Stage 1 source pretraining).
17 target languages × 2 models (53 + Base) = 34 experiments.
"""
import os

BASE = "/mnt/storage/qisheng/github/wav2vec_test"
CDIR53 = '26_6_16'
CDIR_BASE = '26_6_16_base'

TARGETS = ['mt','af','da','ky','tk','kk','sk','id','sq','ltg','ur','cy','gn','tn','am','he','az']
HALF = {'mt':830,'af':65,'da':1379,'ky':807,'tk':285,'kk':268,'sk':2526,'id':1845,
        'sq':958,'ltg':1765,'ur':2541,'cy':2704,'gn':480,'tn':184,'am':126,'he':196,'az':47}

for tgt in TARGETS:
    for ms in ['53', 'base']:
        cdir = CDIR53 if ms == '53' else CDIR_BASE
        model_name = "facebook/wav2vec2-large-xlsr-53" if ms == '53' else "facebook/wav2vec2-base"
        lr = '2e-4' if ms == '53' else '3e-4'
        suffix = f'direct{ms}'
        wdir = f'{tgt}_100-xlsr-{suffix}'
        cfg_name = f'{tgt}_100-{suffix}.py'
        slurm_name = f'{tgt}_100-{suffix}.sh'
        
        cfg = f"""# config_{ms}.py
wandb_project = "wav2vec_26_6_16"
wandb_run = "{tgt}_100-{suffix}"
preprocessing_only = False
resume = False
TRAINING_PARAMS = {{
    "output_dir": "../weights/{wdir}", 
    "overwrite_output_dir":True,        
    "num_train_epochs": 600,                                    
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "learning_rate": {lr},
    "max_grad_norm": 1.5,
    "warmup_steps": 2000,
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
    "model_name_or_path" : "{model_name}",
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
    "use_phoneme":True,
    "vocab_dir":"/mnt/storage/qisheng/github/wav2vec_test/weights/general_phoneme",
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
            f.write(cfg)
        
        sh = f"""#!/bin/bash
#SBATCH --job-name=s2_{tgt}_{suffix}
#SBATCH --output={BASE}/slurm_config/out/s2_{tgt}_{suffix}.out
#SBATCH --error={BASE}/slurm_config/out/s2_{tgt}_{suffix}.err
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
            f.write(sh)
        
        print(f'  {tgt}_direct{ms}')

print(f'\nDone! {len(TARGETS) * 2} direct finetune experiments')
