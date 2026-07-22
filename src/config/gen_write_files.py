"""Read manifest and write all config/slurm files using write_file."""
import json, os
BASE = "/mnt/storage/qisheng/github/wav2vec_test"

with open('/mnt/storage/qisheng/github/wav2vec_test/src/config/gen_manifest.json') as f:
    items = json.load(f)

for item in items:
    # Config content
    cfg = f"""# config_{item['ms']}.py
wandb_project = "wav2vec_26_6_5"
wandb_run = "{item['tgt']}_100-{item['src']}{item['ms']}"
preprocessing_only = False
resume = False
TRAINING_PARAMS = {{
    "output_dir": "../weights/{item['wdir']}", 
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
    "output_dir": "../weights/{item['wdir']}", 
    "model_name_or_path" : "{item['s1']}",
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
    "dataset_config_name" : "{item['tgt']}",
    "train_split" : "train",
    "test_split" : "test",
    "chars_to_ignore" : [',', '?', '.', '!', '-', ';', ':', "'"],
    "text_column":'sentence',
    "audio_column":'audio',
    "max_train_samples_per_language":[100],
    "max_eval_samples_per_language":[{item['eval_size']}],
    "max_train_samples":None,
    "max_eval_samples":None,
    "max_duration_in_seconds":20.0,
    "min_duration_in_seconds":0.0,
    "preprocessing_num_workers":1,
    "eval_metrics" : ["wer"],
    "cache_dir":"/mnt/storage/ldl_linguistics/datasets",
}}
"""
    os.makedirs(os.path.dirname(item['cfg_path']), exist_ok=True)
    with open(item['cfg_path'], 'w') as f:
        f.write(cfg)
    
    # Slurm content
    sh = f"""#!/bin/bash
#SBATCH --job-name=s2_{item['tgt']}_{item['src']}{item['ms']}
#SBATCH --output={BASE}/slurm_config/out/s2_{item['tgt']}_{item['src']}{item['ms']}.out
#SBATCH --error={BASE}/slurm_config/out/s2_{item['tgt']}_{item['src']}{item['ms']}.err
#SBATCH --time=12:00:00
#SBATCH --mem=24G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

export WANDB_API_KEY=67129c4138cabfa6fe40ff02f228f65339bbba0d
export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

nvidia-smi

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd {BASE}/src;python3 main.py --config ./config/lora/{item['cdir']}/{item['cfg_name']}"
"""
    os.makedirs(os.path.dirname(item['slurm_path']), exist_ok=True)
    with open(item['slurm_path'], 'w') as f:
        f.write(sh)

print(f"Written {len(items)} configs and slurm scripts")
