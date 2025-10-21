# config_base.py
create_vocab = True
wandb_project = "wav2vec_test"
wandb_run = "test1"
preprocessing_only = False
TRAINING_PARAMS = {
    "output_dir": "../wav2vec2-common_voice-tr-demo", 
    "overwrite_output_dir":True,        
    "num_train_epochs": 15,                                    
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "learning_rate": 3e-4,
    "warmup_steps": 500,
    "save_steps": 400,                                         
    "save_total_limit": 3,
    "gradient_checkpointing": True,                              
    "fp16": True,                                              
    "group_by_length": True, 
    "do_train": True, 
    "do_eval":True, 
    "report_to":"wandb",
    "logging_strategy":"steps",         
    "evaluation_strategy":"steps",      # 按步数评估
    "eval_steps":200,                   # 每 200 步评估一次
    "logging_steps":10,                 # 每 10 步记录一次训练指标
                             
}
MODEL_PARAMS = {
    "output_dir": "../wav2vec2-common_voice-en-demo", 
    "model_name_or_path" : "facebook/wav2vec2-large-xlsr-53",
    "word_delimiter_token": "|",
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "feat_proj_dropout": 0,
    "attention_dropout": 0,
    "hidden_dropout": 0,
    "final_dropout": 0,
    "mask_time_prob": 0.05,
    "mask_time_length": 10,
    "mask_feature_prob": 0,
    "mask_feature_length": 10,
    "layerdrop": 0,
    "ctc_loss_reduction": "mean",
    "ctc_zero_infinity": False,
    "activation_dropout": 0,
    "add_adapter": False,
    "freeze_feature_encoder":True,
}
DATASET_PARAMS={
    "dataset_name" : "mozilla-foundation/common_voice_17_0",  
    "dataset_config_name" : "en",
    "train_split" : "train+validation",
    "test_split" : "test",
    "chars_to_ignore" : ", ? . ! - ; : \" “ % ‘ ” � ' - ’",
    "text_column":'sentence',
    "audio_column":'audio',
    "max_train_samples":50000,
    "max_eval_samples":5000,
    "max_duration_in_seconds":20.0,
    "min_duration_in_seconds":0.0,
    "preprocessing_num_workers":1,
    "eval_metrics" : ["wer"],
}