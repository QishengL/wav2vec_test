# config_base.py
wandb_project = "wav2vec_test"
wandb_run = "german1"
preprocessing_only = False
resume = True
#checkpoint = "../wav2vec2-common_voice-tr-demo/checkpoint-21105"
TRAINING_PARAMS = {
    "output_dir": "../german_test1", 
    "overwrite_output_dir":True,        
    "num_train_epochs": 200,                                    
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size":1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-4,
    "weight_decay":0.005,
    "warmup_steps": 1000,
    "save_steps": 1000,                                         
    "save_total_limit": 1,
    "gradient_checkpointing": True,                              
    "fp16": True,                                              
    "group_by_length": True, 
    "do_train": True, 
    "do_eval":True, 
    "report_to":"wandb",
    "logging_strategy":"steps",         
    "eval_strategy":"steps",
    "eval_steps":400,                 # 每 1000 步评估一次
    "logging_steps":10, 
                             
}
#"model_name_or_path" : "facebook/wav2vec2-large-xlsr-53",
MODEL_PARAMS = {
    "output_dir": "../german_test1", 
    "model_name_or_path" : "facebook/wav2vec2-large-xlsr-53",
    "word_delimiter_token": "|",
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "feat_proj_dropout": 0.0,
    "attention_dropout": 0.0,
    "hidden_dropout": 0.0,
    "final_dropout": 0.0,
    "mask_time_prob": 0.75,
    "mask_time_length": 10,
    "mask_feature_prob": 0.25,
    "mask_feature_length": 64,
    "layerdrop": 0.1,
    "ctc_loss_reduction": "mean",
    "ctc_zero_infinity": False,
    "activation_dropout": 0.1,
    "add_adapter": False,
    "freeze_feature_encoder":True,
}
DATASET_PARAMS={
    "dataset_name" : "facebook/multilingual_librispeech",  
    "dataset_config_name" : "german",
    "train_split" : "9_hours",
    "test_split" : "test",
    "chars_to_ignore" : [",", "?", ".", "!", "-", ";", ":", "\"", "“", "%", "‘", "”","�", "(", ")", "'"],
    "text_column":'transcript',
    "audio_column":'audio',
    "max_train_samples":None,
    "max_eval_samples":None,
    "max_duration_in_seconds":20.0,
    "min_duration_in_seconds":0.0,
    "preprocessing_num_workers":1,
    "eval_metrics" : ["wer"],
}