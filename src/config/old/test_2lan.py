# config_base.py
#create_vocab = True
wandb_project = "wav2vec_test"
wandb_run = "test2lan"
preprocessing_only = False
resume = False
TRAINING_PARAMS = {
    "output_dir": "../wav2vec2-common_voice-test2lan", 
    "overwrite_output_dir":True,        
    "num_train_epochs": 5,                                    
    "per_device_train_batch_size": 10,
    "gradient_accumulation_steps": 2,
    "learning_rate": 3e-4,
    "warmup_steps": 500,
    "save_steps": 1,                                         
    "save_total_limit": 2,
    "gradient_checkpointing": True,                              
    "fp16": True,                                              
    "group_by_length": True, 
    "do_train": True, 
    "do_eval":True, 
    "report_to":"wandb",
    "logging_strategy":"steps",         
    "eval_strategy":"steps",
    "eval_steps":1,                 # 每 1000 步评估一次
    "logging_steps":1, 
                             
}
MODEL_PARAMS = {
    "output_dir": "../wav2vec2-common_voice-test2lan", 
    "model_name_or_path" : "facebook/wav2vec2-xls-r-300m",
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
    "layerdrop": 0.0,
    "ctc_loss_reduction": "mean",
    "ctc_zero_infinity": False,
    "activation_dropout": 0.1,
    "add_adapter": False,
    "freeze_feature_encoder":True,
    "use_phoneme":True,
    #"phoneme_lang":"tr",
    "vocab_dir":"../vocab_builder/vocab_folder/cv_uk_ro_phoneme"
}
DATASET_PARAMS={
    "dataset_name" : "fixie-ai/common_voice_17_0",  
    "dataset_config_name" : ["uk","ro"],
    "train_split" : "train+validation",
    "test_split" : "test",
    "chars_to_ignore" : [",", "?", ".", "!", "-", ";", ":", "\"", "“", "%", "‘", "”","�", "(", ")", "'"],
    "text_column":'sentence',
    "audio_column":'audio',
    "max_train_samples_per_language":4,
    "max_eval_samples_per_language":4,
    "max_train_samples":2,
    "max_eval_samples":2,
    "max_duration_in_seconds":20.0,
    "min_duration_in_seconds":0.0,
    "preprocessing_num_workers":1,
    "eval_metrics" : ["wer"],
    "cache_dir":"/mnt/storage/ldl_linguistics/datasets",

}