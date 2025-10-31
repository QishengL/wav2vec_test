# config_base.py
#create_vocab = True
wandb_project = "wav2vec_test"
wandb_run = "tr_ph_test2"
preprocessing_only = False
resume = True
TRAINING_PARAMS = {
    "output_dir": "../wav2vec2-common_voice-tr-ph2", 
    "overwrite_output_dir":True,        
    "num_train_epochs": 200,                                    
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 2,
    "learning_rate": 3e-4,
    "warmup_steps": 500,
    "save_steps": 400,                                         
    "save_total_limit": 2,
    "gradient_checkpointing": True,                              
    "fp16": True,                                              
    "group_by_length": True, 
    "do_train": True, 
    "do_eval":True, 
    "report_to":"wandb",
    "logging_strategy":"steps",         
    "eval_strategy":"steps",
    "eval_steps":400,                 # 每 1000 步评估一次
    "logging_steps":1, 
                             
}
MODEL_PARAMS = {
    "output_dir": "../wav2vec2-common_voice-tr-ph2", 
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
    "phoneme_lang":"tr",
    "vocab_dir":"../vocab_builder/vocab_folder/cv_tr_phoneme"
}
DATASET_PARAMS={
    "dataset_name" : "mozilla-foundation/common_voice_17_0",  
    "dataset_config_name" : "tr",
    "train_split" : "train+validation",
    "test_split" : "test",
    "chars_to_ignore" : [",", "?", ".", "!", "-", ";", ":", "\"", "“", "%", "‘", "”","�", "(", ")", "'"],
    "text_column":'sentence',
    "audio_column":'audio',
    "max_train_samples":None,
    "max_eval_samples":None,
    "max_duration_in_seconds":20.0,
    "min_duration_in_seconds":0.0,
    "preprocessing_num_workers":1,
    "eval_metrics" : ["wer"],

}