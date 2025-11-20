# config_base.py
#create_vocab = True
general = True
wandb_project = "wav2vec_300m_lora_11_14_ur"
wandb_run = "hi2ur_100v2"
checkpoint = "/mnt/storage/qisheng/github/wav2vec_test/weights/hi_9000-xlsr-300m_general/checkpoint-20000"
new_vocab_path = "/mnt/storage/qisheng/github/wav2vec_test/vocab_builder/vocab_folder/cv_ur_phoneme"
preprocessing_only = False
resume = False
TRAINING_PARAMS = {
    "output_dir": "../weights/lora/hi2ur_100-xlsr-300m_general", 
    "overwrite_output_dir":True,        
    "num_train_epochs": 2000,                                    
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-3,
    "max_grad_norm": 1.5,
    #"warmup_steps": 1000,
    "save_steps": 2000,                                         
    "save_total_limit": 4,
    "gradient_checkpointing": True,                              
    "fp16": True,                                              
    "group_by_length": True, 
    "do_train": True, 
    "do_eval":True, 
    "report_to":"wandb",
    "logging_strategy":"steps",         
    "eval_strategy":"steps",
    "eval_steps":2000,                 
    "logging_steps":50, 
                             
}
MODEL_PARAMS = {
    "output_dir": "../weights/lora/hi2ur_100-xlsr-300m_general", 
    "model_name_or_path" : "facebook/wav2vec2-xls-r-300m",
    "word_delimiter_token": "|",
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "feat_proj_dropout": 0.1,      # 增加正则化强度
    "attention_dropout": 0.1,      # 提高注意力dropout
    "hidden_dropout": 0.1,         # 提高隐藏层dropout
    "final_dropout": 0.1,          # 输出层dropout
    "mask_time_prob": 0.1,         # 更强的数据增强
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
    #"phoneme_lang":"tr",
    #"vocab_dir":"../vocab_builder/vocab_folder/cv_ru_phoneme"
}
DATASET_PARAMS={
    "dataset_name" : "fixie-ai/common_voice_17_0",  
    "dataset_config_name" : "ur",
    "train_split" : "train+validation",
    "test_split" : "test",
    "chars_to_ignore" : [",", "?", ".", "!", "-", ";", ":", "\"", "“", "%", "‘", "”","�", "(", ")", "'"],
    "text_column":'sentence',
    "audio_column":'audio',
    "max_train_samples_per_language":[100],
    "max_eval_samples_per_language":[],
    "max_train_samples":None,
    "max_eval_samples":None,
    "max_duration_in_seconds":20.0,
    "min_duration_in_seconds":0.0,
    "preprocessing_num_workers":1,
    "eval_metrics" : ["wer"],
    "cache_dir":"/mnt/storage/ldl_linguistics/datasets",

}
LORA_PARAMS = {
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "r": 16,
    "bias": "none",
    "target_modules": ["q_proj", "v_proj"],
}