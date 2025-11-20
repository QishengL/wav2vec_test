wandb_project = "wav2vec_classification"
wandb_run = "class1"
preprocessing_only = False
train_samples = 1000
eval_samples = 200
lan_list=["ar","be","bn","cs","cy","de","en","es","fa","fr","hu","it","ja","ka","lt","lv","nl","pl","pt","ru","ro","sw","ta","th","tr"]
TRAINING_PARAMS = {
    "output_dir": "/mnt/storage/qisheng/github/wav2vec_test/weights/classification/xlsr-300m_class1",
    "overwrite_output_dir":True,        
    "num_train_epochs": 200,                                    
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-4,
    "lr_scheduler_type":"constant",  # Disables learning rate decay
    "weight_decay":0.0,  # No weight decay
    "warmup_steps":0,  # No learning rate warmup
    "max_grad_norm": 1.5,                                 
    "save_total_limit": 4,
    "gradient_checkpointing": True,                              
    "fp16": True,                                              
    "group_by_length": False, 
    "do_train": True, 
    "do_eval":True, 
    "report_to":"wandb",
    "logging_strategy":"steps",         
    "eval_strategy":"steps",
    "eval_steps":500,                 # 每 1000 步评估一次
    "logging_steps":50, 
                             
}
MODEL_PARAMS = {
    "output_dir": "/mnt/storage/qisheng/github/wav2vec_test/weights/classification/xlsr-300m_class1", 
    "model_name_or_path" : "facebook/wav2vec2-xls-r-300m",
    "feat_proj_dropout": 0.1,      # 增加正则化强度
    "attention_dropout": 0.1,      # 提高注意力dropout
    "hidden_dropout": 0.1,         # 提高隐藏层dropout
    "final_dropout": 0.1,          # 输出层dropout
    "mask_time_prob": 0.1,         # 更强的数据增强
    "mask_time_length": 10,
    "mask_feature_prob": 0.1,
    "mask_feature_length": 10,
    "layerdrop": 0.0,
    "activation_dropout": 0.1,
    "add_adapter": False,
    "freeze_feature_encoder":True,
    "num_labels":len(lan_list),
}

DATASET_PARAMS={
    "dataset_name" : "fixie-ai/common_voice_17_0",  
    "dataset_config_name" : "uk",
    "train_split" : "train+validation",
    "test_split" : "test",
    "chars_to_ignore" : [",", "?", ".", "!", "-", ";", ":", "\"", "“", "%", "‘", "”","�", "(", ")", "'"],
    "text_column":'sentence',
    "audio_column":'audio',
    "max_train_samples_per_language":[50],
    "max_eval_samples_per_language":[],
    "max_train_samples":None,
    "max_eval_samples":None,
    "max_duration_in_seconds":20.0,
    "min_duration_in_seconds":0.0,
    "preprocessing_num_workers":1,
    "eval_metrics" : ["accuracy"],
    "cache_dir":"/mnt/storage/ldl_linguistics/datasets",

}