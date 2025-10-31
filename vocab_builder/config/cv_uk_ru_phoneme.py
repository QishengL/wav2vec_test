subfolder = "cv_uk_ru_phoneme"
DATASET_PARAMS={
    "dataset_name" : "fixie-ai/common_voice_17_0",  
    "dataset_config_name" : ["ru","uk"],
    "output_dir" : f"./vocab_folder/{subfolder}",
    "train_split" : "train+validation",
    "test_split" : "test",
    "chars_to_ignore" : ["…","’",",", "?", ".", "!", "-", ";", ":", "\"", "“", "%", "‘", "”","�", "(", ")", "'"],
    "text_column":'sentence',
    "audio_column":'audio',
    "word_delimiter_token": "|",
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "use_phoneme":True,
    "cache_dir":"/mnt/storage/ldl_linguistics/datasets",

}