subfolder = "cv_tr_speech"
DATASET_PARAMS={
    "dataset_name" : "mozilla-foundation/common_voice_17_0",  
    "dataset_config_name" : "tr",
    "output_dir" : f"./vocab_folder/{subfolder}",
    "train_split" : "train+validation",
    "test_split" : "test",
    "chars_to_ignore" : ["…","’",",", "?", ".", "!", "-", ";", ":", "\"", "“", "%", "‘", "”","�", "(", ")", "'"],
    "text_column":'sentence',
    "audio_column":'audio',
    "word_delimiter_token": "|",
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "use_phoneme":False,
    "phoneme_lang":None,
}