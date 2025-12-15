from dataset import load_datasets, preprocess_datasets,vectorize_datasets
from build_vocab import create_vocab
import importlib.util
import argparse

#backend_lang = ['ar', 'be', 'bg', 'bn', 'cs', 'cy', 'da', 'de', 'el', 'es', 'et', 'fa', 'fi', 'hi', 'hu', 'it', 'ja', 'ka', 'ko', 'lt', 'lv', 'mk', 'ml', 'mn', 'mr', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi','en-us','fr-fr']
#config_lang = ['ar', 'be', 'bg', 'bn', 'cs', 'cy', 'da', 'de', 'el', 'es', 'et', 'fa', 'fi', 'hi', 'hu', 'it', 'ja', 'ka', 'ko', 'lt', 'lv', 'mk', 'ml', 'mn', 'mr', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi','en','fr']
backend_lang = ['hu', 'it', 'ja', 'ka', 'ko', 'lt', 'lv', 'mk', 'ml', 'mn', 'mr', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi','en-us','fr-fr']
config_lang = ['hu', 'it', 'ja', 'ka', 'ko', 'lt', 'lv', 'mk', 'ml', 'mn', 'mr', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi','en','fr']

lan_list = ['ar', 'ba', 'eu', 'be', 'bn', 'ca', 'yue', 'cs', 'nl', 'en', 'eo', 'fa', 'fr', 'ka', 'de', 'hu', 'it', 'ja', 'lv', 'lt', 'pl', 'pt', 'ro', 'ru', 'uk', 'es', 'sw', 'ta', 'th', 'tt', 'tr', 'ug', 'ur', 'uz', 'cy','zh-CN']
already_saved = ['ar', 'be', 'bg', 'bn', 'cs', 'cy', 'da', 'de', 'el', 'es', 'et', 'fa', 'fi', 'hi', 'hu', 'it', 'ja', 'ka', 'ko', 'lt', 'lv', 'mk', 'ml', 'mn', 'mr', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi','en','fr']


    
for idx in range(len(lan_list)):
    backend_lan = lan_list[idx]
    config_lan = lan_list[idx]
    if backend_lan in already_saved:
        continue
    subfolder = f"cv_{config_lan}_phoneme"
    DATASET_PARAMS={
        "dataset_name" : "fsicoli/common_voice_22_0",  
        "dataset_config_name" : config_lan,
        "output_dir" : f"./vocab_folder/{subfolder}",
        "train_split" : "train+validation",
        "test_split" : "test",
        "chars_to_ignore" : [],
        "text_column":'sentence',
        "audio_column":'audio',
        "word_delimiter_token": "|",
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "use_phoneme":True,
        "phoneme_lang":backend_lan,
        "cache_dir":"/mnt/storage/ldl_linguistics/datasets",
        "max_train_samples_per_language":50000,
        "max_eval_samples_per_language":10000,

    }
    print(f"{backend_lan},{config_lan},{subfolder}")
    raw_datasets = load_datasets(**DATASET_PARAMS)
    raw_datasets = preprocess_datasets(raw_datasets,**DATASET_PARAMS)
    create_vocab(raw_datasets,**DATASET_PARAMS)








