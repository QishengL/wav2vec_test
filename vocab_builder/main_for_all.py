from dataset import load_datasets, preprocess_datasets,vectorize_datasets
from build_vocab import create_vocab
import importlib.util
import argparse

#backend_lang = ['ar', 'be', 'bg', 'bn', 'cs', 'cy', 'da', 'de', 'el', 'es', 'et', 'fa', 'fi', 'hi', 'hu', 'it', 'ja', 'ka', 'ko', 'lt', 'lv', 'mk', 'ml', 'mn', 'mr', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi','en-us','fr-fr']
#config_lang = ['ar', 'be', 'bg', 'bn', 'cs', 'cy', 'da', 'de', 'el', 'es', 'et', 'fa', 'fi', 'hi', 'hu', 'it', 'ja', 'ka', 'ko', 'lt', 'lv', 'mk', 'ml', 'mn', 'mr', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi','en','fr']
backend_lang = ['hu', 'it', 'ja', 'ka', 'ko', 'lt', 'lv', 'mk', 'ml', 'mn', 'mr', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi','en-us','fr-fr']
config_lang = ['hu', 'it', 'ja', 'ka', 'ko', 'lt', 'lv', 'mk', 'ml', 'mn', 'mr', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi','en','fr']
for idx in range(len(backend_lang)):
    backend_lan = backend_lang[idx]
    config_lan = config_lang[idx]
    subfolder = f"cv_{config_lan}_phoneme"
    DATASET_PARAMS={
        "dataset_name" : "fixie-ai/common_voice_17_0",  
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








