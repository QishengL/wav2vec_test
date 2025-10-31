# dataset.py
import datasets
import re
from datasets import load_dataset
from phonemizer.backend import BACKENDS
from phonemizer.separator import Separator

def load_datasets(**config):
    from datasets import concatenate_datasets
    
    raw_datasets = datasets.DatasetDict()
    
    dataset_configs = config["dataset_config_name"]
    if isinstance(dataset_configs, str):
        dataset_configs = [dataset_configs]
    
    train_datasets = []
    eval_datasets = []
    
    for config_name in dataset_configs:
        print(f"Loading {config_name} dataset...")
        
        # 加载数据
        train_ds = load_dataset(
            config["dataset_name"],
            config_name,
            split=config["train_split"],
            trust_remote_code=True,
            cache_dir=config["cache_dir"],
        )
        
        eval_ds = load_dataset(
            config["dataset_name"],
            config_name,
            split=config["test_split"],
            trust_remote_code=True,
            cache_dir=config["cache_dir"],
        )
        
         # 在添加语言标识前先应用单语言样本限制
        max_train_samples_per_language = config.get("max_train_samples_per_language")
        max_eval_samples_per_language = config.get("max_eval_samples_per_language")
        if max_train_samples_per_language is not None:
            # 对每个语言的训练集和验证集分别限制样本数
            if len(train_ds) > max_train_samples_per_language:
                train_ds = train_ds.select(range(max_train_samples_per_language))
                print(f"Limited {config_name} training set to {max_train_samples_per_language} samples")
        if max_eval_samples_per_language is not None:    
            if len(eval_ds) > max_eval_samples_per_language:
                eval_ds = eval_ds.select(range(max_eval_samples_per_language))
                print(f"Limited {config_name} evaluation set to {max_eval_samples_per_language} samples")


        # 添加语言标识
        train_ds = train_ds.add_column("language", [config_name] * len(train_ds))
        eval_ds = eval_ds.add_column("language", [config_name] * len(eval_ds))
        
        train_datasets.append(train_ds)
        eval_datasets.append(eval_ds)
    
    # 合并数据集
    raw_datasets["train"] = concatenate_datasets(train_datasets).shuffle(seed=42)
    raw_datasets["eval"] = concatenate_datasets(eval_datasets).shuffle(seed=42)
    
    # 应用样本限制
    if config["max_train_samples"] is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(config["max_train_samples"]))
    if config["max_eval_samples"] is not None:
        raw_datasets["eval"] = raw_datasets["eval"].select(range(config["max_eval_samples"]))
    
    return raw_datasets





def preprocess_datasets(raw_datasets, **config):
    # 示例：清理字符

    chars_to_ignore = config["chars_to_ignore"]
    if chars_to_ignore:
        if isinstance(chars_to_ignore, str):
            chars_to_ignore = chars_to_ignore.split()
        chars_to_ignore_regex = f"[{''.join(re.escape(c) for c in chars_to_ignore)}]"
    else:
        chars_to_ignore_regex = None

    def remove_special_characters(batch):
        text = batch[config["text_column"]]
        if chars_to_ignore_regex is not None:
            text = re.sub(chars_to_ignore_regex, "", text)
        batch["target_text"] = text.lower() + " "
        return batch

    return raw_datasets.map(remove_special_characters, desc="Clean text")

"""
no need cause in tokenzier the sentence will be convert to phoneme
def preprocess_datasets_phoneme(raw_datasets,lan, **config):
    # 示例：清理字符
    backend = BACKENDS["espeak"](lan, language_switch="remove-flags")

    def convert_phoneme(batch):
        text = batch['sentence']
        #print(text)
        
        separator = Separator(phone=' ', word="", syllable="")
        phonemes = backend.phonemize(
            [text],
            separator=separator,
        )
        processed_text = phonemes[0].strip()
        #print(processed_text)
        #return
        batch["target_text"] = processed_text
        return batch

    return raw_datasets.map(convert_phoneme,num_proc=1, desc="convert phoneme")
"""






def vectorize_datasets(raw_datasets, tokenizer, feature_extractor, **config):
    
    dataset_sampling_rate = next(iter(raw_datasets.values())).features[config["audio_column"]].sampling_rate
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        raw_datasets = raw_datasets.cast_column(
            config["audio_column"],
            datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
        )

    max_input_length = config["max_duration_in_seconds"] * feature_extractor.sampling_rate
    min_input_length = config["min_duration_in_seconds"] * feature_extractor.sampling_rate
    audio_column_name = config["audio_column"]
    num_workers = config["preprocessing_num_workers"]
    feature_extractor_input_name = feature_extractor.model_input_names[0]

    language_phoneme_map = config.get("language_phoneme_map", {})
    
    if "language" not in next(iter(raw_datasets.values())).column_names:
        raise ValueError("Dataset must contain 'language' column for multi-language processing")

    languages = list(set(raw_datasets["train"]["language"]))
    print(f"Processing datasets for languages: {languages}")

    def prepare_dataset_single(example):
        """处理单个样本"""
        # load audio
        sample = example[audio_column_name]

        # 处理音频特征
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        example[feature_extractor_input_name] = getattr(inputs, feature_extractor_input_name)[0]
        
        # 记录音频长度
        example["input_length"] = len(sample["array"].squeeze())

        # 根据语言设置音素化参数
        additional_kwargs = {}
        if "language" in example:
            language = example["language"]
            phoneme_lang = language_phoneme_map.get(language, language)
            additional_kwargs["phonemizer_lang"] = phoneme_lang

        # 编码文本标签
        example["labels"] = tokenizer(example["target_text"], **additional_kwargs).input_ids
        
        return example

    # 定义要移除的列（保留language列用于评估）
    columns_to_remove = [col for col in next(iter(raw_datasets.values())).column_names 
                        if col not in ["language"]]  # 保留language列

    vectorized_datasets = raw_datasets.map(
        prepare_dataset_single,
        remove_columns=columns_to_remove,
        num_proc=num_workers,
        desc="preprocess datasets with language-specific phoneme processing",
    )

    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_length"],
    )
    
    return vectorized_datasets

    