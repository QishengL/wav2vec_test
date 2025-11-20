# dataset.py
import datasets
import re
from datasets import load_dataset
from phonemizer.backend import BACKENDS
from phonemizer.separator import Separator
def load_datasets(lan,max_train_sample=None,max_eval_sample=None,random=None,**config):
    raw_datasets = datasets.DatasetDict()

    raw_datasets["train"] = load_dataset(
        config["dataset_name"],
        lan,
        split=config["train_split"],
        trust_remote_code=True,
        cache_dir=config["cache_dir"],
    )

    raw_datasets["eval"] = load_dataset(
        config["dataset_name"],
        lan,
        split=config["test_split"],
        trust_remote_code=True,
        cache_dir=config["cache_dir"],
    )
    #need to debug

    ori_train_len = len(raw_datasets["train"])
    ori_eval_len = len(raw_datasets["eval"])
    if max_train_sample is not None:
        if max_train_sample < len(raw_datasets["train"]):
            print(f"select:train{max_train_sample}")
            if random is not None:
                raw_datasets["train"] = raw_datasets["train"].shuffle(seed=random)
            raw_datasets["train"] = raw_datasets["train"].select(range(max_train_sample))
    if max_eval_sample is not None:
        if max_eval_sample < len(raw_datasets["eval"]):
            print(f"select:test{max_train_sample}")
            raw_datasets["eval"] = raw_datasets["eval"].select(range(max_eval_sample))
    #if config["max_train_samples"] is not None:
    #    raw_datasets["train"] = raw_datasets["train"].select(range(config["max_train_samples"]))
    #if config["max_eval_samples"] is not None:
    #    raw_datasets["eval"] = raw_datasets["eval"].select(range(config["max_eval_samples"]))
    print(f"origin train:{ori_train_len}")
    print(f"origin test:{ori_eval_len}")
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

def preprocess_datasets_phoneme(raw_datasets,lan, **config):
    # 示例：清理字符
    backend = BACKENDS["espeak"](lan, language_switch="remove-flags")
    chars_to_ignore = config["chars_to_ignore"]
    if chars_to_ignore:
        if isinstance(chars_to_ignore, str):
            chars_to_ignore = chars_to_ignore.split()
        chars_to_ignore_regex = f"[{''.join(re.escape(c) for c in chars_to_ignore)}]"
    else:
        chars_to_ignore_regex = None

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



def vectorize_datasets(raw_datasets,tokenizer,feature_extractor, **config):
    
    dataset_sampling_rate = next(iter(raw_datasets.values())).features[config["audio_column"]].sampling_rate
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        raw_datasets = raw_datasets.cast_column(
            config["audio_column"],
            datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
        )

    # derive max & min input length for sample rate & max duration
    max_input_length = config["max_duration_in_seconds"] * feature_extractor.sampling_rate
    min_input_length = config["min_duration_in_seconds"] * feature_extractor.sampling_rate
    audio_column_name = config["audio_column"]
    num_workers = config["preprocessing_num_workers"]
    feature_extractor_input_name = feature_extractor.model_input_names[0]

    # `phoneme_language` is only relevant if the model is fine-tuned on phoneme classification
    #phoneme_language = data_args.phoneme_language


    def prepare_dataset(batch):
        # load audio
        sample = batch[audio_column_name]

        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch[feature_extractor_input_name] = getattr(inputs, feature_extractor_input_name)[0]
        # take length of raw audio waveform
        batch["input_length"] = len(sample["array"].squeeze())

        # encode targets
        additional_kwargs = {}
        #if phoneme_language is not None:
        #    additional_kwargs["phonemizer_lang"] = phoneme_language

        batch["labels"] = tokenizer(batch["target_text"], **additional_kwargs).input_ids
        #print(batch["target_text"])
        #print(batch["labels"])
        return batch

    
    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=next(iter(raw_datasets.values())).column_names,
        num_proc=num_workers,
        desc="preprocess datasets",
    )

    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    # filter data that is shorter than min_input_length
    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_length"],
    )
    return vectorized_datasets


def vectorize_datasets_classification(raw_datasets, tokenizer, feature_extractor, **config):
    
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

    def prepare_dataset(batch):
        # load and process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch[feature_extractor_input_name] = getattr(inputs, feature_extractor_input_name)[0]
        batch["input_length"] = len(sample["array"].squeeze())
        
        # 使用语言标签作为分类目标
        batch["labels"] = batch["language_label"]
        
        return batch

    # 处理数据集
    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=[col for col in next(iter(raw_datasets.values())).column_names 
                       if col not in [config["audio_column"], "language_label"]],
        num_proc=num_workers,
        desc="preprocess datasets",
    )

    # 过滤音频长度
    def is_audio_in_length_range(length):
        return min_input_length < length < max_input_length

    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_length"],
    )
    
    return vectorized_datasets


def load_and_combine_multilingual_datasets(languages, max_train_sample=None, max_eval_sample=None, random=None, **config):
    """
    加载并合并多语言数据集
    
    Args:
        languages: 字典，格式为 {'语言代码': 数字标签}，如 {'en': 0, 'zh': 1, 'fr': 2}
        max_train_sample: 每种语言的训练集最大样本数
        max_eval_sample: 每种语言的验证集最大样本数
        random: 随机种子
        config: 数据集配置参数
    """
    from datasets import concatenate_datasets
    
    combined_train = []
    combined_eval = []
    
    for lang_code, lang_label in languages.items():
        print(f"Loading {lang_code} dataset (label: {lang_label})...")
        
        # 加载数据集
        train_ds = load_dataset(
            config["dataset_name"],
            lang_code,
            split=config["train_split"],
            trust_remote_code=True,
            cache_dir=config["cache_dir"],
        )
        
        eval_ds = load_dataset(
            config["dataset_name"],
            lang_code, 
            split=config["test_split"],
            trust_remote_code=True,
            cache_dir=config["cache_dir"],
        )
        
        # 采样
        if max_train_sample is not None and max_train_sample < len(train_ds):
            if random is not None:
                train_ds = train_ds.shuffle(seed=random)
            train_ds = train_ds.select(range(max_train_sample))
            
        if max_eval_sample is not None and max_eval_sample < len(eval_ds):
            eval_ds = eval_ds.select(range(max_eval_sample))
        
        # 添加语言标签 - 使用数字标签
        def add_language_label(example, language_code, language_label):
            example["language_code"] = language_code  # 保留语言代码
            example["language_label"] = language_label  # 数字标签
            return example
            
        train_ds = train_ds.map(lambda x: add_language_label(x, lang_code, lang_label))
        eval_ds = eval_ds.map(lambda x: add_language_label(x, lang_code, lang_label))
        
        combined_train.append(train_ds)
        combined_eval.append(eval_ds)
        
        print(f"Added {len(train_ds)} {lang_code} train samples, {len(eval_ds)} eval samples")
    
    # 合并所有语言数据
    final_train = concatenate_datasets(combined_train)
    final_eval = concatenate_datasets(combined_eval)
    
    # 打乱顺序
    if random is not None:
        final_train = final_train.shuffle(seed=random)
        final_eval = final_eval.shuffle(seed=random)
    
    combined_dataset = datasets.DatasetDict({
        "train": final_train,
        "eval": final_eval
    })
    
    print(f"Final combined dataset: {len(final_train)} train samples, {len(final_eval)} eval samples")
    print(f"Language labels mapping: {languages}")
    
    return combined_dataset



'''
# 使用示例
languages = {
    'en': 0,  # 英语 -> 标签0
    'zh': 1,  # 中文 -> 标签1  
    'fr': 2,  # 法语 -> 标签2
    'es': 3,  # 西班牙语 -> 标签3
    'de': 4,  # 德语 -> 标签4
    'ja': 5,  # 日语 -> 标签5
    'ko': 6   # 韩语 -> 标签6
}

multilingual_dataset = load_and_combine_multilingual_datasets(
    languages=languages,
    max_train_sample=250,  # 每种语言250条训练数据
    max_eval_sample=50,    # 每种语言50条验证数据  
    random=42,
    **config.DATASET_PARAMS
)
'''
    