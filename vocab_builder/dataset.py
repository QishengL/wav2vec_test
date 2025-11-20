# dataset.py
import datasets
import re
from datasets import load_dataset

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
    raw_datasets["eval"] = concatenate_datasets(eval_datasets)
    
    # 应用样本限制
    max_train_samples = config.get("max_train_samples")
    max_eval_samples = config.get("max_eval_samples")
    if max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(config["max_train_samples"]))
    if max_eval_samples is not None:
        raw_datasets["eval"] = raw_datasets["eval"].select(range(config["max_eval_samples"]))
    
    return raw_datasets

def preprocess_datasets(raw_datasets, **config):
    def clean_text(batch):
        batch["target_text"] = batch[config["text_column"]]
        
        #text = batch[config["text_column"]]
        #batch["target_text"] = ' '.join(text.lower().split()) 
        # 直接转换为小写并添加空格
        #batch["target_text"] = text.lower() + " "
        return batch

    return raw_datasets.map(clean_text, desc="Clean text")





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

    