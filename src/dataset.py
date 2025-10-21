# dataset.py
import datasets
import re
from datasets import load_dataset

def load_datasets(**config):
    raw_datasets = datasets.DatasetDict()

    raw_datasets["train"] = load_dataset(
        config["dataset_name"],
        config["dataset_config_name"],
        split=config["train_split"],
        trust_remote_code=True,
    )

    raw_datasets["eval"] = load_dataset(
        config["dataset_name"],
        config["dataset_config_name"],
        split=config["test_split"],
        trust_remote_code=True,
    )
    if "max_train_samples" in config:
        raw_datasets["train"] = raw_datasets["train"].select(range(config["max_train_samples"]))
    if "max_eval_samples" in config:
        raw_datasets["eval"] = raw_datasets["eval"].select(range(config["max_eval_samples"]))
    return raw_datasets

def preprocess_datasets(raw_datasets, **config):
    # 示例：清理字符

    chars_to_ignore = config["chars_to_ignore"]
    escaped_chars = ''.join(re.escape(c) for c in chars_to_ignore)
    regex_pattern = f"[{escaped_chars}]"
    print(regex_pattern)

    def remove_special_characters(batch):
        batch["target_text"] = re.sub(regex_pattern, "", batch[config["text_column"]]).lower() + " "
        return batch

    return raw_datasets.map(remove_special_characters, desc="Clean text")





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

    