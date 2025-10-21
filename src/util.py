
import os
import json
import functools


def create_vocabulary_from_data(
    datasets,
    word_delimiter_token = None,
    unk_token = None,
    pad_token = None,
):
    # Given training and test labels create vocabulary
    def extract_all_chars(batch):
        all_text = " ".join(batch["target_text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = datasets.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=datasets["train"].column_names,
    )

    # take union of all unique characters in each dataset
    vocab_set = functools.reduce(
        lambda vocab_1, vocab_2: set(vocab_1["vocab"][0]) | set(vocab_2["vocab"][0]),
        vocabs.values(),
    )

    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_set))}

    # replace white space with delimiter token
    if word_delimiter_token is not None:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    return vocab_dict




def create_vocab(raw_datasets,**model_config):
    word_delimiter_token = model_config["word_delimiter_token"]
    unk_token = model_config["unk_token"]
    pad_token = model_config["pad_token"]
    
    
    #tokenizer_name_or_path = model_args.tokenizer_name_or_path

    tokenizer_kwargs = {}

    #if tokenizer_name_or_path is None:
        # save vocab in training output dir
    tokenizer_name_or_path = model_config["output_dir"]

    vocab_file = os.path.join(tokenizer_name_or_path, "vocab.json")

    
    if os.path.isfile(vocab_file):
        try:
            os.remove(vocab_file)
        except OSError:
            # in shared file-systems it might be the case that
            # two processes try to delete the vocab file at the some time
            pass
    
    if not os.path.isfile(vocab_file):
        os.makedirs(tokenizer_name_or_path, exist_ok=True)
        vocab_dict = create_vocabulary_from_data(
            raw_datasets,
            word_delimiter_token=word_delimiter_token,
            unk_token=unk_token,
            pad_token=pad_token,
        )

        # save vocab dict to be loaded into tokenizer
        with open(vocab_file, "w") as file:
            json.dump(vocab_dict, file)

    # if tokenizer has just been created
    # it is defined by `tokenizer_class` if present in config else by `model_type`