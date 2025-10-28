import os
import json
import functools
from phonemizer import phonemize
from phonemizer.separator import Separator
from phonemizer.backend import BACKENDS
def create_vocabulary_from_data(
    datasets,
    use_phoneme = False,
    language = None, 
    word_delimiter_token = None,
    unk_token = None,
    pad_token = None,
):
    # Given training and test labels create vocabulary
    def extract_all_chars(batch):
        backend = BACKENDS["espeak"](language, language_switch="remove-flags")
        all_text = " ".join(batch["target_text"])

        
        separator = Separator(phone=' ', word="", syllable="")
        if use_phoneme:
            # 使用音素化处理文本
            #print(all_text)
            phonemes = backend.phonemize(
                [all_text],
                separator=separator,
            )
            phonemized_text = phonemes[0].strip()
            #print(phonemes)
            phonemes = phonemized_text.split(" ")
                # 过滤空字符串
            #phonemes = [p for p in phonemes if p]
            
            
            vocab = list(set(phonemes))
            #print(set(vocab))
            return {"vocab": [vocab], "all_text": [phonemized_text]}

        else:
            # 原始字符处理
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
    # 安全地替换空格为分隔符
    if word_delimiter_token is not None:
        if " " in vocab_dict:
            # 如果空格在词汇表中，替换它
            vocab_dict[word_delimiter_token] = vocab_dict[" "]
            del vocab_dict[" "]
        #else:
            # 如果空格不在词汇表中，直接添加分隔符
        #    vocab_dict[word_delimiter_token] = len(vocab_dict)

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
            model_config['use_phoneme'],
            model_config['phoneme_lang'],
            word_delimiter_token=word_delimiter_token,
            unk_token=unk_token,
            pad_token=pad_token,
        )

        # save vocab dict to be loaded into tokenizer
        with open(vocab_file, "w") as file:
            json.dump(vocab_dict, file)