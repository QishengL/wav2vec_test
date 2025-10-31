import os
import json
import functools
from phonemizer import phonemize
from phonemizer.separator import Separator
from phonemizer.backend import BACKENDS


def create_vocabulary_from_data_multilingual(
    datasets,
    use_phoneme=False,
    languages=None,  # 改为支持多种语言
    word_delimiter_token=None,
    unk_token=None,
    pad_token=None,
    language_phoneme_map=None,  # 语言到音素后端的映射
):
    """为多语言数据创建统一的词汇表"""
    
    def extract_all_chars_multilingual(batch):
        all_phonemes = []
        all_texts = []
        
        # 按语言分组处理
        if "language" in batch:
            # 按语言分组
            language_groups = {}
            for i, (text, lang) in enumerate(zip(batch["target_text"], batch["language"])):
                if lang not in language_groups:
                    language_groups[lang] = []
                language_groups[lang].append(text)
            
            # 对每种语言分别处理
            for lang, texts in language_groups.items():
                combined_text = " ".join(texts)
                
                if use_phoneme:
                    # 获取对应语言的音素后端
                    phoneme_lang = language_phoneme_map.get(lang, lang) if language_phoneme_map else lang
                    print(f'using {phoneme_lang}')
                    backend = BACKENDS["espeak"](phoneme_lang, language_switch="remove-flags")
                    separator = Separator(phone=' ', word="", syllable="")
                    
                    phonemes = backend.phonemize(
                        [combined_text],
                        separator=separator,
                    )
                    phonemized_text = phonemes[0].strip()
                    all_phonemes.extend(phonemized_text.split(" "))
                    all_texts.append(phonemized_text)

                else:
                    # 字符级别处理
                    all_phonemes.extend(list(combined_text))
                    all_texts.append(combined_text)
        else:
            # 如果没有语言信息，统一处理
            all_text = " ".join(batch["target_text"])
            if use_phoneme:
                # 使用默认语言或第一个语言
                default_lang = languages[0] if languages else "en"
                try:
                    backend = BACKENDS["espeak"](default_lang, language_switch="remove-flags")
                    separator = Separator(phone=' ', word="", syllable="")
                    
                    phonemes = backend.phonemize(
                        [all_text],
                        separator=separator,
                    )
                    phonemized_text = phonemes[0].strip()
                    all_phonemes = phonemized_text.split(" ")
                    all_texts = [phonemized_text]
                except Exception as e:
                    print(f"Warning: Failed to phonemize with default language: {e}")
                    all_phonemes = list(all_text)
                    all_texts = [all_text]
            else:
                all_phonemes = list(all_text)
                all_texts = [all_text]
        
        return {"vocab": [list(set(all_phonemes))], "all_text": [" ".join(all_texts)]}

    # 处理所有数据集
    vocabs = datasets.map(
        extract_all_chars_multilingual,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=datasets["train"].column_names,
    )

    # 合并所有数据集的词汇
    vocab_set = functools.reduce(
        lambda vocab_1, vocab_2: set(vocab_1["vocab"][0]) | set(vocab_2["vocab"][0]),
        vocabs.values(),
    )

    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_set))}

    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    # replace white space with delimiter token
    # 安全地替换空格为分隔符
    if word_delimiter_token is not None:
        if " " in vocab_dict:
            # 如果空格在词汇表中，替换它
            vocab_dict[word_delimiter_token] = vocab_dict[" "]
            del vocab_dict[" "]
        if vocab_dict.get("") == 0:
            del vocab_dict[""]
            # 使用字典推导式更新所有值
            vocab_dict = {k: v-1 for k, v in vocab_dict.items()}
        #else:
            # 如果空格不在词汇表中，直接添加分隔符
        #    vocab_dict[word_delimiter_token] = len(vocab_dict)

    return vocab_dict





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
            #phonemes = phonemized_text.split(" ")
                # 过滤空字符串
            phonemes = [p for p in phonemized_text.split(" ") if p != ""]
            
            
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


    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    # replace white space with delimiter token
    # 安全地替换空格为分隔符
    if word_delimiter_token is not None:
        if " " in vocab_dict:
            # 如果空格在词汇表中，替换它
            vocab_dict[word_delimiter_token] = vocab_dict[" "]
            del vocab_dict[" "]
        if vocab_dict.get("") == 0:
            del vocab_dict[""]
            # 使用字典推导式更新所有值
            vocab_dict = {k: v-1 for k, v in vocab_dict.items()}
        #else:
            # 如果空格不在词汇表中，直接添加分隔符
        #    vocab_dict[word_delimiter_token] = len(vocab_dict)

    

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
        if isinstance(model_config['dataset_config_name'], str):
            vocab_dict = create_vocabulary_from_data(
                raw_datasets,
                model_config['use_phoneme'],
                model_config['phoneme_lang'],
                word_delimiter_token=word_delimiter_token,
                unk_token=unk_token,
                pad_token=pad_token,
            )
        else:
            languages = None
            if "language" in raw_datasets["train"].column_names:
                languages = list(set(raw_datasets["train"]["language"]))
                print(f"Detected languages in dataset: {languages}")
            
            # 语言到音素的映射
            language_phoneme_map = model_config.get("language_phoneme_map", None)
            
            vocab_dict = create_vocabulary_from_data_multilingual(
                raw_datasets,
                use_phoneme=model_config['use_phoneme'],
                languages=languages,
                word_delimiter_token=word_delimiter_token,
                unk_token=unk_token,
                pad_token=pad_token,
                language_phoneme_map=language_phoneme_map,
            )

        # save vocab dict to be loaded into tokenizer
        with open(vocab_file, "w") as file:
            json.dump(vocab_dict, file)