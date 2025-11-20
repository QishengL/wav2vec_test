# model.py
from transformers import AutoConfig, AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC, Wav2Vec2PhonemeCTCTokenizer,Wav2Vec2ForSequenceClassification
from phonemizer.backend import BACKENDS

def load_tokenzier(phonemizer_lang,**model_config):
    use_phoneme = model_config.get("use_phoneme", False)
    word_delimiter_token = model_config.get("word_delimiter_token", " ")
    unk_token = model_config.get("unk_token", "[UNK]")
    pad_token = model_config.get("pad_token", "[PAD]")
    output_dir = model_config.get("output_dir", "./output")
    model_name = model_config.get("model_name_or_path", "facebook/wav2vec2-base")
    vocab_dir = model_config.get("vocab_dir", output_dir)
    config = AutoConfig.from_pretrained(
        model_name,
        #cache_dir=model_args.cache_dir,
        #token=None,
        trust_remote_code=True,
    )
    

    tokenizer_kwargs = {
        "config": config if config.tokenizer_class is not None else None,
        "tokenizer_type": (config.model_type if config.tokenizer_class is None else None),
        "unk_token": unk_token,
        "pad_token": pad_token,
        "word_delimiter_token": word_delimiter_token,
    }

    tokenizer_name_or_path = vocab_dir
    #print(tokenizer_name_or_path)
    if use_phoneme:
        print("phonewav!")
        tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                tokenizer_name_or_path,
                unk_token=unk_token,
                pad_token=pad_token,
                phonemizer_lang=phonemizer_lang,
            )
        #backend = BACKENDS["espeak"](phoneme_lang, language_switch="remove-flags")
        #tokenizer.backend = backend
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            #token=data_args.token,
            #trust_remote_code=True,
            **tokenizer_kwargs,
        )
    return tokenizer    

def load_model_and_tokenizer(training_args,**model_config):
    # load config
    output_dir = model_config.get("output_dir", "./output")
    model_name = model_config.get("model_name_or_path", "facebook/wav2vec2-base")
    word_delimiter_token = model_config.get("word_delimiter_token", " ")
    unk_token = model_config.get("unk_token", "[UNK]")
    pad_token = model_config.get("pad_token", "[PAD]")
    feat_proj_dropout = model_config.get("feat_proj_dropout", 0.0)
    attention_dropout = model_config.get("attention_dropout", 0.1)
    hidden_dropout = model_config.get("hidden_dropout", 0.1)
    final_dropout = model_config.get("final_dropout", 0.1)
    mask_time_prob = model_config.get("mask_time_prob", 0.05)
    mask_time_length = model_config.get("mask_time_length", 10)
    mask_feature_prob = model_config.get("mask_feature_prob", 0.0)
    mask_feature_length = model_config.get("mask_feature_length", 10)
    layerdrop = model_config.get("layerdrop", 0.1)
    ctc_loss_reduction = model_config.get("ctc_loss_reduction", "mean")
    ctc_zero_infinity = model_config.get("ctc_zero_infinity", False)
    activation_dropout = model_config.get("activation_dropout", 0.1)
    add_adapter = model_config.get("add_adapter", False)
    freeze_feature_encoder = model_config.get("freeze_feature_encoder", False)
    use_phoneme = model_config.get("use_phoneme", False)
    phoneme_lang = model_config.get("phoneme_lang", "en-us")
    vocab_dir = model_config.get("vocab_dir", output_dir)

    config = AutoConfig.from_pretrained(
        model_name,
        #cache_dir=model_args.cache_dir,
        #token=None,
        trust_remote_code=True,
    )
    

    tokenizer_kwargs = {
        "config": config if config.tokenizer_class is not None else None,
        "tokenizer_type": (config.model_type if config.tokenizer_class is None else None),
        "unk_token": unk_token,
        "pad_token": pad_token,
        "word_delimiter_token": word_delimiter_token,
    }

    tokenizer_name_or_path = vocab_dir
    #print(tokenizer_name_or_path)
    if use_phoneme:
        print("phonewav!")
        tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                tokenizer_name_or_path,
                unk_token=unk_token,
                pad_token=pad_token,
                #phonemizer_lang='uk',
            )
        #backend = BACKENDS["espeak"](phoneme_lang, language_switch="remove-flags")
        #tokenizer.backend = backend
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            #token=data_args.token,
            #trust_remote_code=True,
            **tokenizer_kwargs,
        )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name,
        #cache_dir=model_args.cache_dir,
        #token=data_args.token,
        trust_remote_code=True,
    )


    config.update(
        {
            "feat_proj_dropout": feat_proj_dropout,
            "attention_dropout": attention_dropout,
            "hidden_dropout": hidden_dropout,
            "final_dropout": final_dropout,
            "mask_time_prob": mask_time_prob,
            "mask_time_length": mask_time_length,
            "mask_feature_prob": mask_feature_prob,
            "mask_feature_length": mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": layerdrop,
            "ctc_loss_reduction": ctc_loss_reduction,
            "ctc_zero_infinity": ctc_zero_infinity,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "vocab_size": len(tokenizer),
            "activation_dropout": activation_dropout,
            "add_adapter": add_adapter,
        }
    )

    # create model
    model = AutoModelForCTC.from_pretrained(
        model_name,
        #cache_dir=model_args.cache_dir,
        config=config,
        #token=data_args.token,
        trust_remote_code=True,
    )

    # freeze encoder
    if freeze_feature_encoder:
        model.freeze_feature_encoder()
    return tokenizer, feature_extractor, model, config

def load_model_for_classification(training_args,**model_config):
    # load config
    output_dir = model_config.get("output_dir", "./output")
    model_name = model_config.get("model_name_or_path", "facebook/wav2vec2-base")

    # 保留通用的 dropout 配置
    feat_proj_dropout = model_config.get("feat_proj_dropout", 0.0)
    attention_dropout = model_config.get("attention_dropout", 0.1)
    hidden_dropout = model_config.get("hidden_dropout", 0.1)
    final_dropout = model_config.get("final_dropout", 0.1)
    activation_dropout = model_config.get("activation_dropout", 0.1)

    # 保留 mask 相关配置（对于预训练模型可能还有用）
    mask_time_prob = model_config.get("mask_time_prob", 0.05)
    mask_time_length = model_config.get("mask_time_length", 10)
    mask_feature_prob = model_config.get("mask_feature_prob", 0.0)
    mask_feature_length = model_config.get("mask_feature_length", 10)

    layerdrop = model_config.get("layerdrop", 0.1)
    add_adapter = model_config.get("add_adapter", False)
    freeze_feature_encoder = model_config.get("freeze_feature_encoder", False)

    # 分类任务特定配置
    num_labels = model_config.get("num_labels", 50)  # 添加分类类别数
    problem_type = model_config.get("problem_type", "single_label_classification")  # 问题类型

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        num_labels=num_labels,  # 设置分类类别数
        problem_type=problem_type,  # 设置问题类型
    )

    # 对于分类模型，不再需要 tokenizer
    tokenizer = None

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # 更新配置 - 只保留分类模型需要的配置
    config.update(
        {
            "feat_proj_dropout": feat_proj_dropout,
            "attention_dropout": attention_dropout,
            "hidden_dropout": hidden_dropout,
            "final_dropout": final_dropout,
            "mask_time_prob": mask_time_prob,
            "mask_time_length": mask_time_length,
            "mask_feature_prob": mask_feature_prob,
            "mask_feature_length": mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": layerdrop,
            "activation_dropout": activation_dropout,
            "add_adapter": add_adapter,
            # 删除了所有 CTC 相关的配置
            # 删除了所有 tokenizer 相关的配置
        }
    )

    # 创建分类模型
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
    )

    # freeze encoder (这个可以保留)
    if freeze_feature_encoder:
        model.freeze_feature_encoder()

    return feature_extractor, model, config