# model.py
from transformers import AutoConfig, AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC

    

def load_model_and_tokenizer(training_args,**model_config):
    # 载入配置
    model_name =  model_config["model_name_or_path"]

    config = AutoConfig.from_pretrained(
        model_name,
        #cache_dir=model_args.cache_dir,
        #token=None,
        trust_remote_code=True,
    )
    
    word_delimiter_token = model_config["word_delimiter_token"]
    unk_token = model_config["unk_token"]
    pad_token = model_config["pad_token"]
    tokenizer_kwargs = {
        "config": config if config.tokenizer_class is not None else None,
        "tokenizer_type": (config.model_type if config.tokenizer_class is None else None),
        "unk_token": unk_token,
        "pad_token": pad_token,
        "word_delimiter_token": word_delimiter_token,
    }
    tokenizer_name_or_path = model_config["output_dir"]

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
            "feat_proj_dropout": model_config["feat_proj_dropout"],
            "attention_dropout": model_config["attention_dropout"],
            "hidden_dropout": model_config["hidden_dropout"],
            "final_dropout": model_config["final_dropout"],
            "mask_time_prob": model_config["mask_time_prob"],
            "mask_time_length": model_config["mask_time_length"],
            "mask_feature_prob": model_config["mask_feature_prob"],
            "mask_feature_length": model_config["mask_feature_length"],
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": model_config["layerdrop"],
            "ctc_loss_reduction": model_config["ctc_loss_reduction"],
            "ctc_zero_infinity": model_config["ctc_zero_infinity"],
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
            "activation_dropout": model_config["activation_dropout"],
            "add_adapter": model_config["add_adapter"],
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
    if model_config["freeze_feature_encoder"]:
        model.freeze_feature_encoder()
    return tokenizer, feature_extractor, model, config
