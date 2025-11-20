# main.py



from peft import LoraConfig,TaskType
import importlib.util
import argparse
import sys
from dataset import load_datasets, preprocess_datasets,vectorize_datasets
from model import load_model_and_tokenizer,load_tokenzier
from lora_trainer import save_lora_adapter,load_lora_adapter,create_lora_trainer,resize_wav2vec2_ctc_vocab,get_new_tokens,MyWav2Vec2ForCTC
from util import create_vocab
from transformers import TrainingArguments, Trainer
from trainer import create_trainer
import logging
from transformers.trainer_utils import is_main_process
import transformers
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    Wav2Vec2PhonemeCTCTokenizer,
    set_seed,
)
import os 
import evaluate
import wandb
from phonemizer.separator import Separator
from phonemizer.backend import BACKENDS
import datasets
import re
from datasets import load_dataset
from accelerate import Accelerator

logger = logging.getLogger(__name__)
logging.getLogger('phonemizer').disabled = True
def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module



def main(config_path):
    os.environ["NCCL_P2P_DISABLE"] = "1"
    #os.environ["NCCL_SHM_DISABLE"] = "1"
    accelerator = Accelerator()

    config = load_config(config_path)

    resume = getattr(config, 'resume', False)
    reinit = getattr(config, 'reinit', False)
    resize = getattr(config, 'resize', False)
    general = getattr(config, 'general', False)
    resume_dir = getattr(config, 'resume_dir', None)
    adapter_checkpoint = getattr(config, 'adapter_checkpoint', None)
    random = getattr(config, 'random', None)
    sample_limit_list = config.DATASET_PARAMS.get('max_train_samples_per_language', [])
    eval_limit_list = config.DATASET_PARAMS.get('max_eval_samples_per_language', [])

    training_args = TrainingArguments(**config.TRAINING_PARAMS)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_process_index) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    
    if training_args.local_rank in [-1, 0]:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run,
        )
    
    lora_config = LoraConfig(
        **config.LORA_PARAMS
    )


    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    lan_config = config.DATASET_PARAMS["dataset_config_name"]
    model = MyWav2Vec2ForCTC.from_pretrained(config.checkpoint)
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.checkpoint)
    processor = AutoProcessor.from_pretrained(config.checkpoint)
    if general:
        print("general")
        tokenizer = AutoTokenizer.from_pretrained(config.checkpoint,phonemizer_lang=lan_config)
        model_config = AutoConfig.from_pretrained(config.checkpoint)
        processor.tokenizer = tokenizer

    if reinit:
        print("reinit")
        tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                config.new_vocab_path,
                unk_token="[UNK]",
                pad_token="[PAD]",
                phonemizer_lang=lan_config,
            )
        model.reinit_token_embeddings(len(tokenizer))    
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model_config = model.config
        processor.tokenizer = tokenizer
    if resize:
        print("resize")
        tokenizer = AutoTokenizer.from_pretrained(config.checkpoint,phonemizer_lang=lan_config)
        model_config = AutoConfig.from_pretrained(config.checkpoint)
        new_tokens_list = get_new_tokens(config.new_vocab_path,tokenizer.get_vocab())
        num_added = tokenizer.add_tokens(new_tokens_list)
        new_vocab_size = len(tokenizer)  # 获取新的词汇表大小
        print(f"添加了 {num_added} 个新token")
        #print(f"新词汇表大小: {new_vocab_size}")
        #resize_linear_layer(model.lm_head, model.config.vocab_size, new_vocab_size)
        model.resize_token_embeddings(new_vocab_size)
        processor.tokenizer = tokenizer
    #print(model.config.vocab_size)
    #print(model.lm_head.out_features)
    
    #print(model_config)
    if isinstance(lan_config, str):
        if len(sample_limit_list) == 0:
            sample_limit_list.append(None)
        if len(eval_limit_list) == 0:
            eval_limit_list.append(None)
        #tokenizer = load_tokenzier(lan_config,**config.MODEL_PARAMS)
        raw_datasets = load_datasets(lan_config,max_train_sample=sample_limit_list[0],max_eval_sample=eval_limit_list[0],random=random,**config.DATASET_PARAMS)
        raw_datasets = preprocess_datasets(raw_datasets,**config.DATASET_PARAMS)
        with training_args.main_process_first(desc="dataset map preprocessing"):
            vectorized_datasets = vectorize_datasets(raw_datasets,tokenizer,feature_extractor,**config.DATASET_PARAMS)
            vectorized_datasets["train"] = vectorized_datasets["train"].add_column("language", [lan_config] * len(vectorized_datasets["train"]))
            vectorized_datasets["eval"] = vectorized_datasets["eval"].add_column("language", [lan_config] * len(vectorized_datasets["eval"]))
    else:
        all_train_datasets = []
        all_eval_datasets = []
        print(sample_limit_list)
        if len(sample_limit_list) == 0:
            for idx in range(len(lan_config)):
                sample_limit_list.append(None)
        for idx in range(len(lan_config)):
            print(lan_config[idx])
            #tokenizer = load_tokenzier(lan_config[idx],**config.MODEL_PARAMS)
            raw_datasets = load_datasets(lan_config[idx],max_train_sample=sample_limit_list[idx],**config.DATASET_PARAMS)
            raw_datasets = preprocess_datasets(raw_datasets,**config.DATASET_PARAMS)
            with training_args.main_process_first(desc="dataset map preprocessing"):
                vectorized_datasets = vectorize_datasets(raw_datasets,tokenizer,feature_extractor,**config.DATASET_PARAMS)
                vectorized_datasets["train"] = vectorized_datasets["train"].add_column("language", [lan_config[idx]] * len(vectorized_datasets["train"]))
                vectorized_datasets["eval"] = vectorized_datasets["eval"].add_column("language", [lan_config[idx]] * len(vectorized_datasets["eval"]))
            
            # 把每次得到的vectorized_datasets合并到一起
            all_train_datasets.append(vectorized_datasets["train"])
            all_eval_datasets.append(vectorized_datasets["eval"])
        
        # 合并所有语言的数据集
        from datasets import concatenate_datasets, DatasetDict
        vectorized_datasets = DatasetDict({
            "train": concatenate_datasets(all_train_datasets).shuffle(seed=42),
            "eval": concatenate_datasets(all_eval_datasets)
        })
    


    

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with ``args.preprocessing_only`` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step ``args.preprocessing_only`` can then be set to `False` to load the
    # cached dataset
    if config.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")
        return

    
    

    

        # Now save everything to be able to create a single processor later
        # make sure all processes wait until data is saved
    if not resume and training_args.local_rank in [-1, 0]:
    #with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_process_index):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            model_config.save_pretrained(training_args.output_dir)
    
    trainer = create_lora_trainer(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=processor.feature_extractor,
        dataset=vectorized_datasets,
        training_args=training_args,
        eval_metrics=config.DATASET_PARAMS["eval_metrics"],
        processor=processor,
        lora_config=lora_config,
        adapter_checkpoint=adapter_checkpoint,
    )

    # Training
    if training_args.do_train:
        # use last checkpoint if exist   
        

        #if adapter_checkpoint != None:


        #    logger.info(f"resume from {adapter_checkpoint}")
        #    train_result = trainer.train(resume_from_checkpoint=adapter_checkpoint)
        #else:

        logger.info(f"train from beginning")
        train_result = trainer.train(resume_from_checkpoint=None)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            config.DATASET_PARAMS["max_train_samples"]
            if config.DATASET_PARAMS["max_train_samples"] is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        #trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if training_args.do_train == False:
            logger.info(f"resume from {config.checkpoint}")
            model = Wav2Vec2ForCTC.from_pretrained(config.checkpoint)
            processor = Wav2Vec2Processor.from_pretrained(config.checkpoint)
            trainer = create_trainer(model, tokenizer, feature_extractor, vectorized_datasets, training_args, config.DATASET_PARAMS["eval_metrics"], processor)
        metrics = trainer.evaluate()
        max_eval_samples = (
            config.DATASET_PARAMS["max_eval_samples"] if config.DATASET_PARAMS["max_eval_samples"] is not None else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    save_lora_adapter(trainer, training_args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    main(args.config)