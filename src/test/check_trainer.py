# main.py




import importlib.util
import argparse
import sys
from dataset import load_datasets, preprocess_datasets,vectorize_datasets
from model import load_model_and_tokenizer,load_tokenzier
from trainer import create_trainer
from util import create_vocab
from transformers import TrainingArguments, Trainer
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
    resume_dir = getattr(config, 'resume_dir', None)
    sample_limit_list = config.DATASET_PARAMS.get('max_train_samples_per_language', [])
    
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
    


    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    lan_config = config.DATASET_PARAMS["dataset_config_name"]

    tokenizer, feature_extractor, model, model_config = load_model_and_tokenizer(training_args,**config.MODEL_PARAMS)
    if isinstance(lan_config, str):
        if len(sample_limit_list) == 0:
            sample_limit_list.append(None)
        tokenizer = load_tokenzier(lan_config,**config.MODEL_PARAMS)
        raw_datasets = load_datasets(lan_config,max_train_sample=sample_limit_list[0],**config.DATASET_PARAMS)
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
            tokenizer = load_tokenzier(lan_config[idx],**config.MODEL_PARAMS)
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
    

    
    
    #print(tokenizer)
    #print(feature_extractor)
    #print(model)
    
    
    #return
    #print(vectorized_datasets)
    

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
    
    trainer = create_trainer(model, tokenizer, feature_extractor, vectorized_datasets, training_args, config.DATASET_PARAMS["eval_metrics"])
    # Training
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    main(args.config)