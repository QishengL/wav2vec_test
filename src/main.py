# main.py
import importlib.util
import argparse
import sys
from dataset import load_datasets, preprocess_datasets,vectorize_datasets
from model import load_model_and_tokenizer
from trainer import create_trainer
from util import create_vocab
from transformers import TrainingArguments, Trainer
import logging
from transformers.trainer_utils import is_main_process
import transformers
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

logger = logging.getLogger(__name__)

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module

def main(config_path):
    config = load_config(config_path)

    
    
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
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_process_index):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)
    

    wandb.init(
        project=config.wandb_project,  # 设置 project name
        name=config.wandb_run,     # 设置 run name
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    
    
    raw_datasets = load_datasets(**config.DATASET_PARAMS)
    
    
    
    with training_args.main_process_first(desc="dataset map special characters removal"):
        raw_datasets = preprocess_datasets(raw_datasets,**config.DATASET_PARAMS)
    #print(raw_datasets)

    if config.create_vocab:
        with training_args.main_process_first():
            create_vocab(raw_datasets,**config.MODEL_PARAMS)


    tokenizer, feature_extractor, model, model_config = load_model_and_tokenizer(training_args,**config.MODEL_PARAMS)

    #print(tokenizer)
    #print(feature_extractor)
    #print(model)
    
    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = vectorize_datasets(raw_datasets,tokenizer,feature_extractor,**config.DATASET_PARAMS)

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
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_process_index):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            model_config.save_pretrained(training_args.output_dir)
    
    trainer = create_trainer(model, tokenizer, feature_extractor, vectorized_datasets, training_args, config.DATASET_PARAMS["eval_metrics"])
    
    # Training
    if training_args.do_train:
        # use last checkpoint if exist   
        if os.path.isdir(os.path.join(training_args.output_dir, "checkpoints")):
            checkpoint = os.path.join(training_args.output_dir, "checkpoints")
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
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
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            config.DATASET_PARAMS["max_eval_samples"] if config.DATASET_PARAMS["max_eval_samples"] is not None else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    main(args.config)