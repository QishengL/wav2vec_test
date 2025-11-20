import importlib.util
import argparse
import sys
from dataset import load_datasets, preprocess_datasets,vectorize_datasets,vectorize_datasets_classification,load_and_combine_multilingual_datasets
from model import load_model_for_classification
from trainer import create_trainer
from peft import LoraConfig
from trainer import create_trainer_for_classification
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
    random = getattr(config, 'random', None)
    espeak_config = getattr(config, 'espeak_config', None)
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
    


    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # 加载模型和特征提取器（不再需要tokenizer）
    feature_extractor, model, model_config = load_model_for_classification(training_args, **config.MODEL_PARAMS)
    languages = lan_dict = {code: idx for idx, code in enumerate(config.lan_list)}
    # 加载多语言数据集
    raw_datasets = load_and_combine_multilingual_datasets(
        languages=languages,
        max_train_sample=config.train_samples,  # 每种语言250条训练数据
        max_eval_sample=config.eval_samples,    # 每种语言50条验证数据
        random=training_args.seed,
        **config.DATASET_PARAMS
    )

    # 预处理数据集（如果需要的话）
    #raw_datasets = preprocess_datasets(raw_datasets, **config.DATASET_PARAMS)

    # 向量化数据集 - 不再需要tokenizer
    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = vectorize_datasets_classification(
            raw_datasets, 
            tokenizer=None,  # 传入None或者删除这个参数
            feature_extractor=feature_extractor, 
            **config.DATASET_PARAMS
        )

    
    

    
        # Now save everything to be able to create a single processor later
        # make sure all processes wait until data is saved
    if not resume and training_args.local_rank in [-1, 0]:
    #with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_process_index):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            model_config.save_pretrained(training_args.output_dir)
    
    trainer = create_trainer_for_classification(model, feature_extractor, vectorized_datasets, training_args, config.DATASET_PARAMS["eval_metrics"])
    # Training
    if training_args.do_train:
        # use last checkpoint if exist   
        
        if resume:
            if resume_dir == None:
                logger.info(f"resume from latest checkpoints in output file")
                train_result = trainer.train(resume_from_checkpoint=True)
            else:
                logger.info(f"resume from {resume_dir}")
                train_result = trainer.train(resume_from_checkpoint=resume_dir)
        else:
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
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if training_args.do_train == False:
            if resume_dir:
                logger.info(f"resume from {resume_dir}")
                model = Wav2Vec2ForCTC.from_pretrained(resume_dir)
                processor = Wav2Vec2Processor.from_pretrained(resume_dir)
                trainer = create_trainer_for_classification(model, feature_extractor, vectorized_datasets, training_args, config.DATASET_PARAMS["eval_metrics"])
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