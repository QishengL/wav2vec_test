from dataset import load_datasets, preprocess_datasets,vectorize_datasets,vectorize_datasets_classification,load_and_combine_multilingual_datasets
from model import load_model_for_classification
import importlib.util
import argparse
import logging
import torch
from trainer import create_trainer_for_classification
from transformers import TrainingArguments
logger = logging.getLogger(__name__)
logging.getLogger('phonemizer').disabled = True
def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module



def main(config_path):
    config = load_config(config_path)
    languages = {
        'ro': 0,  # 英语 -> 标签0
        'sl': 1,  # 法语 -> 标签2
        'sr': 2,  # 德语 -> 标签4
        # 添加你需要的其他语言...
        }
    training_args = TrainingArguments(**config.TRAINING_PARAMS)
# 加载多语言数据集
    feature_extractor, model, model_config = load_model_for_classification(training_args, **config.MODEL_PARAMS)
    
    for name, param in model.named_parameters():
        param.requires_grad = True
        print(f"{name}: requires_grad = {param.requires_grad}")
    
    raw_datasets = load_and_combine_multilingual_datasets(
        languages=languages,
        max_train_sample=50,  # 每种语言250条训练数据
        max_eval_sample=50,    # 每种语言50条验证数据
        **config.DATASET_PARAMS
    )

    # 预处理数据集（如果需要的话）
    #raw_datasets = preprocess_datasets(raw_datasets, **config.DATASET_PARAMS)

    # 向量化数据集 - 不再需要tokenizer

    vectorized_datasets = vectorize_datasets_classification(
            raw_datasets, 
            tokenizer=None,  # 传入None或者删除这个参数
            feature_extractor=feature_extractor, 
            **config.DATASET_PARAMS
        )
    
    

    trainer = create_trainer_for_classification(model, feature_extractor, vectorized_datasets, training_args, config.DATASET_PARAMS["eval_metrics"])
    trainer.train(resume_from_checkpoint=None)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    main(args.config)