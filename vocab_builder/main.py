from dataset import load_datasets, preprocess_datasets,vectorize_datasets
from build_vocab import create_vocab
import importlib.util
import argparse


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def main(config_path):
    config = load_config(config_path)
    
    raw_datasets = load_datasets(**config.DATASET_PARAMS)
    raw_datasets = preprocess_datasets(raw_datasets,**config.DATASET_PARAMS)
    create_vocab(raw_datasets,**config.DATASET_PARAMS)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    main(args.config)