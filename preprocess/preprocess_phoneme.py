# preprocess_phonemize.py
import os
import datasets
from phonemizer import phonemize
from phonemizer.separator import Separator
from phonemizer.backend import BACKENDS
from tqdm import tqdm
import time

def preprocess_and_save_phonemized_data(config):
    """
    预处理并保存音素化后的数据
    
    Args:
        config: 包含以下键的字典
            - dataset_name: 数据集名称 (如 "fixie-ai/common_voice_17_0")
            - dataset_config_name: 语言配置列表 (如 ["en", "uk", "ru", "tr"])
            - train_split: 训练集分割名称
            - test_split: 验证集分割名称  
            - text_column: 文本列名 (如 "sentence")
            - cache_dir: 缓存目录
            - language_phoneme_map: 语言到音素后端的映射
    """
    
    dataset_configs = config["dataset_config_name"]
    if isinstance(dataset_configs, str):
        dataset_configs = [dataset_configs]
    
    language_phoneme_map = config.get("language_phoneme_map", {})
    
    print(f"🚀 开始音素化预处理...")
    print(f"数据集: {config['dataset_name']}")
    print(f"语言: {dataset_configs}")
    print(f"缓存目录: {config['cache_dir']}")
    
    for config_name in dataset_configs:
        print(f"\n{'='*50}")
        print(f"处理语言: {config_name}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        try:
            # 加载原始数据
            print(f"📥 加载原始数据...")
            train_ds = datasets.load_dataset(
                config["dataset_name"],
                config_name,
                split=config["train_split"],
                trust_remote_code=True,
                cache_dir=config["cache_dir"],
            )
            
            
            print(f"✅ 加载完成: {len(train_ds)} 样本")
            
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            continue
        
        # 音素化处理函数
        def phonemize_text(text, language):
            """音素化单个文本"""
            try:
                phoneme_lang = language_phoneme_map.get(language, language)
                backend = BACKENDS["espeak"](phoneme_lang, language_switch="remove-flags")
                separator = Separator(phone=' ', word="", syllable="")
                
                phonemes = backend.phonemize([text], separator=separator)
                return phonemes[0].strip()
            except Exception as e:
                print(f"⚠️ 音素化失败 {language}: '{text[:50]}...', 错误: {e}")
                return text  # 失败时返回原文本
        
        def process_dataset(dataset, split_name, language):
            """处理整个数据集"""
            phonemized_texts = []
            failed_count = 0
            
            print(f"🔤 音素化 {split_name} 数据...")
            for i in tqdm(range(len(dataset)), desc=f"Phonemizing {language} {split_name}"):
                text = dataset[i][config["text_column"]]
                phonemized_text = phonemize_text(text, language)
                
                # 检查是否音素化失败（返回了原文本）
                if phonemized_text == text:
                    failed_count += 1
                
                phonemized_texts.append(phonemized_text)
            
            # 添加音素化后的文本列
            dataset = dataset.add_column("phonemized_text", phonemized_texts)
            
            if failed_count > 0:
                print(f"⚠️ {split_name} 有 {failed_count} 个样本音素化失败，使用原文本")
            
            return dataset
        
        # 处理训练集和验证集
        try:
            train_ds_phonemized = process_dataset(train_ds, config["train_split"], config_name)
        except Exception as e:
            print(f"❌ 音素化处理失败: {e}")
            continue
        
        # 保存到文件
        output_dir = os.path.join(config["cache_dir"], "phonemized", config_name)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            print(f"💾 保存音素化数据到: {output_dir}")
            train_ds_phonemized.save_to_disk(os.path.join(output_dir, config["train_split"]))
            
            # 保存配置信息
            config_info = {
                "dataset_name": config["dataset_name"],
                "language": config_name,
                "phonemized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "train_samples": len(train_ds_phonemized),
                "language_phoneme_map": language_phoneme_map.get(config_name, config_name)
            }
            
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                import json
                json.dump(config_info, f, indent=2)
            
            elapsed_time = time.time() - start_time
            print(f"✅ 完成! 耗时: {elapsed_time:.2f}秒")
            print(f"📊 训练集: {len(train_ds_phonemized)} 样本")
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    print(f"\n🎉 所有语言音素化预处理完成!")

def check_phonemized_data(config):
    """检查预音素化数据是否存在"""
    print(f"\n🔍 检查预音素化数据...")
    
    dataset_configs = config["dataset_config_name"]
    if isinstance(dataset_configs, str):
        dataset_configs = [dataset_configs]
    
    available_languages = []
    missing_languages = []
    
    for config_name in dataset_configs:
        phonemized_dir = os.path.join(config["cache_dir"], "phonemized", config_name)
        train_path = os.path.join(phonemized_dir, "train")

        
        if os.path.exists(train_path) and os.path.exists(eval_path):
            try:
                train_ds = datasets.load_from_disk(train_path)
                available_languages.append(f"{config_name} ({len(train_ds)} train")
            except Exception as e:
                missing_languages.append(f"{config_name} (加载失败: {e})")
        else:
            missing_languages.append(config_name)
    
    if available_languages:
        print("✅ 可用的预音素化数据:")
        for lang in available_languages:
            print(f"   - {lang}")
    
    if missing_languages:
        print("❌ 缺失的预音素化数据:")
        for lang in missing_languages:
            print(f"   - {lang}")
    
    return len(missing_languages) == 0

if __name__ == "__main__":
    # 配置参数 - 根据您的需求修改
    config = {
        "dataset_name": "fixie-ai/common_voice_17_0",
        "dataset_config_name": ["uk"],  # 您需要的语言
        "train_split": "test",
        "text_column": "sentence",  # 确认这是正确的文本列名
        "cache_dir": "/mnt/storage/ldl_linguistics/datasets",     # 您想要保存的目录
        "language_phoneme_map": {
            "en": "en-us",
            "uk": "uk", 
            "ru": "ru",
            "tr": "tr",    # 土耳其语
            # 添加其他语言...
        }
    }
    
    # 首先检查是否已经存在预音素化数据
    if check_phonemized_data(config):
        print("\n🎯 预音素化数据已存在，跳过处理")
        response = input("是否重新处理? (y/N): ")
        if response.lower() != 'y':
            exit(0)
    
    # 执行音素化预处理
    preprocess_and_save_phonemized_data(config)