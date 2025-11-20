from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,AutoTokenizer
import torch
import torchaudio
from datasets import load_dataset, Audio
from peft import get_peft_model, LoraConfig,PeftModel
from collections import defaultdict
import json
from evaluate import load

wer_metric = load("wer")

def clean_phoneme_list(phoneme_list):
    """清理音素列表，移除同一元素内的空格"""
    cleaned_list = []
    
    for phoneme in phoneme_list:
        # 移除同一音素内的空格
        cleaned_phoneme = phoneme.replace(" ", "")
        cleaned_list.append(cleaned_phoneme)
    
    return cleaned_list

def create_meta_dataset_asr(lora_models, processor, raw_datasets, num_samples=1000):
    """
    为ASR任务创建元模型训练数据集
    返回: (all_logits_list, all_transcripts)
    """
    all_logits_list = []  # 存储每个样本的所有模型logits
    all_transcripts = []  # 存储真实转录文本
    
    # 选择子集用于训练元模型
    subset = raw_datasets.select(range(min(num_samples, len(raw_datasets))))
    
    with torch.no_grad():
        for i, sample in enumerate(subset):
            if i % 100 == 0:
                print(f"Processing sample {i}/{len(subset)}")
            
            # 预处理音频
            inputs = processor(
                sample["audio"]["array"],
                sampling_rate=sample["audio"]["sampling_rate"],
                return_tensors="pt",
                padding=True
            )
            
            sample_logits = []  # 这个样本的所有模型logits
            for model in lora_models:
                # 获取每个模型的logits输出 [1, seq_len, vocab_size]
                logits = model(inputs.input_values).logits
                sample_logits.append(logits)
            
            # 堆叠: [num_models, seq_len, vocab_size]
            stacked_logits = torch.stack(sample_logits, dim=0)
            all_logits_list.append(stacked_logits)
            all_transcripts.append(sample["sentence"])  # 真实文本
    
    return all_logits_list, all_transcripts

# 使用示例


model_path1 = "/mnt/storage/qisheng/github/wav2vec_test/weights/ru_9000-xlsr-300m_general/checkpoint-7000"
model1 = Wav2Vec2ForCTC.from_pretrained(model_path1)
adapter_path1 = "/mnt/storage/qisheng/github/wav2vec_test/weights/lora/ru2uk_100-xlsr-300m_general/checkpoint-8000"
tokenizer = AutoTokenizer.from_pretrained(adapter_path1,phonemizer_lang='uk')
model1 = PeftModel.from_pretrained(model1, adapter_path1)
processor = Wav2Vec2Processor.from_pretrained(adapter_path1)

model_path2 = "/mnt/storage/qisheng/github/wav2vec_test/weights/ro_9000-xlsr-300m_general/checkpoint-7000"
model2 = Wav2Vec2ForCTC.from_pretrained(model_path2)
adapter_path2 = "/mnt/storage/qisheng/github/wav2vec_test/weights/lora/ro2uk_100-xlsr-300m_general/checkpoint-8000"
model2 = PeftModel.from_pretrained(model2, adapter_path2)

lora_models = []
lora_models.append(model1)
lora_models.append(model2)


# 2. 加载评估数据集
print("加载数据集中...")
raw_datasets = load_dataset(
    "fixie-ai/common_voice_17_0",
    'uk',
    split='test',
    trust_remote_code=True,
    cache_dir="/mnt/storage/ldl_linguistics/datasets",
)
raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16000))


all_logits, all_transcripts = create_meta_dataset_asr(lora_models, processor, raw_datasets, num_samples=10)

print(all_logits)
print(all_transcripts)


