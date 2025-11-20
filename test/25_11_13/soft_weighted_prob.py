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

for i in range(10):
    first_sample = raw_datasets[i]
    inputs = processor(
        first_sample["audio"]["array"],
        sampling_rate=first_sample["audio"]["sampling_rate"],
        return_tensors="pt",
        padding=True
    )

    additional_kwargs = {}
    additional_kwargs["phonemizer_lang"] = 'uk'
    gold_text = first_sample['sentence']
    gold_ids = processor.tokenizer(gold_text, **additional_kwargs).input_ids
    gold_decoded = processor.batch_decode(gold_ids)
    cleaned_gold = clean_phoneme_list(gold_decoded)
    gold_str = ' '.join(cleaned_gold)
    #print(gold_str)







    all_probs = []
    for idx in range(len(lora_models)):
        model = lora_models[idx]
        model.eval()

        with torch.no_grad():
            logits = model(inputs.input_values).logits
        #current pred
        predicted_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(predicted_ids)[0]
        wer = wer_metric.compute(predictions=[pred_str], references=[gold_str])
        print(f"WER_{idx}: {wer:.4f}")
        #add prob
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs)
    # 平均概率
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    final_prediction = torch.argmax(avg_probs, dim=-1)
    final_pred_str = processor.batch_decode(final_prediction)[0]
    wer = wer_metric.compute(predictions=[final_pred_str], references=[gold_str])
    print(f"WER_Final: {wer:.4f}")
