from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
from datasets import load_dataset, Audio
from phonemizer import phonemize
from phonemizer.separator import Separator
from phonemizer.backend import BACKENDS
def clean_phoneme_list(phoneme_list):
    """清理音素列表，移除同一元素内的空格"""
    cleaned_list = []
    
    for phoneme in phoneme_list:
        # 移除同一音素内的空格
        cleaned_phoneme = phoneme.replace(" ", "")
        cleaned_list.append(cleaned_phoneme)
    
    return cleaned_list


# 加载模型和processor
model_path = "../weights/tk_100-xlsr-uzbase/checkpoint-2400"
model = Wav2Vec2ForCTC.from_pretrained(model_path)
processor = Wav2Vec2Processor.from_pretrained(model_path)

# 将模型设置为评估模式
model.eval()
# "fsicoli/common_voice_22_0",
#"fixie-ai/common_voice_17_0",
# 2. 加载评估数据集
print("加载数据集中...")
raw_datasets = load_dataset(
    "fsicoli/common_voice_22_0",
    'tk',
    split='test',
    trust_remote_code=True,
    cache_dir="/mnt/storage/ldl_linguistics/datasets",
)

# 确保音频列是Audio类型
raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16000))

# 3. 获取第一个样本
first_sample = raw_datasets[0]
print("\n" + "="*50)
print("第一个样本的详细信息:")
print("="*50)

additional_kwargs = {}
additional_kwargs["phonemizer_lang"] = 'tk'
gold_text = first_sample['sentence']
gold_ids = processor.tokenizer(gold_text, **additional_kwargs).input_ids

gold_decoded = processor.batch_decode(gold_ids)
# 4. 打印gold label和信息
print(f"📝 原始Gold Label: {gold_text}")
print(f"🔤 Tokenizer处理后Gold: {gold_decoded}")
print(f"🎵 音频路径: {first_sample['audio']['path']}")
print(f"📊 采样率: {first_sample['audio']['sampling_rate']}")
print(f"⏱️  音频长度: {len(first_sample['audio']['array'])} 采样点")
print(f"⏱️  音频时长: {len(first_sample['audio']['array']) / first_sample['audio']['sampling_rate']:.2f} 秒")


#backend = BACKENDS["espeak"]("ba", language_switch="remove-flags")
#separator = Separator(phone=' ', word="", syllable="")

#phonemes = backend.phonemize(
#    [gold_text],
#    separator=separator,
#)
#phonemized_text = phonemes[0].strip()
#print(phonemized_text)

# 预处理音频
inputs = processor(
    first_sample["audio"]["array"],
    sampling_rate=first_sample["audio"]["sampling_rate"],
    return_tensors="pt",
    padding=True
)
with torch.no_grad():
    logits = model(inputs.input_values).logits

 # 解码预测结果

    predicted_ids = torch.argmax(logits, dim=-1)
    pred_str = processor.batch_decode(predicted_ids)[0]
    #prediction = processor.batch_decode(predicted_ids)[0]
    #cleaned_gold = clean_phoneme_list(gold_decoded)
    #gold_str = ' '.join(cleaned_gold)
    print(f"🤖 模型预测: {pred_str}")

def clean_phoneme_list(phoneme_list):
    """清理音素列表，移除同一元素内的空格"""
    cleaned_list = []
    
    for phoneme in phoneme_list:
        # 移除同一音素内的空格
        cleaned_phoneme = phoneme.replace(" ", "")
        cleaned_list.append(cleaned_phoneme)
    
    return cleaned_list

cleaned_gold = clean_phoneme_list(gold_decoded)
gold_str = ' '.join(cleaned_gold)

print(f"✅ Gold Label: {gold_str}")
from evaluate import load
predictions = []
references = []
wer_metric = load("wer")
references.append(gold_str)
predictions.append(pred_str)
wer = wer_metric.compute(predictions=predictions, references=references)
print(f"最终WER: {wer:.4f}")