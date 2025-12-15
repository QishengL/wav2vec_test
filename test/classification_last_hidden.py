from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from datasets import load_dataset, Audio
import torch
from collections import Counter

#model_path = "/mnt/storage/qisheng/github/wav2vec_test/weights/classification/xlsr-300m_train_50/checkpoint-56000"
model_path = "/mnt/storage/qisheng/github/wav2vec_test/weights/classification/xlsr-300m_class1/checkpoint-83000"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
model.eval()
raw_datasets = load_dataset(
    "fsicoli/common_voice_22_0",
    'kk',
    split='train',
    trust_remote_code=True,
    cache_dir="/mnt/storage/ldl_linguistics/datasets",
)


raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16000))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"使用设备: {device}")

# 处理100个样本
for i in range(100):
    sample = raw_datasets[i]
    
    # 预处理音频
    inputs = feature_extractor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt",
        padding=True
    )
    
    # 将输入移动到模型所在的设备
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # 进行推理
    with torch.no_grad():
        outputs = model.wav2vec2(
                    inputs["input_values"],
                    attention_mask=inputs.get("attention_mask", None),
                    output_hidden_states=True
                )
                
                # 使用最后一个隐藏层的平均作为特征
        last_hidden = outputs.last_hidden_state
        time_averaged = last_hidden.mean(dim=1)
        features_norm = torch.nn.functional.normalize(time_averaged, p=2, dim=1)
        features_norm = features_norm.cpu().numpy().flatten()
    print(features_norm.shape)
    print(features_norm)
    break