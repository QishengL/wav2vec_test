from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from datasets import load_dataset, Audio
import torch
from collections import Counter

#model_path = "/mnt/storage/qisheng/github/wav2vec_test/weights/classification/xlsr-300m_train_50/checkpoint-56000"
model_path = "/mnt/storage/qisheng/github/wav2vec_test/weights/classification/xlsr-300m_class1/checkpoint-83000"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
model.eval()

# 加载数据集
print("加载数据集中...")
'''
raw_datasets = load_dataset(
    "fixie-ai/common_voice_17_0",
    'ur',
    split='train',
    trust_remote_code=True,
    cache_dir="/mnt/storage/ldl_linguistics/datasets",
)
'''
raw_datasets = load_dataset(
    "fsicoli/common_voice_22_0",
    'kk',
    split='train',
    trust_remote_code=True,
    cache_dir="/mnt/storage/ldl_linguistics/datasets",
)


raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16000))

# 初始化计数器
top1_counter = Counter()
top5_counter = Counter()

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
        outputs = model(**inputs)
    
    # 获取logits并应用softmax
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # 获取前5个最高概率和对应的索引
    top5_probs, top5_indices = torch.topk(probabilities, 5, dim=-1)
    
    # 转换为numpy
    top5_probs = top5_probs.cpu().numpy()[0]
    top5_indices = top5_indices.cpu().numpy()[0]
    
    # 获取top1预测
    top1_pred = top5_indices[0]
    
    # 更新计数器
    top1_counter[top1_pred] += 1
    for idx in top5_indices:
        top5_counter[idx] += 1
    
    # 打印当前样本结果
    #print(f"\n样本 {i+1} Top 5 预测结果:")
    #for j, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
    #    print(f"  {idx}: {prob:.4f}")

# 统计结果
print("\n" + "="*50)
print("统计结果 (100个样本)")
print("="*50)

# Top1 统计
print("\nTop1 预测统计:")
top1_most_common = top1_counter.most_common()
for idx, count in top1_most_common:
    print(f"类别 {idx}: {count}次 ({count}%)")

# Top5 统计
print("\nTop5 预测统计:")
top5_most_common = top5_counter.most_common()
for idx, count in top5_most_common:
    print(f"类别 {idx}: {count}次 ({count}%)")

# 计算最高次数
max_top1_count = max(top1_counter.values()) if top1_counter else 0
max_top5_count = max(top5_counter.values()) if top5_counter else 0

print(f"\n最高出现次数统计:")
print(f"Top1 最高出现次数: {max_top1_count}次")
print(f"Top5 最高出现次数: {max_top5_count}次")

# 找出出现次数最多的类别
if top1_counter:
    most_common_top1 = top1_counter.most_common(1)[0]
    print(f"\n最常被预测为Top1的类别: {most_common_top1[0]} (出现{most_common_top1[1]}次)")

if top5_counter:
    most_common_top5 = top5_counter.most_common(1)[0]
    print(f"最常出现在Top5的类别: {most_common_top5[0]} (出现{most_common_top5[1]}次)")

# 计算类别覆盖率
total_top1_predictions = sum(top1_counter.values())
total_top5_predictions = sum(top5_counter.values())
unique_top1_classes = len(top1_counter)
unique_top5_classes = len(top5_counter)

print(f"\n类别覆盖率:")
print(f"Top1 覆盖类别数: {unique_top1_classes}")
print(f"Top5 覆盖类别数: {unique_top5_classes}")