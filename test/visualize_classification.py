import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, Audio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import os
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from datasets import load_dataset, Audio

def extract_embeddings_for_language(model, feature_extractor, dataset, lang_name, num_samples=300):
    """
    为特定语言提取嵌入特征（分类层之前的特征）
    
    Args:
        model: Wav2Vec2ForSequenceClassification 模型
        feature_extractor: 特征提取器
        dataset: 音频数据集
        lang_name: 语言名称
        num_samples: 要提取的样本数量
    """
    model.eval()
    embeddings = []
    
    # 限制样本数量
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
       
        # 预处理音频
        inputs = feature_extractor(
            sample["audio"]["array"],
            sampling_rate=sample["audio"]["sampling_rate"],
            return_tensors="pt",
            padding=True
        )
        
        # 移动到设备
        device = 'cuda'
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 获取分类层之前的特征
            # 方法1：获取Wav2Vec2的隐藏状态
            wav2vec_outputs = model.wav2vec2(
                inputs["input_values"],
                attention_mask=inputs.get("attention_mask", None),
                output_hidden_states=True
            )
            
            # 使用最后一个隐藏层的平均作为特征
            last_hidden = wav2vec_outputs.last_hidden_state  # [batch, seq_len, hidden]
            
            
            pooled_features = last_hidden.mean(dim=1)
            
            # 分类层之前的特征（可选项：也可以获取分类器的输入）
            # features = model.classifier[0](pooled_features)  # 如果分类器有多个层
            features = pooled_features
            
            # L2归一化
            features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
            
            embeddings.append(features_norm.cpu().numpy().flatten())
                
            print()
    
    return np.array(embeddings)

def save_simple_tsne(known_language_embeddings, output_path='tsne_plot.png'):
    """
    只保存一个简单的t-SNE图
    """
    # 合并数据
    all_embeddings = []
    all_labels = []
    
    for lang_name, embeddings in known_language_embeddings.items():
        all_embeddings.append(embeddings)
        all_labels.extend([lang_name] * embeddings.shape[0])
    
    X = np.vstack(all_embeddings)
    
    # 运行t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    # 创建图形
    plt.figure(figsize=(14, 12))
    
    # 为每种语言创建颜色
    languages = list(known_language_embeddings.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(languages)))
    
    # 绘制每个点
    start_idx = 0
    for lang_name, color in zip(languages, colors):
        n_samples = known_language_embeddings[lang_name].shape[0]
        end_idx = start_idx + n_samples
        
        plt.scatter(X_tsne[start_idx:end_idx, 0], 
                   X_tsne[start_idx:end_idx, 1],
                   s=30, alpha=0.6, color=color, label=lang_name)
        
        # 标记中心点
        center_x = np.mean(X_tsne[start_idx:end_idx, 0])
        center_y = np.mean(X_tsne[start_idx:end_idx, 1])
        plt.scatter(center_x, center_y, s=200, color=color, 
                   edgecolor='black', linewidth=2, marker='*')
        plt.text(center_x, center_y, lang_name, fontsize=10, 
                fontweight='bold', ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        start_idx = end_idx
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Language Embeddings\n(Wav2Vec2 Classification Model)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE图已保存到: {output_path}")
    return X_tsne, all_labels

def visualize_classification_results(model, feature_extractor, languages, num_samples=50):
    """
    主函数：可视化分类效果
    """
    known_language_embeddings = {}
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"使用设备: {device}")
    print(f"处理 {len(languages)} 种语言，每种语言 {num_samples} 个样本")
    
    for lang in languages:
        print(f"\n处理语言: {lang}")
        
        try:
            # 加载数据集
            train_ds = load_dataset(
                "fixie-ai/common_voice_17_0",
                lang,
                split="train",
                trust_remote_code=True,
                cache_dir="/mnt/storage/ldl_linguistics/datasets",
            )
            train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
            # 提取嵌入特征
            embeddings = extract_embeddings_for_language(
                model, feature_extractor, train_ds, lang, num_samples
            )
            
            if len(embeddings) > 0:
                known_language_embeddings[lang] = embeddings
                print(f"  ✓ 成功提取 {len(embeddings)} 个嵌入特征")
            else:
                print(f"  ✗ 无法提取嵌入特征")
                
        except Exception as e:
            print(f"  ✗ 处理语言 {lang} 时出错: {e}")
            continue
    
    # 生成t-SNE图
    if known_language_embeddings:
        output_path = f'wav2vec2_language_tsne_{len(known_language_embeddings)}_langs.png'
        tsne_results, labels = save_simple_tsne(known_language_embeddings, output_path)
        
        # 返回结果
        return {
            'embeddings': known_language_embeddings,
            'tsne_results': tsne_results,
            'labels': labels
        }
    else:
        print("没有成功提取到任何嵌入特征")
        return None

# 使用示例
if __name__ == "__main__":
    # 加载模型和特征提取器
    model_path = "/mnt/storage/qisheng/github/wav2vec_test/weights/classification/xlsr-300m_class1/checkpoint-83000"
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    
    # 定义要处理的语言列表
    lan_list = ["ar","be","bn","cs","cy","de","en","es","fa","fr","hu","hi","it","ja","ka","lt","lv","nl","pl","pt","ru","ro","sw","ta","th","tr"]
    
    # 可以选择部分语言进行快速测试
    #lan_list = ["en", "es", "fr", "de", "it"]  # 常用语言
    
    # 可视化分类效果
    results = visualize_classification_results(
        model, feature_extractor, lan_list, num_samples=50
    )
    
    # 分析结果
    if results:
        print(f"\n可视化完成！")
        print(f"共处理了 {len(results['embeddings'])} 种语言")
        print(f"总样本数: {len(results['labels'])}")