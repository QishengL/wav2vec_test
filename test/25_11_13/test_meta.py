from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer
import torch
import torch.nn as nn
from datasets import load_dataset, Audio
from peft import PeftModel
from evaluate import load
from torch.nn import CTCLoss
# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

wer_metric = load("wer")

def debug_phoneme_conversion(processor, sample_texts):
    """调试音素转换过程"""
    print("=== 音素转换调试 ===")
    
    for i, text in enumerate(sample_texts[:3]):  # 检查前3个样本
        print(f"\n样本 {i+1}:")
        print(f"  原始文本: {text}")
        
        
        
        # 对比直接使用tokenizer
        direct_encoded = processor.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        )
        print(f"  direct_encoded: {direct_encoded.input_ids.squeeze(0)}")
        direct_phoneme = processor.tokenizer.decode(direct_encoded.input_ids.squeeze(0))
        print(f"  直接tokenizer音素: {direct_phoneme}")


def clean_phoneme_list(phoneme_list):
    """清理音素列表，移除同一元素内的空格"""
    return [phoneme.replace(" ", "") for phoneme in phoneme_list]

def create_meta_dataset_asr_gpu(lora_models, processor, raw_datasets, num_samples=1000, batch_size=4):
    """
    GPU版本的ASR元模型数据集创建
    """
    # 将所有模型移到GPU并设置为评估模式
    for i, model in enumerate(lora_models):
        lora_models[i] = model.to(device)
        lora_models[i].eval()
    
    all_logits_list = []  # 存储每个样本的所有模型logits
    all_transcripts = []  # 存储真实转录文本
    all_audio_data = []   # 存储音频数据
    
    # 选择子集用于训练元模型
    subset = raw_datasets.select(range(min(num_samples, len(raw_datasets))))
    print(f"创建元模型数据集，共 {len(subset)} 个样本，使用设备: {device}")
    
    # 首先收集所有音频数据
    for i, sample in enumerate(subset):
        all_audio_data.append({
            'array': sample["audio"]["array"],
            'sampling_rate': sample["audio"]["sampling_rate"],
            'sentence': sample["sentence"]
        })
    
    # 批量处理以提高GPU利用率
    for batch_start in range(0, len(all_audio_data), batch_size):
        batch_end = min(batch_start + batch_size, len(all_audio_data))
        batch_samples = all_audio_data[batch_start:batch_end]
        
        print(f"处理批次 {batch_start}-{batch_end-1}")
        
        # 预处理整个batch的音频
        audio_arrays = [sample['array'] for sample in batch_samples]
        sampling_rates = [sample['sampling_rate'] for sample in batch_samples]
        
        # 假设所有音频采样率相同
        inputs = processor(
            audio_arrays,
            sampling_rate=sampling_rates[0],
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        
        # 将输入数据移到GPU
        #inputs = {k: v.to(device) for k, v in inputs.items()}
        
        batch_logits = []  # 存储这个batch的所有模型logits
        
        with torch.no_grad():
            for model in lora_models:
                # 获取每个模型的logits输出 [batch_size, seq_len, vocab_size]
                logits = model(inputs.input_values.to('cuda')).logits
                batch_logits.append(logits.cpu())  # 移回CPU以节省GPU内存
        
        # 清理GPU内存
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 重新组织数据：按样本而不是按batch
        for i in range(len(batch_samples)):
            sample_logits = []
            for model_logits in batch_logits:
                # 提取单个样本的logits [1, seq_len, vocab_size]
                sample_logits.append(model_logits[i].unsqueeze(0))
            
            # 堆叠: [1, num_models, seq_len, vocab_size]
            stacked_logits = torch.stack(sample_logits, dim=1)  # 从 dim=0 改为 dim=1
            all_logits_list.append(stacked_logits)
            all_transcripts.append(batch_samples[i]['sentence'])
    
    print("元模型数据集创建完成!")
    return all_logits_list, all_transcripts

# 简单的静态权重元模型
class StaticWeightMetaModel(nn.Module):
    def __init__(self, num_models):
        super().__init__()
        # 可学习的静态权重
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(self, x):
        # x: [batch_size, num_models, seq_len, vocab_size]
        # 归一化权重
        normalized_weights = torch.softmax(self.weights, dim=0)
        # 加权平均: [batch_size, seq_len, vocab_size]
        weighted_logits = torch.einsum('m,bsmv->bsv', normalized_weights, x)
        return weighted_logits

# 动态权重元模型
class DynamicWeightMetaModel(nn.Module):
    def __init__(self, num_models, vocab_size, hidden_dim=128):
        super().__init__()
        self.num_models = num_models
        self.vocab_size = vocab_size
        
        # 基于每个时间步的特征计算权重
        self.feature_net = nn.Sequential(
            nn.Linear(vocab_size * num_models, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_models)
        )
    
    def forward(self, x):
        # x: [batch_size, num_models, seq_len, vocab_size]
        batch_size, num_models, seq_len, vocab_size = x.shape
        
        # 为每个时间步计算权重
        all_weights = []
        for t in range(seq_len):
            # 获取当前时间步所有模型的logits [batch_size, num_models, vocab_size]
            time_step_logits = x[:, :, t, :]
            # 展平: [batch_size, num_models * vocab_size]
            flattened = time_step_logits.reshape(batch_size, -1)
            # 计算权重 [batch_size, num_models]
            weights = self.feature_net(flattened)
            weights = torch.softmax(weights, dim=-1)
            all_weights.append(weights.unsqueeze(1))
        
        # 堆叠权重 [batch_size, seq_len, num_models]
        weights = torch.cat(all_weights, dim=1)
        
        # 应用加权平均
        # x: [batch_size, num_models, seq_len, vocab_size]
        # weights: [batch_size, seq_len, num_models] -> [batch_size, num_models, seq_len, 1]
        weights = weights.transpose(1, 2).unsqueeze(-1)
        weighted_logits = torch.sum(x * weights, dim=1)  # [batch_size, seq_len, vocab_size]
        
        return weighted_logits

def train_meta_model_ctc(meta_model, all_logits, all_transcripts, processor, num_epochs=20):
    """使用CTC Loss训练元模型"""
    meta_model = meta_model.to(device)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)
    
    # 使用CTC损失函数
    ctc_loss = CTCLoss(blank=processor.tokenizer.pad_token_id, zero_infinity=True)
    
    print("开始训练元模型(CTC Loss)...")
    print(f"训练数据: {len(all_logits)} 个样本")
    
    for epoch in range(num_epochs):
        meta_model.train()
        total_ctc_loss = 0
        total_wer = 0
        
        for i, sample_logits in enumerate(all_logits):
            # sample_logits: [1, num_models, seq_len, vocab_size]
            sample_logits = sample_logits.to(device)
            
            # 元模型融合 - 这个操作有梯度
            fused_logits = meta_model(sample_logits)  # [1, seq_len, vocab_size]
            
            # 准备CTC Loss的输入
            # fused_logits: [batch_size, seq_len, vocab_size] -> [seq_len, batch_size, vocab_size]
            log_probs = torch.log_softmax(fused_logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)  # CTC需要 [seq_len, batch_size, vocab_size]
            
            # 准备目标标签
            target_text = all_transcripts[i]
            # 将文本转换为token IDs
            labels = processor.tokenizer(
                target_text, 
                return_tensors="pt", 
                padding=True,
                truncation=True
            ).input_ids.squeeze(0)  # [target_length]
            
            # CTC Loss需要的输入格式
            input_lengths = torch.tensor([log_probs.size(0)], dtype=torch.long)  # 序列长度
            target_lengths = torch.tensor([labels.size(0)], dtype=torch.long)    # 目标长度
            
            # 移到GPU
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            # 计算CTC Loss
            loss = ctc_loss(log_probs, labels, input_lengths, target_lengths)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_ctc_loss += loss.item()
            target_phoneme = processor.tokenizer.decode(labels)
            # 计算WER用于监控（不需要梯度）
            with torch.no_grad():
                predicted_ids = torch.argmax(fused_logits, dim=-1)
                prediction = processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0]
                
                wer = wer_metric.compute(
                    predictions=[prediction], 
                    references=[target_phoneme]
                )
                total_wer += wer
            
            if i % 10 == 0:
                #print(f'Epoch {epoch+1}, Sample {i}, CTC Loss: {loss.item():.4f}, WER: {wer:.4f}')
                if i == 0:
                    print(f"  真实: {target_phoneme}")
                    print(f"  预测: {prediction}")
        
        avg_ctc_loss = total_ctc_loss / len(all_logits)
        avg_wer = total_wer / len(all_logits)
        print(f'Epoch [{epoch+1}/{num_epochs}], Avg CTC Loss: {avg_ctc_loss:.4f}, Avg WER: {avg_wer:.4f}')
        
        # 打印当前学习的权重
        if hasattr(meta_model, 'weights'):
            weights = torch.softmax(meta_model.weights, dim=0)
            print(f"  当前模型权重: {weights.detach().cpu().numpy()}")
    
    return meta_model

# 主函数
def main():
    # 1. 加载模型和处理器
    print("加载模型中...")
    
    model_path1 = "/mnt/storage/qisheng/github/wav2vec_test/weights/ru_9000-xlsr-300m_general/checkpoint-7000"
    model1 = Wav2Vec2ForCTC.from_pretrained(model_path1)
    adapter_path1 = "/mnt/storage/qisheng/github/wav2vec_test/weights/lora/ru2uk_100-xlsr-300m_general/checkpoint-8000"
    tokenizer = AutoTokenizer.from_pretrained(adapter_path1, phonemizer_lang='uk')
    model1 = PeftModel.from_pretrained(model1, adapter_path1)
    processor = Wav2Vec2Processor.from_pretrained(adapter_path1)

    model_path2 = "/mnt/storage/qisheng/github/wav2vec_test/weights/ro_9000-xlsr-300m_general/checkpoint-7000"
    model2 = Wav2Vec2ForCTC.from_pretrained(model_path2)
    adapter_path2 = "/mnt/storage/qisheng/github/wav2vec_test/weights/lora/ro2uk_100-xlsr-300m_general/checkpoint-8000"
    model2 = PeftModel.from_pretrained(model2, adapter_path2)

    model_path3 = "/mnt/storage/qisheng/github/wav2vec_test/weights/bg_9000-xlsr-300m_general/checkpoint-20000"
    model3 = Wav2Vec2ForCTC.from_pretrained(model_path3)
    adapter_path3 = "/mnt/storage/qisheng/github/wav2vec_test/weights/lora/bg2uk_100-xlsr-300m_general/checkpoint-8000"
    model3 = PeftModel.from_pretrained(model3, adapter_path3)

    model_path4 = "/mnt/storage/qisheng/github/wav2vec_test/weights/sk_9000-xlsr-300m_general/checkpoint-16000"
    model4 = Wav2Vec2ForCTC.from_pretrained(model_path4)
    adapter_path4 = "/mnt/storage/qisheng/github/wav2vec_test/weights/lora/sk2uk_100-xlsr-300m_general/checkpoint-8000"
    model4 = PeftModel.from_pretrained(model4, adapter_path4)

    model_path5 = "/mnt/storage/qisheng/github/wav2vec_test/weights/de_9000-xlsr-300m_general/checkpoint-20000"
    model5 = Wav2Vec2ForCTC.from_pretrained(model_path5)
    adapter_path5 = "/mnt/storage/qisheng/github/wav2vec_test/weights/lora/de2uk_100-xlsr-300m_general/checkpoint-8000"
    model5 = PeftModel.from_pretrained(model5, adapter_path5)

    lora_models = [model1, model2,model3,model4,model5]
    
    # 2. 加载数据集
    print("加载数据集中...")
    raw_datasets = load_dataset(
        "fixie-ai/common_voice_17_0",
        'uk',
        split='test',
        trust_remote_code=True,
        cache_dir="/mnt/storage/ldl_linguistics/datasets",
    )
    raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16000))
    
    # 3. 创建元模型数据集（GPU加速）
    print("创建元模型数据集...")
    all_logits, all_transcripts = create_meta_dataset_asr_gpu(
        lora_models, 
        processor, 
        raw_datasets, 
        num_samples=100,  # 先用100个样本测试
        batch_size=8
    )
    
    print(f"创建了 {len(all_logits)} 个样本的元模型数据集")
    print(f"Logits形状示例: {all_logits[0].shape}")
    print(f"转录示例: {all_transcripts[0]}")
    
    # 4. 初始化元模型
    vocab_size = all_logits[0].shape[-1]  # 从logits获取词汇表大小
    num_models = len(lora_models)
    
    
    # 选择元模型类型
    #meta_model = StaticWeightMetaModel(num_models)
    # 或者使用动态权重模型:
    meta_model = DynamicWeightMetaModel(num_models, vocab_size)
    print(f"初始化元模型: {type(meta_model).__name__}")
    
    #sample_texts = [all_transcripts[0], all_transcripts[1], all_transcripts[2]]
    #debug_phoneme_conversion(processor, sample_texts)
    
    # 5. 训练元模型
    trained_meta_model = train_meta_model_ctc(
        meta_model,
        all_logits,
        all_transcripts,
        processor,
        num_epochs=10
    )
    '''
    # 6. 测试融合效果
    print("\n测试融合效果...")
    test_sample = raw_datasets[0]
    test_inputs = processor(
        test_sample["audio"]["array"],
        sampling_rate=test_sample["audio"]["sampling_rate"],
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        # 获取各个模型的logits
        all_model_logits = []
        for model in lora_models:
            logits = model(test_inputs.input_values).logits
            all_model_logits.append(logits.unsqueeze(0))
        
        # 堆叠logits [1, num_models, seq_len, vocab_size]
        stacked_logits = torch.stack(all_model_logits, dim=1)
        
        # 使用元模型融合
        fused_logits = trained_meta_model(stacked_logits)
        fused_prediction = processor.batch_decode(
            torch.argmax(fused_logits, dim=-1),
            skip_special_tokens=True
        )[0]
    
    print(f"真实文本: {test_sample['sentence']}")
    print(f"融合预测: {fused_prediction}")
    
    # 7. 保存元模型
    #torch.save(trained_meta_model.state_dict(), "asr_meta_model.pth")
    #print("元模型训练完成并已保存!")
'''
if __name__ == "__main__":
    main()