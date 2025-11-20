from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,AutoTokenizer
import torch
import torchaudio
from datasets import load_dataset, Audio
from peft import get_peft_model, LoraConfig,PeftModel
from collections import defaultdict
import json

class MyWav2Vec2ForCTC(Wav2Vec2ForCTC):
    def get_input_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, new_num_tokens = None):
        """
        调整词汇表大小
        """
        if new_num_tokens is None:
            return self.lm_head
        
        old_lm_head = self.lm_head
        old_num_tokens = old_lm_head.out_features
        
        if new_num_tokens == old_num_tokens:
            return self.lm_head
        
        # 创建新的线性层
        new_lm_head = torch.nn.Linear(
            old_lm_head.in_features, 
            new_num_tokens, 
            bias=old_lm_head.bias is not None
        )
        
        # 复制权重
        with torch.no_grad():
            if new_num_tokens > old_num_tokens:
                new_lm_head.weight.data[:old_num_tokens] = old_lm_head.weight.data
                # 新 token 随机初始化
                std_dev = 0.02
                new_lm_head.weight.data[old_num_tokens:] = torch.randn(
                    new_num_tokens - old_num_tokens, old_lm_head.in_features
                ) * std_dev
                
                if old_lm_head.bias is not None:
                    new_lm_head.bias.data[:old_num_tokens] = old_lm_head.bias.data
                    new_lm_head.bias.data[old_num_tokens:] = 0
            else:
                new_lm_head.weight.data = old_lm_head.weight.data[:new_num_tokens]
                if old_lm_head.bias is not None:
                    new_lm_head.bias.data = old_lm_head.bias.data[:new_num_tokens]
        
        self.lm_head = new_lm_head
        self.config.vocab_size = new_num_tokens


def get_new_tokens(vocab_path,existing_dict):
    
    with open(f"{vocab_path}/vocab.json", 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)
    existing_tokens = set(existing_dict.keys())
    new_tokens = [token for token in vocab_dict.keys() if token not in existing_tokens]
    return new_tokens

def text_to_ids_using_vocab(text, tokenizer):
    """使用vocab字典将文本转回token IDs"""
    # 将文本分割成phoneme
    phonemes = text.split()
    
    # 使用vocab转换每个phoneme
    ids = []
    for phoneme in phonemes:
        if phoneme in tokenizer.get_vocab():
            ids.append(tokenizer.get_vocab()[phoneme])
        else:
            # 处理OOV (Out-of-Vocabulary) token
            ids.append(tokenizer.unk_token_id)
    
    return ids


# 初始化phoneme频率统计字典
phoneme_frequency = defaultdict(lambda: {'gold_count': 0, 'pred_count': 0})
phoneme_frequency_id = defaultdict(lambda: {'gold_count': 0, 'pred_count': 0})
def count_phoneme_frequency(gold_phonemes, pred_phonemes, freq_dict):
    """
    统计phoneme在gold和pred中的出现频率
    """
    # 将字符串转换为phoneme列表
    gold_list = gold_phonemes.split()
    pred_list = pred_phonemes.split()
    
    # 统计gold中的phoneme频率
    for phoneme in gold_list:
        freq_dict[phoneme]['gold_count'] += 1
    
    # 统计pred中的phoneme频率
    for phoneme in pred_list:
        freq_dict[phoneme]['pred_count'] += 1


def count_phoneme_frequency_id(gold_phonemes, pred_phonemes, freq_dict):
    """
    统计phoneme在gold和pred中的出现频率
    """

    
    # 统计gold中的phoneme频率
    for phoneme in gold_phonemes:
        freq_dict[phoneme]['gold_count'] += 1
    
    # 统计pred中的phoneme频率
    for phoneme in pred_phonemes:
        freq_dict[phoneme]['pred_count'] += 1

def clean_phoneme_list(phoneme_list):
    """清理音素列表，移除同一元素内的空格"""
    cleaned_list = []
    
    for phoneme in phoneme_list:
        # 移除同一音素内的空格
        cleaned_phoneme = phoneme.replace(" ", "")
        cleaned_list.append(cleaned_phoneme)
    
    return cleaned_list


# 加载模型和processor
model_path = "/mnt/storage/qisheng/github/wav2vec_test/weights/ru_9000-xlsr-300m_general/checkpoint-7000"
model = MyWav2Vec2ForCTC.from_pretrained(model_path)
adapter_path = "/mnt/storage/qisheng/github/wav2vec_test/weights/lora/ru2uk_100-xlsr-300m_general/checkpoint-8000"
tokenizer = AutoTokenizer.from_pretrained(adapter_path,phonemizer_lang='uk')
model = PeftModel.from_pretrained(model, adapter_path)
processor = Wav2Vec2Processor.from_pretrained(adapter_path)



# 将模型设置为评估模式
model.eval()

# 2. 加载评估数据集
print("加载数据集中...")
raw_datasets = load_dataset(
    "fixie-ai/common_voice_17_0",
    'uk',
    split='test',
    trust_remote_code=True,
    cache_dir="/mnt/storage/ldl_linguistics/datasets",
)

# 确保音频列是Audio类型
raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16000))

# 3. 获取第一个样本

for i in range(10):
    first_sample = raw_datasets[i]
    print("\n" + "="*50)
    print("第一个样本的详细信息:")
    print("="*50)

    additional_kwargs = {}
    additional_kwargs["phonemizer_lang"] = 'uk'
    gold_text = first_sample['sentence']
    gold_ids = processor.tokenizer(gold_text, **additional_kwargs).input_ids
    gold_decoded = processor.batch_decode(gold_ids)
    # 4. 打印gold label和信息
    print(f"📝 原始Gold Label: {gold_text}")
    print(f"🔤 Tokenizer处理后Gold: {gold_decoded}")
    #print(f"🎵 音频路径: {first_sample['audio']['path']}")
    #print(f"📊 采样率: {first_sample['audio']['sampling_rate']}")
    #print(f"⏱️  音频长度: {len(first_sample['audio']['array'])} 采样点")
    #print(f"⏱️  音频时长: {len(first_sample['audio']['array']) / first_sample['audio']['sampling_rate']:.2f} 秒")

    # 5. 进行推理
    print("\n🔄 进行推理...")

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
        #print(processor.tokenizer.vocab)
        
        #prediction = processor.batch_decode(predicted_ids)[0]
        cleaned_gold = clean_phoneme_list(gold_decoded)
        gold_str = ' '.join(cleaned_gold)
        # 6. 输出结果对比
        print("\n" + "="*50)
        print("识别结果对比:")
        print("="*50)
        print(f"✅ Gold Label: {gold_str}")
        print(f"🤖 模型预测: {pred_str}")

        from evaluate import load
        predictions = []
        references = []
        wer_metric = load("wer")
        references.append(gold_str)
        predictions.append(pred_str)

        gold_ids_from_vocab = text_to_ids_using_vocab(gold_str, processor.tokenizer)
        #print(pred_ids_from_vocab)
        pred_ids_from_vocab = text_to_ids_using_vocab(pred_str, processor.tokenizer)
        #print(pred_ids_from_vocab)

        wer = wer_metric.compute(predictions=predictions, references=references)
        print(f"最终WER: {wer:.4f}")


        count_phoneme_frequency_id(gold_ids_from_vocab, pred_ids_from_vocab, phoneme_frequency_id)
        count_phoneme_frequency(gold_str, pred_str, phoneme_frequency)
sorted_phonemes = sorted(phoneme_frequency.keys(), 
                        key=lambda x: phoneme_frequency[x]['gold_count'], 
                        reverse=True)
sorted_phonemes_id = sorted(phoneme_frequency_id.keys(), 
                        key=lambda x: phoneme_frequency_id[x]['gold_count'], 
                        reverse=True)
for phoneme in sorted_phonemes:
    freq = phoneme_frequency[phoneme]
    gold_freq = freq['gold_count']
    pred_freq = freq['pred_count']

    
    print(f"{phoneme:<10} {gold_freq:<12} {pred_freq:<12} ")
for phoneme in sorted_phonemes_id:
    freq = phoneme_frequency_id[phoneme]
    gold_freq = freq['gold_count']
    pred_freq = freq['pred_count']

    
    print(f"{phoneme:<10} {gold_freq:<12} {pred_freq:<12} ")
