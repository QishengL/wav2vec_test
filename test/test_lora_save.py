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
    
    def set_input_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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

def resize_wav2vec2_ctc_vocab(model, new_vocab_size):
    """调整Wav2Vec2ForCTC模型的词汇表大小"""
    
    print(f"调整Wav2Vec2 CTC模型词汇表: {model.config.vocab_size} -> {new_vocab_size}")
    
    # 获取当前的lm_head
    old_lm_head = model.lm_head
    
    # 创建新的lm_head
    new_lm_head = torch.nn.Linear(
        old_lm_head.in_features,
        new_vocab_size,
        bias=(old_lm_head.bias is not None)
    )
    
    # 复制旧权重
    old_vocab_size = old_lm_head.out_features
    new_lm_head.weight.data[:old_vocab_size] = old_lm_head.weight.data
    
    # 初始化新token的权重
    if new_vocab_size > old_vocab_size:
        torch.nn.init.normal_(
            new_lm_head.weight.data[old_vocab_size:],
            mean=0.0,
            std=0.02
        )
    
    # 复制bias（如果有）
    if old_lm_head.bias is not None:
        new_lm_head.bias.data[:old_vocab_size] = old_lm_head.bias.data
        if new_vocab_size > old_vocab_size:
            new_lm_head.bias.data[old_vocab_size:].zero_()
    
    # 替换lm_head
    model.lm_head = new_lm_head
    
    # 更新config
    model.config.vocab_size = new_vocab_size
    
    print(f"✓ 词汇表扩展完成: {old_vocab_size} -> {new_vocab_size}")
    return model

checkpoint = "/mnt/storage/qisheng/github/wav2vec_test/weights/ru_18000-xlsr-300m/checkpoint-60000"
new_vocab_path = "/mnt/storage/qisheng/github/wav2vec_test/vocab_builder/vocab_folder/cv_uk_phoneme"
model = MyWav2Vec2ForCTC.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint,phonemizer_lang='uk')
processor = Wav2Vec2Processor.from_pretrained(checkpoint)
#feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
#model_config = AutoConfig.from_pretrained(checkpoint)

new_tokens_list = get_new_tokens(new_vocab_path,tokenizer.get_vocab())
num_added = tokenizer.add_tokens(new_tokens_list)
new_vocab_size = len(tokenizer)  # 获取新的词汇表大小
print(f"添加了 {num_added} 个新token")
print(f"新词汇表大小: {new_vocab_size}")
model.resize_token_embeddings(new_vocab_size)
print(model.config.vocab_size)
print(model.lm_head.weight.shape)
#model = resize_wav2vec2_ctc_vocab(model,new_vocab_size)
processor.tokenizer = tokenizer

LORA_PARAMS = {
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "r": 16,
    "bias": "none",
    "target_modules": ["q_proj", "v_proj"],
    "save_embedding_layers":True,
}

lora_config = LoraConfig(
        **LORA_PARAMS
    )

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
peft_model.forward = model.forward
peft_model.save_pretrained("./lora_test1")
