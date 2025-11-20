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

checkpoint = "/mnt/storage/qisheng/github/wav2vec_test/weights/ru_18000-xlsr-300m/checkpoint-60000"
new_vocab_path = "/mnt/storage/qisheng/github/wav2vec_test/vocab_builder/vocab_folder/cv_uk_phoneme"
model = MyWav2Vec2ForCTC.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint,phonemizer_lang='uk')
processor = Wav2Vec2Processor.from_pretrained(checkpoint)
new_tokens_list = get_new_tokens(new_vocab_path,tokenizer.get_vocab())
num_added = tokenizer.add_tokens(new_tokens_list)
new_vocab_size = len(tokenizer)
model.resize_token_embeddings(new_vocab_size)
print(model.config.vocab_size)
print(model.lm_head.weight.shape)
peft_model = PeftModel.from_pretrained(model, "./lora_test1")