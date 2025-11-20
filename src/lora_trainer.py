from peft import LoraConfig, get_peft_model, TaskType,PeftModel
import torch
from transformers import AutoModelForCTC, AutoProcessor,Trainer,Wav2Vec2ForCTC
from datasets import DatasetDict
import evaluate
from collator import DataCollatorCTCWithPadding
import wandb
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

    def reinit_token_embeddings(self, new_num_tokens = None):
        """
        完全重新设置词汇表，只保留特殊token
        """
        if new_num_tokens is None:
            return self.lm_head
        
        old_lm_head = self.lm_head
        old_num_tokens = old_lm_head.out_features
        
        print(f"=== Complete Vocabulary Reset ===")
        print(f"Old vocab size: {old_num_tokens}")
        print(f"New vocab size: {new_num_tokens}")
        
        # 创建新的线性层
        new_lm_head = torch.nn.Linear(
            old_lm_head.in_features, 
            new_num_tokens, 
            bias=old_lm_head.bias is not None
        )
        
        print(f"New lm_head shape: {new_lm_head.weight.shape}")
        
        # 完全重新初始化所有权重
        with torch.no_grad():
            # 所有token都重新初始化
            std_dev = 0.02
            new_lm_head.weight.data = torch.randn(
                new_num_tokens, old_lm_head.in_features
            ) * std_dev
            
            if old_lm_head.bias is not None:
                new_lm_head.bias.data.zero_()
        
        self.lm_head = new_lm_head
        self.config.vocab_size = new_num_tokens
        
        print(f"✅ Complete vocabulary reset completed")
        print(f"✅ All {new_num_tokens} tokens are newly initialized")
        

class MultiLanguageEvaluationTrainer(Trainer):
    def __init__(self, *args, language_column="language", languages=None, **kwargs):
        #print("🔧 MultiLanguageEvaluationTrainer.__init__ 被调用")
        super().__init__(*args, **kwargs)
        self.language_column = language_column
        self.languages = languages or []
        #print(f"🔧 初始化参数: language_column={language_column}, languages={languages}")
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        #print("🎯🎯🎯 MultiLanguageEvaluationTrainer.evaluate 被调用!")

        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        #print(f"🎯 eval_dataset: {eval_dataset}")
        #print(f"🎯 metric_key_prefix: {metric_key_prefix}")
        
        # 如果有多语言数据，直接进行多语言评估并计算加权平均
        if eval_dataset is not None and self.language_column in eval_dataset.column_names:
            #print("🎯 使用多语言加权评估...")
            metrics = self._evaluate_with_weighted_average(eval_dataset, ignore_keys, metric_key_prefix)
        else:
            # 没有语言信息，回退到默认评估
            #print("🎯 使用默认评估...")
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        #print(f"🎯 最终返回的 metrics: {metrics}")
        return metrics
    
    def _evaluate_with_weighted_average(self, eval_dataset, ignore_keys, metric_key_prefix):
        """使用加权平均进行多语言评估"""
        if not self.languages:
            self.languages = list(set(eval_dataset[self.language_column]))
        
        #print(f"🎯 检测到的语言: {self.languages}")
        
        all_lang_metrics = {}
        total_samples = 0
        
        # 分别评估每种语言
        for lang_name in self.languages:
            #print(f"🎯 评估语言: {lang_name}")
            lang_dataset = eval_dataset.filter(
                lambda example: example[self.language_column] == lang_name
            )
            lang_samples = len(lang_dataset)
            #print(f"🎯 {lang_name} 样本数: {lang_samples}")
            
            if lang_samples > 0:
                lang_metrics = super().evaluate(lang_dataset, ignore_keys, f"eval_{lang_name}")
                
                # 简化指标名称：从 uk/eval_uk_wer 改为 uk/wer
                for k, v in lang_metrics.items():
                    # 移除重复的语言前缀
                    if k.startswith(f"eval_{lang_name}_"):
                        simplified_key = k.replace(f"eval_{lang_name}_", "")
                    else:
                        simplified_key = k
                    all_lang_metrics[f"{lang_name}/{simplified_key}"] = v
                
                all_lang_metrics[f"{lang_name}/num_samples"] = lang_samples
                
                # 累加样本数
                total_samples += lang_samples
        
        #print("--------------")
        #print("简化后的指标:", all_lang_metrics)
        #print("语言:", self.languages)
        #print("总样本数:", total_samples)
        #print("--------------")
        
        # 计算加权平均指标
        weighted_metrics = self._compute_weighted_average(all_lang_metrics, self.languages, total_samples)
        
        # 合并所有指标
        final_metrics = {**weighted_metrics, **all_lang_metrics}
        return final_metrics

    def _compute_weighted_average(self, lang_metrics, languages, total_samples):
        """计算加权平均指标 - 使用简化后的名称"""
        weighted_metrics = {}
        
        for metric in ['wer', 'loss']:  # 根据您实际有的指标调整
            weighted_sum = 0
            valid_langs = 0
            
            for lang_name in languages:
                lang_metric_key = f"{lang_name}/{metric}"
                lang_samples_key = f"{lang_name}/num_samples"
                
                if lang_metric_key in lang_metrics and lang_samples_key in lang_metrics:
                    lang_value = lang_metrics[lang_metric_key]
                    lang_samples = lang_metrics[lang_samples_key]
                    
                    # 加权累加
                    weighted_sum += lang_value * lang_samples
                    valid_langs += 1
            
            # 计算加权平均
            if valid_langs > 0 and total_samples > 0:
                weighted_avg = weighted_sum / total_samples
                weighted_metrics[metric] = weighted_avg
                #print(f"🎯 加权平均 {metric}: {weighted_avg} (基于 {valid_langs} 种语言, {total_samples} 样本)")
                
                wandb.log({f"weighted_avg/{metric}": weighted_avg})
                #print(f"📊 已记录加权平均 {metric} 到 wandb: {weighted_avg}")
        weighted_metrics["eval_samples"] = total_samples
        return weighted_metrics



def save_lora_adapter(trainer, adapter_path):
    """保存LoRA适配器"""
    trainer.model.save_pretrained(adapter_path)

def load_lora_adapter(model, adapter_path):
    """加载LoRA适配器到基础模型"""
    
    model = PeftModel.from_pretrained(model, adapter_path)
    return model



def setup_lora_for_ctc(model, lora_config=None,adapter_checkpoint=None):
    """为CTC模型设置LoRA配置"""
    
    if adapter_checkpoint != None:
        print("load adapter!")
        peft_model = load_lora_adapter(model,adapter_checkpoint)
        return peft_model
    # 默认LoRA配置
    if lora_config is None:
        lora_config = LoraConfig(
            inference_mode=False,
            r=8,  # LoRA秩
            lora_alpha=32,  # LoRA alpha
            lora_dropout=0.1,  # LoRA dropout
            target_modules=["k_proj", "v_proj", "q_proj", "out_proj"]  # 针对transformer层的投影层
        )
    
    # 应用LoRA到模型
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    peft_model.forward = model.forward
    return peft_model

def create_lora_trainer(model, tokenizer, feature_extractor, dataset, training_args, eval_metrics, processor=None, lora_config=None,adapter_checkpoint=None):
    
    # 设置LoRA
    model = setup_lora_for_ctc(model, lora_config,adapter_checkpoint)
    
    # 加载评估指标
    eval_metrics = {metric: evaluate.load(metric) for metric in eval_metrics}

    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels

    def compute_metrics(pred):
        pred_ids = pred.predictions[0]
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        #print(f"pred:{pred_str}")
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)
        #print(f"label:{label_str}")
        metrics = {k: v.compute(predictions=pred_str, references=label_str) for k, v in eval_metrics.items()}

        return metrics

    
    if processor is None:
        processor = AutoProcessor.from_pretrained(training_args.output_dir)
    
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    trainer = MultiLanguageEvaluationTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["eval"] if training_args.do_eval else None,
        processing_class=processor,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    return trainer



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


def resize_linear_layer(layer, old_size, new_size):
    """调整线性层的输出大小"""
    import torch
    import torch.nn as nn

    if isinstance(layer, nn.Linear):
        if new_size > old_size:
            print(f"扩展输出层: {old_size} -> {new_size}")
            
            # 保存旧权重
            old_weight = layer.weight.data
            old_bias = layer.bias.data if layer.bias is not None else None
            
            # 创建新层
            new_layer = nn.Linear(
                layer.in_features, 
                new_size, 
                bias=layer.bias is not None
            )
            
            # 复制旧权重
            new_layer.weight.data[:old_size] = old_weight
            
            # 初始化新权重（使用与原始模型相同的初始化策略）
            torch.nn.init.normal_(
                new_layer.weight.data[old_size:], 
                mean=0.0, 
                std=0.02  # 常见初始化标准差
            )
            
            # 处理bias
            if old_bias is not None:
                new_layer.bias.data[:old_size] = old_bias
                new_layer.bias.data[old_size:].zero_()
            
            return new_layer

    return layer

def get_new_tokens(vocab_path,existing_dict):
    
    with open(f"{vocab_path}/vocab.json", 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)
    existing_tokens = set(existing_dict.keys())
    new_tokens = [token for token in vocab_dict.keys() if token not in existing_tokens]
    return new_tokens
