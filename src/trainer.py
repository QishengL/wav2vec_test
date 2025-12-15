# trainer.py
from transformers import Trainer, TrainingArguments
import evaluate
import torch
from collator import DataCollatorCTCWithPadding,AudioClassificationDataCollator
from transformers import AutoProcessor
from accelerate import Accelerator
import numpy as np
import wandb
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

def create_trainer(model, tokenizer, feature_extractor, dataset, training_args,eval_metrics, processor=None):
    
    
    #accelerator = Accelerator()
    
    # 确保训练参数正确设置
    #training_args.ddp_find_unused_parameters = False
    #training_args.remove_unused_columns = False

    # 定义评价指标
    #eval_metrics = {metric: evaluate.load(metric) for metric in ["wer"]}
    # Define evaluation metrics during training, *i.e.* word error rate, character error rate
    # cache_dir can be added here
    eval_metrics = {metric: evaluate.load(metric) for metric in eval_metrics}

    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels

    def compute_metrics(pred):
        pred_ids = pred.predictions[0]
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        #print(f"pred:{pred_str}")
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)
        #print(f"label:{label_str}")
        metrics = {k: v.compute(predictions=pred_str, references=label_str) for k, v in eval_metrics.items()}

        return metrics

    
    if processor == None:
        processor = AutoProcessor.from_pretrained(training_args.output_dir)
    else:
        processor = processor
    
    data_collator = DataCollatorCTCWithPadding(
        processor = processor
    )

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

    '''
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["eval"] if training_args.do_eval else None,
        processing_class=processor,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    '''
    return trainer

def create_trainer_for_classification(model, feature_extractor, dataset, training_args, eval_metrics,language_dict, processor=None):
    
    # 对于分类任务，我们使用准确率等标准分类指标
    # eval_metrics 现在应该是 ["accuracy", "f1"] 等分类指标
    eval_metrics = {metric: evaluate.load(metric) for metric in eval_metrics}
    label_to_code = {v: k for k, v in language_dict.items()}

    def compute_metrics(pred):
        preds = pred.predictions
        labels = pred.label_ids
        
        # 获取预测结果
        if preds.ndim == 2:
            preds = np.argmax(preds, axis=1)
        else:
            preds = np.argmax(preds, axis=-1)
        
        accuracy_metric = evaluate.load("accuracy")
        unique_langs = np.unique(labels)
        metrics = {}
        total_samples = len(labels)
        
        # 整体准确率
        overall_acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        metrics["overall_accuracy"] = overall_acc
        
        # 按语言分组计算
        weighted_sum = 0
        for lang in unique_langs:
            lang_mask = labels == lang
            lang_count = lang_mask.sum()
            
            if lang_count > 0:
                lang_acc = accuracy_metric.compute(
                    predictions=preds[lang_mask], 
                    references=labels[lang_mask]
                )["accuracy"]
                lang_code = label_to_code[lang]
                metrics[f"{lang_code}/accuracy"] = lang_acc
                weighted_sum += lang_acc * lang_count
        
        
        
        return metrics
    data_collator = AudioClassificationDataCollator(
        feature_extractor = feature_extractor
    )
    # 使用标准的 Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["eval"] if training_args.do_eval else None,
    )
    
    return trainer