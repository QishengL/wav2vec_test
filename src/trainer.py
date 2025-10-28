# trainer.py
from transformers import Trainer, TrainingArguments
import evaluate
import torch
from collator import DataCollatorCTCWithPadding
from transformers import AutoProcessor
from accelerate import Accelerator
def create_trainer(model, tokenizer, feature_extractor, dataset, training_args, eval_metrics):
    
    
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
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        metrics = {k: v.compute(predictions=pred_str, references=label_str) for k, v in eval_metrics.items()}

        return metrics

    

    processor = AutoProcessor.from_pretrained(training_args.output_dir)
    
    data_collator = DataCollatorCTCWithPadding(
        processor = processor
    )

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
    return trainer