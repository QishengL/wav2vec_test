# %%
import torch
import torch.nn as nn
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
from typing import Optional, Union, Tuple
from dataclasses import dataclass
import wandb

wandb.init(
            project="wav2vec_contrastive",
            name="class1",
        )


@dataclass
class ContrastiveLearningOutput:
    loss: Optional[torch.FloatTensor] = None
    embeddings: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

class Wav2Vec2ContrastiveConfig(Wav2Vec2Config):
    def __init__(self, contrastive_proj_size=128, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_proj_size = contrastive_proj_size

class Wav2Vec2ForContrastiveLearning(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Contrastive learning does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )
        
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # 投影头 - 参照官方结构
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.contrastive_proj_size)  # 128 for contrastive learning
        )
        
        # 使用你的SupConLoss
        self.contrastive_loss = SupConLoss(temperature=0.07)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        mode: str = "supervised_contrastive",  # 主要使用监督对比学习
    ) -> Union[tuple, ContrastiveLearningOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # 平均池化
        if attention_mask is not None:
            # 计算特征向量的注意力掩码
            attention_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            pooled_output = hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled_output = hidden_states.mean(dim=1)

        # 投影到对比学习空间
        embeddings = self.projector(pooled_output)
        
        # 计算损失
        loss = None
        if labels is not None:
            if mode == "supervised_contrastive":
                # 为SupConLoss准备特征形状 [batch_size, 1, feature_dim]
                features = embeddings.unsqueeze(1)
                loss = self.contrastive_loss(features, labels)
            elif mode == "contrastive":
                loss = self._compute_simple_contrastive_loss(embeddings)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        if not return_dict:
            output = (embeddings,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "embeddings": embeddings,
            "hidden_states": outputs.hidden_states if output_hidden_states else None,
            "attentions": outputs.attentions if output_attentions else None,
        }

    def _compute_simple_contrastive_loss(self, embeddings):
        """
        简化的标准对比损失，用于无标签情况
        假设批次中的样本是成对的: [aug1, aug2, aug1, aug2, ...]
        """
        batch_size = embeddings.shape[0]
        
        if batch_size % 2 != 0:
            raise ValueError("Batch size must be even for contrastive learning")
        
        # 归一化嵌入
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / 0.07
        
        # 创建标签：正样本对是 (2i, 2i+1) 和 (2i+1, 2i)
        labels = torch.arange(batch_size, device=embeddings.device)
        labels = (labels // 2) * 2
        labels = torch.cat([labels[1:], labels[:1]])  # 移位对齐正样本对
        
        # 计算对比损失
        loss = nn.functional.cross_entropy(similarity_matrix, labels)
        
        return loss

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        """从官方代码复制的辅助方法"""
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """从官方代码复制的辅助方法"""
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

# %%

config = Wav2Vec2ContrastiveConfig.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    contrastive_proj_size=128
)

model = Wav2Vec2ForContrastiveLearning.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    config=config,
    ignore_mismatched_sizes=True
)

# %%
from datasets import load_dataset
import datasets

# %%
def load_and_combine_multilingual_datasets(languages, max_train_sample=None, max_eval_sample=None, random=None):
    """
    加载并合并多语言数据集
    
    Args:
        languages: 字典，格式为 {'语言代码': 数字标签}，如 {'en': 0, 'zh': 1, 'fr': 2}
        max_train_sample: 每种语言的训练集最大样本数
        max_eval_sample: 每种语言的验证集最大样本数
        random: 随机种子
        config: 数据集配置参数
    """
    from datasets import concatenate_datasets
    
    combined_train = []
    combined_eval = []
    
    for lang_code, lang_label in languages.items():
        print(f"Loading {lang_code} dataset (label: {lang_label})...")
        
        # 加载数据集
        train_ds = load_dataset(
            "fixie-ai/common_voice_17_0",
            lang_code,
            split="train",
            trust_remote_code=True,
            cache_dir = "/mnt/storage/ldl_linguistics/datasets"
        )
        
        eval_ds = load_dataset(
            "fixie-ai/common_voice_17_0",
            lang_code, 
            split="test",
            trust_remote_code=True,
            cache_dir = "/mnt/storage/ldl_linguistics/datasets"
        )
        
        # 采样
        if max_train_sample is not None and max_train_sample < len(train_ds):
            if random is not None:
                train_ds = train_ds.shuffle(seed=random)
            train_ds = train_ds.select(range(max_train_sample))
            
        if max_eval_sample is not None and max_eval_sample < len(eval_ds):
            eval_ds = eval_ds.select(range(max_eval_sample))
        
        # 添加语言标签 - 使用数字标签
        def add_language_label(example, language_code, language_label):
            example["language_code"] = language_code  # 保留语言代码
            example["language_label"] = language_label  # 数字标签
            return example
            
        train_ds = train_ds.map(lambda x: add_language_label(x, lang_code, lang_label))
        eval_ds = eval_ds.map(lambda x: add_language_label(x, lang_code, lang_label))
        
        combined_train.append(train_ds)
        combined_eval.append(eval_ds)
        
        print(f"Added {len(train_ds)} {lang_code} train samples, {len(eval_ds)} eval samples")
    
    # 合并所有语言数据
    final_train = concatenate_datasets(combined_train)
    final_eval = concatenate_datasets(combined_eval)
    
    # 打乱顺序
    if random is not None:
        final_train = final_train.shuffle(seed=random)
        final_eval = final_eval.shuffle(seed=random)
    
    combined_dataset = datasets.DatasetDict({
        "train": final_train,
        "eval": final_eval
    })
    
    print(f"Final combined dataset: {len(final_train)} train samples, {len(final_eval)} eval samples")
    print(f"Language labels mapping: {languages}")
    
    return combined_dataset

# %%
def vectorize_datasets_classification(raw_datasets, tokenizer, feature_extractor):
    
    dataset_sampling_rate = next(iter(raw_datasets.values())).features["audio"].sampling_rate
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        raw_datasets = raw_datasets.cast_column(
            "audio",
            datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
        )

    max_input_length = 20.0 * feature_extractor.sampling_rate
    min_input_length = 0.0 * feature_extractor.sampling_rate
    audio_column_name = "audio"
    num_workers = 1
    feature_extractor_input_name = feature_extractor.model_input_names[0]

    def prepare_dataset(batch):
        # load and process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch[feature_extractor_input_name] = getattr(inputs, feature_extractor_input_name)[0]
        batch["input_length"] = len(sample["array"].squeeze())
        
        # 使用语言标签作为分类目标
        batch["labels"] = batch["language_label"]
        
        return batch

    # 处理数据集
    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=[col for col in next(iter(raw_datasets.values())).column_names 
                       if col not in ["audio", "language_label"]],
        num_proc=num_workers,
        desc="preprocess datasets",
    )

    # 过滤音频长度
    def is_audio_in_length_range(length):
        return min_input_length < length < max_input_length

    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_length"],
    )
    
    return vectorized_datasets
lan_list=["ar","be","bn","cs","cy","de","en","es","fa"]
languages_dict =  {code: idx for idx, code in enumerate(lan_list)}
# %%
raw_datasets = load_and_combine_multilingual_datasets(
        languages=languages_dict,
        max_train_sample=1000,  # 每种语言250条训练数据
        max_eval_sample=100,    # 每种语言50条验证数据
        random=42,
    )


# %%
from transformers import AutoFeatureExtractor

# %%
feature_extractor = AutoFeatureExtractor.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        trust_remote_code=True,
    )

# %%
vectorized_datasets = vectorize_datasets_classification(
            raw_datasets, 
            tokenizer=None,  # 传入None或者删除这个参数
            feature_extractor=feature_extractor, 
        )



# %%
class AudioClassificationDataCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def __call__(self, features):
        # 提取输入特征 - Wav2Vec2ForSequenceClassification 需要 input_values
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        
        # 提取标签
        labels = [feature["labels"] for feature in features]
        
        # 使用特征提取器处理填充
        batch = self.feature_extractor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # 添加标签
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        
        return batch

# %%
from transformers import Trainer

# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./contrastive_wav2vec",
    overwrite_output_dir=True,
    num_train_epochs=200,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    lr_scheduler_type="constant",
    weight_decay=0.0,
    warmup_steps=0,
    max_grad_norm=1.5,
    save_total_limit=2,
    report_to="wandb",
    save_steps = 500, 
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=False,
    do_train=True,
    do_eval=True,
    logging_strategy="steps",
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=50,
)

# %%
class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        自定义损失计算，处理对比学习
        """
        # 提取输入
        input_values = inputs.get("input_values")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        
        # 前向传播
        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        loss = outputs.get("loss")
        
        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        自定义预测步骤
        """
        # 提取输入
        input_values = inputs.get("input_values")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        
        # 前向传播
        with torch.no_grad():
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        
        loss = outputs.get("loss")
        embeddings = outputs.get("embeddings")
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, embeddings, labels)

# %%
data_collator = AudioClassificationDataCollator(
        feature_extractor = feature_extractor
    )
# 使用标准的 Trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
    eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
)

# %%
train_result = trainer.train(resume_from_checkpoint=None)


