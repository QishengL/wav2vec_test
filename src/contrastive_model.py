import torch
import torch.nn as nn
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
from typing import Optional, Union, Tuple
from dataclasses import dataclass

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

        return ContrastiveLearningOutput(
            loss=loss,
            embeddings=embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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