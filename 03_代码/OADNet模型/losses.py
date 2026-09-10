# -*- coding: utf-8 -*-
"""损失函数与梯度调制工具（二区不平衡基线）。

- CE       : 类别加权交叉熵（默认主损失）
- Focal    : Lin et al. 2017
- CB       : Class-Balanced Loss (Cui et al. 2019)，有效样本数加权
- LDAM     : Label-Distribution-Aware Margin Loss (Cao et al. 2019)
- OGM_GE   : OGM-GE 逐模态梯度调制 (Peng et al. 2022)，辅助分类器+系数
- Kang     : 解耦重训（两阶段，见 train.py）
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# 类别权重
# ---------------------------------------------------------------------
def class_counts(labels) -> np.ndarray:
    return np.bincount(np.asarray(labels), minlength=2).astype(np.float64)


def inverse_freq_weights(labels) -> torch.Tensor:
    """逆频权重（主损失用）。"""
    cnt = class_counts(labels)
    w = cnt.sum() / (len(cnt) * cnt)
    return torch.tensor(w, dtype=torch.float32)


# ---------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight          # (2,) 张量或 None
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none", weight=self.weight)
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------
# Class-Balanced Loss (Cui et al. 2019)
# ---------------------------------------------------------------------
class ClassBalancedLoss(nn.Module):
    def __init__(self, cls_num_list, beta=0.999, weight=None):
        super().__init__()
        n = np.asarray(cls_num_list, dtype=np.float64)
        effective = (1.0 - beta) / (1.0 - beta ** n) if beta < 1.0 else np.ones(len(n))
        cb_weights = torch.tensor(effective, dtype=torch.float32)
        if weight is not None:
            cb_weights = cb_weights * torch.tensor(weight, dtype=torch.float32)
        self.register_buffer("cb_weights", cb_weights)

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        w = self.cb_weights[targets]
        return (ce * w).mean()


# ---------------------------------------------------------------------
# LDAM Loss (Cao et al. 2019)
# ---------------------------------------------------------------------
class LDAMLoss(nn.Module):
    """LDAM (Cao et al. 2019)，官方实现：对正确类 logit 减 margin，损失乘温度 s。"""
    def __init__(self, cls_num_list, max_m=0.5, s=30.0, weight=None):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(np.asarray(cls_num_list, dtype=np.float64)))
        m_list = m_list * (max_m / np.max(m_list))
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.s = s
        self.weight = torch.tensor(weight, dtype=torch.float32) if weight is not None else None

    def forward(self, inputs, targets):
        index = torch.zeros_like(inputs, dtype=torch.uint8).scatter_(
            1, targets.view(-1, 1), 1)
        index_float = index.float()
        # batch_m: 每样本其正确类对应的 margin（(B,1)）
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1)).view(-1, 1)
        x_m = inputs - batch_m
        outputs = torch.where(index.bool(), x_m, inputs)
        return F.cross_entropy(outputs, targets, weight=self.weight) * self.s


# ---------------------------------------------------------------------
# OGM-GE 梯度调制系数 (Peng et al. 2022)
# ---------------------------------------------------------------------
def compute_ogm_coeffs(mod_logits_list, targets, alpha=0.8):
    """对每个模态，按其正确类概率与全体均值之比调制梯度：
       ratio > 1（过度自信模态）-> m = 1 - tanh(alpha * ratio)，否则 m = 1。"""
    with torch.no_grad():
        probs = [F.softmax(l, dim=1) for l in mod_logits_list]
        cp = [p.gather(1, targets.unsqueeze(1)).squeeze(1) for p in probs]
        avg = sum(cp) / len(cp)
        return [torch.where(r > 1.0, 1.0 - torch.tanh(alpha * r), torch.ones_like(r))
                for r in [c / (avg + 1e-8) for c in cp]]
