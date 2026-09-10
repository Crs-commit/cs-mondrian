# -*- coding: utf-8 -*-
"""指标与显著性检验（二区标准）。

- threshold_selection : 验证集上网格搜索，最大化 Macro-F2
- evaluate_binary     : 测试集全套指标（Acc / Macro-F1 / Macro-F2 / 两类 P·R / AUC）
- mcnemar_per_fold    : 主模型 vs 基线，按折配对 McNemar 检验
- wilcoxon_paired     : 按 (fold×seed) 逐运行分数的配对 Wilcoxon
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, fbeta_score,
                             precision_recall_fscore_support, roc_auc_score)
from scipy.stats import wilcoxon

import config


def probs_from_logits(logits, T=config.TEMP):
    """温度缩放 + softmax，返回 P(class=1)。"""
    return F.softmax(torch.as_tensor(logits, dtype=torch.float32) / T, dim=1)[:, 1].numpy()


def threshold_selection(logits, labels, grid=None):
    """在验证集上选最优阈值：最大化 Macro-F2(β=2)。"""
    if grid is None:
        grid = np.arange(config.THRESH_GRID_START, config.THRESH_GRID_STOP + 1e-9,
                         config.THRESH_GRID_STEP)
    p = probs_from_logits(logits)
    best_t, best_f2 = grid[0], -1.0
    for t in grid:
        preds = (p > t).astype(int)
        f2 = fbeta_score(labels, preds, beta=2.0, average="macro", labels=[0, 1], zero_division=0)
        if f2 > best_f2:
            best_t, best_f2 = t, f2
    return best_t


def evaluate_binary(labels, preds, probs=None):
    """测试集全套指标。labels/preds 为 0/1 数组。"""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", labels=[0, 1], zero_division=0)
    macro_f2 = fbeta_score(labels, preds, beta=2.0, average="macro", labels=[0, 1], zero_division=0)
    p, r, _, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0)
    p0, p1 = p[0], p[1]
    r0, r1 = r[0], r[1]
    auc = roc_auc_score(labels, probs) if (probs is not None and len(np.unique(labels)) > 1) else float("nan")
    return {
        "Acc": acc, "Macro_F1": macro_f1, "Macro_F2": macro_f2,
        "P0": p0, "R0": r0, "P1": p1, "R1": r1, "AUC": auc,
    }


# ---------------------------------------------------------------------
# 显著性检验
# ---------------------------------------------------------------------
def mcnemar_per_fold(main_preds_folds, base_preds_folds, labels_folds):
    """对每个折做 McNemar（连续性校正），返回 [(chi2, p, b, c, n), ...]。
    main_preds_folds/base_preds_folds/labels_folds 为按折的预测/标签列表。"""
    if not (len(main_preds_folds) == len(base_preds_folds) == len(labels_folds)):
        raise ValueError("McNemar 输入折列表长度不一致")
    results = []
    for main_p, base_p, y in zip(main_preds_folds, base_preds_folds, labels_folds):
        main_p = np.asarray(main_p); base_p = np.asarray(base_p); y = np.asarray(y)
        if not (len(main_p) == len(base_p) == len(y)):
            raise ValueError("McNemar 单折预测/标签长度不一致")
        # 仅统计分类器判错的样本（McNemar 用于比较两个分类器）
        main_err = (main_p != y).astype(int)
        base_err = (base_p != y).astype(int)
        b = int(((main_err == 1) & (base_err == 0)).sum())   # 主错、基对
        c = int(((main_err == 0) & (base_err == 1)).sum())   # 主对、基错
        if b + c == 0:
            results.append((0.0, 1.0, b, c, len(y)))
            continue
        # 连续性校正卡方（等价于 McNemar）
        stat = (abs(b - c) - 1.0) ** 2 / (b + c)
        p = 1.0 - _chi2_cdf(stat, 1)
        results.append((stat, p, b, c, len(y)))
    return results


def _chi2_cdf(x, df):
    from scipy.stats import chi2
    return chi2.cdf(x, df)


def wilcoxon_paired(main_scores, base_scores):
    """配对 Wilcoxon（按 fold×seed 逐运行分数）。返回 (stat, p)。"""
    main_scores = np.asarray(main_scores, dtype=float)
    base_scores = np.asarray(base_scores, dtype=float)
    if main_scores.shape != base_scores.shape:
        raise ValueError("Wilcoxon 配对数组 shape 不一致")
    if not np.isfinite(main_scores).all() or not np.isfinite(base_scores).all():
        raise ValueError("Wilcoxon 输入含 NaN/Inf")
    diff = main_scores - base_scores
    if len(diff) < 2 or np.all(diff == 0):
        return (np.nan, 1.0)
    try:
        return wilcoxon(main_scores, base_scores, alternative="two-sided")
    except ValueError:
        return (np.nan, 1.0)


def format_mean_std(vals):
    """'0.7123 ± 0.0184'（保留 4 位）。"""
    vals = np.asarray(vals, dtype=float)
    return f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"


def confusion_matrix(labels, preds):
    """返回 (2,2) 混淆矩阵 [[TN, FP], [FN, TP]]（供可视化）。"""
    from sklearn.metrics import confusion_matrix as _cm
    return _cm(labels, preds, labels=[0, 1])
