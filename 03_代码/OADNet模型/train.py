# -*- coding: utf-8 -*-
"""单次训练评估：一个 (dataset, loss, fold, seed) 的训练+验证+测试。

返回 ModelResult，含测试集全套指标与逐样本预测（供配对检验）。
Kang 解耦重训在 loss_type="Kang" 时启用两阶段训练。
OGM_GE 在 loss_type="OGM_GE" 时启用逐模态梯度调制。
"""
import contextlib
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataclasses import dataclass, field
from collections import deque
from sklearn.metrics import fbeta_score, accuracy_score, precision_recall_fscore_support

import config


def _autocast(device):
    """AMP 混合精度上下文；CPU（冒烟测试）时退化为 no-op。"""
    if config.USE_AMP and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return contextlib.nullcontext()
from data import DeceptionDataset, collate_fn
from model import OADNet, SimpleMLP, TextGatedFusion, TextSafeBehavioralResidual
from losses import (inverse_freq_weights, FocalLoss, ClassBalancedLoss,
                    LDAMLoss, compute_ogm_coeffs)
from metrics import threshold_selection, evaluate_binary, probs_from_logits


@dataclass
class ModelResult:
    config_name: str
    dataset: str
    loss: str
    fold: int
    seed: int
    metrics: dict = field(default_factory=dict)
    threshold: float = 0.5
    test_preds: np.ndarray = None
    test_probs: np.ndarray = None
    test_labels: np.ndarray = None
    val_preds: np.ndarray = None
    val_probs: np.ndarray = None
    val_labels: np.ndarray = None
    calib_probs: np.ndarray = None
    calib_labels: np.ndarray = None
    epochs: int = 0
    model_state: dict = None


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: (move_to_device(v, device) if not isinstance(v, list) else v)
                for k, v in batch.items()}
    return batch


def make_loaders(dataset, df, train_idx, val_idx, test_idx, seed, feat_root, prior_csv):
    """构建 train(均匀采样)/val/test 三个 DataLoader。

    注意：主模型与 CE/Focal/CB/LDAM/OGM_GE 均用均匀采样（单变量对照——只改损失）。
    Kang 解耦重训的阶段2在训练循环内切换到类别平衡采样。
    """
    train_ds = DeceptionDataset(dataset, df, train_idx, feat_root, prior_csv)
    val_ds = DeceptionDataset(dataset, df, val_idx, feat_root, prior_csv)
    test_ds = DeceptionDataset(dataset, df, test_idx, feat_root, prior_csv)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              generator=g, collate_fn=collate_fn, num_workers=config.NUM_WORKERS,
                              pin_memory=True, persistent_workers=(config.NUM_WORKERS > 0))
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=config.NUM_WORKERS,
                            pin_memory=True, persistent_workers=(config.NUM_WORKERS > 0))
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=config.NUM_WORKERS,
                             pin_memory=True, persistent_workers=(config.NUM_WORKERS > 0))
    return train_loader, val_loader, test_loader


def make_balanced_train_loader(dataset, df, train_idx, seed, feat_root, prior_csv):
    """Kang 阶段2：类别平衡采样器（每类权重 ∝ 1/该类样本数）。"""
    train_ds = DeceptionDataset(dataset, df, train_idx, feat_root, prior_csv)
    train_labels = df.iloc[train_idx]["label"].values
    cnt = np.bincount(train_labels, minlength=2)
    bal_weights = np.array([1.0 / max(cnt[c], 1) for c in train_labels], dtype=float)
    sampler = WeightedRandomSampler(weights=bal_weights, num_samples=len(bal_weights),
                                    replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    return DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler,
                      collate_fn=collate_fn, num_workers=config.NUM_WORKERS,
                      pin_memory=True, persistent_workers=(config.NUM_WORKERS > 0))


def make_model(dataset, device, ablate=None, use_mlp=False, use_text_gated=False,
               use_text_safe=False, text_safe_ablate=None):
    cfg = config.get_dataset_cfg(dataset)
    modality_dims = {m: cfg[m] for m in cfg
                     if m.endswith(("vision", "text_llm", "text_classic", "audio_llm", "audio_classic"))}
    if use_text_safe:
        return TextSafeBehavioralResidual(
            modality_dims, d=config.MODEL_DIM,
            use_prior=cfg.get("use_prior", False),
            personality_dim=config.PERSONALITY_DIM,
            prior_emb_dim=config.PRIOR_EMB_DIM,
            use_gate=(text_safe_ablate != "no_gate"),
            bound_residual=(text_safe_ablate != "no_residual_bound"),
            residual_max_scale=config.TSBRL_RESIDUAL_MAX_SCALE).to(device)
    if use_text_gated:
        return TextGatedFusion(modality_dims, d=config.MODEL_DIM,
                               use_prior=cfg.get("use_prior", False),
                               personality_dim=config.PERSONALITY_DIM,
                               prior_emb_dim=config.PRIOR_EMB_DIM).to(device)
    if use_mlp:
        return SimpleMLP(modality_dims, hidden=config.MLP_HIDDEN).to(device)
    # 组件消融：ablate 指定关掉的组件（orthogonal/cross_attn/adaptive_mask/pseudo_cnn），其余保留
    use_orthogonal = config.USE_ORTHOGONAL and (ablate != "orthogonal")
    use_cross_attn = config.USE_CROSS_ATTN and (ablate != "cross_attn")
    use_pseudo_cnn = config.USE_PSEUDO_CNN and (ablate != "pseudo_cnn")
    use_adaptive_mask = config.USE_ADAPTIVE_MASK and (ablate != "adaptive_mask")
    model = OADNet(
        modality_dims=modality_dims,
        d=config.MODEL_DIM, aux_hidden=config.AUX_HIDDEN,
        use_orthogonal=use_orthogonal,
        use_cross_attn=use_cross_attn,
        use_pseudo_cnn=use_pseudo_cnn,
        use_adaptive_mask=use_adaptive_mask,
        use_gated_fusion=cfg["use_prior"] and config.USE_GATED_FUSION,
        personality_dim=config.PERSONALITY_DIM,
        prior_emb_dim=config.PRIOR_EMB_DIM,
    ).to(device)
    return model


def build_loss(loss_type, train_labels, device):
    cnt = np.bincount(np.asarray(train_labels), minlength=2).astype(float)
    ifw = inverse_freq_weights(train_labels).to(device) if config.USE_CLASS_WEIGHTS else None
    if loss_type in ("CE", "KD", "ConfKD"):
        return nn.CrossEntropyLoss(weight=ifw)
    if loss_type == "Focal":
        return FocalLoss(weight=inverse_freq_weights(train_labels).to(device),
                         gamma=config.FOCAL_GAMMA)
    if loss_type == "CB":
        return ClassBalancedLoss(cls_num_list=cnt, beta=config.CB_BETA).to(device)
    if loss_type == "LDAM":
        return LDAMLoss(cls_num_list=cnt, max_m=config.LDAM_MAX_MARGIN, s=30.0).to(device)
    if loss_type == "Kang":
        return nn.CrossEntropyLoss(weight=ifw)   # 阶段1实例平衡，阶段2见训练循环
    if loss_type == "OGM_GE":
        return nn.CrossEntropyLoss(weight=ifw)   # 主损失 + OGM 辅助调制
    raise ValueError(f"unknown loss {loss_type}")


def text_safe_regularized_loss(criterion, logits, aux, labels):
    """TS-BRL: final CE + 文本分支 CE + 残差幅度约束。"""
    loss = criterion(logits, labels)
    loss = loss + config.TSBRL_TEXT_LOSS_WEIGHT * criterion(aux["text_logits"], labels)
    if aux.get("residual_bounded", True):
        loss = loss + config.TSBRL_RESIDUAL_L2_WEIGHT * aux["residual_logits"].pow(2).mean()
    return loss


def build_optimizer(model, ogm_aux=None):
    no_decay = ["bias", "LayerNorm.weight", "BatchNorm1d.weight", "BatchNorm2d.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": config.WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if ogm_aux is not None:
        groups.append({"params": ogm_aux.parameters()})
    return optim.AdamW(groups, lr=config.LEARNING_RATE)


def evaluate_loader(model, loader, device, with_feat=False, zero_prior=False):
    model.eval()
    logits_list, labels_list, feat_list = [], [], []
    with torch.no_grad(), _autocast(device):
        for batch in loader:
            batch = move_to_device(batch, device)
            if zero_prior and "personality" in batch:
                batch["personality"] = torch.zeros_like(batch["personality"])
                batch["emotion"] = torch.zeros_like(batch["emotion"])
            out = model(batch)
            logits, feat = out[0], out[1]
            logits_list.append(logits.cpu())
            labels_list.extend(batch["labels"].cpu().numpy())
            if with_feat:
                feat_list.append(feat.cpu())
    all_logits = torch.cat(logits_list, dim=0)
    all_feat = torch.cat(feat_list, dim=0) if feat_list else None
    return all_logits, np.asarray(labels_list), all_feat


def train_one_run(dataset, df, train_idx, val_idx, test_idx, seed,
                  calib_idx=None,
                  loss_type="CE", zero_prior=False, feat_root=None, prior_csv=None,
                  log_dir=None, config_name=None, ablate=None, use_mlp=False,
                  use_text_gated=False, use_text_safe=False, text_safe_ablate=None, fold=None,
                  teacher_state=None, return_state=False):
    """训练+验证+测试单次运行；KD 配置使用已冻结的文本教师。"""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = make_loaders(
        dataset, df, train_idx, val_idx, test_idx, seed, feat_root, prior_csv)

    model = make_model(
        dataset, device, ablate=ablate, use_mlp=use_mlp,
        use_text_gated=use_text_gated,
        use_text_safe=use_text_safe,
        text_safe_ablate=text_safe_ablate)
    teacher_model = None
    if loss_type in ("KD", "ConfKD"):
        if teacher_state is None:
            raise ValueError("KD/ConfKD 必须提供按 fold/seed 训练得到的 teacher_state")
        teacher_model = make_model(f"{config.base_dataset(dataset)}_text", device)
        teacher_model.load_state_dict(teacher_state, strict=True)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
    ogm_aux = nn.Linear(config.MODEL_DIM, 2, bias=False).to(device) if loss_type == "OGM_GE" else None
    criterion = build_loss(loss_type, df.iloc[train_idx]["label"].values, device)
    optimizer = build_optimizer(model, ogm_aux)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE)

    # AMP：GradScaler（CPU 冒烟时 no-op；torch>=1.10 用 torch.amp.GradScaler，兼容新版）
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP and device.type == "cuda")

    # 训练曲线 / 融合特征落盘目录（按基础数据集归目录，供可视化）
    base = config.base_dataset(dataset)
    res_dir = os.path.join(config.RESULTS_DIR, base)
    curves_dir = os.path.join(res_dir, "curves")
    feats_dir = os.path.join(res_dir, "feats")
    ckpt_dir = os.path.join(res_dir, "ckpt")
    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(feats_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_score, best_state, best_thresh = 0.0, None, 0.5
    recent = deque(maxlen=3)
    early = 0
    epochs_run = 0
    total_epochs = config.EPOCHS
    curve_rows = []   # (epoch, train_loss, val_f2)
    f_tag = fold if fold is not None else -1
    # Kang：两阶段。阶段2只训分类头 + 平衡采样，前 total_epochs//2 轮训特征。
    kang_stage2_start = config.EPOCHS // 2 if loss_type == "Kang" else 10 ** 9

    for epoch in range(1, total_epochs + 1):
        epochs_run = epoch
        # ---- Kang 阶段2：冻结骨干，仅训分类头 + 类别平衡采样 ----
        if epoch == kang_stage2_start:
            for n, p in model.named_parameters():
                p.requires_grad = False
            for p in model.classifier.parameters():
                p.requires_grad = True
            if loss_type == "Kang":
                train_loader = make_balanced_train_loader(dataset, df, train_idx, seed,
                                                          feat_root, prior_csv)
            # 重建优化器（只含可训练参数）与 scheduler（旧 scheduler 指向旧优化器，必须重建）
            optimizer = build_optimizer(model, None)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=config.SCHEDULER_FACTOR,
                patience=config.SCHEDULER_PATIENCE)

        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = move_to_device(batch, device)
            if zero_prior and "personality" in batch:
                batch["personality"] = torch.zeros_like(batch["personality"])
                batch["emotion"] = torch.zeros_like(batch["emotion"])
            optimizer.zero_grad()
            with _autocast(device):
                out = model(batch)
                logits, _, aux = out
                if loss_type in ("KD", "ConfKD"):
                    with torch.no_grad():
                        teacher_batch = dict(batch)
                        if zero_prior and "personality" in teacher_batch:
                            teacher_batch["personality"] = torch.zeros_like(batch["personality"])
                            teacher_batch["emotion"] = torch.zeros_like(batch["emotion"])
                        teacher_logits = teacher_model(teacher_batch)[0]
                        teacher_prob = F.softmax(teacher_logits / config.KD_TEMPERATURE, dim=-1)
                        if loss_type == "ConfKD":
                            entropy = -(teacher_prob * teacher_prob.clamp_min(1e-8).log()).sum(dim=-1)
                            confidence = 1.0 - entropy / np.log(2.0)
                        else:
                            confidence = torch.ones_like(teacher_prob[:, 0])
                    kd_each = F.kl_div(
                        F.log_softmax(logits / config.KD_TEMPERATURE, dim=-1),
                        teacher_prob, reduction="none").sum(dim=-1)
                    kd_loss = (confidence * kd_each).sum() / confidence.sum().clamp_min(1e-8)
                    loss = criterion(logits, batch["labels"]) + config.KD_WEIGHT * (
                        config.KD_TEMPERATURE ** 2) * kd_loss
                elif loss_type == "OGM_GE":
                    # 逐模态梯度调制：以主损失 CE + 辅助分类器 OGM 调制
                    t_f, a_f, v_f = aux
                    ml = [ogm_aux(t_f), ogm_aux(a_f), ogm_aux(v_f)]
                    coeffs = compute_ogm_coeffs(ml, batch["labels"], alpha=config.OGM_ALPHA)
                    aux_loss = sum((c.detach() * criterion(m, batch["labels"])).mean()
                                   for c, m in zip(coeffs, ml))
                    loss = criterion(logits, batch["labels"]) + config.OGM_AUX_LAMBDA * aux_loss
                elif use_text_safe:
                    loss = text_safe_regularized_loss(criterion, logits, aux, batch["labels"])
                else:
                    loss = criterion(logits, batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # ---- 验证：网格选阈值（最大化 Macro-F2） ----
        val_logits, val_labels, _ = evaluate_loader(model, val_loader, device, zero_prior=zero_prior)
        thresh = threshold_selection(val_logits, val_labels)
        p = probs_from_logits(val_logits)
        val_preds = (p > thresh).astype(int)
        val_f2 = fbeta_score(val_labels, val_preds, beta=2.0, average="macro",
                             labels=[0, 1], zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)
        val_pr, val_rc, _, _ = precision_recall_fscore_support(
            val_labels, val_preds, labels=[0, 1], zero_division=0)
        curve_rows.append((epoch, train_loss, val_f2, val_acc, val_pr[1], val_rc[1]))
        recent.append(val_f2)
        smoothed = sum(recent) / len(recent)
        scheduler.step(smoothed)
        if smoothed > best_score:
            best_score = smoothed
            best_thresh = thresh
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ogm_aux is not None:
                best_state["ogm_aux.weight"] = ogm_aux.weight.detach().cpu().clone()
            early = 0
        else:
            early += 1
            if early >= config.EARLY_STOP_PATIENCE:
                break

    # ---- 测试 ----
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_thresh = 0.5
    model.load_state_dict({k: v for k, v in best_state.items() if k != "ogm_aux.weight"}, strict=False)
    if ogm_aux is not None and "ogm_aux.weight" in best_state:
        ogm_aux.weight.data = best_state["ogm_aux.weight"].to(device)

    val_logits, val_labels, _ = evaluate_loader(
        model, val_loader, device, zero_prior=zero_prior)
    val_probs = probs_from_logits(val_logits)
    val_preds = (val_probs > best_thresh).astype(int)

    test_logits, test_labels, test_feat = evaluate_loader(
        model, test_loader, device, with_feat=True, zero_prior=zero_prior)
    test_probs = probs_from_logits(test_logits)
    test_preds = (test_probs > best_thresh).astype(int)
    metrics = evaluate_binary(test_labels, test_preds, test_probs)

    # 独立校准集前向（受试者与 val 不相交）：供 conformal 校准。
    # 受试者内片段相关，结果按经验片段级覆盖解释。
    calib_probs = calib_labels = None
    if calib_idx is not None:
        calib_ds = DeceptionDataset(dataset, df, calib_idx, feat_root, prior_csv)
        calib_loader = DataLoader(calib_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                                  collate_fn=collate_fn, num_workers=0)
        calib_logits, calib_labels, _ = evaluate_loader(model, calib_loader, device, zero_prior=zero_prior)
        calib_probs = probs_from_logits(calib_logits)

    # 落盘训练曲线 + 融合特征 + 模型 checkpoint（全落盘，任何时点可恢复/复现）
    name = config_name or f"{dataset}_{loss_type}"
    pd.DataFrame(curve_rows, columns=["epoch", "train_loss", "val_f2", "val_acc", "val_p1", "val_r1"]).to_csv(
        os.path.join(curves_dir, f"{name}_f{f_tag}_s{seed}.csv"), index=False, encoding="utf-8-sig")
    if test_feat is not None:
        np.savez(os.path.join(feats_dir, f"{name}_f{f_tag}_s{seed}.npz"),
                 feat=test_feat, labels=test_labels)
    torch.save(best_state, os.path.join(ckpt_dir, f"{name}_f{f_tag}_s{seed}.pt"))

    result = ModelResult(
        config_name=name,
        dataset=dataset, loss=loss_type, fold=-1, seed=seed,
        metrics=metrics, threshold=best_thresh,
        test_preds=test_preds, test_probs=test_probs, test_labels=test_labels,
        val_preds=val_preds, val_probs=val_probs, val_labels=val_labels,
        calib_probs=calib_probs, calib_labels=calib_labels,
        epochs=epochs_run,
        model_state=best_state if return_state else None)
    return result
