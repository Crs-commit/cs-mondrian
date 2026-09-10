# -*- coding: utf-8 -*-
"""OAD-Net 二元版本（诚实版，去掉 4 类软标签 / 语义权重技巧）。

保留原架构的五个组件（作为论文贡献，描述已诚实化）：
  - 正交抗塌缩投影（QR 硬约束，AntiCollapseProjector）
  - 文本锚定交叉注意力（attn_aud / attn_vis，query=text）
  - 自适应掩码（零填充帧不参与注意力/池化）
  - 伪二维卷积（时序 CNN）
  - 残差门控融合（先验通道，仅 MDPE 使用真实 personality/emotion）

模态可配置：按 config.DATASET_MODALITIES[dataset] 只实例化存在的模态投影。
forward 返回 (logits, fusion_feat, aux_feats)：
  - logits     : (B, 2) 二元 logits
  - fusion_feat: (B, D) 分类前融合特征（用于 t-SNE 等）
  - aux_feats  : (t_pool, a_pool, v_pool) 逐模态池化特征，供 OGM-GE 梯度调制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize


class OrthogonalConstraint(nn.Module):
    """对权重执行 QR，返回正交因子 Q（行正交，QQ^T=I）。"""
    def forward(self, weight):
        if weight.shape[0] < weight.shape[1]:
            Q, _ = torch.linalg.qr(weight.T, mode='reduced')
            return Q.T
        else:
            Q, _ = torch.linalg.qr(weight, mode='reduced')
            return Q


class AntiCollapseProjector(nn.Module):
    """正交抗塌缩投影：Linear(d_in, d_out*2) -> LN -> GELU -> Linear(d_out*2, d_out)。"""
    def __init__(self, in_dim, out_dim, use_orthogonal=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2), nn.LayerNorm(out_dim * 2),
            nn.GELU(), nn.Linear(out_dim * 2, out_dim))
        if use_orthogonal:
            parametrize.register_parametrization(self.net[0], "weight", OrthogonalConstraint())
            parametrize.register_parametrization(self.net[3], "weight", OrthogonalConstraint())

    def forward(self, x):
        return self.net(x)


def align_seq(src, tgt):
    """把 src 序列长度对齐到 tgt（短则右补零，长则截断）。"""
    s_len, t_len = src.size(1), tgt.size(1)
    if s_len == t_len:
        return src
    elif s_len < t_len:
        return F.pad(src, (0, 0, 0, t_len - s_len))
    return src[:, :t_len, :]


class OADNet(nn.Module):
    def __init__(self, modality_dims: dict, d=256, aux_hidden=512,
                 use_orthogonal=True, use_cross_attn=True, use_pseudo_cnn=True,
                 use_adaptive_mask=True, use_gated_fusion=True,
                 personality_dim=60, prior_emb_dim=128):
        super().__init__()
        self.d = d
        self.use_cross_attn = use_cross_attn
        self.use_pseudo_cnn = use_pseudo_cnn
        self.use_adaptive_mask = use_adaptive_mask
        self.use_gated_fusion = use_gated_fusion
        self.use_prior = use_gated_fusion  # 门控依赖先验输入

        # 模态投影（只实例化存在的模态）
        self.proj = nn.ModuleDict()
        self.has = {}
        for m, dim in modality_dims.items():
            self.proj[m] = AntiCollapseProjector(dim, d, use_orthogonal=use_orthogonal)
            self.has[m] = True

        self.has_text_llm = self.has.get("text_llm", False)
        self.has_audio_llm = self.has.get("audio_llm", False)

        if use_cross_attn:
            self.attn_aud = nn.MultiheadAttention(d, 8, batch_first=True)
            self.attn_vis = nn.MultiheadAttention(d, 8, batch_first=True)
        self.cross_attn_fallback = nn.Sequential(
            nn.Linear(d * 3, d * 2), nn.GELU(), nn.Linear(d * 2, d))

        if use_pseudo_cnn:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, (3, 1), padding=(1, 0)), nn.BatchNorm2d(64), nn.ReLU())
        else:
            self.cnn_1d_fallback = nn.Sequential(
                nn.Conv1d(d, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU())

        if use_gated_fusion:
            self.pers_gate = nn.Sequential(nn.Linear(64, personality_dim), nn.Sigmoid())
            self.emo_gate = nn.Sequential(nn.Linear(64, prior_emb_dim), nn.Sigmoid())
            cls_in = 64 + personality_dim + prior_emb_dim
        else:
            cls_in = 64

        self.classifier = nn.Sequential(nn.Linear(cls_in, 128), nn.ReLU(), nn.Linear(128, 2))

    # ------------------------------------------------------------------
    def forward(self, b):
        n_mods = (self.has.get("vision", False)
                  + (self.has_text_llm or self.has.get("text_classic", False))
                  + (self.has_audio_llm or self.has.get("audio_classic", False)))
        if n_mods == 1:
            return self._forward_single(b)
        return self._forward_multi(b)

    def _project_text(self, b):
        """投影文本模态（LLM 文本存在则以之为锚，经典文本对齐相加）。"""
        if self.has_text_llm:
            t = self.proj["text_llm"](b["text_llm"])
            if self.has.get("text_classic", False):
                tc = self.proj["text_classic"](b["text_classic"])
                t = t + align_seq(tc, t)
            return t
        return self.proj["text_classic"](b["text_classic"])

    def _project_audio(self, b):
        """投影音频模态（LLM 音频存在则以之为锚，经典音频对齐相加）。"""
        if self.has_audio_llm:
            a = self.proj["audio_llm"](b["audio_llm"])
            if self.has.get("audio_classic", False):
                ac = self.proj["audio_classic"](b["audio_classic"])
                a = a + align_seq(ac, a)
            return a
        return self.proj["audio_classic"](b["audio_classic"])

    def _fuse_gate(self, b, cnn_out):
        """残差门控融合（先验通道）；无先验则返回 cnn_out 本身。"""
        if self.use_gated_fusion:
            p, e = b["personality"], b["emotion"]
            p_g = p + p * self.pers_gate(cnn_out)
            e_g = e + e * self.emo_gate(cnn_out)
            return torch.nan_to_num(
                torch.cat([cnn_out, p_g, e_g], dim=-1), nan=0.0, posinf=10.0, neginf=-10.0)
        return torch.nan_to_num(cnn_out, nan=0.0, posinf=10.0, neginf=-10.0)

    def _forward_multi(self, b):
        v = self.proj["vision"](b["vision"])
        t = self._project_text(b)
        a = self._project_audio(b)

        if self.use_adaptive_mask:
            audio_mask_key = (
                "audio_llm_mask" if self.has_audio_llm else "audio_classic_mask"
            )
            is_padding_a = ~b[audio_mask_key]
            is_padding_v = ~b["vision_mask"]
        else:
            is_padding_a = is_padding_v = None

        if self.use_cross_attn:
            # 文本锚定：以 t 为 query，跨音频/视觉序列做注意力
            a_fused, _ = self.attn_aud(t, a, a, key_padding_mask=is_padding_a)
            v_fused, _ = self.attn_vis(t, v, v, key_padding_mask=is_padding_v)
            fused_3d = a_fused + v_fused
        else:
            mask_a = (~is_padding_a).float().unsqueeze(-1) if self.use_adaptive_mask else torch.ones_like(a)
            mask_v = (~is_padding_v).float().unsqueeze(-1) if self.use_adaptive_mask else torch.ones_like(v)
            a_mean = (a * mask_a).sum(dim=1, keepdim=True) / mask_a.sum(dim=1, keepdim=True).clamp(min=1e-9)
            v_mean = (v * mask_v).sum(dim=1, keepdim=True) / mask_v.sum(dim=1, keepdim=True).clamp(min=1e-9)
            fused_3d = self.cross_attn_fallback(
                torch.cat([t, a_mean.expand(-1, t.size(1), -1), v_mean.expand(-1, t.size(1), -1)], dim=-1))

        if self.use_pseudo_cnn:
            feat_seq = self.cnn(fused_3d.unsqueeze(1)).mean(dim=3)   # (B, 64, T)
        else:
            feat_seq = self.cnn_1d_fallback(fused_3d.transpose(1, 2))

        if self.use_adaptive_mask:
            mask_key = "text_llm_mask" if self.has_text_llm else "text_classic_mask"
            valid = b[mask_key].float().unsqueeze(1)      # 文本长度为锚（LLM 或经典）
            cnn_out = (feat_seq * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1e-9)
        else:
            cnn_out = feat_seq.mean(dim=2)

        aux = (t.mean(dim=1), a.mean(dim=1), v.mean(dim=1))
        fusion_feat = self._fuse_gate(b, cnn_out)
        logits = self.classifier(fusion_feat)
        return logits, fusion_feat, aux

    def _forward_single(self, b):
        """单模态前向：投影单一路径，跳过交叉注意力，直接进时序 CNN。"""
        if self.has_text_llm or self.has.get("text_classic", False):
            feat = self._project_text(b)
            mask_key = "text_llm_mask" if self.has_text_llm else "text_classic_mask"
        elif self.has.get("vision", False):
            feat = self.proj["vision"](b["vision"])
            mask_key = "vision_mask"
        else:
            feat = self._project_audio(b)
            mask_key = "audio_llm_mask" if self.has_audio_llm else "audio_classic_mask"

        if self.use_pseudo_cnn:
            feat_seq = self.cnn(feat.unsqueeze(1)).mean(dim=3)   # (B, 64, T)
        else:
            feat_seq = self.cnn_1d_fallback(feat.transpose(1, 2))

        if self.use_adaptive_mask:
            valid = b[mask_key].float().unsqueeze(1)
            cnn_out = (feat_seq * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1e-9)
        else:
            cnn_out = feat_seq.mean(dim=2)

        fusion_feat = self._fuse_gate(b, cnn_out)
        logits = self.classifier(fusion_feat)
        pool = feat.mean(dim=1)
        aux = (pool, pool, pool)   # 单模态无 OGM-GE，占位保持接口
        return logits, fusion_feat, aux


class SimpleMLP(nn.Module):
    """简单融合基线（作为"复杂架构是否有价值"的下限对照）。

    各模态时序特征按掩码池化 → concat → MLP → 2 分类。不包含 OAD-Net 的
    交叉注意力/伪二维卷积/门控融合，仅做线性融合，用于证明 OAD-Net 的复杂设计带来增益。
    """
    def __init__(self, modality_dims: dict, hidden=(512, 256), dropout=0.3):
        super().__init__()
        self.modality_dims = modality_dims
        in_dim = sum(modality_dims.values())
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.mlp = nn.Sequential(*layers)

    def forward(self, b):
        feats = []
        for m in self.modality_dims:
            x = b[m]
            if x.dim() == 2:
                x = x.unsqueeze(1)
            mask = b.get(f"{m}_mask", None)
            if mask is not None:
                x = (x * mask.float().unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp(min=1e-9)
            else:
                x = x.mean(dim=1)
            feats.append(x)
        fused = torch.cat(feats, dim=-1)
        logits = self.mlp(fused)
        aux = (fused, fused, fused)
        return logits, fused, aux


class TextGatedFusion(nn.Module):
    """文本主导门控融合（针对欺骗场景低信噪比）。

    诊断：欺骗场景下非言语模态（AU/MFCC/CLIP/HuBERT）被说谎者主动压制，简单融合
    （concat/对称注意力/文本锚定）会让噪声稀释文本的强判别力——这已在消融中证实
    （文本单模态 F2 高于任何融合方案）。

    设计：文本模态始终作为主干保留；视觉/音频经"由文本决定的门控"选择性引入。
    门控可学到关闭（g≈0）→ 退化为文本单模态（下限不亏）；或打开（g>0）→ 引入
    非文本的增量信息（有望超过文本单模态）。
    """
    def __init__(self, modality_dims: dict, d=256, use_prior=False,
                 personality_dim=60, prior_emb_dim=8):
        super().__init__()
        self.d = d
        self.use_prior = use_prior
        self.has_text_llm = "text_llm" in modality_dims
        self.has_audio_llm = "audio_llm" in modality_dims
        self.text_key = "text_llm" if self.has_text_llm else "text_classic"
        self.audio_key = "audio_llm" if self.has_audio_llm else "audio_classic"

        # 投影（简单 Linear + LN + GELU；不用正交——正交已证无用）
        self.proj = nn.ModuleDict()
        for m, dim in modality_dims.items():
            self.proj[m] = nn.Sequential(nn.Linear(dim, d), nn.LayerNorm(d), nn.GELU())

        # 文本主干判别头（文本判别力始终保留）
        self.text_head = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        # 门控：由文本决定非文本的引入强度
        self.gate_v = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.gate_a = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())

        cls_in = d
        if use_prior:
            self.pers_gate = nn.Sequential(nn.Linear(d, personality_dim), nn.Sigmoid())
            self.emo_gate = nn.Sequential(nn.Linear(d, prior_emb_dim), nn.Sigmoid())
            cls_in = d + personality_dim + prior_emb_dim
        self.classifier = nn.Sequential(nn.Linear(cls_in, 128), nn.ReLU(), nn.Linear(128, 2))

    def _pool(self, x, mask):
        if mask is not None:
            return (x * mask.float().unsqueeze(-1)).sum(1) / mask.float().sum(1, keepdim=True).clamp(min=1e-9)
        return x.mean(1)

    def forward(self, b):
        # 文本主干
        t = self.proj[self.text_key](b[self.text_key])
        t_pool = self._pool(t, b.get(f"{self.text_key}_mask"))
        t_feat = self.text_head(t_pool)
        # 视觉
        v = self.proj["vision"](b["vision"])
        v_pool = self._pool(v, b.get("vision_mask"))
        # 音频
        a = self.proj[self.audio_key](b[self.audio_key])
        a_pool = self._pool(a, b.get(f"{self.audio_key}_mask"))
        # 门控融合：文本主干 + 门控选择性引入非文本
        g_v = self.gate_v(t_feat)
        g_a = self.gate_a(t_feat)
        fused = t_feat + g_v * v_pool + g_a * a_pool
        # 先验（可选，仅 MDPE）
        if self.use_prior:
            p, e = b["personality"], b["emotion"]
            p_g = p + p * self.pers_gate(fused)
            e_g = e + e * self.emo_gate(fused)
            fused = torch.nan_to_num(torch.cat([fused, p_g, e_g], dim=-1),
                                     nan=0.0, posinf=10.0, neginf=-10.0)
        else:
            fused = torch.nan_to_num(fused, nan=0.0, posinf=10.0, neginf=-10.0)
        logits = self.classifier(fused)
        aux = (t_feat, a_pool, v_pool)   # 占位，保持接口
        return logits, fused, aux


class TextSafeBehavioralResidual(nn.Module):
    """Text-Safe Behavioral Residual Learning (TS-BRL).

    文本分支输出主 logits，音视频分支只输出行为残差：
        final_logits = text_logits + gate * residual_logits

    gate 由文本置信度、音视频表征和文本/音视频预测冲突共同决定。默认使用 tanh
    限制残差幅度，避免弱噪声模态大幅覆盖文本判断。
    """
    def __init__(self, modality_dims: dict, d=256, use_prior=False,
                 personality_dim=60, prior_emb_dim=8,
                 use_gate=True, bound_residual=True, residual_max_scale=1.0):
        super().__init__()
        self.d = d
        self.use_prior = use_prior
        self.use_gate = use_gate
        self.bound_residual = bound_residual
        self.residual_max_scale = residual_max_scale
        self.has_text_llm = "text_llm" in modality_dims
        self.has_audio_llm = "audio_llm" in modality_dims
        self.text_key = "text_llm" if self.has_text_llm else "text_classic"
        self.audio_key = "audio_llm" if self.has_audio_llm else "audio_classic"

        self.proj = nn.ModuleDict()
        for m, dim in modality_dims.items():
            self.proj[m] = nn.Sequential(nn.Linear(dim, d), nn.LayerNorm(d), nn.GELU())

        self.text_encoder = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(0.2))
        self.text_classifier = nn.Linear(d, 2)

        self.av_encoder = nn.Sequential(
            nn.Linear(d * 3, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.2))
        self.av_probe = nn.Linear(d, 2)

        prior_extra = d if use_prior else 0
        self.prior_proj = nn.Sequential(
            nn.Linear(personality_dim + prior_emb_dim, d), nn.LayerNorm(d), nn.GELU()
        ) if use_prior else None

        residual_in = d * 2 + prior_extra
        self.residual_head = nn.Sequential(
            nn.Linear(residual_in, d), nn.ReLU(), nn.Dropout(0.2), nn.Linear(d, 2))
        self.gate = nn.Sequential(
            nn.Linear(d * 2 + prior_extra + 2, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
            nn.Sigmoid(),
        )

    def _pool(self, x, mask):
        if mask is not None:
            valid = mask.float().unsqueeze(-1)
            return (x * valid).sum(1) / valid.sum(1).clamp(min=1e-9)
        return x.mean(1)

    def _prior_context(self, b):
        if not self.use_prior:
            return None
        prior = torch.cat([b["personality"], b["emotion"]], dim=-1)
        return self.prior_proj(prior)

    def forward(self, b):
        t = self.proj[self.text_key](b[self.text_key])
        t_pool = self._pool(t, b.get(f"{self.text_key}_mask"))
        text_feat = self.text_encoder(t_pool)
        text_logits = self.text_classifier(text_feat)

        v = self.proj["vision"](b["vision"])
        v_pool = self._pool(v, b.get("vision_mask"))

        a = self.proj[self.audio_key](b[self.audio_key])
        a_pool = self._pool(a, b.get(f"{self.audio_key}_mask"))

        av_feat = self.av_encoder(
            torch.cat([a_pool, v_pool, torch.abs(a_pool - v_pool)], dim=-1))
        av_logits = self.av_probe(av_feat)

        prior_feat = self._prior_context(b)
        residual_inputs = [text_feat, av_feat]
        if prior_feat is not None:
            residual_inputs.append(prior_feat)
        residual_raw = self.residual_head(torch.cat(residual_inputs, dim=-1))
        if self.bound_residual:
            residual_logits = torch.tanh(residual_raw) * self.residual_max_scale
        else:
            residual_logits = residual_raw

        text_prob = F.softmax(text_logits, dim=-1)
        av_prob = F.softmax(av_logits, dim=-1)
        text_conf = text_prob.max(dim=-1, keepdim=True).values.detach()
        conflict = torch.abs(text_prob[:, 1:2] - av_prob[:, 1:2]).detach()

        gate_inputs = [text_feat, av_feat]
        if prior_feat is not None:
            gate_inputs.append(prior_feat)
        gate_inputs.extend([text_conf, conflict])
        gate = self.gate(torch.cat(gate_inputs, dim=-1)) if self.use_gate else torch.ones_like(text_conf)

        logits = text_logits + gate * residual_logits
        fusion_feat = torch.cat([text_feat, av_feat, gate], dim=-1)
        aux = {
            "text_logits": text_logits,
            "av_logits": av_logits,
            "residual_raw": residual_raw,
            "residual_logits": residual_logits,
            "gate": gate,
            "residual_bounded": self.bound_residual,
            "modality_feats": (text_feat, a_pool, v_pool),
        }
        fusion_feat = torch.nan_to_num(fusion_feat, nan=0.0, posinf=10.0, neginf=-10.0)
        return logits, fusion_feat, aux
