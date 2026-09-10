# -*- coding: utf-8 -*-
"""双数据集二区重跑 —— 全局配置（单点改，处处生效）。

两个数据集分开跑（跨数据集验证，不联合训练）：
  - SEU-MLD: AU + text(768) + MFCC，无心理先验
  - MDPE  : clip + baichuan + sbert + wavlm + egemaps，有真实 personality/emotion 先验
"""
import os

# =====================================================================
# 路径（服务器 / 本地）
#
# 公开版不含实际部署路径。运行时通过环境变量注入数据根目录：
#   export OADNET_SERVER_ROOT=/path/to/My_Deception_Project
# 该根目录下需包含（详见本目录 README「数据准备」）：
#   data/SEUMLD/SEUMLD/                 SEU-MLD 原始数据（Preprocess/Video|Audio）
#   data/processed/seumld/              SEU-MLD 特征（{vid}_au/text/mfcc.npy）
#   data/processed/mdpe/                MDPE 特征（{vid}.npy / {vid}.csv）
#   data/mdpe_raw/                      MDPE 原始特征下载目录
#   data/*_with_soft_labels.csv         索引/标注 CSV
#   data/personality.csv                MDPE 人格量表
# =====================================================================
SERVER_ROOT = os.environ.get("OADNET_SERVER_ROOT", "<SERVER_ROOT>")

# SEU-MLD 原始数据根目录（包含 Preprocess/Video、Preprocess/Audio）
SEUMLD_ROOT = os.path.join(SERVER_ROOT, "data", "SEUMLD", "SEUMLD")

# 特征输出目录
FEAT_ROOT = os.path.join(SERVER_ROOT, "data", "processed")
FEAT_SEUMLD = os.path.join(FEAT_ROOT, "seumld")       # {vid}_au.npy / {vid}_text.npy / {vid}_mfcc.npy（单目录）
FEAT_MDPE   = os.path.join(SERVER_ROOT, "data", "processed", "mdpe")

# MDPE 原始特征根目录（文本/视觉/音频的都从一个 mdpe_raw 下读取）
MDPE_RAW   = os.path.join(SERVER_ROOT, "data", "mdpe_raw")   # 内含 baichuan13B-base/ clipVIT-B16/ chinese-hubert-base/ ...
# MDPE 模态 → 特征目录名（与 HF 解压后的目录一致）
MDPE_MODALITY_SUBDIRS = {
    "vision": "clipVIT-B16",           # 视觉特征（也试 clipVIT-L14 作对照）
    "text_llm": "baichuan13B-base",    # 文本（论文最强）
    "audio_llm": "chinese-hubert-base",# 音频（论文最优组合用 HuBERT-base）
}
# 各模态特征文件名用什么分隔符（MDPE 命名混乱：clipVIT 用下划线，其它连字符）
MDPE_NAME_SEP = {
    "vision": "_",          # clipVIT-B16: 001_10_8.npy
    "text_llm": "-",        # baichuan13B-base: 001-1-1.csv
    "audio_llm": "-",       # chinese-hubert-base: 001-1-1.npy
}
# 各模态特征文件扩展名
MDPE_MODALITY_EXT = {
    "vision": ".npy",
    "text_llm": ".csv",
    "audio_llm": ".npy",
}

# 索引 / 标注 CSV
CSV_SEUMLD = os.path.join(SERVER_ROOT, "data", "seumld_final_dataset_with_soft_labels.csv")
CSV_MDPE   = os.path.join(SERVER_ROOT, "data", "mdpe_final_dataset_with_expert_soft_labels.csv")
CSV_PERS   = os.path.join(SERVER_ROOT, "data", "personality.csv")

# 结果目录（本代码运行的工作目录 = rerun/）
RESULTS_DIR = "results"

# =====================================================================
# 每数据集模态配置：仅列出该数据集实际存在的模态
#   维度 = 特征文件最后一维（缺失时按此填充到统一序列长度，见 data.py）
# =====================================================================
DATASET_MODALITIES = {
    "SEUMLD": {
        "vision": 52,          # MediaPipe blendshapes AU
        "text_classic": 768,   # text2vec-base-chinese
        "audio_classic": 60,   # MFCC+delta+delta2
        # 无 text_llm / audio_llm（SEU-MLD 未发布 LLM 级特征）
        "use_prior": False,    # SEU-MLD 无心理/情绪标注 → 不设先验通道
    },
    "MDPE": {
        "vision": 512,         # CLIP ViT-B16 (float16, 需转float32)
        "text_llm": 5120,      # Baichuan-13B 文本（5120 维 csv）
        "audio_llm": 768,      # HuBERT-base 音频（768 维 npy）
        # 暂不含 text_classic/audio_classic（MDPE 核心组合用 LLM/深度特征）
        "use_prior": True,     # MDPE 有真实 personality(60) + emotion(8) 先验
    },
    # ---- 单模态基线（消融：证明多模态融合 > 单打）----
    "SEUMLD_text":   {"text_classic": 768, "use_prior": False},
    "SEUMLD_vision": {"vision": 52,        "use_prior": False},
    "SEUMLD_audio":  {"audio_classic": 60, "use_prior": False},
    "MDPE_text":   {"text_llm": 5120, "use_prior": True},
    "MDPE_vision": {"vision": 512,    "use_prior": True},
    "MDPE_audio":  {"audio_llm": 768, "use_prior": True},
}

# =====================================================================
# 实验协议（二区标准）
# =====================================================================
N_FOLDS = 5            # 受试者级 StratifiedGroupKFold
SEEDS   = [42, 123, 2024, 7, 2026]   # 多种子，5 个
FOLD_RANDOM_STATE = 42               # 折叠划分固定（跨种子一致，便于配对检验）
VAL_FRACTION = 0.10                  # 每折内，从训练受试者中划出【模型选择】验证集比例（早停+阈值，受试者级）
CALIB_FRACTION = 0.25                # 每折内，从训练受试者中划出【conformal 校准集】比例（受试者级，与 val 不相交）
                                      # 0.15→0.25：Mondrian 谎言类校准样本约 145→240，覆盖 std ~3pt→~1.5pt
                                      # 四路划分 train/val/calib/test；校准独立于模型选择。
                                      # 受试者内片段相关，因此覆盖结果按经验片段级解释。

# 阈值搜索：在验证集上最大化 Macro-F2
THRESH_GRID_START, THRESH_GRID_STOP, THRESH_GRID_STEP = 0.05, 0.95, 0.05

# DataLoader 并行（服务器 25 核；特征内存缓存后 IO 大幅下降，worker 不必拉满）
#   单进程可 12；SEU-MLD 与 MDPE 并发跑时每进程 8（2×8+主进程 ≈ 25 核安全）。本地冒烟测试覆盖为 0。
NUM_WORKERS = 4

# =====================================================================
# 训练超参（沿用原协议并适配二元任务）
# =====================================================================
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
EARLY_STOP_PATIENCE = 12
GRAD_CLIP = 1.0
TEMP = 0.85                # 推理期温度缩放（锐化后验）

# 类别不平衡基线可切换的损失
LOSSES = ["CE", "Focal", "CB", "LDAM", "Kang", "OGM_GE"]   # 全部候选
FOCAL_GAMMA = 2.0
CB_BETA = 0.999            # Class-Balanced 有效样本数 beta
LDAM_MAX_MARGIN = 0.5      # LDAM 最大间隔（Cao et al. 2019 默认 0.5）
OGM_ALPHA = 0.8            # OGM-GE 梯度调制强度
OGM_AUX_LAMBDA = 0.1       # OGM-GE 辅助损失权重
OGM_GAMMA = 2.0            # OGM-GE 辅助损失 gamma（原实现用 CE，保留为 CE 即可）

# 文本教师蒸馏（教师在每个 fold/seed 内独立训练）
KD_WEIGHT = 0.5
KD_TEMPERATURE = 2.0

# Text-Safe Behavioral Residual Learning (TS-BRL)
# 文本 logits 作为主预测，音视频只通过受限残差修正文本判断。
TSBRL_TEXT_LOSS_WEIGHT = 0.5
TSBRL_RESIDUAL_L2_WEIGHT = 0.02
TSBRL_RESIDUAL_MAX_SCALE = 1.0

# 主模型默认损失（硬标签 CE；不加权，与旧最终协议 CLASS_PENALTY_WEIGHTS=全1.0 一致，
# 也使 Focal/CB/LDAM/Kang 各自作为"单变量对照"——只改损失/训练方案）
MAIN_LOSS = "CE"
USE_CLASS_WEIGHTS = False   # 主 CE 是否加逆频权重；False 为不加权（与基线对照更干净）

# =====================================================================
# 模型
# =====================================================================
MODEL_DIM = 256            # 统一投影维度 d
AUX_HIDDEN = 512           # AntiCollapseProjector 中间维
USE_ORTHOGONAL = True      # 正交抗塌缩投影（QR 硬约束）
USE_CROSS_ATTN = True      # 文本锚定交叉注意力
USE_PSEUDO_CNN = True      # 伪二维卷积（时序 CNN）
USE_ADAPTIVE_MASK = True   # 自适应掩码（零填充帧不参与池化/注意力）
USE_GATED_FUSION = True    # 先验门控融合（仅 MDPE 开启；SEU-MLD 无先验则关闭）
USE_AMP = True             # 混合精度训练（FP16 autocast + GradScaler；确定性，指标与 FP32 差异在噪声内）
MLP_HIDDEN = [512, 256]    # SimpleMLP baseline 隐藏层维度
PRIOR_EMB_DIM = 8          # 情绪先验维：8 类情绪强度自评（1-5）均值（受试者级）
                           # ⚠ 不是 HF emotion_features（70GB 深度特征）——已确认用标注即可，免下载
PERSONALITY_DIM = 60       # 人格向量维（MDPE personality，BFI-2 前 60 题 1-5 分）
EMO_LABELS_DIR = os.path.join(MDPE_RAW, "emotion", "labels")
                           # 193 受试者 × 16 情绪诱导视频 × 8 类强度（labels.zip 解压目录）

# =====================================================================
# 显著性检验
# =====================================================================
SIG_TEST = "mcnemar"       # 主模型 vs 基线：按折配对 McNemar（p<0.05 标 *)
# 另附按 (fold×seed) 逐运行指标的配对 Wilcoxon（在 experiments.py 输出）

def get_dataset_cfg(dataset: str) -> dict:
    return DATASET_MODALITIES[dataset]


def base_dataset(dataset: str) -> str:
    """单模态变体名（SEUMLD_text / MDPE_vision）映射回基础数据集名。"""
    return "MDPE" if str(dataset).startswith("MDPE") else "SEUMLD"
