# OAD-Net 基准预测器（双数据集二元版）

> 对应主稿：**CS-Mondrian v1.2**
> 最后修订：2026-09-10
> 本目录是 CS-Mondrian 复现链路的**上游**：产出 CS-Mondrian 所需的基础预测张量（NPZ）。

---

## 一、本目录在复现链路中的位置

```
受限数据（SEU-MLD / MDPE）+ 预训练特征
        │
        ├─ [本目录] OAD-Net 训练  ──► 每 config × fold × seed 的 NPZ 预测
        │                             （probs / labels / calib_* / val_*）
        ▼
[03_代码/01_核心严格重算] + [02_Singh对照臂] + [03_替代调度消融] + [04_辅助脚本]
        │
        ▼
[05_结果] 汇总数字（论文表格的最终来源）
```

| 链路环节 | 能否用本仓库直接复现 |
|---|---|
| **NPZ 预测 → CS-Mondrian 共形拒判 → 汇总结果** | ✅ 可以。`03_代码/01` 起的所有脚本只读 NPZ，不读原始数据 |
| **受限数据 → OAD-Net 训练 → NPZ 预测** | ⚠️ 本目录提供**完整训练代码**，但受控数据集需自行申请授权（见 §五） |

**关键说明**：基础预测器（OAD-Net）的输出 NPZ 由受控数据集派生，受数据使用协议约束，
因此不随本仓库分发。本目录公开的是**训练这套预测器的代码**，使具备数据授权的第三方
能够独立复现 NPZ，进而复现 CS-Mondrian 的全部结论。

---

## 二、文件清单

| 文件 | 作用 |
|---|---|
| `config.py` | 全局配置：数据根目录（环境变量注入）、模态维度、实验协议、训练超参 |
| `data.py` | 双数据集加载、受试者级 `StratifiedGroupKFold` 划分、特征完整性检查、`Dataset` / `collate_fn` |
| `model.py` | OAD-Net 二元版（模态可配置） |
| `losses.py` | CE / Focal / Class-Balanced / LDAM / Kang / OGM-GE 系数 |
| `metrics.py` | Acc / Macro-F1 / Macro-F2 / 两类 P·R / AUC，阈值网格搜索，McNemar / Wilcoxon |
| `train.py` | 单次 (fold × seed) 训练与评估 |
| `experiments.py` | 实验编排：跑完整配置矩阵 → `summary` / `significance` / `report` / `preds` NPZ |

### OAD-Net 架构（`model.py`）

保留五个组件，均可在 `config.py` 中单独开关：

1. **正交抗塌缩投影**（`AntiCollapseProjector`，QR 硬约束）
2. **文本锚定交叉注意力**（`attn_aud` / `attn_vis`，query = text）
3. **自适应掩码**（零填充帧不参与注意力/池化）
4. **伪二维卷积**（时序 CNN）
5. **残差门控融合**（先验通道，仅 MDPE 使用真实 personality / emotion）

`forward` 返回 `(logits, fusion_feat, aux_feats)`；`aux_feats` 供 OGM-GE 梯度调制使用。

### 实验配置矩阵（`experiments.py::experiment_matrix`）

| 类别 | 配置名 |
|---|---|
| 主模型 | `OADNet_CE`（SEU-MLD）／ `OADNet_CE_prior`（MDPE，真实先验） |
| 损失对照 | `OADNet_Focal`、`OADNet_CB`、`OADNet_LDAM`、`OADNet_Kang`、`OADNet_OGM_GE` |
| 先验消融 | `OADNet_CE_no_prior`（先验置零，架构不变） |
| 组件消融 | `OADNet_woOrthogonal`、`OADNet_woCrossAttn`、`OADNet_woAdaptiveMask`、`OADNet_woPseudoCNN` |
| 单模态基线 | `OADNet_text`、`OADNet_vision`、`OADNet_audio` |
| 文本门控 | `OADNet_TGated` |
| TS-BRL | `TS_BRL`、`TS_BRL_woGate`、`TS_BRL_woResidualBound` |
| 教师蒸馏 | `OADNet_KD`、`OADNet_ConfKD` |
| 简单融合基线 | `MLP_fusion` |

---

## 三、环境

Python 3.11（与主稿运行环境一致）。

```bash
pip install torch numpy pandas scipy scikit-learn
```

有 CUDA 时自动使用 GPU（`config.USE_AMP` 控制混合精度）；CPU 亦可运行小规模冒烟测试。

---

## 四、数据准备与运行

### 4.1 配置数据根目录

公开版 `config.py` 不含任何实际部署路径。运行前通过环境变量注入：

```bash
export OADNET_SERVER_ROOT=/path/to/My_Deception_Project
```

该根目录下需要存在：

```
$OADNET_SERVER_ROOT/
├── data/
│   ├── SEUMLD/SEUMLD/                        SEU-MLD 原始数据（Preprocess/Video、Preprocess/Audio）
│   ├── processed/
│   │   ├── seumld/                           {vid}_au.npy / {vid}_text.npy / {vid}_mfcc.npy
│   │   └── mdpe/                             {vid}.npy / {vid}.csv（按模态后缀区分）
│   ├── mdpe_raw/                             MDPE 原始特征目录（clipVIT-B16 / baichuan13B-base / chinese-hubert-base）
│   ├── seumld_final_dataset_with_soft_labels.csv
│   ├── mdpe_final_dataset_with_expert_soft_labels.csv
│   └── personality.csv                       MDPE 人格量表（BFI-2 前 60 题）
└── results/                                  训练输出（自动创建）
```

目录名与模态子目录映射见 `config.py` 的 `MDPE_MODALITY_SUBDIRS` 与 `DATASET_MODALITIES`。

### 4.2 特征完整性检查

```bash
python -c "from data import integrity_check; integrity_check('SEUMLD'); integrity_check('MDPE')"
```

缺失特征会**显式报错**，不会静默填充噪声。

### 4.3 跑实验

```bash
# 全量：两个数据集 × 5 折 × 5 种子 × 完整配置矩阵
python experiments.py --datasets SEUMLD,MDPE

# 只跑主模型 + 先验消融
python experiments.py --datasets SEUMLD,MDPE --main-only

# 小规模快速验证
python experiments.py --datasets SEUMLD --folds 2 --seeds 42,123

# 指定配置 / 损失
python experiments.py --datasets MDPE --configs OADNet_CE_prior,OADNet_text
python experiments.py --datasets SEUMLD --losses CE,Focal
```

已有 `per_run` CSV 会自动跳过（断点续跑）；`--force` 强制重跑。

### 4.4 输出

`results/<dataset>/` 下：

- `per_run_<config>.csv` — 逐 (fold, seed) 指标
- `summary.csv` — 每配置 mean ± std
- `significance.csv` — 主模型 vs 各基线的显著性检验
- `report.md` — 人类可读汇总
- `preds/<config>_f<fold>_s<seed>.npz` — **CS-Mondrian 的直接输入**，含
  `preds` / `probs` / `labels` / `calib_probs` / `calib_labels` / `val_probs` / `val_labels`

---

## 五、数据集授权与伦理

SEU-MLD 与 MDPE 为**受控数据集**，须向原始数据持有者申请授权后方可使用，
使用范围受其数据使用协议约束。

- 本目录**不含**任何受控原始数据、派生预测张量或受试者级可识别信息。
- 复现完整链路需要：① 向数据持有者取得授权；② 按 §4.1 准备目录结构；③ 运行本文档的代码。
- 本目录的代码经过路径脱敏，不含任何服务器地址、凭据或部署路径。

---

## 六、与本地完整包的关系

作者本地完整复现包中另有 `03_代码/05_共形拒判实验脚本/`，是在外部 GPU 服务器上
跑这批实验的**整套工作流**（含服务器同步、上传、受试者映射生成、进度抓取等运维脚本）。
该目录因含服务器地址/凭据/受控数据路径，**不随本仓库分发**。

本目录是从中抽出的**纯算法子集**，去掉了全部运维与服务器相关代码，
只保留复现 OAD-Net 训练所必需的模块。

---

## 七、可移植性

本目录脚本已消除本机绝对路径：

- **包内路径** → 由脚本自身位置推导，整体复制到任何机器均可运行
- **数据根目录** → 由环境变量 `OADNET_SERVER_ROOT` 注入（见 §4.1）
- **模型权重 / 预测张量** → 不随仓库分发

若发现仍存在硬编码路径，请提交 issue。
