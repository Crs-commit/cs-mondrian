# Open-Domain Deception 修复后结果报告

依据：`CS_Mondrian_OpenDomain_外部验证修复方案.md` v1.0（2026-09-08，实施前冻结）
代码：`work/csmondrian_ext/fixed_stage1.py` / `fixed_stage2.py` / `fixed_stage3.py`
数据：Open-Domain Deception Dataset v1.0（UMich 直链下载，SHA-256 见 audit）
旧结果：`outputs/ext_validation/` 全部保留为历史产物，论文只引用本目录结果。

## 1. 修复内容对照

| 问题 | 旧执行（ext_validation/） | 修复后（本目录） |
|---|---|---|
| test 覆盖 | 每折独立抽 20% 受试者，同一受试者可能多次或从未进 test | `KFold(5, shuffle, random_state=seed)` 于全部 511 人；每 seed 每人恰好一次 test |
| 锚点推断 n | 名义 511 但覆盖不保证 | **严格 n=511**（每人恰 5 次 test，先跨 seed 平均） |
| Plan A | 只按 CS 的 selection 拒判率选一个 alpha，两方法共用 | 两方法**各自**在 selection 上按预算选 alpha（网格 0.025–0.300，步长 0.025） |
| Plan B | 单一 C_rev* | 拆分 `C_rev,all-reject*`（对全拒判基线）与 `C_rev,CS-vs-M*`（差分），ΔReject=0 时记 undefined |

数据口径、规范化重复删除（170 组/438 行→6,730 条/511 人）、backbone、哈希校准抽取（salt 机制）与冻结协议完全一致，未改动。

## 2. 拆分审计（§10 验收项 1–3）

- 每个 seed：五个 test fold 并集 = 511 人，两两互斥（程序断言通过）；
- 每名受试者恰好 5 次 test（5 seeds），无遗漏、无重复；
- 每 run 的 fit/selection/calibration/test 四分区受试者集合两两互斥（程序断言通过）；
- 规范化重复文本：全局为零 + 每 run 跨分区为零（断言通过）；
- 内层比例（占外层池）：fit 37.5% / selection 12.5% / calibration 50%，随机状态 `RandomState([seed, fold])`。

## 3. Backbone（25 次运行，无事后删除）

| 指标 | 均值 | 标准差 | 范围 |
|---|---|---|---|
| ROC-AUC | 0.6246 | 0.0149 | [0.5971, 0.6654] |
| Balanced accuracy | 0.5893 | 0.0138 | [0.5637, 0.6189] |
| Brier | 0.2391 | 0.0037 | [0.2296, 0.2468] |

## 4. 固定锚点主分析（α=0.10, r=3, C_FP=1, C_rev=0.5）

方法级（25 次运行均值）：

| 方法 | 期望成本 | 拒判率 | FP 率 | FN 率 | 覆盖率 |
|---|---|---|---|---|---|
| CS-Mondrian | **0.4912** | 0.7771 | 0.1134 | 0.0308 | 0.9288 |
| Mondrian（统一 α） | 0.5521 | 0.6736 | 0.1134 | 0.1044 | 0.8912 |

主检验（受试者级，n=511，先跨 seed 平均）：

```text
Δcost = −0.0617，95% CI [−0.0685, −0.0549]，cluster bootstrap 10,000
paired sign-flip p = 1.0e-4；与最终修复的 Cross-Cultural 锚点（n=1200）共同做 Holm 校正后 p = 2.0e-4
```

机制与 v2.0 历史结果一致且数值更强：CS 通过压低 lie 侧 qhat 把 FN 率从 0.104 降到 0.031，代价是拒判率上升 0.103，在 C_rev=0.5 下成本净降。

## 5. Plan A：selection 集固定拒判预算上限下的工作点比较

两方法各自选 alpha（selection 受试者等权拒判率 ≤ B 后最小化 |B−rr|，并列取 selection 成本更低、alpha 更小者）：

| B | α_CS | α_MD | sel RR CS/MD | test RR CS/MD | Δ test RR | Δcost | 95% CI | p（Holm，OD 内） | 预算违约率 CS/MD |
|---|---|---|---|---|---|---|---|---|---|
| 0.60 | 0.211 | 0.147 | 0.578/0.561 | 0.576/0.560 | +0.016 | **−0.0729** | [−0.0792, −0.0664] | 3e-4 | 0.20 / 0.12 |
| 0.70 | 0.155 | 0.108 | 0.677/0.656 | 0.673/0.652 | +0.021 | **−0.0576** | [−0.0635, −0.0516] | 3e-4 | 0.20 / 0.08 |
| 0.80 | 0.108 | 0.073 | 0.762/0.762 | 0.758/0.755 | +0.002 | **−0.0392** | [−0.0438, −0.0346] | 3e-4 | 0.04 / 0.16 |

报告边界（§6.3）：selection 上满足同一预算不保证 test 拒判率相同——test Δreject 在 +0.002 到 +0.021 之间（CS 略高），预算违约率两方法均在 4%–20%，属小样本 selection（约 51 人）的噪声量级。正式表述为“selection 集固定拒判预算上限下的工作点比较”，不使用“严格固定容量”。

## 6. Plan B：部署经济学（固定锚点，不重新选 α）

| 方法 | err_per_auto | C_rev,all-reject* |
|---|---|---|
| CS-Mondrian | 0.3183 | 0.4572 |
| Mondrian | 0.3322 | 0.6574 |

差分盈亏平衡：`C_rev,CS-vs-M* = −(ΔFP + r·ΔFN)/ΔReject`，25 次运行 ΔReject 均非零（均值 +139.4 条/运行），无 undefined；跨运行均值 **1.088**。解释：当复核成本 C_rev < 1.088 时 CS-Mondrian 总成本更低；C_rev=0.5 的锚点在该区间内。两个盈亏平衡值口径不同，未共用列名。

## 7. r 敏感性（次要，点估计 + cluster bootstrap CI）

| r | Δcost | 95% CI |
|---|---|---|
| 1 | +0.0000（r=1 数学锚点，两方法逐元素一致） | [0, 0]，p=1.0 |
| 2 | −0.0184 | [−0.0219, −0.0150] |
| 3（锚点） | −0.0617 | [−0.0685, −0.0549] |
| 5 | −0.1587 | [−0.1727, −0.1446] |
| 10 | −0.4167 | [−0.4499, −0.3834] |

单调性符合机制预期：r 越大（FN 越贵），CS 的成本优势越大。

## 8. 验收门槛核对（§10）

| 门槛 | 结果 |
|---|---|
| 每 seed 511 人在 5 个 test folds 恰好一次 | ✅ 程序断言 |
| 每人恰 5 个 test 结果（5 seeds） | ✅ test_coverage min=max=5 |
| 四分区两两互斥 | ✅ 程序断言 |
| 规范化重复文本主分析为零 | ✅ 全局 + 逐 run 断言 |
| 特征变换仅在 fit 拟合 | ✅（构造保证） |
| 所有 qhat 有限 | ✅ strict_q 断言 |
| r=1 预测集合与成本逐元素一致 | ✅ 50 项检查 max diff = 0.0 |
| 锚点推断 n=511 | ✅ |
| Plan A 两方法各自选 alpha、两套 selection trace | ✅ trace 1,800 行 |
| 成本分解残差 < 1e-12 | ✅ max = 0.0 |
| 25 次运行全部完成，无事后删除 | ✅ |
| 正向、反向和不显著结果均保留 | ✅（本轮无反向项；r=1 零效应如实报告） |

## 9. 论文使用边界

- 修复后固定锚点可作为正文“英文受试者级外部验证”主证据（n=511，与 CC 组成双数据集 Holm 校正 p=2e-4）；
- Plan A 为预算约束主证据，Plan B 为部署经济学分析；
- Open-Domain 基础模型在该数据集上重新训练（AUC≈0.62），验证的是 CS-Mondrian 后处理机制的外部复现，不得表述为多模态、高风险真实欺骗或跨语言零样本验证；
- 标签为伴随属性（陈述本身真/假），非“预测时点之后的结果”，与主实验同源局限，limitations 需保留。

## 10. 产物清单

`split_audit.csv/json`、`backbone_metrics.csv`、`preds/`（25 npz）、
`subject_map_OpenDeception_fixed.csv`、`run_level.csv`、`anchor_subject_effects.csv`、
`primary_test.csv`、`planA_selection_trace.csv`、`planA_run_level.csv`、
`planA_subject_effects.csv`、`planA_tests.csv`、`planB_summary.csv`、
`planB_summary_stats.csv`、`r_sensitivity_runs.csv`、`r_sensitivity.csv`、
`r1_anchor_check.json`、`stage2_summary.json`、`stage3_results_fixed.json`、
`protocol_fixed.json`、本报告。
