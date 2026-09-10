# Singh[10] 对照臂 — 可复现性索引

> 生成：2026-09-09　｜　最后修订：2026-09-10
> 性质：纯离线后处理，不训练、不连服务器、不改原始数据
> **注意**：本索引为公开版本。受控数据集派生的预测输入不随本仓库分发，
> 详见 `00_说明与索引/00_README_复现包说明.md` 的数据获取章节。

---

## 1. 实际运行的代码与依赖

### 1.1 文件清单

| 文件 | SHA-256 (前16位) | 大小 | 角色 |
|---|---|---|---|
| `singh_arm_recompute.py` | `e17228679ed32a21` | 30,381 B | **主计算**：中文四配置，自包含 |
| `singh_external_recompute.py` | `518e40256ce81803` | 16,348 B | 外部两数据集，从主脚本导入 |
| `singh_self_check.py` | `fc61c2bb36259cdf` | 8,860 B | 自检清单验证（全部 PASS，含 subject-level 混淆计数检查） |
| `fix_stats_eps_hum.py` | `5bd59b6afa84d6a2` | 5,605 B | 中间修复：统计筛选 bug 后重跑 |
| `strict_recompute.py` | `0e3a9a0bc7e6fa9f` | 46,006 B | **核心函数来源**（逐字复制，非 import） |

上述文件均位于 `03_代码/02_Singh对照臂_新实验/`。

### 1.2 依赖关系

- **`singh_arm_recompute.py` 自包含**：`conformal_quantile`、`build_flags`、`method_mondrian`、`method_cs_mondrian`、`compute_metrics_base` 从 `strict_recompute.py` 逐字复制，仅依赖 `numpy / pandas / scipy / os / sys / time`。
- **`singh_external_recompute.py`**：通过 `SCRIPT_DIR = Path(__file__).resolve().parent` + `sys.path.insert(0, str(SCRIPT_DIR))` + `from singh_arm_recompute import (...)` 导入核心函数。运行前需准备授权输入并核对脚本中的相对目录配置。
- **`INTEGRATION_ROOT`**：优先读取环境变量 `CS_MONDRIAN_ROOT`，默认回退到包根目录。
- **运行环境**：Python 3.11（numpy, pandas, scipy）。

### 1.3 公共模块说明

| 模块 | 来源 | 复用方式 |
|---|---|---|
| `conformal_quantile(scores, alpha)` | strict_recompute.py | 逐字复制到 singh_arm_recompute.py |
| `build_flags(include_truth, include_lie)` | strict_recompute.py | 逐字复制 |
| `method_mondrian(...)` | strict_recompute.py | 逐字复制 |
| `method_cs_mondrian(...)` | strict_recompute.py | 逐字复制 |
| `compute_metrics_base(...)` | 基于 strict compute_metrics 扩展 | 新增 eps_hum 参数，eps=0 时逐位一致 |
| `bayes_threshold_cost(...)` | 新增（Singh §3.4） | τ*=1/(1+r) 硬决策 |
| `singh_breakeven_c_rev(...)` | 新增 | 线性插值求根 |
| `subject_level_stats(...)` | 新增 | cluster bootstrap + sign-flip permutation + Wilcoxon + sign test |
| `holm_adjust(pvalues)` | 新增 | Holm-Bonferroni 逐步校正 |

### 1.4 运行命令

```bash
cd 03_代码/02_Singh对照臂_新实验

# 中文四配置主计算（~220s，输出 run-level + subject-level + 统计 + break-even）
python singh_arm_recompute.py

# 修复统计筛选 bug 后重跑统计（读取已保存 CSV，不重算核心指标）
python fix_stats_eps_hum.py

# 外部两数据集（~130s，Open-Domain + Cross-Cultural）
python singh_external_recompute.py

# 自检清单（r=1 退化、成本非负、lie 覆盖率、Holm 正确性等）
python singh_self_check.py
```

> 运行前需自备受控数据集的派生预测，并通过 `CS_MONDRIAN_ROOT` 或脚本内
> 输入路径变量指向之。仓库不提供该输入。

---

## 2. 代码版本

以当前 Git 提交中的 `.py` 文件为准。旧 ZIP 快照已移除，避免继续分发过期路径和不同版本的代码。

---

## 3. 输入数据

### 3.1 数据性质与获取

本臂的输入为**基础预测器在受控数据集上的输出张量**，非原始数据：

| 数据 | 内容 | 获取方式 |
|---|---|---|
| 中文四配置 | SEUMLD / MDPE 两数据集 × text/audio 两模态，2 config × 5 fold × 5 seed 的预测 | 受控数据集，须向原始数据持有者申请授权后自行生成 |
| Open-Domain | 外部 OpenDeception 语料的预测 | 见论文数据来源章节 |
| Cross-Cultural v1 / v2 | 外部 Cross-Cultural 语料的两个校准版本 | 见 §4 |

- subject 级配对映射（受试者 × fold × role）随受控数据一并提供，**不随本仓库分发**。
- 复现本臂结果需自备上述输入；脚本本身不依赖任何本机绝对路径。

### 3.2 预测张量字段规范

| 字段 | 说明 |
|---|---|
| `probs` | 测试集 lie 后验概率（float32/64） |
| `labels` | 测试集真实标签（0=truth, 1=lie） |
| `preds` | 测试集预测标签 |
| `calib_probs` / `calib_labels` | 校准集概率与标签 |
| `val_probs` / `val_labels` | 验证集概率与标签 |
| `test_subject_ids` | （仅外部）测试集受试者/单位 ID |

### 3.3 Open-Domain 复现对账

- 统计单位：受试者级（n=511），跨 5 seed 平均后 cluster bootstrap
- 对账：ε=0 时 Δcost=−0.0617 CI[−.0687,−.0550]，与论文锚点 −0.0617 CI[−.0685,−.0549] 一致

---

## 4. Cross-Cultural 两个版本

两个版本并存，主稿与 Singh 对照臂**使用不同版本**，这是刻意选择，不是笔误。

### 4.1 版本 A：v1 初版（论文锚点 **−0.0315** CI[−.0424,−.0213]）

- 用于主稿 Cross-Cultural 主结果
- 对应协议参数、审计与锚点校验记录均已归档于本包 `05_结果/05_外部验证结果/`

### 4.2 版本 B：v2 修复版（Singh 对照结果 **−0.0389** CI[−.0503,−.0276]）

- 用于 Singh 对照臂的全部计算
- 修复内容见 `05_结果/05_外部验证结果/` 下的回归校验记录

### 4.3 两版本差异

| 维度 | v1 初版 | v2 修复版 |
|---|---|---|
| CS-vs-Singh Δcost (r=3, Crev=.5) | **−0.0315** | **−0.0389** |
| 95% CI | [−.0424, −.0213] | [−.0503, −.0276] |
| 方向 | CS 更优（负） | CS 更优（负），效应量增大 |
| 使用位置 | 主稿 Cross-Cultural 主结果 | Singh 对照臂全部计算 |

v1 → v2 的修复涉及 GlobalAlpha 校准逻辑，导致预测概率分布变化，因此两版数值不同。
若需对齐论文锚点 −0.0315，改用 v1 初版预测重新运行 `singh_external_recompute.py`
（仅需修改输入预测路径变量）。

---

## 5. 产物总览

### 5.1 结果文件（`05_结果/02_Singh对照臂_新实验/`）

| 文件 | 行数 | 说明 | 是否公开 |
|---|---|---|---|
| `singh_eps_hum_comparison.csv` | 320 | 中文四配置 CS-vs-Singh Δcost + CI + p + Holm | ✅ |
| `singh_breakeven_vs_bayes.csv` | 20 | 中文 Singh 口径 C*_rev | ✅ |
| `singh_external_eps_hum_comparison.csv` | 160 | 外部两数据集 CS-vs-Singh | ✅ |
| `singh_external_breakeven_vs_bayes.csv` | 10 | 外部 Singh C*_rev | ✅ |
| `singh_run_level.csv` | 18,000 | 中文 run-level（fold×seed 级，不含个体标识） | ✅ |
| `singh_subject_level.csv` | 480,600 | 中文 subject-level 明细（含 subject_id） | ❌ 仅本地 |

### 5.2 文档

| 文件 | 说明 |
|---|---|
| `Singh对照臂_结果与可粘贴段落.md` | 主报告：方法/结果/限制三段可直接粘贴论文 |
| `Singh对照臂_可复现性索引.md` | 本文档 |

---

## 6. 核验方法

```bash
# 代码完整性（zip 快照内文件应与仓库内一致）
sha256sum 03_代码/02_Singh对照臂_新实验/singh_arm_recompute.py

# 复现全部结果
cd 03_代码/02_Singh对照臂_新实验
python singh_arm_recompute.py
python fix_stats_eps_hum.py
python singh_external_recompute.py
python singh_self_check.py
```

---

## 7. 修复记录（2026-09-09 第二次修复）

### 7.1 修复内容

| # | 问题 | 修复方式 | 影响范围 |
|---|---|---|---|
| 1 | **Bayes subject-level 混淆计数错误**：`n_tp`/`n_tn` 硬编码为 0 | 修改 `bayes_threshold_cost()` 函数增加返回 `n_tp`/`n_tn`；run-level 和 subject-level 统一改用函数返回值 | 仅 subject-level 明细中 Bayes 行的 `n_tp`/`n_tn` 列（诊断性计数，不影响成本/p 值） |
| 2 | **外部脚本导入路径硬编码** | 改为 `SCRIPT_DIR = Path(__file__).resolve().parent` + `sys.path.insert(0, str(SCRIPT_DIR))`；`INTEGRATION_ROOT` 支持 `CS_MONDRIAN_ROOT` 环境变量 | 代码可移植性，不影响计算结果 |
| 3 | **fix_stats CS 未按 eps_hum 筛选**：CS 数据包含所有 eps_hum，合并时产生笛卡尔积 | CS 也按相同 `eps_hum` 筛选，与 Singh 公平配对 | 修复前统计结果错误（ε=0 时 Δcost 符号相反），修复后与主计算内部统计一致 |
| 4 | **自检脚本增强**：缺少 subject-level 混淆计数检查 | 增加：①`n_accept = n_fp+n_fn+n_tp+n_tn` 恒等式；②`n_test = n_accept+n_reject`；③Bayes `n_reject=0`；④Bayes `n_tp>0` 且 `n_tn>0`；⑤run/subject `total_cost` 一致性 | 防止未来类似问题漏检 |

### 7.2 结果处理原则

- **不改变方法定义**：CS-Mondrian、Singh、Bayes-τ* 的算法定义未变
- **不改变输入预测**：中文四配置、Open-Domain、Cross-Cultural v2 的输入未变
- **不改变版本选择**：Cross-Cultural 仍使用 v2 修复版
- **不改变统计结论**：修复后 ε=0 主终点与 strict 权威数字逐位一致（max diff=1.11e-16）
- **只替换含错误计数的明细文件**
- **诊断性修复**：Bayes n_tp/n_tn 是诊断性计数，不影响 `expected_cost`/`Δcost`/p 值；
  **"diagnostic-count fix only; inferential outputs unchanged"**

### 7.3 修复后验证

- [x] 全部自检 PASS（10 大类，含新增 subject-level 混淆计数检查）
- [x] ε=0 与 strict 权威结果逐项一致（max diff=1.11e-16）
- [x] r=1 时 CS 与 Singh 逐项一致（Δcost=0）
- [x] Bayes subject-level `n_tp` 非零（max=35）、`n_tn` 非零（max=46）
- [x] subject-level 混淆计数恒等式 `n_accept = n_fp+n_fn+n_tp+n_tn`（violations=0）
- [x] run-level 与 subject-level `total_cost` 一致（max diff=4.55e-13）
- [x] Cross-Cultural 使用 v2 修复版
