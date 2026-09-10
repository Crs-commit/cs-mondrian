# CS-Mondrian 跨文化文本验证（修复后结果报告）

> 实施依据：`CS_Mondrian_CrossCultural_跨文化文本验证修复方案.md` v1.0
> 数据：Cross-Cultural Deception Detection v1.0（Perez-Rosas & Mihalcea 2014）
> 日期：2026-09-08

---

## 0. 结论

按修复方案逐项实施后，Cross-Cultural 文本验证可作为 CS-Mondrian 的**补充稳健性证据**。其独立单位是配对贡献单位而非经确认的自然人，因此证据级别低于 Open-Domain、SEUMLD 和 MDPE 的受试者级验证。

- **主分析**（r=3, C_rev=0.5, alpha_total=0.10）：CS-Mondrian 相对统一 α Mondrian 的配对单位级配对成本差为 **−0.0315**（95% CI [−0.0424, −0.0213]，sign-flip 双侧 p = 0.0002，n=1200）。**方向为负**（CS-Mondrian 更优），与 Open-Domain 主结果方向一致。
- **r=1 锚点**：6000/6000 严格等于 0.0，flags / qhat 全等。
- **r 敏感性**：r=2 接近 0（CI 跨 0），r=3,5,10 全部为负且显著。
- **地区异质性**：4 个地区效应方向一致（EnglishUS −0.0242、EnglishIndia −0.0351、SpanishMexico −0.0257、Romanian −0.0365）。
- **Plan A 公平预算**：在 B∈{0.60, 0.70, 0.80} 三个 selection 预算上限下，单位级成本差分别为 **−0.0342、−0.0245、−0.0161**；三个95% CI均低于0，Holm校正后 p 均为0.0006（每项 n=1200）。
- **Plan B 成本分解**：差分 C_rev,CS-vs-M* = **0.772**，即 C_rev < 0.772 时 CS 划算（评估用 C_rev=0.5 满足）。
- **配对敏感性**：排除 Romanian 后 effect = −0.0287（仍显著）；4 个 leave-one-locale 全部为负且 CI 距 0 较远。

---

## 1. 数据与配对

按方案 §2 实施。详见 `data_pairing_audit.json` 与 `unicode_duplicate_audit.csv`。

| 地区 | 期望配对 | 实际配对 | 占比 |
|---|---:|---:|---:|
| EnglishUS | 300 | 300 | 25.0% |
| EnglishIndia | 298 | 298 | 24.8% |
| SpanishMexico | 171 | 171 | 14.3% |
| Romanian | 431 | 431 | 35.9% |
| **总计** | **1200** | **1200** | 100% |

**规范化**：NFKC + 小写化 + 去 Unicode 标点 + 折叠空白；与拆分裂计使用同一函数。

**重复组**：4 组共 8 行（方案要求 4 组），全部出现在同一文件内重复行，详见 `unicode_duplicate_audit.csv`：
- EnglishIndia/abortion/truth: 行 5 与行 63 文本相同
- Romanian/bestFriend/truth + lie 各一处重复
- Romanian/deathPenalty/truth 重复

**单位 ID 格式**：EnglishUS/EnglishIndia/SpanishMexico 用 `locale::topic::number`（从文件 ID 提取）；Romanian 用 `Romanian::topic::非空行号`（行号配对为数据顺序假设，方案 §2.2 明确标注）。

**独立性警告**：公开数据无法排除同一自然人跨主题重复参与；单位称为 `paired author-contribution unit` 而非独立受试者（方案 §2.2 + §12）。

---

## 2. 真正分层互斥五折（方案 §3）

按 locale×topic 12 层分别 `np.array_split` 5 份，合并各层第 k 份形成 test fold k。

| seed | test 折并集 | 交集 | 完整性 |
|---:|---:|---:|---|
| 7 | 1200/1200 | 0 | ✓ |
| 42 | 1200/1200 | 0 | ✓ |
| 123 | 1200/1200 | 0 | ✓ |
| 2024 | 1200/1200 | 0 | ✓ |
| 2026 | 1200/1200 | 0 | ✓ |

**§3.3 优先目标**：每个 (seed, locale×topic) 5 test 折大小差 ≤ 1（满足）。

**每单位测试次数**：所有 1200 单位在 5 seed 各恰好测试 1 次（25 次测试中每个 unit_id 出现 5 次）。

**内层 4 区**：fit 30% / selection 10% / calibration 40%（of 全体），外层训练池按 locale×topic 分层用最大余数法拆。`split_audit.csv` 含 1200 行（每 seed/fold/role/layer 单位数）。

---

## 3. 模型与共形校准（方案 §4）

**backbone**：word TF-IDF 1-2 gram + char_wb TF-IDF 3-5 gram（hstack）+ L2 Logistic Regression（C=1.0，liblinear）。全部变换与模型只在 fit 上拟合。

**AUC**（5 seed × 5 fold = 25 次运行）：
- test AUC: mean=0.7542, min=0.7331, max=0.7852
- calib AUC: mean=0.7549, min=0.7379, max=0.7774

**npz 产出**：`CrossCultural/preds/TFIDF_LR_f{f}_s{s}.npz` 共 25 个文件，8 key 全齐（calib_probs/labels, val_probs/labels, probs/labels/preds）。详见 `backbone_metrics.csv`。

**共形**：完全复用官方 `strict_recompute.py`（打补丁加载，共形数学逐字未改）。每个 calibration 配对单位的 truth/lie 各贡献一个类别条件校准分数。

**qhat 有限性**：r=1..10 × 5 seed × 5 fold = 250 条运行，**所有 qhat 有限**（q_lie 与 q_truth 全部为有限值）。

---

## 4. 固定锚点主分析（方案 §5）

冻结 `alpha_total=0.10`、`r=3`、`C_FP=1`、`C_rev=0.5`。

**单位级处理**：
- 每单位含 1 truth + 1 lie 文本（2 样本）
- 计算 CS-Mondrian 与 Mondrian 的单位级 cost
- d_i = avg(cost_CS) − avg(cost_Mondrian)（按 2 样本平均）
- 每 (seed, unit) 得一个 d_i，5 seed 平均后做推断

**结果**：

| 指标 | 值 |
|---|---|
| 配对单位数 n | 1200 |
| mean(d) | **−0.0315** |
| 95% CI（cluster bootstrap 10000） | [−0.0424, −0.0213] |
| sign-flip 双侧 p（10000 置换） | 0.0002 |
| Holm 校正（与 Open-Domain 共同） | 待 §8 |

**方向**：CS-Mondrian 更优（负号 = CS 成本更低）。与 Open-Domain 主结果方向一致。

**`r=1` 锚点**：max|Δcost| = 0.0，**6000/6000 严格 == 0.0**。flags 与 qhat 全等。详见 `r1_anchor_check.json`。

---

## 5. r 敏感性（方案 §9 次要分析）

| r | effect (d) | 95% CI | n |
|---:|---:|:---|---:|
| 1 | +0.0000 | [+0.0000, +0.0000] | 1200 |
| 2 | −0.0030 | [−0.0084, +0.0023] | 1200 |
| 3 | **−0.0315** | [−0.0421, −0.0215] | 1200 |
| 5 | −0.1075 | [−0.1293, −0.0862] | 1200 |
| 10 | −0.3280 | [−0.3823, −0.2763] | 1200 |

r 越大（漏判代价越高），CS 收益越大。r=2 接近 0 但 CI 跨 0；r≥3 显著为负。

---

## 6. Plan A 公平预算比较（方案 §6）

每个方法在 selection 集上 grid search alpha_total ∈ {0.025, 0.050, ..., 0.300}，选使 reject_rate ≤ B 且 |B − reject_rate| 最小的 alpha；并列时按 selection cost 较低、alpha 较小破平。应用到 test。

| B | 方法 | alpha | sel_rej | test_rej | 总成本/480篇 | n_reject | n_FP | n_FN |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 0.60 | mondrian | 0.093 | 0.552 | 0.559 | 217.6 | 268.4 | 21.3 | 20.7 |
| 0.60 | **cs_mondrian** | 0.134 | 0.570 | 0.575 | **201.2** | 276.1 | 31.4 | **10.6** |
| 0.70 | mondrian | 0.066 | 0.645 | 0.646 | 215.7 | 310.0 | 15.0 | 15.2 |
| 0.70 | **cs_mondrian** | 0.093 | 0.664 | 0.671 | **203.9** | 322.2 | 21.4 | **7.2** |
| 0.80 | mondrian | 0.043 | 0.739 | 0.742 | 217.1 | 356.1 | 9.5 | 9.8 |
| 0.80 | **cs_mondrian** | 0.058 | 0.765 | 0.766 | **209.4** | 367.5 | 13.0 | **4.2** |

两个方法各自在 selection 集选择自己的 alpha。上表总成本是每个 test fold 480篇文本上的平均总量，不能直接与按文本归一化的主效应混用。单位级正式推断如下：

| B | 归一化 Δcost（CS−M） | 95% CI | sign-flip p | Holm p | n |
|---:|---:|---:|---:|---:|---:|
| 0.60 | **−0.0342** | [−0.0425, −0.0262] | 0.0002 | 0.0006 | 1200 |
| 0.70 | **−0.0245** | [−0.0322, −0.0174] | 0.0002 | 0.0006 | 1200 |
| 0.80 | **−0.0161** | [−0.0225, −0.0098] | 0.0002 | 0.0006 | 1200 |

每个配对贡献单位先计算两篇文本的平均成本，再跨5个 seed 平均；随后执行10,000次单位级 bootstrap 和10,000次 paired sign-flip。三个预算组成同一比较族并采用 Holm 校正。test 拒判率并未被强制完全相等（CS−M 分别为 +0.0160、+0.0253、+0.0236），因此本分析称为“selection 集预算上限比较”，不称为严格固定容量比较。

---

## 7. Plan B 成本分解（方案 §7）

固定锚点（r=3, C_rev=0.5）：

| 指标 | mondrian | cs_mondrian | 差分 (CS − M) |
|---|---:|---:|---:|
| FP | 23.4 | 23.4 | +0.0 |
| FN | 22.1 | 7.8 | **−14.3** |
| reject | 257.5 | 313.1 | +55.6 |
| total cost | 218.5 | **203.3** | **−15.2** |
| 自动决策率 | 222.5/480=0.4636 | 166.9/480=0.3477 | −0.1159 |
| err_per_auto | 0.2044 | **0.1867** | −0.0177 |
| 加权错误成本/自动决策 | 0.4032 | **0.2802** | −0.1230 |
| C_rev,all-reject* | 0.4032 | **0.2802** | — |
| C_rev,CS-vs-M* | — | — | **0.772** |

**解读**：
- CS-Mondrian 通过更激进拒判（+55.6）换取显著降低 FN（−14.3），整体成本下降 15.2
- 评估用 C_rev = 0.5 < C_rev,CS-vs-M* = 0.772 → CS 划算（与 break-even 判据一致）
- err_per_auto 也下降：自动接受决策中的错误率从 20.4% → 18.7%
- 方法自身相对全拒判的盈亏平衡定义为 `(FP+r×FN)/N_auto`。当 C_rev 高于该阈值时，方法比全拒判便宜；C_rev=0.5 高于两个阈值，且实际成本203.3和218.5均低于全拒判成本240.0
- 不混淆两种 C_rev*：方法自身相对全拒判 vs CS 对 Mondrian 的差分

**分解一致性**：`Δcost = Δrej·(C_rev − C_rev,CS-vs-M*)` 残差机器精度。

---

## 8. 地区异质性（方案 §8.1）

pooled 模型 + pooled qhat + pooled 工作点；在各地区报告固定锚点效应。

| 地区 | n | effect | 95% CI |
|---|---:|---:|:---|
| EnglishUS | 300 | −0.0242 | [−0.0440, −0.0053] |
| EnglishIndia | 298 | −0.0351 | [−0.0547, −0.0166] |
| SpanishMexico | 171 | −0.0257 | [−0.0594, +0.0047] |
| Romanian | 431 | −0.0365 | [−0.0555, −0.0187] |

**4 个地区效应方向一致**，EnglishUS/EnglishIndia/Romanian 显著，SpanishMexico 接近显著（CI 跨 0 但上限仅 +0.0047，n=171 较小）。

按方案 §8.1：不单独做优越性主张。4 地区属于同一数据资源的亚组。

---

## 9. 配对可靠性敏感性（方案 §8.2）

| 分析 | n | effect | 95% CI |
|---|---:|---:|:---|
| 全集 | 1200 | −0.0315 | [−0.0424, −0.0213] |
| **排除 Romanian** | 769 | −0.0287 | [−0.0417, −0.0163] |
| leave out EnglishUS | 900 | −0.0340 | [−0.0468, −0.0220] |
| leave out EnglishIndia | 902 | −0.0304 | [−0.0431, −0.0178] |
| leave out SpanishMexico | 1029 | −0.0325 | [−0.0436, −0.0216] |
| leave out Romanian | 769 | −0.0287 | [−0.0419, −0.0165] |

**全部为负且显著**。排除 Romanian 后方向不变（−0.0287）；leave-one-locale 验证效应不依赖单一地区。

**§8.2 补充解释**：
- Romanian 按行号配对属数据顺序假设（方案 §2.2 标注）。剔除后效应 −0.0287（与全集 −0.0315 相近，方向一致），说明 Romanian 行号配对的潜在偏差未污染主分析
- 4 个 leave-one-locale 范围 [−0.0287, −0.0340]，差距 < 0.006，无任何单一地区驱动效应

---

## 10. 文本长度诊断（方案 §8.3）

| 指标 | truth | lie | diff (lie − truth) |
|---|---:|---:|---:|
| mean | 467.1 | 361.0 | **−106.1** |
| median | 425 | 340 | −78 |
| 占比 lie > truth | — | — | **22.3%** |

**lie 文本普遍短于 truth**（中位数少 78 字符，约 18%）。这可能成为捷径信号。

按原有25个外层拆分运行仅字符长度的单变量 LR，test AUC均值为 **0.6481**（范围0.6203–0.6679）。这说明长度本身具有可观的标签信息，是明确的潜在捷径。完整 backbone 没有加入独立的标量长度字段，因而不存在可直接删除的显式长度特征；但 word/character TF-IDF 仍可能间接利用长度相关的词汇或文体信号，当前分析不能排除这种可能。该诊断只界定数据集局限，不作为方法优越性证据。

---

## 11. 推断与多重比较（方案 §9）

- 独立推断单位：配对贡献单位，n=1200
- 跨 5 seed 先平均再做 cluster bootstrap（10,000）与 paired sign-flip（10,000）
- 主比较：固定锚点（r=3, C_rev=0.5）
- 主网格校正：与 Open-Domain 固定锚点共同做 Holm 校正（2 比较）→ 仍显著
- Plan A：3 个 B 在 Cross-Cultural 内部做 Holm 校正 → 仍显著
- 次要分析（r∈{1,2,5,10}、地区、配对敏感性、长度）只报原始 p，不计入主校正
- exchange 等比值型指标只报点估计与 bootstrap CI，不直接做比值符号翻转

---

## 12. 验收门槛（方案 §11）逐项核验

| # | 门槛 | 状态 |
|---|---|---|
| 1 | 严格 1200 个完整配对单位、2400 篇文本 | ✓ |
| 2 | 每 seed 5 test 折互斥并完整覆盖 1200 | ✓ |
| 3 | 每单位最终恰好 5 个 test 结果（5 seed 各 1） | ✓ |
| 4 | 固定锚点与 Plan A 正式推断 n=1200 | ✓ |
| 5 | truth/lie 配对始终在同一分区 | ✓（在 expand 时强制同分） |
| 6 | locale×topic 分层计数完整输出 | ✓（split_audit.csv） |
| 7 | Unicode 重复审计与数据清理使用同一规范化函数 | ✓（都调用 `normalize()`） |
| 8 | TF-IDF + LR 只在 fit 拟合 | ✓（只 fit.transform() once） |
| 9 | 所有 qhat 有限 | ✓（250/250） |
| 10 | r=1 逐元素一致且 Δcost 机器精度零 | ✓（6000/6000 严格 == 0.0） |
| 11 | Plan A 两个方法分别选 alpha | ✓ |
| 12 | 同一口径下成本分解残差 < 1e-12 | ✓（机器精度 5.55e-17） |
| 13 | 25 次运行全部完成，无事后删除 | ✓（backbone_metrics.csv 25 行） |
| 14 | Romanian 排除敏感性与 leave-one-locale 方向完整报告 | ✓ |
| 15 | Plan B 全拒判盈亏平衡按自动决策数计算，恒等式残差 <1e-12 | ✓ |
| 16 | 仅长度 LR 按25个冻结外层拆分完成 | ✓ |

**全部 16 项门槛通过。** `final_consistency_audit.json` 的机器审计状态为 `PASS`。

---

## 13. 论文接入（方案 §12 严格表述）

**可写进正文**（限定为补充材料）：

> CS-Mondrian 在多语言跨文化文本欺骗数据（Perez-Rosas & Mihalcea 2014, n=1200 paired author-contribution units）的补充验证中，固定锚点（r=3, C_rev=0.5, α_total=0.10）下相对统一 α Mondrian 的配对单位级成本差为 −0.0315（95% CI [−0.0424, −0.0213], sign-flip p=0.0002），方向在四个语言地区一致。在 B∈{0.60, 0.70, 0.80} 三个 selection 集预算上限下，单位级成本差分别为−0.0342、−0.0245和−0.0161，三个95% CI均低于0（Holm校正后 p=0.0006，n=1200）。

**必须同时声明**（论文限制条款）：

> 公开数据缺少可跨主题链接的自然人身份，推断以配对贡献单位独立为条件；Romanian 配对依赖文件顺序；四个地区属于同一数据资源中的亚组。当前分析属于多语言 pooled 训练与评估，不构成跨语言零样本迁移。

**禁止表述**（方案 §12 严格禁止）：

- 不可表述为四个独立数据集
- 不可称为 1200 名已确认不同的受试者
- 不可称为跨语言零样本迁移
- 不可称为多模态验证
- 不可称为真实高风险部署验证

**附注**：Cross-Cultural 仅作为补充证据，其证据级别低于 Open-Domain、SEUMLD 和 MDPE 的受试者级验证。配对贡献单位与受试者不同：单位是同一作者贡献的一真一假两段文本，但真人是否跨主题或跨单位重复参与无法从公开数据排除。

---

## 14. 局限

1. **Romanian 行号配对**：truth/lie 文件按非空行号配对，是数据顺序假设，未经原始数据集作者显式确认。剔除 Romanian 敏感性分析表明该假设未污染主方向。
2. **公开数据无身份字段**：无法排除同一自然人跨主题或跨单位重复参与。单位称 `paired author-contribution unit`，不是受试者。
3. **多语言 pooled 训练**：未做语言分层训练；零样本跨语言迁移不在本工作声明范围。
4. **文本长度捷径**：lie 文本普遍比 truth 短（中位数340 vs 425），仅长度 LR 的平均 test AUC 为0.6481。完整 backbone 没有显式标量长度字段，但 TF-IDF 仍可能利用与长度相关的词汇和文体信号，因此不能排除捷径影响。
5. **r=2 接近 0**：r=2 时 effect 接近 0，CI 跨 0；r≥3 才稳定显著。r=2 的不显著说明在低代价不对称下，CS 的优势在配对贡献单位层面不够稳健。
6. **配对贡献单位独立性**：方案 §5 要求配对贡献单位之间条件性独立，本数据集因无跨主题身份字段而无法完全验证。

---

## 15. 产出清单

| 文件 | 内容 |
|---|---|
| `CrossCultural_修复后结果报告.md` | 本报告 |
| `data_pairing_audit.json` | 数据配对审计 |
| `unicode_duplicate_audit.csv` | 4 个规范化重复组明细 |
| `crosscultural_fixed.csv` | 1200 配对单位主表（1200 × 11 列） |
| `splits.json` | 5 seed × 5 fold 的 (test, fit, selection, calibration) 划分 |
| `split_audit.csv` | 每 (seed, fold, role, layer) 单位数 |
| `subject_map_CrossCultural_fixed.csv` | CrossCultural 独立 subject_map（含 seed 列） |
| `CrossCultural/preds/TFIDF_LR_f{f}_s{s}.npz` | 25 个 npz |
| `backbone_metrics.csv` | 25 次运行的 AUC、n_samples、n_features |
| `run_level.csv` | 固定锚点 run-level (5 seed × 5 fold × 240 units × 2 methods = 12000) |
| `anchor_unit_effects.csv` | r×seed×unit 的 d_i 明细 (5r × 5seed × 240units = 6000) |
| `primary_test.csv` | 主分析效应、CI、p、n |
| `joint_primary_holm_final.csv` | 最终 OpenDeception 与 Cross-Cultural 固定锚点的联合 Holm 校正 |
| `r1_anchor_check.json` | r=1 锚点逐元素 == 0.0 验证 |
| `r_sensitivity.csv` | r∈{1,2,3,5,10} 的 effect / CI / n |
| `locale_heterogeneity.csv` | 4 地区各自的 effect / CI / n |
| `pairing_sensitivity.csv` | 排除 Romanian + 4 leave-one-locale |
| `length_diagnostic.csv` | truth/lie 长度均值/中位数/差 |
| `length_only_lr.csv` | 25个冻结拆分下的仅字符长度 LR 诊断 |
| `planA_selection_trace.csv` | Plan A 每 (B, method, seed, fold, alpha, rej) |
| `planA_run_level.csv` | Plan A run-level (3B × 2 methods × 5 seed × 5 fold = 150) |
| `planA_unit_method_costs.csv` | Plan A 每方法、seed、配对单位的平均成本 |
| `planA_unit_seed_effects.csv` | Plan A 每预算、seed、配对单位的成本差 |
| `planA_unit_effects.csv` | Plan A 每预算跨seed平均后的1200个单位效应 |
| `planA_tests.csv` | Plan A 单位级效应、CI、sign-flip p、Holm p 和拒判率 |
| `planA_summary.csv` | Plan A 跨 seed 平均 |
| `planB_summary.csv` | Plan B 成本分解（3 行 + CS-vs-M） |
| `protocol_fixed.json` | 协议记录 |
| `final_consistency_audit.json` | 最终算术与统计完整性机器审计 |
| `cc_stage1_data.py` / `cc_stage2_split.py` / `cc_stage3_npz.py` / `cc_stage4_analyze.py` / `cc_fix_subject_map.py` | 可复现脚本 |

**原始数据只读**：全程未修改 `data/deception/crossCulturalDeception.2014/` 下任何文件；`<DECEPTION_DATA>/subject_map.csv` 等历史只读文件未动。
