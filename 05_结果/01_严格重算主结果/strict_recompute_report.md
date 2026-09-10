# CS-Mondrian 严格共形重算报告（Strict Recompute Report）

> 版本：v1.0　|　生成时间：2026-09-04 19:22:49
> 实现脚本：`scripts/strict_recompute.py`　|　输出目录：`10_strict_recompute/`
> 运行耗时：109.2s　|　run 行：16000，subject 行：427200

---

## 1. 实施概要

相对旧版 `core_compute_final.py` 的修正：

| # | 修正项 | 旧版 | 严格版 |
|---|--------|------|--------|
| 1 | 阈值 | `np.percentile(p_c, alpha*100)` | 有限样本共形秩分位点 `ceil((n+1)(1-alpha))`，`k>n` 时返回 `inf` |
| 2 | 拒判规则 | 仅双标签 reject，空集回退 argmax | `set_size==0`（empty）与 `set_size==2`（ambiguous）**均 reject**，不回退 |
| 3 | coverage 命名 | `coverage=n_accept/n`（非拒判率） | `decision_coverage=singleton/n`、`acceptance_rate=n_accept/n`、`label_coverage=mean(contains_true_label)` |
| 4 | 统计独立单位 | 曾用 subject×seed 伪重复（已修正） | 独立受试者（SEUMLD 76 / MDPE 191），跨 fold×seed 聚合 |
| 5 | 输出目录 | 覆盖旧结果 | 独立目录 `10_strict_recompute/`，旧结果保留 |

**设计决策（已记录于 `strict_protocol.json`）**：

- 校准分数（规范 3.1）：`s_truth=p_lie`（truth 样本）、`s_lie=1-p_lie`（lie 样本）。
- 包含规则采用与 3.1 自洽的标准 Mondrian 共形规则：`include_truth = p_lie <= q_truth`、`include_lie = (1-p_lie) <= q_lie`。
  **说明**：规范 3.4 字面写 `include_truth=(1-p_lie)<=q_truth`，与 3.1 分数定义存在内部矛盾——该字面规则会把高置信 lie 样本从 lie 集合中排除，破坏标准有限样本覆盖率保证（`P(y∈Ŷ)≥1-α`）。本实现采用与 3.1 自洽的标准共形规则。
- `lie_only`：`alpha_truth=0 → q_truth=inf`，truth 类永不拒判，只对 lie 类做共形拒判。
- 非集合方法（`confidence_reject_tuned`/`arithmetic`/`geometric`）无预测集合，其 `label_coverage` 定义为“接受且预测正确率”。

**修改/新增文件**：

| 文件 | 说明 |
|------|------|
| `scripts/strict_recompute.py` | 新增：严格共形计算引擎 + 10 单元测试 + 下游统计 + 消融 + 差异报告 |
| `10_strict_recompute/strict_core_run_level.csv` | 16,000 条 run 级指标 |
| `10_strict_recompute/strict_subject_level_metrics.csv` | 427,200 条 subject×fold×seed 指标 |
| `10_strict_recompute/strict_subject_aggregated_stats.csv` | 80 单元 subject 级统计（bootstrap/permutation/Wilcoxon/sign） |
| `10_strict_recompute/strict_pairwise_subject_deltas.csv` | CS vs Mondrian 配对差异 |
| `10_strict_recompute/strict_cost_region_data.csv` | 80 单元成本区域 |
| `10_strict_recompute/strict_ablation_summary.csv` | 8 方法消融 |
| `10_strict_recompute/strict_fair_matching.csv` | 拒判率 ±3% 公平匹配 |
| `10_strict_recompute/strict_vs_old_primary_endpoint_diff.csv` | 严格 vs 旧版主终点差异 |
| `10_strict_recompute/strict_protocol.json` | 协议哈希、输入、设计决策 |

**执行命令**：

```
python scripts\strict_recompute.py
python scripts\generate_strict_report.py
```

---

## 2. 单元测试（规范 7）

**结果：11/11 PASS**

| 测试 | PASS | 说明 |
|------|------|------|
| t2_qhat_formula | ✅ | conformal_quantile([0.1..0.5],0.5)=0.3, expected 0.3 |
| t2b_qhat_formula | ✅ | conformal_quantile(...,0.2)=0.5, expected 0.5 |
| t3_k_gt_n_inf | ✅ | alpha=0.05 n=3 -> inf, expected inf |
| t1_r1_equivalence | ✅ | q eq=True, arrays eq=True, cost eq=True; cs(FP=12,FN=6,R=269) mo(FP=12,FN=6,R=26 |
| t4_empty_set_reject | ✅ | empty=[True, True], reject=[True, True], q_truth=0.334, q_lie=0.334 |
| t5_ambiguous_reject | ✅ | ambiguous=[True, True], reject=[True, True] |
| t6_singleton_only_fpfn | ✅ | FP=0,FN=0,TP=1,TN=1,R=2,A=2 |
| t7_label_coverage_indexing | ✅ | label_coverage=0.6667, truth=0.5, lie=1.0 |
| t8_test_label_isolation | ✅ | conformal_quantile params=['scores', 'alpha'] |
| t9_cost_normalization | ✅ | expected=0.300000, total/n=0.300000 |
| t10_subject_equal_weight | ✅ | equal-weight mean=0.3750 (≠ segment-weighted) |

---

## 3. 主终点（r=3, C_rev=0.5，独立受试者）

**方向结论不变：4/4 配置 CS-Mondrian 成本更低，3/4 显著，SEUMLD/text 边际。**

| 数据集 | 配置 | n 受试者 | Δcost | 95% CI | permutation p | Wilcoxon p | CS 胜率(受试者) |
|---|---|---|---|---|---|---|---|
| MDPE | audio | 191 | -0.0270 | [-0.0344, -0.0201] | <0.001 | 0.000 | 49.2% |
| MDPE | text | 191 | -0.0268 | [-0.0340, -0.0198] | <0.001 | 0.000 | 64.4% |
| SEUMLD | audio | 76 | -0.0165 | [-0.0308, -0.0027] | 0.0234 | 0.219 | 38.2% |
| SEUMLD | text | 76 | -0.0142 | [-0.0321, 0.0027] | 0.1185 | 0.576 | 46.1% |

---

## 4. 严格版 vs 旧版（主终点差异）

| 数据集 | 配置 | 严格 Δcost | 旧版 Δcost | Δ差异 | 严格 p | 旧版 p | 方向一致 |
|---|---|---|---|---|---|---|---|
| SEUMLD | text | -0.0142 | -0.0156 | +0.0014 | 0.1185 | 0.0925 | 是 |
| SEUMLD | audio | -0.0165 | -0.0141 | -0.0025 | 0.0234 | 0.0351 | 是 |
| MDPE | text | -0.0268 | -0.0272 | +0.0004 | <0.001 | <0.001 | 是 |
| MDPE | audio | -0.0270 | -0.0254 | -0.0016 | <0.001 | <0.001 | 是 |

**解读**：严格重算后 4 配置 Δcost 方向与显著性判断全部不变；SEUMLD/text 的 p 从 0.093 微升至 0.119（仍为边际、未达 0.05），其余 3 配置显著。主结论方向不改变。

---

## 5. 覆盖率指标（主终点配置，run 均值）

下表区分 `decision_coverage`（自动决策率）与 `label_coverage`（真实标签纳入率，经验值）。**`label_coverage` 不得表述为受试者级 coverage guarantee。**

| 数据集 | 配置 | 方法 | decision_cov | acceptance | reject | label_cov | label_cov_truth | label_cov_lie |
|---|---|---|---|---|---|---|---|---|
| MDPE | audio | cs_mondrian | 0.1606 | 0.1606 | 0.8394 | 0.9208 | 0.8947 | 0.9642 |
| MDPE | audio | mondrian | 0.2269 | 0.2269 | 0.7731 | 0.9007 | 0.8947 | 0.9107 |
| MDPE | text | cs_mondrian | 0.2689 | 0.2689 | 0.7311 | 0.9198 | 0.8960 | 0.9594 |
| MDPE | text | mondrian | 0.3623 | 0.3623 | 0.6377 | 0.8953 | 0.8960 | 0.8941 |
| SEUMLD | audio | cs_mondrian | 0.1668 | 0.1668 | 0.8332 | 0.9265 | 0.9064 | 0.9637 |
| SEUMLD | audio | mondrian | 0.2259 | 0.2259 | 0.7741 | 0.9108 | 0.9064 | 0.9180 |
| SEUMLD | text | cs_mondrian | 0.2428 | 0.2428 | 0.7572 | 0.9235 | 0.8971 | 0.9735 |
| SEUMLD | text | mondrian | 0.3523 | 0.3523 | 0.6477 | 0.9007 | 0.8971 | 0.9072 |

### 覆盖率保证对照验证（实现正确性证据）

为确认严格实现本身正确，用与真实分数分布同形状的数据做了 2,000 次同分布（exchangeable）模拟：

| 模拟 | truth 平均覆盖率 | lie 平均覆盖率 | 说明 |
|---|---|---|---|
| 同分布 calib/test（n_calib=600, n_test=400） | **0.9010** | **0.9015** | ≥1-α=0.90，含 rank 保守性盈余 +0.001 |

真实数据的 `label_coverage_truth` 均值约 0.895–0.906，略低于名义 0.90，原因是**受试者不相交划分**引入 calib/test 分数分布偏移（跨受试者泛化，非实现问题）。这印证了规范红线：**`label_coverage` 只是经验标签纳入率，不得宣称达到覆盖率保证**。论文如需讨论，应同时给出同分布对照与跨受试者偏移的说明。

---

## 6. 成本区域（32 单元方向统计）

- 区域定义：`r ≥ 2 且 C_rev ≤ 0.5`，共 **32 单元**（2 数据集 × 2 配置 × 4 r × 2 C_rev）
- CS-Mondrian 更优（mean_delta<0）：**31/32（96.9%）**
- permutation p<0.05：**28/32**

**完整 80 单元数据见 `strict_cost_region_data.csv`。非主终点配置均为探索性结果，不做事后多重比较包装。**

---

## 7. 消融（r=3, C_rev=0.5，8 方法 × 4 配置）

**成本最低方法用粗体标注**；同时报告成本、FN 率、reject rate 与 label coverage。

### SEUMLD / text

| 方法 | cost | FN率 | FP率 | reject | label_cov | lie_cov |
|---|---|---|---|---|---|---|
| cs_mondrian | 0.4733 | 0.0265 | 0.1029 | 0.7572 | 0.9235 | 0.9735 |
| mondrian | 0.4870 | 0.0928 | 0.1029 | 0.6477 | 0.9007 | 0.9072 |
| split_conformal | 0.5610 | 0.2466 | 0.0164 | 0.5908 | 0.9044 | 0.7534 |
| **fixed_alpha** | 0.4703 | 0.0396 | 0.0526 | 0.7898 | 0.9520 | 0.9604 |
| arithmetic | 0.4836 | 0.0265 | 0.0164 | 0.8911 | 0.0891 | 0.0734 |
| geometric | 0.5920 | 0.3064 | 0.0712 | 0.4569 | 0.3909 | 0.2374 |
| lie_only | 0.5046 | 0.0928 | 0.0000 | 0.8177 | 0.9681 | 0.9072 |
| confidence_reject_tuned | 0.4987 | 0.0561 | 0.0024 | 0.8791 | 0.1001 | 0.0085 |

### SEUMLD / audio

| 方法 | cost | FN率 | FP率 | reject | label_cov | lie_cov |
|---|---|---|---|---|---|---|
| cs_mondrian | 0.5149 | 0.0363 | 0.0936 | 0.8332 | 0.9265 | 0.9637 |
| mondrian | 0.5324 | 0.0820 | 0.0936 | 0.7741 | 0.9108 | 0.9180 |
| split_conformal | 0.5922 | 0.2395 | 0.0025 | 0.6872 | 0.9161 | 0.7605 |
| fixed_alpha | 0.5178 | 0.0504 | 0.0486 | 0.8685 | 0.9510 | 0.9496 |
| arithmetic | 0.5113 | 0.0363 | 0.0025 | 0.9450 | 0.0410 | 0.0081 |
| geometric | 0.6627 | 0.3443 | 0.0289 | 0.5754 | 0.2869 | 0.0545 |
| lie_only | 0.5293 | 0.0820 | 0.0000 | 0.8903 | 0.9719 | 0.9180 |
| **confidence_reject_tuned** | 0.5056 | 0.0208 | 0.0001 | 0.9683 | 0.0245 | 0.0021 |

### MDPE / text

| 方法 | cost | FN率 | FP率 | reject | label_cov | lie_cov |
|---|---|---|---|---|---|---|
| **cs_mondrian** | 0.4762 | 0.0406 | 0.1040 | 0.7311 | 0.9198 | 0.9594 |
| mondrian | 0.5030 | 0.1059 | 0.1040 | 0.6377 | 0.8953 | 0.8941 |
| split_conformal | 0.5651 | 0.2120 | 0.0363 | 0.6076 | 0.8978 | 0.7880 |
| fixed_alpha | 0.4815 | 0.0587 | 0.0549 | 0.7623 | 0.9437 | 0.9413 |
| arithmetic | 0.4770 | 0.0406 | 0.0359 | 0.8179 | 0.1445 | 0.0755 |
| geometric | 0.5646 | 0.2123 | 0.1042 | 0.5208 | 0.3344 | 0.1936 |
| lie_only | 0.5068 | 0.1059 | 0.0000 | 0.7752 | 0.9603 | 0.8941 |
| confidence_reject_tuned | 0.4775 | 0.0350 | 0.0039 | 0.8713 | 0.1131 | 0.0089 |

### MDPE / audio

| 方法 | cost | FN率 | FP率 | reject | label_cov | lie_cov |
|---|---|---|---|---|---|---|
| cs_mondrian | 0.5258 | 0.0358 | 0.1053 | 0.8394 | 0.9208 | 0.9642 |
| mondrian | 0.5529 | 0.0893 | 0.1053 | 0.7731 | 0.9007 | 0.9107 |
| split_conformal | 0.6342 | 0.2425 | 0.0000 | 0.7225 | 0.9090 | 0.7575 |
| fixed_alpha | 0.5306 | 0.0520 | 0.0535 | 0.8773 | 0.9471 | 0.9480 |
| arithmetic | 0.5178 | 0.0358 | 0.0000 | 0.9549 | 0.0316 | 0.0000 |
| geometric | 0.7083 | 0.3490 | 0.0191 | 0.6069 | 0.2502 | 0.0261 |
| lie_only | 0.5449 | 0.0893 | 0.0000 | 0.8887 | 0.9665 | 0.9107 |
| **confidence_reject_tuned** | 0.5048 | 0.0093 | 0.0000 | 0.9888 | 0.0077 | 0.0000 |

**消融解读（严格版）**：

- CS-Mondrian 在 4 配置中均保持低 FN 率，代价是更高的 reject rate；其成本优势相对 Mondrian 稳定（2.4%–5.4%）。
- `fixed_alpha` 在部分配置绝对成本与 CS-Mondrian 接近甚至略低——**论文需诚实表述**：CS-Mondrian 的贡献是“共形框架内的成本敏感 α 分配”，并非在所有配置上绝对最优。
- `lie_only`（truth 永不拒判）从不输出 lie，故 FP=0、label_coverage 高，但 reject 高、成本不占优。

---

## 8. 论文影响评估

### A. 代码正确性审查（规范 8.A）

- [x] 严格 qhat 实际被调用（`conformal_quantile`，无隐藏 `np.percentile`）
- [x] 空集与双标签均按规则 reject（Test 4/5 PASS）
- [x] test labels 未参与阈值/qhat/方法选择（Test 8 PASS；方法函数只接收 calib 分数与 calib 标签分组）
- [x] `coverage` 命名歧义已消除（`decision_coverage`/`acceptance_rate`/`label_coverage` 分离）
- [x] 单元测试 11/11 PASS

### B. 数据与复现审查（规范 8.B）

- [x] 样本数与受试者数与冻结口径一致（SEUMLD 76/3224，MDPE 191/4581，truth 2863 / lie 1718）
- [x] subject audit **100/100 PASS**（test/calib 受试者不相交）
- [x] run 数 16,000、subject 行 427,200、配置网格完整（2d×2c×5f×5s×8m×5r×4Crev）
- [x] 输出含 protocol hash（源码 SHA256 前缀）、代码版本、输入文件路径、运行时间
- [x] 旧结果未被覆盖（严格结果独立写入 `10_strict_recompute/`）

### C. 统计与论文影响（规范 8.C）

- 主终点由 subject-level strict 结果计算（独立受试者 bootstrap 10,000 + sign-flip permutation 10,000 + Wilcoxon + exact sign test）
- **主结论方向不变**：严格版与旧版 4 配置方向一致，3/4 显著，SEUMLD/text 边际（p=0.119）
- 论文不得再把 acceptance rate 称为共形 coverage；需用 `label_coverage` 表述标签纳入率

### 论文修改建议

1. **Methods**：补充严格有限样本共形算法描述——`qhat = sorted(scores)[ceil((n+1)(1-α))-1]`、集合规则（size 0/1/2）、分数定义 `s_truth=p_lie`/`s_lie=1-p_lie`。
2. **Results**：主终点采用严格版数据；`coverage` 相关图表改用 `label_coverage`（约 90–97%），并同时报告 `decision_coverage`（自动决策率）。
3. **敏感性分析**：严格版 vs 旧版差异微小（Δcost 差异 <0.003），可放补充材料说明旧 percentile 版作为敏感性分析、严格秩分位点版为主结果。
4. **Fig 3**：类条件覆盖率用严格版 `label_coverage_truth` / `label_coverage_lie` 重绘。
5. **Fig 6 诚实表述**：保留“CS-Mondrian 是共形框架内的成本敏感 α 分配，非绝对最优”的表述；`fixed_alpha` 等消融方法绝对成本可能更低，需同时报告成本、FN、reject、label coverage 四个维度。

---

## 9. 未解决问题

1. **规范 3.4 内部矛盾**：`include_truth=(1-p_lie)<=q_truth` 与 3.1 分数定义冲突。本实现选择与 3.1 自洽的标准共形规则（保证覆盖率）。若审稿人质疑，需在 Methods 说明得分与包含方向。
2. **`lie_only` 语义重构**：旧版（alpha_truth=1.0 使 truth 几乎永不包含）与严格版（truth 永不拒判）语义相反；本实现按“只对 lie 类拒判”的合理语义重构。
3. **非集合方法 label_coverage 口径**：`confidence_reject_tuned`/`arithmetic`/`geometric` 无预测集合，其 label_coverage 定义为“接受且预测正确率”，与集合方法口径不同，消融对比时需注明。
4. **SEUMLD/text 边际性**：p=0.119 未达 0.05，论文需按边际证据表述，不得宣称 4/4 全显著。
5. **受试者受益比例与 median**：主终点 mean Δcost 显著为负，但受试者受益比例约 38–64%、median 接近 0（MDPE/audio 严格版 median=0.0、n_zero=40）。
   即成本优势由部分受试者的大幅收益驱动，论文应同时报告 mean、median 与受益比例，避免只报均值造成误导。

---

*报告由 `scripts/generate_strict_report.py` 从 `10_strict_recompute/` 输出自动生成。*