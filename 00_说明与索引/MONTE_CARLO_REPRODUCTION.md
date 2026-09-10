# Monte Carlo 修订与后处理复现

随机 sign-flip 检验使用 `p=(b+1)/(B+1)`，其中 `B=10000`，
`b` 为模拟的配对差均值绝对值大于或等于观测均值绝对值的次数。
保持原随机种子、受试者/贡献单位聚合、双侧定义和重采样次数不变。
零差值情形得到 p=1；最小可报告 Monte Carlo p 值为 1/10001。
此修订不改变精确二项符号检验或 Wilcoxon 检验。

中文分析先在同一受试者内跨 fold×seed 聚合，再对独立受试者重采样。
Open-Domain 使用受试者单位，Cross-Cultural 使用脚本定义的配对贡献单位；
不能把贡献单位直接表述为已确认的独立受试者。
Bootstrap 区间为原实现的百分位区间。各脚本已有的 Holm 比较族随原始 p 值重算，
不新增比较族或把探索性比较升级为预先指定的主比较。

## 运行

在 Python 3.11 的全新虚拟环境中：

```bash
python -m pip install -r 03_代码/requirements-repro.txt
python 03_代码/test_monte_carlo.py
python 03_代码/run_postprocessing.py --data /path/to/private/04_数据 --output /path/to/new-empty-output
```

输入目录保留完整包的四个子目录：

- `01_中文_SEUMLD与MDPE`：两个 `*_preds` 目录和 `subject_map.csv`。
- `02_OpenDomain_修复后权威`：25 个 NPZ。
- `03_CrossCultural_v1_论文锚点`：历史版本，本运行器不使用。
- `04_CrossCultural_v2_Singh口径`：25 个 NPZ，供 Singh 和替代调度外部分析使用。

运行器重算严格核心、Singh 中文及外部对照、替代调度中文及外部消融、容量扩展。
每步独立运行，并记录退出状态、源代码哈希、耗时和环境版本。
输出目录必须为空，禁止把旧结果混入新结果。单位明细、输入哈希和执行日志仅保留在私有输出目录。

本范围不包括基础模型训练、PlanE、原始外部 backbone 流水线及 Cross-Cultural v1/v2 合并版本论证。
PlanE 原实现已使用加一校正，本次不机械修改其检验。
可选历史 pre-strict 比较通过 `STRICT_OLD_STATS` 提供原始旧统计表；未提供时明确跳过。
它不影响当前严格版本的统计计算。

## 验证结果（2026-09-10）

Python 3.11.9，全新虚拟环境、空输出目录：十步全部通过。
严格核心 11/11 自检通过；容量扩展 100/100 回归一致（容差 1e-8），100/100 泄漏审计通过。
20 份同名公开 CSV 完成比对，结构和行数一致，除 permutation_p 和 holm_p 外无数值变化（rtol=1e-10，atol=1e-12）。

| 聚合表 | 更新的原始 p 值数 | 原零 p 值数 |
|---|---:|---:|
| strict_subject_aggregated_stats | 64 | 44 |
| singh_eps_hum_comparison | 256 | 167 |
| singh_external_eps_hum_comparison | 128 | 109 |
| schedule_ablation_comparison | 480 | 238 |
| schedule_ablation_external_comparison | 128 | 124 |

五表共消除 682 个 Monte Carlo 零值，原始及 Holm p 值均无跨越 0.05 的变化。
全部原始 p 值与 `(old_p*10000+1)/10001` 一致；均值和置信区间不变。
MDPE 两个严格主终点的 p 值为 1/10001，约 0.000100；SEUMLD text 为 0.118588，audio 为 0.023498。
SEUMLD text 未达到 0.05，不应表述为显著或接近显著。
未纳入运行器的历史报告、图件及其他结果不属于此次复算验证范围。
