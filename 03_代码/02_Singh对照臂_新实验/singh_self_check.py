# -*- coding: utf-8 -*-
"""自检清单：验证 Singh 对照臂结果的正确性。"""
import os

_HOME = os.path.expanduser("~")
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
import pandas as pd
import os

OUT = os.path.join(_PKG_ROOT, '05_结果', '01_严格重算主结果')

print("=" * 60)
print("Singh 对照臂自检清单")
print("=" * 60)

# 加载数据
stat = pd.read_csv(os.path.join(OUT, 'singh_eps_hum_comparison.csv'))
run = pd.read_csv(os.path.join(OUT, 'singh_run_level.csv'))
be = pd.read_csv(os.path.join(OUT, 'singh_breakeven_vs_bayes.csv'))
subj = pd.read_csv(os.path.join(OUT, 'singh_subject_level.csv'))

all_pass = True

def check(name, condition, detail=""):
    global all_pass
    status = "PASS" if condition else "FAIL"
    if not condition:
        all_pass = False
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))

# 1. r=1 时 CS==Singh 逐元素一致
print("\n--- 1. r=1 退化一致性 ---")
r1 = stat[stat['cost_ratio'] == 1.0]
check("r=1 全部单元 Δcost=0", np.allclose(r1['mean_delta'], 0, atol=1e-10),
      f"max|Δ|={r1['mean_delta'].abs().max():.2e}, n={len(r1)}")
check("r=1 permutation_p=1.0", np.allclose(r1['permutation_p'], 1.0),
      f"min p={r1['permutation_p'].min()}")

# 2. ε_hum=0 时 Singh==uni（与 strict 权威数字一致）
print("\n--- 2. ε=0 复现对账（与 strict 权威数字比对）---")
strict_stat = pd.read_csv(os.path.join(OUT, 'strict_subject_aggregated_stats.csv'))
eps0 = stat[stat['eps_hum'] == 0.0].copy()
# 对齐键
merged = eps0.merge(strict_stat, on=['dataset', 'config', 'cost_ratio', 'C_rev'],
                     suffixes=('_singh', '_strict'))
check("ε=0 与 strict 全部单元对齐", len(merged) == len(eps0),
      f"singh={len(eps0)}, merged={len(merged)}")
check("ε=0 mean_delta 与 strict 一致",
      np.allclose(merged['mean_delta_singh'], merged['mean_delta_strict'], atol=1e-4),
      f"max diff={np.abs(merged['mean_delta_singh']-merged['mean_delta_strict']).max():.2e}")
ci_diff_lo = np.abs(merged['ci_2.5'] - merged['bootstrap_ci_2.5']).max()
ci_diff_hi = np.abs(merged['ci_97.5'] - merged['bootstrap_ci_97.5']).max()
check("ε=0 CI 与 strict 一致",
      np.allclose(merged['ci_2.5'], merged['bootstrap_ci_2.5'], atol=1e-3) and
      np.allclose(merged['ci_97.5'], merged['bootstrap_ci_97.5'], atol=1e-3),
      f"CI max diff={max(ci_diff_lo, ci_diff_hi):.2e}")

# 3. 所有成本非负
print("\n--- 3. 成本非负性 ---")
check("run-level expected_cost >= 0", (run['expected_cost'] >= 0).all(),
      f"min={run['expected_cost'].min():.6f}")
check("run-level total_cost >= 0", (run['total_cost'] >= 0).all(),
      f"min={run['total_cost'].min():.2f}")

# 4. lie 类覆盖率 CS >= Singh（CS 对 lie 更严格）
print("\n--- 4. lie 类覆盖率方向（CS >= Singh）---")
cs_run = run[run['method'] == 'cs_mondrian'].copy()
singh_run = run[(run['method'] == 'singh_eps0')].copy()
cov_merged = cs_run.merge(singh_run, on=['dataset', 'config', 'fold', 'seed', 'cost_ratio', 'C_rev'],
                           suffixes=('_cs', '_singh'))
# 只看 r>1（r=1 时两者相同）
cov_r_gt1 = cov_merged[cov_merged['cost_ratio'] > 1.0]
check("r>1 时 CS lie 覆盖率 >= Singh",
      (cov_r_gt1['label_coverage_lie_cs'] >= cov_r_gt1['label_coverage_lie_singh'] - 1e-6).all(),
      f"CS lie cov mean={cov_r_gt1['label_coverage_lie_cs'].mean():.4f}, "
      f"Singh lie cov mean={cov_r_gt1['label_coverage_lie_singh'].mean():.4f}")
check("r>1 时 CS reject 率 >= Singh",
      (cov_r_gt1['reject_rate_cs'] >= cov_r_gt1['reject_rate_singh'] - 1e-6).all(),
      f"CS rej mean={cov_r_gt1['reject_rate_cs'].mean():.4f}, "
      f"Singh rej mean={cov_r_gt1['reject_rate_singh'].mean():.4f}")
check("r>1 时 CS FN 率 <= Singh",
      (cov_r_gt1['fn_rate_cs'] <= cov_r_gt1['fn_rate_singh'] + 1e-6).all(),
      f"CS FN mean={cov_r_gt1['fn_rate_cs'].mean():.4f}, "
      f"Singh FN mean={cov_r_gt1['fn_rate_singh'].mean():.4f}")

# 5. Holm 校正正确性
print("\n--- 5. Holm 校正 ---")
for eps in [0.0, 0.05, 0.1, 0.2]:
    mask = ((stat['cost_ratio'] == 3) & (stat['C_rev'] == 0.5) & (stat['eps_hum'] == eps))
    pvals = stat.loc[mask, 'permutation_p'].values
    holm = stat.loc[mask, 'holm_p'].values
    # Holm: adjusted >= raw, adjusted <= 1, monotonic
    check(f"ε={eps} Holm >= raw", (holm >= pvals - 1e-10).all(),
          f"max(raw-holm)={np.max(pvals-holm):.2e}")
    check(f"ε={eps} Holm <= 1", (holm <= 1.0 + 1e-10).all())
    check(f"ε={eps} Holm 单调（排序后非降）",
          np.all(np.diff(np.sort(holm)) >= -1e-10))

# 6. Bayes-τ* 无 reject
print("\n--- 6. Bayes-τ* 性质 ---")
bayes_run = run[run['method'] == 'bayes_tau']
check("Bayes reject_rate=0", (bayes_run['reject_rate'] == 0).all())
check("Bayes decision_coverage=1", np.allclose(bayes_run['decision_coverage'], 1.0))
check("Bayes tau_star=1/(1+r)",
      np.allclose(bayes_run['q_truth'], 1.0/(1.0+bayes_run['cost_ratio']), atol=1e-10))

# 7. ε_hum 单调性（Singh 成本随 ε 增加而增加）
print("\n--- 7. ε_hum 单调性 ---")
singh_all = run[run['method'].isin(['singh_eps0', 'singh'])].copy()
eps_grid = [0.0, 0.05, 0.1, 0.2]
for i in range(1, len(eps_grid)):
    eps = eps_grid[i]
    prev_eps = eps_grid[i-1]
    prev = singh_all[singh_all['eps_hum'] == prev_eps]['expected_cost'].mean()
    curr = singh_all[singh_all['eps_hum'] == eps]['expected_cost'].mean()
    check(f"Singh ε={eps} 成本 > ε={prev_eps}",
          curr > prev, f"{prev:.4f} -> {curr:.4f}")

# 8. C*_rev 合理性
print("\n--- 8. C*_rev 合理性 ---")
check("C*_rev 有限", be['C_star_rev'].notna().all() and np.isfinite(be['C_star_rev']).all())
check("C*_rev > 0", (be['C_star_rev'] > 0).all(), f"min={be['C_star_rev'].min():.4f}")
check("C*_rev CI 包含点估计",
      ((be['C_star_ci_2.5'] <= be['C_star_rev'] + 1e-6) &
       (be['C_star_rev'] <= be['C_star_ci_97.5'] + 1e-6)).all())

# 9. 外部数据
print("\n--- 9. 外部数据基本检查 ---")
ext_stat = pd.read_csv(os.path.join(OUT, 'singh_external_eps_hum_comparison.csv'))
check("外部 Open-Domain n=511",
      (ext_stat[ext_stat['dataset']=='Open-Domain']['n_units'] == 511).all())
check("外部 Cross-Cultural n=1200",
      (ext_stat[ext_stat['dataset']=='Cross-Cultural']['n_units'] == 1200).all())
ext_be = pd.read_csv(os.path.join(OUT, 'singh_external_breakeven_vs_bayes.csv'))
check("外部 C*_rev 有限", ext_be['C_star_rev'].notna().all())

# 10. subject-level 明细检查
print("\n--- 10. subject-level 明细检查 ---")
# 混淆计数恒等式：n_accept = n_fp + n_fn + n_tp + n_tn（accept 样本才计入混淆矩阵）
# n_test = n_accept + n_reject（reject 样本不计入混淆矩阵）
check("subject-level 混淆计数恒等式 (n_accept = n_fp+n_fn+n_tp+n_tn)",
      (subj['n_accept'] == subj['n_fp'] + subj['n_fn'] + subj['n_tp'] + subj['n_tn']).all(),
      f"violations={(subj['n_accept'] != subj['n_fp']+subj['n_fn']+subj['n_tp']+subj['n_tn']).sum()}")
check("subject-level n_test = n_accept + n_reject",
      (subj['n_test'] == subj['n_accept'] + subj['n_reject']).all(),
      f"violations={(subj['n_test'] != subj['n_accept']+subj['n_reject']).sum()}")
check("Bayes subject-level reject rate = 0",
      (subj[subj['method'] == 'bayes_tau']['n_reject'] == 0).all(),
      f"max reject={subj[subj['method']=='bayes_tau']['n_reject'].max()}")
check("Bayes subject-level n_tp > 0 (not hardcoded to 0)",
      (subj[subj['method'] == 'bayes_tau']['n_tp'] > 0).any(),
      f"max n_tp={subj[subj['method']=='bayes_tau']['n_tp'].max()}")
check("Bayes subject-level n_tn > 0 (not hardcoded to 0)",
      (subj[subj['method'] == 'bayes_tau']['n_tn'] > 0).any(),
      f"max n_tn={subj[subj['method']=='bayes_tau']['n_tn'].max()}")
# run-level 与 subject-level 成本一致性：比较 total_cost 之和（按 run 聚合）
run_agg = run.groupby(['dataset','config','fold','seed','method','cost_ratio','C_rev','eps_hum']).agg(
    run_total_cost=('total_cost','sum'),
    run_n_test=('n_test','sum')).reset_index()
subj_agg = subj.groupby(['dataset','config','fold','seed','method','cost_ratio','C_rev','eps_hum']).agg(
    subj_total_cost=('total_cost','sum'),
    subj_n_test=('n_test','sum')).reset_index()
cost_merge = run_agg.merge(subj_agg, on=['dataset','config','fold','seed','method','cost_ratio','C_rev','eps_hum'])
check("run-level 与 subject-level total_cost 一致 (<1e-6)",
      np.allclose(cost_merge['run_total_cost'], cost_merge['subj_total_cost'], atol=1e-6),
      f"max diff={np.abs(cost_merge['run_total_cost']-cost_merge['subj_total_cost']).max():.2e}")
check("run-level 与 subject-level n_test 一致",
      (cost_merge['run_n_test'] == cost_merge['subj_n_test']).all(),
      f"max diff={np.abs(cost_merge['run_n_test']-cost_merge['subj_n_test']).max()}")

print("\n" + "=" * 60)
print(f"自检结果: {'全部 PASS' if all_pass else '存在 FAIL'}")
print("=" * 60)
