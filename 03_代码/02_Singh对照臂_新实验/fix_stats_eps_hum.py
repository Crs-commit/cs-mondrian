# -*- coding: utf-8 -*-
"""
修复脚本：重新运行 CS vs Singh 统计，正确按 eps_hum 筛选。
读取已保存的 singh_subject_level.csv，不重新计算核心指标。
"""
import os

_HOME = os.path.expanduser("~")
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
import pandas as pd
import os
from scipy import stats

INTEGRATION_ROOT = _PKG_ROOT
OUT_DIR = os.path.join(INTEGRATION_ROOT, '02_严格重算输出')

DATASETS = ['SEUMLD', 'MDPE']
CONFIGS = ['OADNet_text', 'OADNet_audio']
COST_RATIOS = [1.0, 2.0, 3.0, 5.0, 10.0]
C_REV_VALUES = [0.25, 0.5, 1.0, 2.0]
EPS_HUM_VALUES = [0.0, 0.05, 0.1, 0.2]

RNG_SEED = 42
N_BOOT = 10000
N_PERM = 10000


def subject_level_stats(deltas, rng, n_boot=N_BOOT, n_perm=N_PERM):
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    if n < 5:
        return None
    obs_mean = float(np.mean(deltas))
    obs_median = float(np.median(deltas))
    boot_idx = rng.integers(0, n, (n_boot, n))
    boot_means = np.mean(deltas[boot_idx], axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    perm_signs = rng.choice([-1, 1], size=(n_perm, n))
    perm_means = np.mean(deltas[None, :] * perm_signs, axis=1)
    p_perm = float(np.mean(np.abs(perm_means) >= np.abs(obs_mean)))
    try:
        if np.all(deltas == 0):
            w_p = 1.0
        else:
            with np.errstate(all='ignore'):
                w_p = float(stats.wilcoxon(deltas).pvalue)
    except Exception:
        w_p = np.nan
    n_pos = int(np.sum(deltas > 0))
    n_neg = int(np.sum(deltas < 0))
    n_zero = int(np.sum(deltas == 0))
    if n_pos + n_neg > 0:
        sign_p = float(stats.binomtest(min(n_pos, n_neg), n_pos + n_neg, 0.5).pvalue)
    else:
        sign_p = 1.0
    return {
        'n_subjects': n, 'mean_delta': obs_mean, 'median_delta': obs_median,
        'ci_2.5': float(ci_low), 'ci_97.5': float(ci_high),
        'permutation_p': p_perm, 'wilcoxon_p': w_p, 'sign_p': sign_p,
        'n_positive': n_pos, 'n_negative': n_neg, 'n_zero': n_zero,
        'cs_wins_pct': float(np.mean(deltas < 0) * 100),
    }


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, p[idx] * (n - rank))
        if rank > 0:
            adjusted[idx] = max(adjusted[idx], adjusted[order[rank - 1]])
    return adjusted


def main():
    print("=== Fix: CS vs Singh stats with correct eps_hum filtering ===")
    subj_df = pd.read_csv(os.path.join(OUT_DIR, 'singh_subject_level.csv'))
    print(f"Loaded subject-level: {len(subj_df)} rows")
    print(f"methods: {subj_df['method'].unique()}")
    print(f"eps_hum values: {sorted(subj_df['eps_hum'].unique())}")

    rng = np.random.default_rng(RNG_SEED)
    records = []

    for eps_hum in EPS_HUM_VALUES:
        # CS 和 Singh 都按相同 eps_hum 筛选（公平比较：两者共用相同人工成本模型）
        cs = subj_df[(subj_df['method'] == 'cs_mondrian') &
                     (subj_df['eps_hum'] == eps_hum)].copy()
        singh_method = 'singh' if eps_hum > 0 else 'singh_eps0'
        # 同时按 method 和 eps_hum 筛选
        si = subj_df[(subj_df['method'] == singh_method) &
                     (subj_df['eps_hum'] == eps_hum)].copy()
        print(f"\neps_hum={eps_hum}: singh rows={len(si)}, cs rows={len(cs)}")

        merged = cs.merge(
            si, on=['dataset', 'config', 'fold', 'seed', 'cost_ratio', 'C_rev',
                    'subject_id', 'n_segments'], suffixes=('_cs', '_singh'))
        merged['delta_cost'] = merged['expected_cost_cs'] - merged['expected_cost_singh']
        print(f"  merged rows: {len(merged)}")

        agg = merged.groupby(['dataset', 'config', 'subject_id', 'cost_ratio', 'C_rev']).agg(
            delta_cost=('delta_cost', 'mean'),
            n_fold_seed=('delta_cost', 'count')).reset_index()

        for dataset in DATASETS:
            for config in CONFIGS:
                for r in COST_RATIOS:
                    for C_rev in C_REV_VALUES:
                        mask = ((agg['dataset'] == dataset) & (agg['config'] == config) &
                                (agg['cost_ratio'] == r) & (agg['C_rev'] == C_rev))
                        deltas = agg.loc[mask, 'delta_cost'].values
                        st = subject_level_stats(deltas, rng)
                        if st is None:
                            continue
                        records.append({
                            'dataset': dataset, 'config': config, 'cost_ratio': r,
                            'C_rev': C_rev, 'eps_hum': eps_hum, **st})

    stat_df = pd.DataFrame(records)

    # Holm 校正：四个中文配置主族（r=3, C_rev=0.5），按 eps_hum 分组
    for eps_hum in EPS_HUM_VALUES:
        mask = ((stat_df['cost_ratio'] == 3) & (stat_df['C_rev'] == 0.5) &
                (stat_df['eps_hum'] == eps_hum))
        if mask.sum() > 0:
            pvals = stat_df.loc[mask, 'permutation_p'].values
            stat_df.loc[mask, 'holm_p'] = holm_adjust(pvals)

    stat_df.to_csv(os.path.join(OUT_DIR, 'singh_eps_hum_comparison.csv'), index=False)
    print(f"\nStats cells: {len(stat_df)}")

    # 打印主终点
    print("\n=== PRIMARY ENDPOINT (r=3, C_rev=0.5) ===")
    primary = stat_df[(stat_df['cost_ratio'] == 3) & (stat_df['C_rev'] == 0.5)]
    print(primary[['dataset', 'config', 'eps_hum', 'n_subjects', 'mean_delta',
                    'ci_2.5', 'ci_97.5', 'permutation_p', 'holm_p', 'cs_wins_pct']].to_string(index=False))

    return stat_df


if __name__ == '__main__':
    main()
