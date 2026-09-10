# -*- coding: utf-8 -*-
"""
外部两数据集替代调度消融（只跑 S1 reverse、S2 oracle-scalar-α）
- Open-Domain: 受试者级 (n=511)
- Cross-Cultural: 配对贡献单位级 (n=1200)
复用 schedule_ablation_recompute.py 的策略函数和 singh_arm_recompute.py 的核心函数。
"""
import os

_HOME = os.path.expanduser("~")
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
import pandas as pd
import os
import sys
import time
from pathlib import Path
from scipy import stats

SINGH_DIR = os.path.join(_PKG_ROOT, '03_代码', '02_Singh对照臂_新实验')
WORKDIR = os.path.join(_PKG_ROOT, '03_代码', '02_Singh对照臂_新实验')
ABLATION_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, SINGH_DIR)
sys.path.insert(0, WORKDIR)
sys.path.insert(0, ABLATION_DIR)

from singh_arm_recompute import (
    conformal_quantile, build_flags, method_mondrian, method_cs_mondrian,
    compute_metrics_base, bayes_threshold_cost, holm_adjust
)
from schedule_ablation_recompute import (
    strategy_s1_reverse, strategy_s2_oracle_scalar, strategy_cs_budget_selected, cs_budget,
    ALPHA_TOTAL, C_FP, COST_RATIOS, C_REV_VALUES, SELECTION_BUDGETS,
    RNG_SEED, N_BOOT, N_PERM
)

INTEGRATION_ROOT = _PKG_ROOT
EXT_ROOT = os.path.join(INTEGRATION_ROOT, '12_外部验证与扩展实验_2026-09-07')
OUT_DIR = os.path.join(INTEGRATION_ROOT, '02_严格重算输出')

OPEN_DOMAIN_PREDS = os.path.join(EXT_ROOT, r'05_双外部验证_结果\修复后_权威\ext_validation_fixed\OpenDeception\preds')
CROSS_CULTURAL_PREDS = os.path.join(EXT_ROOT, r'09_GlobalAlpha_v2_修复定稿\global_alpha_opt_v2\CrossCultural_preds_v2')

FOLDS = [0, 1, 2, 3, 4]
SEEDS = [7, 42, 123, 2024, 2026]


def load_ext_npz(preds_dir, fold, seed):
    path = os.path.join(preds_dir, f'TFIDF_LR_f{fold}_s{seed}.npz')
    d = np.load(path, allow_pickle=True)
    return {
        'probs': d['probs'], 'labels': d['labels'],
        'calib_probs': d['calib_probs'], 'calib_labels': d['calib_labels'],
        'val_probs': d['val_probs'], 'val_labels': d['val_labels'],
        'test_unit_ids': d['test_subject_ids'],
    }


def compute_unit_level_costs(labels, flags, unit_ids, C_FP, C_FN, C_rev):
    """单位级平均成本（受试者或配对贡献单位）。"""
    labels = np.asarray(labels)
    reject = flags['reject']
    singleton = flags['singleton']
    pred_label = flags['pred_label']
    unit_ids = np.asarray(unit_ids)
    sample_cost = np.zeros(len(labels), dtype=float)
    auto_mask = singleton
    sample_cost[auto_mask & (pred_label == 1) & (labels == 0)] = C_FP
    sample_cost[auto_mask & (pred_label == 0) & (labels == 1)] = C_FN
    sample_cost[reject] = C_rev
    units, inv = np.unique(unit_ids, return_inverse=True)
    unit_cost = np.zeros(len(units), dtype=float)
    unit_count = np.zeros(len(units), dtype=int)
    np.add.at(unit_cost, inv, sample_cost)
    np.add.at(unit_count, inv, 1)
    return dict(zip(units, unit_cost / unit_count)), units


def unit_level_stats(deltas, rng, n_boot=N_BOOT, n_perm=N_PERM):
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    if n < 5:
        return None
    obs_mean = float(np.mean(deltas))
    boot_idx = rng.integers(0, n, (n_boot, n))
    boot_means = np.mean(deltas[boot_idx], axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    perm_signs = rng.choice([-1, 1], size=(n_perm, n))
    perm_means = np.mean(deltas[None, :] * perm_signs, axis=1)
    p_perm = float(np.mean(np.abs(perm_means) >= np.abs(obs_mean)))
    try:
        w_p = float(stats.wilcoxon(deltas).pvalue) if not np.all(deltas == 0) else 1.0
    except Exception:
        w_p = np.nan
    return {
        'n_units': n, 'mean_delta': obs_mean,
        'ci_2.5': float(ci_low), 'ci_97.5': float(ci_high),
        'permutation_p': p_perm, 'wilcoxon_p': w_p,
        'cs_wins_pct': float(np.mean(deltas < 0) * 100),
    }


def run_external_dataset(name, preds_dir, unit_type):
    """运行单个外部数据集的 S1+S2。"""
    print(f"\n{'='*60}")
    print(f"External: {name} ({unit_type})")
    print(f"{'='*60}")
    t0 = time.time()

    # 收集单位级成本
    unit_cost_data = {}  # key=(r, C_rev, method, B) -> {(fold,seed): {unit_id: cost}}
    all_units = set()
    run_rows = []

    for fold in FOLDS:
        for seed in SEEDS:
            d = load_ext_npz(preds_dir, fold, seed)
            test_probs = d['probs']
            test_labels = d['labels']
            calib_probs = d['calib_probs']
            calib_labels = d['calib_labels']
            val_probs = d['val_probs']
            val_labels = d['val_labels']
            test_unit_ids = d['test_unit_ids']
            all_units.update(test_unit_ids)

            for r in COST_RATIOS:
                C_FN = r * C_FP
                # CS
                flags_cs, _, _ = method_cs_mondrian(
                    calib_probs, calib_labels, test_probs, ALPHA_TOTAL, r)
                # S1 reverse
                at_s1, al_s1 = strategy_s1_reverse(ALPHA_TOTAL, r)
                flags_s1, _, _ = method_mondrian(
                    calib_probs, calib_labels, test_probs, at_s1, al_s1)

                for C_rev in C_REV_VALUES:
                    # CS run-level
                    m_cs = compute_metrics_base(test_labels, flags_cs, C_FP, C_FN, C_rev, 0.0)
                    run_rows.append({'dataset': name, 'fold': fold, 'seed': seed,
                                     'method': 'cs_mondrian', 'cost_ratio': r, 'C_rev': C_rev,
                                     'B': np.nan, **m_cs})
                    uc_cs, _ = compute_unit_level_costs(
                        test_labels, flags_cs, test_unit_ids, C_FP, C_FN, C_rev)
                    unit_cost_data[(r, C_rev, 'cs_mondrian', np.nan)] = \
                        unit_cost_data.get((r, C_rev, 'cs_mondrian', np.nan), {})
                    unit_cost_data[(r, C_rev, 'cs_mondrian', np.nan)][(fold, seed)] = uc_cs

                    # S1
                    m_s1 = compute_metrics_base(test_labels, flags_s1, C_FP, C_FN, C_rev, 0.0)
                    run_rows.append({'dataset': name, 'fold': fold, 'seed': seed,
                                     'method': 's1_reverse', 'cost_ratio': r, 'C_rev': C_rev,
                                     'B': np.nan, **m_s1})
                    uc_s1, _ = compute_unit_level_costs(
                        test_labels, flags_s1, test_unit_ids, C_FP, C_FN, C_rev)
                    key = (r, C_rev, 's1_reverse', np.nan)
                    unit_cost_data[key] = unit_cost_data.get(key, {})
                    unit_cost_data[key][(fold, seed)] = uc_s1

                    # S2 (每个 B)
                    for B in SELECTION_BUDGETS:
                        a_cs_b, _, _ = strategy_cs_budget_selected(
                            calib_probs, calib_labels, val_probs, val_labels,
                            r, C_rev, B)
                        flags_cs_b, _, _ = method_cs_mondrian(
                            calib_probs, calib_labels, test_probs, a_cs_b, r)
                        uc_cs_b, _ = compute_unit_level_costs(
                            test_labels, flags_cs_b, test_unit_ids, C_FP, C_FN, C_rev)
                        key = (r, C_rev, 'cs_budget_selected', B)
                        unit_cost_data[key] = unit_cost_data.get(key, {})
                        unit_cost_data[key][(fold, seed)] = uc_cs_b
                        m_cs_b = compute_metrics_base(test_labels, flags_cs_b, C_FP, C_FN, C_rev, 0.0)
                        run_rows.append({'dataset': name, 'fold': fold, 'seed': seed,
                                         'method': 'cs_budget_selected', 'cost_ratio': r,
                                         'C_rev': C_rev, 'B': B, **m_cs_b})
                        alpha0_s2, _, _ = strategy_s2_oracle_scalar(
                            calib_probs, calib_labels, val_probs, val_labels, r, C_rev, B, ALPHA_TOTAL)
                        flags_s2, _, _ = method_mondrian(
                            calib_probs, calib_labels, test_probs, alpha0_s2, alpha0_s2)
                        m_s2 = compute_metrics_base(test_labels, flags_s2, C_FP, C_FN, C_rev, 0.0)
                        run_rows.append({'dataset': name, 'fold': fold, 'seed': seed,
                                         'method': 's2_oracle_scalar_alpha', 'cost_ratio': r,
                                         'C_rev': C_rev, 'B': B, **m_s2})
                        uc_s2, _ = compute_unit_level_costs(
                            test_labels, flags_s2, test_unit_ids, C_FP, C_FN, C_rev)
                        key = (r, C_rev, 's2_oracle_scalar_alpha', B)
                        unit_cost_data[key] = unit_cost_data.get(key, {})
                        unit_cost_data[key][(fold, seed)] = uc_s2

    print(f"  Core compute done in {time.time()-t0:.1f}s, {len(all_units)} units")
    run_df = pd.DataFrame(run_rows)

    # 统计：策略 vs CS；S2 uses the same validation-budget-selected CS comparator.
    rng = np.random.default_rng(RNG_SEED)
    stat_records = []
    sorted_units = sorted(all_units)

    for method in ['s1_reverse', 's2_oracle_scalar_alpha']:
        b_list = SELECTION_BUDGETS if method == 's2_oracle_scalar_alpha' else [np.nan]
        for B in b_list:
            for r in COST_RATIOS:
                for C_rev in C_REV_VALUES:
                    cs_method = 'cs_budget_selected' if method == 's2_oracle_scalar_alpha' else 'cs_mondrian'
                    cs_B = B if method == 's2_oracle_scalar_alpha' else np.nan
                    cs_key = (r, C_rev, cs_method, cs_B)
                    strat_key = (r, C_rev, method, B)
                    if cs_key not in unit_cost_data or strat_key not in unit_cost_data:
                        continue
                    cs_data = unit_cost_data[cs_key]
                    strat_data = unit_cost_data[strat_key]
                    unit_deltas = []
                    for unit in sorted_units:
                        cs_vals, strat_vals = [], []
                        for fs in cs_data:
                            if unit in cs_data[fs] and unit in strat_data.get(fs, {}):
                                cs_vals.append(cs_data[fs][unit])
                                strat_vals.append(strat_data[fs][unit])
                        if cs_vals:
                            unit_deltas.append(np.mean(strat_vals) - np.mean(cs_vals))
                    deltas = np.array(unit_deltas)
                    st = unit_level_stats(deltas, rng)
                    if st is None:
                        continue
                    stat_records.append({
                        'dataset': name, 'unit_type': unit_type, 'method': method,
                        'B': B, 'cost_ratio': r, 'C_rev': C_rev, **st})

    stat_df = pd.DataFrame(stat_records)
    # Holm（外部为次要族，按 method+B 分组，2 配置比较）
    for method in ['s1_reverse', 's2_oracle_scalar_alpha']:
        b_list = SELECTION_BUDGETS if method == 's2_oracle_scalar_alpha' else [np.nan]
        for B in b_list:
            mask = ((stat_df['cost_ratio'] == 3) & (stat_df['C_rev'] == 0.5) &
                    (stat_df['method'] == method))
            if method == 's2_oracle_scalar_alpha':
                mask = mask & (stat_df['B'] == B)
            if mask.sum() > 0:
                pvals = stat_df.loc[mask, 'permutation_p'].values
                stat_df.loc[mask, 'holm_p'] = holm_adjust(pvals)

    print(f"  Stats: {len(stat_df)} cells")
    # 主终点
    print(f"\n=== {name} PRIMARY (r=3, Crev=.5) ===")
    primary = stat_df[(stat_df['cost_ratio'] == 3) & (stat_df['C_rev'] == 0.5)]
    primary = primary[(primary['method'] != 's2_oracle_scalar_alpha') | (primary['B'] == 0.7)]
    print(primary[['method', 'B', 'n_units', 'mean_delta', 'ci_2.5', 'ci_97.5',
                    'permutation_p', 'holm_p']].to_string(index=False))

    print(f"\n{name} elapsed: {time.time()-t0:.1f}s")
    return run_df, stat_df


def main():
    t_start = time.time()
    od_run, od_stat = run_external_dataset('Open-Domain', OPEN_DOMAIN_PREDS, 'subject')
    cc_run, cc_stat = run_external_dataset('Cross-Cultural', CROSS_CULTURAL_PREDS, 'paired_unit')

    ext_run = pd.concat([od_run, cc_run], ignore_index=True)
    ext_stat = pd.concat([od_stat, cc_stat], ignore_index=True)

    ext_run.to_csv(os.path.join(OUT_DIR, 'schedule_ablation_external_run_level.csv'), index=False)
    ext_stat.to_csv(os.path.join(OUT_DIR, 'schedule_ablation_external_comparison.csv'), index=False)

    print(f"\n{'='*60}")
    print(f"Total elapsed: {time.time()-t_start:.1f}s")
    print(f"Output: {OUT_DIR}")
    print(f"  schedule_ablation_external_run_level.csv ({len(ext_run)} rows)")
    print(f"  schedule_ablation_external_comparison.csv ({len(ext_stat)} rows)")


if __name__ == '__main__':
    main()
