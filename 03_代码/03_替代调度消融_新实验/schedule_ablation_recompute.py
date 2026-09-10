# -*- coding: utf-8 -*-
"""
CS-Mondrian「替代调度消融」主计算
====================================
实现 5 个替代 α 调度策略，与 CS-Mondrian 同协议对比：
  S1 reverse-schedule:    α_t=a/r, α_l=a（方向相反，不锚预算）
  S2 oracle-scalar-α:     α_t=α_l=α0*，val 集 selection 一维搜优，等 B 预算
  S3 empirical-equal-risk: calib 上经验错误率均等，锚定总预算=CS
  S4 prevalence-inverse:   α_c∝1/π_c，锚定总预算=CS，不看成本 r
  S5 exact-budget-match:   α_t=α_l，π_t α_t+π_l α_l=CS 逐位相等

纯离线校准层 α 重算，不训练、不连服务器、不改原始数据/既有结果。
核心函数复用自 singh_arm_recompute.py（04_脚本/Singh对照臂/）。
"""
import os

_HOME = os.path.expanduser("~")
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
import pandas as pd
import os
import sys
import time
from scipy import stats

# 导入 Singh 框架核心函数
SINGH_DIR = os.path.join(_PKG_ROOT, '03_代码', '02_Singh对照臂_新实验')
sys.path.insert(0, SINGH_DIR)
from singh_arm_recompute import (
    conformal_quantile, build_flags, method_mondrian, method_cs_mondrian,
    compute_metrics_base, get_subject_ids, load_npz,
    subject_level_stats, holm_adjust
)

# ============================================================
# 0. 配置（冻结口径，与 strict_protocol.json 一致）
# ============================================================
INTEGRATION_ROOT = _PKG_ROOT
NPZ_ROOT = os.path.join(os.environ.get("CS_MONDRIAN_DATA", os.path.join(_PKG_ROOT, "04_数据")), "01_中文_SEUMLD与MDPE")
SUBJECT_MAP_PATH = os.path.join(NPZ_ROOT, 'subject_map.csv')
OUT_DIR = os.environ.get("CS_MONDRIAN_OUTPUT", os.path.join(_PKG_ROOT, "recomputed"))

DATASETS = ['SEUMLD', 'MDPE']
CONFIGS = ['OADNet_text', 'OADNet_audio']
FOLDS = [0, 1, 2, 3, 4]
SEEDS = [7, 42, 123, 2024, 2026]
ALPHA_TOTAL = 0.10
C_FP = 1.0
COST_RATIOS = [1.0, 2.0, 3.0, 5.0, 10.0]
C_REV_VALUES = [0.25, 0.5, 1.0, 2.0]
SELECTION_BUDGETS = [0.6, 0.7, 0.8]  # S2 最大拒判率 B

RNG_SEED = 42
N_BOOT = 10000
N_PERM = 10000

# S2 α 网格
ALPHA_GRID = np.arange(0.005, 0.501, 0.005)
# S3 二维搜索网格
S3_ALPHA_GRID = np.arange(0.001, 0.301, 0.002)

SUBJECT_MAP = pd.read_csv(SUBJECT_MAP_PATH)


# ============================================================
# 1. 策略 α 分配函数
# ============================================================
def cs_budget(calib_labels, a=ALPHA_TOTAL, r=3.0):
    """CS 的总名义失覆盖预算：π_t*a + π_l*(a/r)。"""
    n = len(calib_labels)
    pi_t = np.sum(calib_labels == 0) / n
    pi_l = np.sum(calib_labels == 1) / n
    return pi_t * a + pi_l * (a / r), pi_t, pi_l


def strategy_s1_reverse(a=ALPHA_TOTAL, r=3.0):
    """反向调度：α_t=a/r, α_l=a（不锚预算）。"""
    return a / r, a


def strategy_s2_oracle_scalar(calib_probs, calib_labels, val_probs, val_labels,
                               r=3.0, C_rev=0.5, B=0.7, a=ALPHA_TOTAL):
    """
    一维全局最优单一 α：在 val(selection) 集上搜索 α0*，
    满足拒判率 ≤ B 且期望成本最小。
    返回 (α0*, selected_alpha_list, costs_list, reject_rates_list)
    """
    C_FN = r * C_FP
    best_alpha = None
    best_cost = np.inf
    best_reject = np.inf
    alpha_costs = []
    alpha_rejects = []

    for alpha0 in ALPHA_GRID:
        # 用 calib 计算 qhat
        flags, _, _ = method_mondrian(calib_probs, calib_labels, val_probs, alpha0, alpha0)
        m = compute_metrics_base(val_labels, flags, C_FP, C_FN, C_rev, 0.0)
        if m is None:
            continue
        alpha_costs.append(m['expected_cost'])
        alpha_rejects.append(m['reject_rate'])
        # 选满足拒判率 ≤ B 且成本最小的
        if m['reject_rate'] <= B + 1e-9:
            if m['expected_cost'] < best_cost - 1e-12:
                best_cost = m['expected_cost']
                best_alpha = alpha0
                best_reject = m['reject_rate']

    # 如果没有 α 满足 B（所有 α 拒判率都 > B），取拒判率最小的
    if best_alpha is None:
        idx_min_rej = np.argmin(alpha_rejects)
        best_alpha = ALPHA_GRID[idx_min_rej]
        best_reject = alpha_rejects[idx_min_rej]

    return best_alpha, best_cost, best_reject


def strategy_cs_budget_selected(calib_probs, calib_labels, val_probs, val_labels,
                                r=3.0, C_rev=0.5, B=0.7):
    """Select the CS scalar a on validation under the same reject budget as S2."""
    C_FN = r * C_FP
    best_a = None
    best_cost = np.inf
    best_reject = np.inf
    for a0 in ALPHA_GRID:
        flags, _, _ = method_cs_mondrian(calib_probs, calib_labels, val_probs, a0, r)
        m = compute_metrics_base(val_labels, flags, C_FP, C_FN, C_rev, 0.0)
        if m is not None and m['reject_rate'] <= B + 1e-9:
            if m['expected_cost'] < best_cost - 1e-12:
                best_a, best_cost, best_reject = a0, m['expected_cost'], m['reject_rate']
    if best_a is None:
        # Keep the fallback deterministic and choose the feasible point closest to B.
        candidates = []
        for a0 in ALPHA_GRID:
            flags, _, _ = method_cs_mondrian(calib_probs, calib_labels, val_probs, a0, r)
            m = compute_metrics_base(val_labels, flags, C_FP, C_FN, C_rev, 0.0)
            if m is not None:
                candidates.append((abs(m['reject_rate'] - B), a0, m['expected_cost'], m['reject_rate']))
        _, best_a, best_cost, best_reject = min(candidates)
    return best_a, best_cost, best_reject


def strategy_s3_empirical_equal_risk(calib_probs, calib_labels, val_probs, val_labels, budget,
                                       r=3.0, C_rev=0.5):
    """
    在独立 validation 集上选 (α_t, α_l) 使两类自动接受样本的
    经验条件错误率（FP_rate vs FN_rate）相等，同时锚定总预算 π_t α_t + π_l α_l = budget。
    calibration 集只用于最终 qhat 估计。
    二维网格搜索。
    """
    n = len(calib_labels)
    pi_t = np.sum(calib_labels == 0) / n
    pi_l = np.sum(calib_labels == 1) / n

    best_at, best_al = None, None
    best_diff = np.inf
    best_budget_err = np.inf

    for at in S3_ALPHA_GRID:
        # 由预算约束解 α_l
        if pi_l > 0:
            al = (budget - pi_t * at) / pi_l
        else:
            al = 0.0
        if al < 0 or al > 1:
            continue
        # 在 calib 上计算经验错误率
        flags, _, _ = method_mondrian(calib_probs, calib_labels, val_probs, at, al)
        m = compute_metrics_base(val_labels, flags, C_FP, r * C_FP, C_rev, 0.0)
        if m is None:
            continue
        fp_rate = m['fp_rate']
        fn_rate = m['fn_rate']
        diff = abs(fp_rate - fn_rate)
        budget_actual = pi_t * at + pi_l * al
        budget_err = abs(budget_actual - budget)
        # 优先预算锚定，其次错误率均等
        if budget_err < 1e-8:
            if diff < best_diff - 1e-12:
                best_diff = diff
                best_at, best_al = at, al
                best_budget_err = budget_err

    # 如果没找到精确预算锚定的，放宽容差
    if best_at is None:
        for at in S3_ALPHA_GRID:
            if pi_l > 0:
                al = (budget - pi_t * at) / pi_l
            else:
                al = 0.0
            if al < 0 or al > 1:
                continue
            flags, _, _ = method_mondrian(calib_probs, calib_labels, val_probs, at, al)
            m = compute_metrics_base(val_labels, flags, C_FP, r * C_FP, C_rev, 0.0)
            if m is None:
                continue
            diff = abs(m['fp_rate'] - m['fn_rate'])
            budget_actual = pi_t * at + pi_l * al
            budget_err = abs(budget_actual - budget)
            if budget_err < best_budget_err - 1e-12 or (abs(budget_err - best_budget_err) < 1e-12 and diff < best_diff):
                best_budget_err = budget_err
                best_diff = diff
                best_at, best_al = at, al

    return best_at, best_al, best_diff, best_budget_err


def strategy_s4_prevalence_inverse(calib_labels, budget):
    """
    类别频率反比：α_c ∝ 1/π_c，归一化后锚定总预算=budget。
    α_t = budget / (2 π_t), α_l = budget / (2 π_l)
    """
    n = len(calib_labels)
    pi_t = np.sum(calib_labels == 0) / n
    pi_l = np.sum(calib_labels == 1) / n
    at = budget / (2 * pi_t) if pi_t > 0 else 1.0
    al = budget / (2 * pi_l) if pi_l > 0 else 1.0
    # 截断到 [0,1]
    at = min(max(at, 0.0), 1.0)
    al = min(max(al, 0.0), 1.0)
    return at, al


def strategy_s5_exact_budget_match(calib_labels, budget):
    """
    精确预算守恒：α_t=α_l=α0，π_t α_t + π_l α_l = budget。
    因为 π_t + π_l = 1，所以 α0 = budget。
    """
    alpha0 = budget
    alpha0 = min(max(alpha0, 0.0), 1.0)
    return alpha0, alpha0


# ============================================================
# 2. 主计算
# ============================================================
def run_all_strategies():
    """对每个 dataset/config/fold/seed/r 计算 CS + 5 策略的指标。"""
    t0 = time.time()
    run_rows = []
    subj_rows = []
    total = len(DATASETS) * len(CONFIGS) * len(FOLDS) * len(SEEDS) * len(COST_RATIOS)
    count = 0

    for dataset in DATASETS:
        for config in CONFIGS:
            for fold in FOLDS:
                for seed in SEEDS:
                    d = load_npz(dataset, config, fold, seed)
                    test_probs = d['probs']
                    test_labels = d['labels']
                    calib_probs = d['calib_probs']
                    calib_labels = d['calib_labels']
                    val_probs = d['val_probs']
                    val_labels = d['val_labels']
                    test_subjects = get_subject_ids(dataset, fold, 'test', len(test_labels))

                    for r in COST_RATIOS:
                        C_FN = r * C_FP
                        count += 1
                        if count % 50 == 0:
                            print(f"  {count}/{total} ({time.time()-t0:.0f}s)")

                        # CS 总预算
                        budget_cs, pi_t, pi_l = cs_budget(calib_labels, ALPHA_TOTAL, r)

                        # --- CS (本文) ---
                        flags_cs, qt_cs, ql_cs = method_cs_mondrian(
                            calib_probs, calib_labels, test_probs, ALPHA_TOTAL, r)

                        # --- S1 reverse ---
                        at_s1, al_s1 = strategy_s1_reverse(ALPHA_TOTAL, r)
                        flags_s1, qt_s1, ql_s1 = method_mondrian(
                            calib_probs, calib_labels, test_probs, at_s1, al_s1)

                        # --- S3 empirical-equal-risk ---
                        at_s3, al_s3, s3_diff, s3_budget_err = strategy_s3_empirical_equal_risk(
                            calib_probs, calib_labels, val_probs, val_labels, budget_cs, r, 0.5)
                        flags_s3, qt_s3, ql_s3 = method_mondrian(
                            calib_probs, calib_labels, test_probs, at_s3, al_s3)

                        # --- S4 prevalence-inverse ---
                        at_s4, al_s4 = strategy_s4_prevalence_inverse(calib_labels, budget_cs)
                        flags_s4, qt_s4, ql_s4 = method_mondrian(
                            calib_probs, calib_labels, test_probs, at_s4, al_s4)

                        # --- S5 exact-budget-match ---
                        at_s5, al_s5 = strategy_s5_exact_budget_match(calib_labels, budget_cs)
                        flags_s5, qt_s5, ql_s5 = method_mondrian(
                            calib_probs, calib_labels, test_probs, at_s5, al_s5)

                        # --- S2 oracle-scalar-α（对每个 B 和 C_rev）---
                        # S2 需要在 val 集上选 α0*，依赖 C_rev（成本函数）
                        # 所以对每个 C_rev 分别搜索
                        s2_results = {}  # (C_rev, B) -> (alpha0, flags)
                        cs_budget_results = {}  # same-budget CS comparator
                        for C_rev in C_REV_VALUES:
                            for B in SELECTION_BUDGETS:
                                alpha0_s2, _, _ = strategy_s2_oracle_scalar(
                                    calib_probs, calib_labels, val_probs, val_labels,
                                    r, C_rev, B, ALPHA_TOTAL)
                                flags_s2, qt_s2, ql_s2 = method_mondrian(
                                    calib_probs, calib_labels, test_probs, alpha0_s2, alpha0_s2)
                                s2_results[(C_rev, B)] = (alpha0_s2, flags_s2, qt_s2, ql_s2)
                                a_cs, _, _ = strategy_cs_budget_selected(
                                    calib_probs, calib_labels, val_probs, val_labels,
                                    r, C_rev, B)
                                flags_cs_b, qt_cs_b, ql_cs_b = method_cs_mondrian(
                                    calib_probs, calib_labels, test_probs, a_cs, r)
                                cs_budget_results[(C_rev, B)] = (
                                    a_cs, flags_cs_b, qt_cs_b, ql_cs_b)

                        # 收集所有策略的 flags
                        all_strategies = {
                            'cs_mondrian': (flags_cs, qt_cs, ql_cs, ALPHA_TOTAL, ALPHA_TOTAL / r, budget_cs),
                            's1_reverse': (flags_s1, qt_s1, ql_s1, at_s1, al_s1, pi_t * at_s1 + pi_l * al_s1),
                            's3_empirical_equal_risk': (flags_s3, qt_s3, ql_s3, at_s3, al_s3, pi_t * at_s3 + pi_l * al_s3),
                            's4_prevalence_inverse': (flags_s4, qt_s4, ql_s4, at_s4, al_s4, pi_t * at_s4 + pi_l * al_s4),
                            's5_exact_budget_match': (flags_s5, qt_s5, ql_s5, at_s5, al_s5, pi_t * at_s5 + pi_l * al_s5),
                        }

                        for C_rev in C_REV_VALUES:
                            # 非 S2 策略
                            for method_name, (flags, qt, ql, at, al, total_budget) in all_strategies.items():
                                m = compute_metrics_base(test_labels, flags, C_FP, C_FN, C_rev, 0.0)
                                run_rows.append({
                                    'dataset': dataset, 'config': config, 'fold': fold, 'seed': seed,
                                    'method': method_name, 'cost_ratio': r, 'C_rev': C_rev,
                                    'B': np.nan, 'alpha_truth': at, 'alpha_lie': al,
                                    'total_nominal_budget': total_budget,
                                    'q_truth': qt, 'q_lie': ql, **m})
                                # subject-level
                                for subj in np.unique(test_subjects):
                                    mask = test_subjects == subj
                                    sm = compute_metrics_base(
                                        test_labels[mask],
                                        {k: flags[k][mask] for k in flags},
                                        C_FP, C_FN, C_rev, 0.0)
                                    subj_rows.append({
                                        'dataset': dataset, 'config': config, 'fold': fold, 'seed': seed,
                                        'method': method_name, 'cost_ratio': r, 'C_rev': C_rev,
                                        'B': np.nan, 'alpha_truth': at, 'alpha_lie': al,
                                        'total_nominal_budget': total_budget,
                                        'subject_id': subj, 'n_segments': int(np.sum(mask)), **sm})

                            # S2 策略（每个 B）
                            for B in SELECTION_BUDGETS:
                                a_cs_b, flags_cs_b, qt_cs_b, ql_cs_b = cs_budget_results[(C_rev, B)]
                                m_cs_b = compute_metrics_base(test_labels, flags_cs_b, C_FP, C_FN, C_rev, 0.0)
                                run_rows.append({
                                    'dataset': dataset, 'config': config, 'fold': fold, 'seed': seed,
                                    'method': 'cs_budget_selected', 'cost_ratio': r, 'C_rev': C_rev,
                                    'B': B, 'alpha_truth': a_cs_b, 'alpha_lie': a_cs_b / r,
                                    'total_nominal_budget': pi_t * a_cs_b + pi_l * (a_cs_b / r),
                                    'q_truth': qt_cs_b, 'q_lie': ql_cs_b, **m_cs_b})
                                for subj in np.unique(test_subjects):
                                    mask = test_subjects == subj
                                    sm = compute_metrics_base(
                                        test_labels[mask],
                                        {k: flags_cs_b[k][mask] for k in flags_cs_b},
                                        C_FP, C_FN, C_rev, 0.0)
                                    subj_rows.append({
                                        'dataset': dataset, 'config': config, 'fold': fold, 'seed': seed,
                                        'method': 'cs_budget_selected', 'cost_ratio': r, 'C_rev': C_rev,
                                        'B': B, 'alpha_truth': a_cs_b, 'alpha_lie': a_cs_b / r,
                                        'total_nominal_budget': pi_t * a_cs_b + pi_l * (a_cs_b / r),
                                        'subject_id': subj, 'n_segments': int(np.sum(mask)), **sm})
                                alpha0_s2, flags_s2, qt_s2, ql_s2 = s2_results[(C_rev, B)]
                                m = compute_metrics_base(test_labels, flags_s2, C_FP, C_FN, C_rev, 0.0)
                                run_rows.append({
                                    'dataset': dataset, 'config': config, 'fold': fold, 'seed': seed,
                                    'method': 's2_oracle_scalar_alpha', 'cost_ratio': r, 'C_rev': C_rev,
                                    'B': B, 'alpha_truth': alpha0_s2, 'alpha_lie': alpha0_s2,
                                    'total_nominal_budget': alpha0_s2,  # π_t+π_l=1
                                    'q_truth': qt_s2, 'q_lie': ql_s2, **m})
                                for subj in np.unique(test_subjects):
                                    mask = test_subjects == subj
                                    sm = compute_metrics_base(
                                        test_labels[mask],
                                        {k: flags_s2[k][mask] for k in flags_s2},
                                        C_FP, C_FN, C_rev, 0.0)
                                    subj_rows.append({
                                        'dataset': dataset, 'config': config, 'fold': fold, 'seed': seed,
                                        'method': 's2_oracle_scalar_alpha', 'cost_ratio': r, 'C_rev': C_rev,
                                        'B': B, 'alpha_truth': alpha0_s2, 'alpha_lie': alpha0_s2,
                                        'total_nominal_budget': alpha0_s2,
                                        'subject_id': subj, 'n_segments': int(np.sum(mask)), **sm})

    run_df = pd.DataFrame(run_rows)
    subj_df = pd.DataFrame(subj_rows)
    print(f"\nDone in {time.time()-t0:.1f}s. run={len(run_df)}, subject={len(subj_df)}")
    return run_df, subj_df


# ============================================================
# 3. 统计：策略 vs CS 配对差值
# ============================================================
def strategy_vs_cs_stats(subj_df):
    """对每个策略，计算受试者级配对差值（策略 − CS），bootstrap/permutation/Holm。"""
    print("\n=== Strategy vs CS subject-level stats ===")
    rng = np.random.default_rng(RNG_SEED)
    records = []

    cs = subj_df[subj_df['method'] == 'cs_mondrian'].copy()
    strategy_methods = ['s1_reverse', 's2_oracle_scalar_alpha', 's3_empirical_equal_risk',
                         's4_prevalence_inverse', 's5_exact_budget_match']

    for method in strategy_methods:
        si = subj_df[subj_df['method'] == method].copy()
        # S2 需要按 B 分组
        if method == 's2_oracle_scalar_alpha':
            b_values = SELECTION_BUDGETS
        else:
            b_values = [np.nan]

        for B in b_values:
            if method == 's2_oracle_scalar_alpha':
                si_b = si[si['B'] == B].copy()
            else:
                si_b = si.copy()

            # S2 uses the same validation-budget-selected CS comparator.
            merge_on = ['dataset', 'config', 'fold', 'seed', 'cost_ratio', 'C_rev', 'subject_id', 'n_segments']
            cs_base = (subj_df[subj_df['method'] == 'cs_budget_selected'].copy()
                       if method == 's2_oracle_scalar_alpha' else cs)
            if method == 's2_oracle_scalar_alpha':
                cs_base = cs_base[cs_base['B'] == B].copy()
            merged = cs_base.merge(si_b, on=merge_on, suffixes=('_cs', '_strategy'))
            merged['delta_cost'] = merged['expected_cost_strategy'] - merged['expected_cost_cs']

            # 按受试者聚合（跨 fold×seed 平均）
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
                                'dataset': dataset, 'config': config, 'method': method,
                                'B': B, 'cost_ratio': r, 'C_rev': C_rev, **st})

    stat_df = pd.DataFrame(records)

    # Holm 校正：四中文配置主族（r=3, C_rev=0.5），按 method+B 分组
    for method in strategy_methods:
        if method == 's2_oracle_scalar_alpha':
            b_list = SELECTION_BUDGETS
        else:
            b_list = [np.nan]
        for B in b_list:
            mask = ((stat_df['cost_ratio'] == 3) & (stat_df['C_rev'] == 0.5) &
                    (stat_df['method'] == method))
            if method == 's2_oracle_scalar_alpha':
                mask = mask & (stat_df['B'] == B)
            if mask.sum() > 0:
                pvals = stat_df.loc[mask, 'permutation_p'].values
                stat_df.loc[mask, 'holm_p'] = holm_adjust(pvals)

    print(f"  Stats cells: {len(stat_df)}")
    return stat_df


# ============================================================
# 4. Pareto 数据
# ============================================================
def build_pareto_data(run_df):
    """构建成本–拒判率 Pareto 数据：每策略×配置×r 的均值点。"""
    print("\n=== Pareto data ===")
    # 固定锚点 r=3, C_rev=0.5
    anchor = run_df[(run_df['cost_ratio'] == 3) & (run_df['C_rev'] == 0.5)].copy()
    # S2 取 B=0.7
    s2_anchor = anchor[anchor['method'] == 's2_oracle_scalar_alpha']
    s2_b07 = s2_anchor[s2_anchor['B'] == 0.7]
    cs_b07 = anchor[(anchor['method'] == 'cs_budget_selected') & (anchor['B'] == 0.7)]
    other_anchor = anchor[~anchor['method'].isin(['s2_oracle_scalar_alpha', 'cs_budget_selected'])]
    pareto_df = pd.concat([other_anchor, cs_b07, s2_b07], ignore_index=True)

    # 按 dataset/config/method 聚合（跨 fold×seed 平均）
    pareto_agg = pareto_df.groupby(['dataset', 'config', 'method']).agg(
        mean_expected_cost=('expected_cost', 'mean'),
        mean_reject_rate=('reject_rate', 'mean'),
        mean_fn_rate=('fn_rate', 'mean'),
        mean_fp_rate=('fp_rate', 'mean'),
        mean_label_coverage=('label_coverage', 'mean'),
        mean_alpha_truth=('alpha_truth', 'mean'),
        mean_alpha_lie=('alpha_lie', 'mean'),
        mean_total_budget=('total_nominal_budget', 'mean'),
        n_runs=('expected_cost', 'count'),
    ).reset_index()

    print(f"  Pareto points: {len(pareto_agg)}")
    return pareto_agg


# ============================================================
# 5. 主流程
# ============================================================
def main():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== CS-Mondrian Alternative Schedule Ablation ===")
    print(f"NPZ root: {NPZ_ROOT}")
    print(f"Output: {OUT_DIR}")

    # 核心计算
    run_df, subj_df = run_all_strategies()
    run_df.to_csv(os.path.join(OUT_DIR, 'schedule_ablation_run_level.csv'), index=False)
    subj_df.to_csv(os.path.join(OUT_DIR, 'schedule_ablation_subject_level.csv'), index=False)

    # 统计
    stat_df = strategy_vs_cs_stats(subj_df)
    stat_df.to_csv(os.path.join(OUT_DIR, 'schedule_ablation_comparison.csv'), index=False)

    # Pareto
    pareto_df = build_pareto_data(run_df)
    pareto_df.to_csv(os.path.join(OUT_DIR, 'schedule_ablation_pareto.csv'), index=False)

    # 打印主终点
    print("\n=== PRIMARY ENDPOINT (r=3, C_rev=0.5) ===")
    primary = stat_df[(stat_df['cost_ratio'] == 3) & (stat_df['C_rev'] == 0.5)]
    # S2 只看 B=0.7
    primary_show = primary[(primary['method'] != 's2_oracle_scalar_alpha') |
                           (primary['B'] == 0.7)]
    print(primary_show[['dataset', 'config', 'method', 'B', 'n_subjects', 'mean_delta',
                         'ci_2.5', 'ci_97.5', 'permutation_p', 'holm_p']].to_string(index=False))

    print("\n=== PARETO (r=3, C_rev=0.5) ===")
    print(pareto_df[['dataset', 'config', 'method', 'mean_expected_cost',
                      'mean_reject_rate', 'mean_fn_rate', 'mean_alpha_truth',
                      'mean_alpha_lie']].to_string(index=False))

    print(f"\nTotal elapsed: {time.time()-t_start:.1f}s")
    return run_df, subj_df, stat_df, pareto_df


if __name__ == '__main__':
    main()
