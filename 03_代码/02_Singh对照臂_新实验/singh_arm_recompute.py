# -*- coding: utf-8 -*-
"""
CS-Mondrian Singh[10] 严格同协议对照臂重算
=============================================
在本地已有 npz 上，把 Singh et al. 2026 实现成一条可命名的对照臂：
- Singh 臂 = 统一 α=.10 的 Mondrian deferral（与 uni 同机制）+ 非理想人工 ε_hum
- Bayes 成本阈值器 τ* = 1/(1+r)，硬决策无 defer
- Singh 口径 break-even C*_rev：deferral vs Bayes-τ* 临界复核成本

纯离线后处理，不训练、不连服务器、不改原始数据。
核心函数复用自 strict_recompute.py（04_脚本/严格重算与报告/）。
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

# ============================================================
# 0. 配置（冻结口径，与 strict_protocol.json 一致）
# ============================================================
INTEGRATION_ROOT = _PKG_ROOT
NPZ_ROOT = os.path.join(INTEGRATION_ROOT, '06_实验数据')
SUBJECT_MAP_PATH = os.path.join(NPZ_ROOT, 'subject_map.csv')
OUT_DIR = os.path.join(INTEGRATION_ROOT, '02_严格重算输出')  # 结果 CSV 放这里，独立命名

DATASETS = ['SEUMLD', 'MDPE']
CONFIGS = ['OADNet_text', 'OADNet_audio']
FOLDS = [0, 1, 2, 3, 4]
SEEDS = [7, 42, 123, 2024, 2026]
ALPHA_TOTAL = 0.10
C_FP = 1.0
COST_RATIOS = [1.0, 2.0, 3.0, 5.0, 10.0]
C_REV_VALUES = [0.25, 0.5, 1.0, 2.0]
EPS_HUM_VALUES = [0.0, 0.05, 0.1, 0.2]

RNG_SEED = 42
N_BOOT = 10000
N_PERM = 10000

# ============================================================
# 1. 核心函数（严格复用自 strict_recompute.py，逐字一致）
# ============================================================
def conformal_quantile(scores, alpha):
    """有限样本共形秩分位点：sorted(scores)[ceil((n+1)(1-alpha))-1]。"""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        raise ValueError("empty calibration class")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf
    return np.sort(scores)[k - 1]


def build_flags(include_truth, include_lie):
    """根据 include 布尔数组构造完整集合决策标志。"""
    include_truth = np.asarray(include_truth, dtype=bool)
    include_lie = np.asarray(include_lie, dtype=bool)
    singleton_t = include_truth & (~include_lie)
    singleton_l = include_lie & (~include_truth)
    empty_set = (~include_truth) & (~include_lie)
    ambiguous_set = include_truth & include_lie
    singleton = singleton_t | singleton_l
    reject = empty_set | ambiguous_set
    pred_label = np.zeros(len(include_truth), dtype=int)
    pred_label[singleton_l] = 1
    return {
        'include_truth': include_truth,
        'include_lie': include_lie,
        'singleton': singleton,
        'singleton_truth': singleton_t,
        'singleton_lie': singleton_l,
        'empty_set': empty_set,
        'ambiguous_set': ambiguous_set,
        'reject': reject,
        'pred_label': pred_label,
    }


def method_mondrian(calib_probs, calib_labels, test_probs, alpha_truth, alpha_lie):
    """标准 Mondrian：类条件 qhat（严格秩分位点）。"""
    calib_probs = np.asarray(calib_probs, dtype=float)
    calib_labels = np.asarray(calib_labels)
    test_probs = np.asarray(test_probs, dtype=float)
    s_truth_calib = calib_probs[calib_labels == 0]
    s_lie_calib = 1.0 - calib_probs[calib_labels == 1]
    if alpha_truth == 0.0:
        q_truth = np.inf
    else:
        q_truth = conformal_quantile(s_truth_calib, alpha_truth)
    if alpha_lie == 0.0:
        q_lie = np.inf
    else:
        q_lie = conformal_quantile(s_lie_calib, alpha_lie)
    s_truth_test = test_probs.copy()
    s_lie_test = 1.0 - test_probs
    include_truth = s_truth_test <= q_truth
    include_lie = s_lie_test <= q_lie
    flags = build_flags(include_truth, include_lie)
    return flags, q_truth, q_lie


def method_cs_mondrian(calib_probs, calib_labels, test_probs, alpha_total, r):
    """CS-Mondrian：alpha_truth=alpha_total, alpha_lie=alpha_total/r。r=1 时退化为 Mondrian。"""
    alpha_truth = alpha_total
    alpha_lie = alpha_total / r
    return method_mondrian(calib_probs, calib_labels, test_probs, alpha_truth, alpha_lie)


# ============================================================
# 2. 成本计算（支持 ε_hum 非理想人工）
# ============================================================
def compute_metrics_base(labels, flags, C_FP=1.0, C_FN=1.0, C_rev=0.5, eps_hum=0.0):
    """
    计算 run / subject 级指标。只在 singleton（自动接受）上计 FP/FN。
    reject 样本成本：C_rev + eps_hum * (y=lie ? C_FN : C_FP)
    eps_hum=0 时与 strict compute_metrics 逐位一致。
    """
    labels = np.asarray(labels)
    n = len(labels)
    if n == 0:
        return None
    reject = flags['reject']
    singleton = flags['singleton']
    accepted = ~reject
    pred_label = flags['pred_label']

    n_truth = int(np.sum(labels == 0))
    n_lie = int(np.sum(labels == 1))
    n_reject = int(np.sum(reject))
    n_accept = n - n_reject
    n_empty = int(np.sum(flags['empty_set']))
    n_ambiguous = int(np.sum(flags['ambiguous_set']))
    n_singleton = int(np.sum(singleton))

    acc = accepted
    fp = int(np.sum((pred_label[acc] == 1) & (labels[acc] == 0)))
    fn = int(np.sum((pred_label[acc] == 0) & (labels[acc] == 1)))
    tp = int(np.sum((pred_label[acc] == 1) & (labels[acc] == 1)))
    tn = int(np.sum((pred_label[acc] == 0) & (labels[acc] == 0)))

    contains_true_label = np.where(labels == 1, flags['include_lie'], flags['include_truth'])
    label_coverage = float(np.mean(contains_true_label))
    label_coverage_truth = float(np.mean(contains_true_label[labels == 0])) if n_truth > 0 else np.nan
    label_coverage_lie = float(np.mean(contains_true_label[labels == 1])) if n_lie > 0 else np.nan

    # 自动样本成本
    auto_cost = C_FP * fp + C_FN * fn
    # reject 样本成本（非理想人工）
    if eps_hum == 0.0:
        reject_cost = C_rev * n_reject
    else:
        reject_labels = labels[reject]
        reject_extra = eps_hum * (
            C_FN * np.sum(reject_labels == 1) + C_FP * np.sum(reject_labels == 0)
        )
        reject_cost = C_rev * n_reject + reject_extra

    total_cost = auto_cost + reject_cost
    return {
        'n_test': n, 'n_truth': n_truth, 'n_lie': n_lie,
        'n_reject': n_reject, 'n_accept': n_accept,
        'n_singleton': n_singleton, 'n_empty': n_empty, 'n_ambiguous': n_ambiguous,
        'n_fp': fp, 'n_fn': fn, 'n_tp': tp, 'n_tn': tn,
        'fp_rate': fp / n_truth if n_truth > 0 else 0.0,
        'fn_rate': fn / n_lie if n_lie > 0 else 0.0,
        'reject_rate': n_reject / n if n > 0 else 0.0,
        'decision_coverage': n_singleton / n if n > 0 else 0.0,
        'acceptance_rate': n_accept / n if n > 0 else 0.0,
        'label_coverage': label_coverage,
        'label_coverage_truth': label_coverage_truth,
        'label_coverage_lie': label_coverage_lie,
        'total_cost': float(total_cost),
        'expected_cost': float(total_cost) / n if n > 0 else 0.0,
    }


# ============================================================
# 3. Bayes 成本阈值器 τ*（硬决策，无 defer）
# ============================================================
def bayes_threshold_cost(test_probs, test_labels, r, C_FP=1.0):
    """
    Bayes-optimal 硬阈值：τ* = C_FP/(C_FP+C_FN) = 1/(1+r)
    判 lie 当 p_lie >= τ*，无 reject、无复核。
    返回 expected_cost（每样本平均成本）及完整混淆计数。
    """
    test_probs = np.asarray(test_probs, dtype=float)
    test_labels = np.asarray(test_labels)
    C_FN = r * C_FP
    tau_star = C_FP / (C_FP + C_FN)  # = 1/(1+r)
    pred_lie = test_probs >= tau_star
    fp = int(np.sum(pred_lie & (test_labels == 0)))
    fn = int(np.sum((~pred_lie) & (test_labels == 1)))
    tp = int(np.sum(pred_lie & (test_labels == 1)))
    tn = int(np.sum((~pred_lie) & (test_labels == 0)))
    total_cost = C_FP * fp + C_FN * fn
    n = len(test_labels)
    return {
        'tau_star': tau_star,
        'n_fp': fp, 'n_fn': fn, 'n_tp': tp, 'n_tn': tn,
        'fp_rate': fp / int(np.sum(test_labels == 0)) if np.sum(test_labels == 0) > 0 else 0.0,
        'fn_rate': fn / int(np.sum(test_labels == 1)) if np.sum(test_labels == 1) > 0 else 0.0,
        'total_cost': float(total_cost),
        'expected_cost': float(total_cost) / n if n > 0 else 0.0,
    }


# ============================================================
# 4. Singh 口径 break-even C*_rev（数值求根）
# ============================================================
def singh_breakeven_c_rev(deferral_expected_cost_at_crev_grid, bayes_expected_cost, crev_grid):
    """
    求临界 C*_rev：deferral 方案期望成本 = Bayes-τ* 硬决策期望成本。
    deferral 成本随 C_rev 线性增长（reject 样本每样本成本 = C_rev + 常数），
    所以可以用线性插值求根。
    返回 None 如果无解（deferral 始终高于/低于 Bayes）。
    """
    deferral = np.asarray(deferral_expected_cost_at_crev_grid, dtype=float)
    bayes = float(bayes_expected_cost)
    grid = np.asarray(crev_grid, dtype=float)
    diff = deferral - bayes
    # 找符号变化的区间
    for i in range(len(grid) - 1):
        if diff[i] == 0:
            return float(grid[i])
        if diff[i] * diff[i + 1] < 0:
            # 线性插值
            t = -diff[i] / (diff[i + 1] - diff[i])
            return float(grid[i] + t * (grid[i + 1] - grid[i]))
    # 检查端点
    if diff[-1] == 0:
        return float(grid[-1])
    # 如果 deferral 始终 < bayes，说明在最大 C_rev 仍更优，返回 > max
    if np.all(diff < 0):
        return float(grid[-1])  # 标记：实际 C*_rev > 网格上限
    # 如果 deferral 始终 > bayes，说明在最小 C_rev 已更差，返回 < min
    if np.all(diff > 0):
        return float(grid[0])  # 标记：实际 C*_rev < 网格下限
    return None


# ============================================================
# 5. 数据加载
# ============================================================
SUBJECT_MAP = pd.read_csv(SUBJECT_MAP_PATH)


def get_subject_ids(dataset, fold, role, n_samples):
    sub = SUBJECT_MAP[(SUBJECT_MAP['dataset'] == dataset) &
                      (SUBJECT_MAP['fold'] == fold) &
                      (SUBJECT_MAP['role'] == role)].sort_values('pos')
    if len(sub) != n_samples:
        raise ValueError(
            f"Size mismatch: {dataset} fold{fold} {role}: expected {n_samples}, got {len(sub)}")
    return sub['subject'].values


def load_npz(dataset, config, fold, seed):
    path = os.path.join(NPZ_ROOT, f'{dataset}_preds', f'{config}_f{fold}_s{seed}.npz')
    return np.load(path, allow_pickle=True)


# ============================================================
# 6. 统计检验（受试者级 cluster bootstrap + permutation + Wilcoxon + sign）
# ============================================================
def subject_level_stats(deltas, rng, n_boot=N_BOOT, n_perm=N_PERM):
    """对一组受试者级配对差值做统计推断。"""
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    if n < 5:
        return None
    obs_mean = float(np.mean(deltas))
    obs_median = float(np.median(deltas))

    # cluster bootstrap
    boot_idx = rng.integers(0, n, (n_boot, n))
    boot_means = np.mean(deltas[boot_idx], axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    # paired sign-flip permutation
    perm_signs = rng.choice([-1, 1], size=(n_perm, n))
    perm_means = np.mean(deltas[None, :] * perm_signs, axis=1)
    p_perm = float(np.mean(np.abs(perm_means) >= np.abs(obs_mean)))

    # Wilcoxon signed-rank
    try:
        if np.all(deltas == 0):
            w_p = 1.0
        else:
            with np.errstate(all='ignore'):
                w_p = float(stats.wilcoxon(deltas).pvalue)
    except Exception:
        w_p = np.nan

    # exact sign test
    n_pos = int(np.sum(deltas > 0))
    n_neg = int(np.sum(deltas < 0))
    n_zero = int(np.sum(deltas == 0))
    if n_pos + n_neg > 0:
        sign_p = float(stats.binomtest(min(n_pos, n_neg), n_pos + n_neg, 0.5).pvalue)
    else:
        sign_p = 1.0

    return {
        'n_subjects': n,
        'mean_delta': obs_mean,
        'median_delta': obs_median,
        'ci_2.5': float(ci_low),
        'ci_97.5': float(ci_high),
        'permutation_p': p_perm,
        'wilcoxon_p': w_p,
        'sign_p': sign_p,
        'n_positive': n_pos,
        'n_negative': n_neg,
        'n_zero': n_zero,
        'cs_wins_pct': float(np.mean(deltas < 0) * 100),
    }


def holm_adjust(pvalues):
    """Holm-Bonferroni 逐步校正。"""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, p[idx] * (n - rank))
        if rank > 0:
            adjusted[idx] = max(adjusted[idx], adjusted[order[rank - 1]])
    return adjusted


# ============================================================
# 7. 主计算：逐 run 计算三方法（CS / Singh / Bayes）
# ============================================================
def run_all_methods():
    """
    对每个 dataset/config/fold/seed/r 计算：
    - CS-Mondrian（α_truth=.1, α_lie=.1/r）
    - Singh/Mondrian（α_truth=α_lie=.1，即 uni）
    - Bayes-τ* 硬阈值
    对每个 C_rev 和 ε_hum 计算成本。
    返回 run-level 长表和 subject-level 长表。
    """
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
                    test_subjects = get_subject_ids(dataset, fold, 'test', len(test_labels))

                    for r in COST_RATIOS:
                        C_FN = r * C_FP
                        count += 1
                        if count % 50 == 0:
                            print(f"  {count}/{total} ({time.time()-t0:.0f}s)")

                        # --- CS-Mondrian ---
                        flags_cs, qt_cs, ql_cs = method_cs_mondrian(
                            calib_probs, calib_labels, test_probs, ALPHA_TOTAL, r)
                        # --- Singh / Mondrian (uni) ---
                        flags_singh, qt_s, ql_s = method_mondrian(
                            calib_probs, calib_labels, test_probs, ALPHA_TOTAL, ALPHA_TOTAL)
                        # --- Bayes-τ* ---
                        bayes_res = bayes_threshold_cost(test_probs, test_labels, r, C_FP)

                        for C_rev in C_REV_VALUES:
                            # CS 成本：各 ε_hum（CS 与 Singh 共用相同人工成本模型，公平比较）
                            for eps_hum in EPS_HUM_VALUES:
                                m_cs = compute_metrics_base(test_labels, flags_cs, C_FP, C_FN, C_rev, eps_hum)
                                run_rows.append({
                                    'dataset': dataset, 'config': config, 'fold': fold, 'seed': seed,
                                    'method': 'cs_mondrian', 'cost_ratio': r, 'C_rev': C_rev,
                                    'eps_hum': eps_hum, 'q_truth': qt_cs, 'q_lie': ql_cs, **m_cs})

                            # Singh 臂：各 ε_hum
                            for eps_hum in EPS_HUM_VALUES:
                                m_singh = compute_metrics_base(
                                    test_labels, flags_singh, C_FP, C_FN, C_rev, eps_hum)
                                method_name = 'singh' if eps_hum > 0 else 'singh_eps0'
                                run_rows.append({
                                    'dataset': dataset, 'config': config, 'fold': fold, 'seed': seed,
                                    'method': method_name, 'cost_ratio': r, 'C_rev': C_rev,
                                    'eps_hum': eps_hum, 'q_truth': qt_s, 'q_lie': ql_s, **m_singh})

                            # Bayes-τ*（无 C_rev 依赖，但为了对齐网格也记录）
                            run_rows.append({
                                'dataset': dataset, 'config': config, 'fold': fold, 'seed': seed,
                                'method': 'bayes_tau', 'cost_ratio': r, 'C_rev': C_rev,
                                'eps_hum': 0.0, 'q_truth': bayes_res['tau_star'],
                                'q_lie': bayes_res['tau_star'],
                                'n_test': bayes_res.get('n_test', len(test_labels)),
                                'n_truth': int(np.sum(test_labels == 0)),
                                'n_lie': int(np.sum(test_labels == 1)),
                                'n_reject': 0, 'n_accept': len(test_labels),
                                'n_singleton': len(test_labels), 'n_empty': 0, 'n_ambiguous': 0,
                                'n_fp': bayes_res['n_fp'], 'n_fn': bayes_res['n_fn'],
                                'n_tp': bayes_res['n_tp'], 'n_tn': bayes_res['n_tn'],
                                'fp_rate': bayes_res['fp_rate'], 'fn_rate': bayes_res['fn_rate'],
                                'reject_rate': 0.0, 'decision_coverage': 1.0,
                                'acceptance_rate': 1.0, 'label_coverage': np.nan,
                                'label_coverage_truth': np.nan, 'label_coverage_lie': np.nan,
                                'total_cost': bayes_res['total_cost'],
                                'expected_cost': bayes_res['expected_cost'],
                            })

                            # --- Subject-level ---
                            for subj in np.unique(test_subjects):
                                mask = test_subjects == subj
                                n_seg = int(np.sum(mask))
                                # CS 各 ε_hum
                                for eps_hum in EPS_HUM_VALUES:
                                    sm_cs = compute_metrics_base(
                                        test_labels[mask],
                                        {k: flags_cs[k][mask] for k in flags_cs},
                                        C_FP, C_FN, C_rev, eps_hum)
                                    subj_rows.append({
                                        'dataset': dataset, 'config': config, 'fold': fold, 'seed': seed,
                                        'method': 'cs_mondrian', 'cost_ratio': r, 'C_rev': C_rev,
                                        'eps_hum': eps_hum, 'subject_id': subj, 'n_segments': n_seg, **sm_cs})
                                # Singh 各 ε
                                for eps_hum in EPS_HUM_VALUES:
                                    sm_singh = compute_metrics_base(
                                        test_labels[mask],
                                        {k: flags_singh[k][mask] for k in flags_singh},
                                        C_FP, C_FN, C_rev, eps_hum)
                                    method_name = 'singh' if eps_hum > 0 else 'singh_eps0'
                                    subj_rows.append({
                                        'dataset': dataset, 'config': config, 'fold': fold, 'seed': seed,
                                        'method': method_name, 'cost_ratio': r, 'C_rev': C_rev,
                                        'eps_hum': eps_hum, 'subject_id': subj,
                                        'n_segments': n_seg, **sm_singh})
                                # Bayes
                                bayes_subj = bayes_threshold_cost(
                                    test_probs[mask], test_labels[mask], r, C_FP)
                                subj_rows.append({
                                    'dataset': dataset, 'config': config, 'fold': fold, 'seed': seed,
                                    'method': 'bayes_tau', 'cost_ratio': r, 'C_rev': C_rev,
                                    'eps_hum': 0.0, 'subject_id': subj, 'n_segments': n_seg,
                                    'n_test': n_seg,
                                    'n_truth': int(np.sum(test_labels[mask] == 0)),
                                    'n_lie': int(np.sum(test_labels[mask] == 1)),
                                    'n_reject': 0, 'n_accept': n_seg,
                                    'n_singleton': n_seg, 'n_empty': 0, 'n_ambiguous': 0,
                                    'n_fp': bayes_subj['n_fp'], 'n_fn': bayes_subj['n_fn'],
                                    'n_tp': bayes_subj['n_tp'], 'n_tn': bayes_subj['n_tn'],
                                    'fp_rate': bayes_subj['fp_rate'], 'fn_rate': bayes_subj['fn_rate'],
                                    'reject_rate': 0.0, 'decision_coverage': 1.0,
                                    'acceptance_rate': 1.0, 'label_coverage': np.nan,
                                    'label_coverage_truth': np.nan, 'label_coverage_lie': np.nan,
                                    'total_cost': bayes_subj['total_cost'],
                                    'expected_cost': bayes_subj['expected_cost'],
                                })

    run_df = pd.DataFrame(run_rows)
    subj_df = pd.DataFrame(subj_rows)
    print(f"\nDone in {time.time()-t0:.1f}s. run={len(run_df)}, subject={len(subj_df)}")
    return run_df, subj_df


# ============================================================
# 8. CS vs Singh 配对统计（受试者级）
# ============================================================
def cs_vs_singh_stats(subj_df):
    """
    对每个 dataset/config/r/C_rev/eps_hum，计算 CS - Singh 的受试者级配对差值，
    做 bootstrap/permutation/Wilcoxon/sign，并对四个中文配置做主族 Holm。
    """
    print("\n=== CS vs Singh subject-level stats ===")
    rng = np.random.default_rng(RNG_SEED)
    records = []

    for eps_hum in EPS_HUM_VALUES:
        # CS 和 Singh 都按相同 eps_hum 筛选（公平比较：两者共用相同人工成本模型）
        cs = subj_df[(subj_df['method'] == 'cs_mondrian') &
                     (subj_df['eps_hum'] == eps_hum)].copy()
        singh_method = 'singh' if eps_hum > 0 else 'singh_eps0'
        si = subj_df[(subj_df['method'] == singh_method) &
                     (subj_df['eps_hum'] == eps_hum)].copy()
        merged = cs.merge(
            si, on=['dataset', 'config', 'fold', 'seed', 'cost_ratio', 'C_rev',
                    'subject_id', 'n_segments'], suffixes=('_cs', '_singh'))
        merged['delta_cost'] = merged['expected_cost_cs'] - merged['expected_cost_singh']

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

    print(f"  stats cells: {len(stat_df)}")
    return stat_df


# ============================================================
# 9. Singh 口径 C*_rev break-even
# ============================================================
def singh_breakeven_analysis(run_df, subj_df):
    """
    对每个 dataset/config/r，计算 Singh(ε=0) deferral vs Bayes-τ* 的 C*_rev。
    用 run-level 均值做点估计，用 subject-level 做 bootstrap CI。
    """
    print("\n=== Singh C*_rev break-even analysis ===")
    rng = np.random.default_rng(RNG_SEED)
    crev_grid = np.array(C_REV_VALUES)
    records = []

    for dataset in DATASETS:
        for config in CONFIGS:
            for r in COST_RATIOS:
                # Run-level: Singh eps0 在各 C_rev 的平均 expected_cost
                singh_run = run_df[(run_df['method'] == 'singh_eps0') &
                                   (run_df['dataset'] == dataset) &
                                   (run_df['config'] == config) &
                                   (run_df['cost_ratio'] == r)]
                bayes_run = run_df[(run_df['method'] == 'bayes_tau') &
                                   (run_df['dataset'] == dataset) &
                                   (run_df['config'] == config) &
                                   (run_df['cost_ratio'] == r)]

                singh_by_crev = singh_run.groupby('C_rev')['expected_cost'].mean().reindex(crev_grid)
                bayes_cost = bayes_run['expected_cost'].mean()  # Bayes 不依赖 C_rev

                c_star = singh_breakeven_c_rev(
                    singh_by_crev.values, bayes_cost, crev_grid)

                # Subject-level bootstrap CI for C*_rev
                singh_subj = subj_df[(subj_df['method'] == 'singh_eps0') &
                                      (subj_df['dataset'] == dataset) &
                                      (subj_df['config'] == config) &
                                      (subj_df['cost_ratio'] == r)]
                bayes_subj = subj_df[(subj_df['method'] == 'bayes_tau') &
                                      (subj_df['dataset'] == dataset) &
                                      (subj_df['config'] == config) &
                                      (subj_df['cost_ratio'] == r)]

                # 按受试者聚合（跨 fold×seed 平均）
                singh_agg = singh_subj.groupby(['subject_id', 'C_rev'])['expected_cost'].mean().unstack('C_rev')
                bayes_agg = bayes_subj.groupby('subject_id')['expected_cost'].mean()

                # 对齐受试者
                common_subj = singh_agg.index.intersection(bayes_agg.index)
                singh_mat = singh_agg.loc[common_subj, crev_grid].values  # (n_subj, n_crev)
                bayes_vec = bayes_agg.loc[common_subj].values  # (n_subj,)

                n_subj = len(common_subj)
                c_star_boots = []
                for _ in range(N_BOOT):
                    idx = rng.integers(0, n_subj, n_subj)
                    s_mean = np.mean(singh_mat[idx], axis=0)
                    b_mean = np.mean(bayes_vec[idx])
                    cs = singh_breakeven_c_rev(s_mean, b_mean, crev_grid)
                    if cs is not None:
                        c_star_boots.append(cs)

                ci_low = float(np.percentile(c_star_boots, 2.5)) if c_star_boots else np.nan
                ci_high = float(np.percentile(c_star_boots, 97.5)) if c_star_boots else np.nan

                records.append({
                    'dataset': dataset, 'config': config, 'cost_ratio': r,
                    'n_subjects': n_subj,
                    'singh_cost_at_crev025': float(singh_by_crev.loc[0.25]),
                    'singh_cost_at_crev05': float(singh_by_crev.loc[0.5]),
                    'singh_cost_at_crev1': float(singh_by_crev.loc[1.0]),
                    'singh_cost_at_crev2': float(singh_by_crev.loc[2.0]),
                    'bayes_tau_cost': float(bayes_cost),
                    'C_star_rev': c_star,
                    'C_star_ci_2.5': ci_low,
                    'C_star_ci_97.5': ci_high,
                    'tau_star': 1.0 / (1.0 + r),
                })

    be_df = pd.DataFrame(records)
    print(f"  break-even cells: {len(be_df)}")
    return be_df


# ============================================================
# 10. 主流程
# ============================================================
def main():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== CS-Mondrian Singh[10] Control Arm Recompute ===")
    print(f"NPZ root: {NPZ_ROOT}")
    print(f"Output: {OUT_DIR}")

    # --- 核心计算 ---
    run_df, subj_df = run_all_methods()
    run_df.to_csv(os.path.join(OUT_DIR, 'singh_run_level.csv'), index=False)
    subj_df.to_csv(os.path.join(OUT_DIR, 'singh_subject_level.csv'), index=False)

    # --- CS vs Singh 统计 ---
    stat_df = cs_vs_singh_stats(subj_df)
    stat_df.to_csv(os.path.join(OUT_DIR, 'singh_eps_hum_comparison.csv'), index=False)

    # --- Singh C*_rev break-even ---
    be_df = singh_breakeven_analysis(run_df, subj_df)
    be_df.to_csv(os.path.join(OUT_DIR, 'singh_breakeven_vs_bayes.csv'), index=False)

    # --- 打印主终点 ---
    print("\n=== PRIMARY ENDPOINT (r=3, C_rev=0.5) ===")
    primary = stat_df[(stat_df['cost_ratio'] == 3) & (stat_df['C_rev'] == 0.5)]
    print(primary[['dataset', 'config', 'eps_hum', 'n_subjects', 'mean_delta',
                    'ci_2.5', 'ci_97.5', 'permutation_p', 'holm_p', 'cs_wins_pct']].to_string(index=False))

    print("\n=== Singh C*_rev (r=3) ===")
    be_primary = be_df[be_df['cost_ratio'] == 3]
    print(be_primary[['dataset', 'config', 'n_subjects', 'bayes_tau_cost',
                       'C_star_rev', 'C_star_ci_2.5', 'C_star_ci_97.5']].to_string(index=False))

    print(f"\nTotal elapsed: {time.time()-t_start:.1f}s")
    return run_df, subj_df, stat_df, be_df


if __name__ == '__main__':
    main()
