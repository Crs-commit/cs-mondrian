# -*- coding: utf-8 -*-
"""
外部两数据集 Singh 对照臂计算
- Open-Domain: 受试者级 (n=511)，内嵌 test_subject_ids
- Cross-Cultural: 配对贡献单位级 (n=1200)，每单位含 truth+lie 两文本先平均
复用 singh_arm_recompute.py 的核心函数。
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

# 基于当前脚本目录的导入路径（代码快照解压后即可运行）
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from singh_arm_recompute import (
    conformal_quantile, build_flags, method_mondrian, method_cs_mondrian,
    compute_metrics_base, bayes_threshold_cost, singh_breakeven_c_rev,
    holm_adjust
)

INTEGRATION_ROOT = os.environ.get(
    'CS_MONDRIAN_ROOT',
    _PKG_ROOT
)
EXT_ROOT = os.path.join(INTEGRATION_ROOT, '12_外部验证与扩展实验_2026-09-07')
OUT_DIR = os.environ.get("CS_MONDRIAN_OUTPUT", os.path.join(_PKG_ROOT, "recomputed"))

OPEN_DOMAIN_PREDS = os.path.join(os.environ.get("CS_MONDRIAN_DATA", os.path.join(_PKG_ROOT, "04_数据")), "02_OpenDomain_修复后权威")
CROSS_CULTURAL_PREDS = os.path.join(os.environ.get("CS_MONDRIAN_DATA", os.path.join(_PKG_ROOT, "04_数据")), "04_CrossCultural_v2_Singh口径")

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


def load_ext_npz(preds_dir, fold, seed):
    """加载外部 npz，返回 probs/labels/calib_probs/calib_labels/test_unit_ids."""
    path = os.path.join(preds_dir, f'TFIDF_LR_f{fold}_s{seed}.npz')
    d = np.load(path, allow_pickle=True)
    return {
        'probs': d['probs'],
        'labels': d['labels'],
        'calib_probs': d['calib_probs'],
        'calib_labels': d['calib_labels'],
        'test_unit_ids': d['test_subject_ids'],
    }


def compute_unit_level_costs(labels, flags, unit_ids, C_FP, C_FN, C_rev, eps_hum):
    """
    计算单位级（受试者或配对贡献单位）的平均成本。
    对每个单位，先计算该单位所有样本的 total_cost / n_samples。
    """
    labels = np.asarray(labels)
    reject = flags['reject']
    singleton = flags['singleton']
    pred_label = flags['pred_label']
    unit_ids = np.asarray(unit_ids)

    # per-sample cost
    auto_mask = singleton
    fp_mask = auto_mask & (pred_label == 1) & (labels == 0)
    fn_mask = auto_mask & (pred_label == 0) & (labels == 1)
    sample_cost = np.zeros(len(labels), dtype=float)
    sample_cost[fp_mask] = C_FP
    sample_cost[fn_mask] = C_FN
    # reject samples
    rej_mask = reject
    if eps_hum == 0.0:
        sample_cost[rej_mask] = C_rev
    else:
        rej_labels = labels[rej_mask]
        rej_extra = eps_hum * np.where(rej_labels == 1, C_FN, C_FP)
        sample_cost[rej_mask] = C_rev + rej_extra

    # aggregate by unit
    units, inv = np.unique(unit_ids, return_inverse=True)
    unit_cost = np.zeros(len(units), dtype=float)
    unit_count = np.zeros(len(units), dtype=int)
    np.add.at(unit_cost, inv, sample_cost)
    np.add.at(unit_count, inv, 1)
    unit_avg_cost = unit_cost / unit_count

    return dict(zip(units, unit_avg_cost)), units


def run_external_dataset(name, preds_dir, unit_type):
    """
    运行单个外部数据集的 Singh 对照臂计算。
    unit_type: 'subject' (Open-Domain) 或 'paired_unit' (Cross-Cultural)
    """
    print(f"\n{'='*60}")
    print(f"External dataset: {name} ({unit_type})")
    print(f"{'='*60}")
    t0 = time.time()

    # 收集每个 (fold, seed, r, method, eps_hum, C_rev) 的单位级成本
    # 结构: unit_cost_data[(r, C_rev, eps_hum, method)][(fold, seed)] = {unit_id: avg_cost}
    unit_cost_data = {}
    all_units = set()

    for fold in FOLDS:
        for seed in SEEDS:
            d = load_ext_npz(preds_dir, fold, seed)
            test_probs = d['probs']
            test_labels = d['labels']
            calib_probs = d['calib_probs']
            calib_labels = d['calib_labels']
            test_unit_ids = d['test_unit_ids']
            all_units.update(test_unit_ids)

            for r in COST_RATIOS:
                C_FN = r * C_FP
                # CS
                flags_cs, _, _ = method_cs_mondrian(
                    calib_probs, calib_labels, test_probs, ALPHA_TOTAL, r)
                # Singh / Mondrian
                flags_singh, _, _ = method_mondrian(
                    calib_probs, calib_labels, test_probs, ALPHA_TOTAL, ALPHA_TOTAL)
                # Bayes
                bayes_res = bayes_threshold_cost(test_probs, test_labels, r, C_FP)

                for C_rev in C_REV_VALUES:
                    for eps_hum in EPS_HUM_VALUES:
                        # CS
                        uc_cs, _ = compute_unit_level_costs(
                            test_labels, flags_cs, test_unit_ids, C_FP, C_FN, C_rev, eps_hum)
                        key = (r, C_rev, eps_hum, 'cs_mondrian')
                        if key not in unit_cost_data:
                            unit_cost_data[key] = {}
                        unit_cost_data[key][(fold, seed)] = uc_cs

                        # Singh
                        uc_singh, _ = compute_unit_level_costs(
                            test_labels, flags_singh, test_unit_ids, C_FP, C_FN, C_rev, eps_hum)
                        singh_method = 'singh' if eps_hum > 0 else 'singh_eps0'
                        key = (r, C_rev, eps_hum, singh_method)
                        if key not in unit_cost_data:
                            unit_cost_data[key] = {}
                        unit_cost_data[key][(fold, seed)] = uc_singh

                    # Bayes (eps_hum=0, no reject)
                    bayes_per_sample = np.zeros(len(test_labels), dtype=float)
                    pred_lie = test_probs >= bayes_res['tau_star']
                    bayes_per_sample[pred_lie & (test_labels == 0)] = C_FP
                    bayes_per_sample[(~pred_lie) & (test_labels == 1)] = C_FN
                    units_b, inv_b = np.unique(test_unit_ids, return_inverse=True)
                    bayes_cost = np.zeros(len(units_b), dtype=float)
                    bayes_count = np.zeros(len(units_b), dtype=int)
                    np.add.at(bayes_cost, inv_b, bayes_per_sample)
                    np.add.at(bayes_count, inv_b, 1)
                    bayes_avg = bayes_cost / bayes_count
                    uc_bayes = dict(zip(units_b, bayes_avg))
                    key = (r, C_rev, 0.0, 'bayes_tau')
                    if key not in unit_cost_data:
                        unit_cost_data[key] = {}
                    unit_cost_data[key][(fold, seed)] = uc_bayes

    print(f"  Core compute done in {time.time()-t0:.1f}s, {len(all_units)} unique units")

    # ============================================================
    # 统计：CS vs Singh 配对差值
    # ============================================================
    rng = np.random.default_rng(RNG_SEED)
    stat_records = []
    sorted_units = sorted(all_units)

    for eps_hum in EPS_HUM_VALUES:
        singh_method = 'singh' if eps_hum > 0 else 'singh_eps0'
        for r in COST_RATIOS:
            for C_rev in C_REV_VALUES:
                # 聚合：对每个单位，跨 fold×seed 平均
                cs_key = (r, C_rev, eps_hum, 'cs_mondrian')
                singh_key = (r, C_rev, eps_hum, singh_method)
                cs_data = unit_cost_data[cs_key]
                singh_data = unit_cost_data[singh_key]

                # 对每个单位，计算跨 fold×seed 的平均成本差
                unit_deltas = []
                for unit in sorted_units:
                    cs_vals = []
                    singh_vals = []
                    for (fold, seed) in cs_data:
                        if unit in cs_data[(fold, seed)] and unit in singh_data.get((fold, seed), {}):
                            cs_vals.append(cs_data[(fold, seed)][unit])
                            singh_vals.append(singh_data[(fold, seed)][unit])
                    if cs_vals:
                        unit_deltas.append(np.mean(cs_vals) - np.mean(singh_vals))

                deltas = np.array(unit_deltas)
                n_units = len(deltas)
                if n_units < 5:
                    continue

                obs_mean = float(np.mean(deltas))
                obs_median = float(np.median(deltas))

                # cluster bootstrap
                boot_idx = rng.integers(0, n_units, (N_BOOT, n_units))
                boot_means = np.mean(deltas[boot_idx], axis=1)
                ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

                # paired sign-flip permutation
                perm_signs = rng.choice([-1, 1], size=(N_PERM, n_units))
                perm_means = np.mean(deltas[None, :] * perm_signs, axis=1)
                # Monte Carlo random sign flips: include the observed arrangement.
                p_perm = float((np.count_nonzero(np.abs(perm_means) >= np.abs(obs_mean)) + 1) / (len(perm_means) + 1))

                # Wilcoxon
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

                stat_records.append({
                    'dataset': name, 'config': 'TFIDF_LR', 'cost_ratio': r,
                    'C_rev': C_rev, 'eps_hum': eps_hum,
                    'n_units': n_units, 'unit_type': unit_type,
                    'mean_delta': obs_mean, 'median_delta': obs_median,
                    'ci_2.5': float(ci_low), 'ci_97.5': float(ci_high),
                    'permutation_p': p_perm, 'wilcoxon_p': w_p,
                    'n_positive': n_pos, 'n_negative': n_neg, 'n_zero': n_zero,
                    'cs_wins_pct': float(np.mean(deltas < 0) * 100),
                })

    stat_df = pd.DataFrame(stat_records)

    # Holm 校正：外部数据集固定锚点（r=3, C_rev=0.5），按 eps_hum 分组
    # 外部为次要族，报告原始 p，并注明未族内校正（按方案要求）
    # 但仍计算 holm_p 供参考
    for eps_hum in EPS_HUM_VALUES:
        mask = ((stat_df['cost_ratio'] == 3) & (stat_df['C_rev'] == 0.5) &
                (stat_df['eps_hum'] == eps_hum))
        if mask.sum() > 0:
            pvals = stat_df.loc[mask, 'permutation_p'].values
            stat_df.loc[mask, 'holm_p'] = holm_adjust(pvals)

    print(f"  Stats cells: {len(stat_df)}")

    # ============================================================
    # Singh C*_rev break-even
    # ============================================================
    be_records = []
    crev_grid = np.array(C_REV_VALUES)

    for r in COST_RATIOS:
        # Singh eps0 在各 C_rev 的单位级平均成本
        singh_by_crev = {}
        for C_rev in C_REV_VALUES:
            key = (r, C_rev, 0.0, 'singh_eps0')
            data = unit_cost_data[key]
            # 跨 fold×seed 平均后再跨单位平均
            unit_avgs = []
            for unit in sorted_units:
                vals = []
                for (fold, seed) in data:
                    if unit in data[(fold, seed)]:
                        vals.append(data[(fold, seed)][unit])
                if vals:
                    unit_avgs.append(np.mean(vals))
            singh_by_crev[C_rev] = float(np.mean(unit_avgs))

        # Bayes 成本（不依赖 C_rev）
        bayes_key = (r, 0.5, 0.0, 'bayes_tau')  # C_rev 不影响 Bayes
        bayes_data = unit_cost_data[bayes_key]
        bayes_unit_avgs = []
        for unit in sorted_units:
            vals = []
            for (fold, seed) in bayes_data:
                if unit in bayes_data[(fold, seed)]:
                    vals.append(bayes_data[(fold, seed)][unit])
            if vals:
                bayes_unit_avgs.append(np.mean(vals))
        bayes_cost = float(np.mean(bayes_unit_avgs))

        singh_costs = np.array([singh_by_crev[cr] for cr in crev_grid])
        c_star = singh_breakeven_c_rev(singh_costs, bayes_cost, crev_grid)

        # bootstrap CI for C*_rev
        # 收集每个单位在各 C_rev 的 Singh 成本和 Bayes 成本（跨 fold×seed 平均）
        singh_unit_mat = []  # (n_units, n_crev)
        bayes_unit_vec = []
        for unit in sorted_units:
            s_row = []
            for C_rev in C_REV_VALUES:
                key = (r, C_rev, 0.0, 'singh_eps0')
                data = unit_cost_data[key]
                vals = []
                for (fold, seed) in data:
                    if unit in data[(fold, seed)]:
                        vals.append(data[(fold, seed)][unit])
                s_row.append(np.mean(vals) if vals else np.nan)
            b_key = (r, 0.5, 0.0, 'bayes_tau')
            b_data = unit_cost_data[b_key]
            b_vals = []
            for (fold, seed) in b_data:
                if unit in b_data[(fold, seed)]:
                    b_vals.append(b_data[(fold, seed)][unit])
            if not np.any(np.isnan(s_row)) and b_vals:
                singh_unit_mat.append(s_row)
                bayes_unit_vec.append(np.mean(b_vals))

        singh_mat = np.array(singh_unit_mat)
        bayes_vec = np.array(bayes_unit_vec)
        n_u = len(bayes_vec)
        c_star_boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n_u, n_u)
            s_mean = np.mean(singh_mat[idx], axis=0)
            b_mean = np.mean(bayes_vec[idx])
            cs = singh_breakeven_c_rev(s_mean, b_mean, crev_grid)
            if cs is not None:
                c_star_boots.append(cs)

        ci_low_be = float(np.percentile(c_star_boots, 2.5)) if c_star_boots else np.nan
        ci_high_be = float(np.percentile(c_star_boots, 97.5)) if c_star_boots else np.nan

        be_records.append({
            'dataset': name, 'config': 'TFIDF_LR', 'cost_ratio': r,
            'n_units': n_u, 'unit_type': unit_type,
            'bayes_tau_cost': bayes_cost,
            'singh_cost_at_crev025': singh_by_crev[0.25],
            'singh_cost_at_crev05': singh_by_crev[0.5],
            'singh_cost_at_crev1': singh_by_crev[1.0],
            'singh_cost_at_crev2': singh_by_crev[2.0],
            'C_star_rev': c_star,
            'C_star_ci_2.5': ci_low_be,
            'C_star_ci_97.5': ci_high_be,
            'tau_star': 1.0 / (1.0 + r),
        })

    be_df = pd.DataFrame(be_records)
    print(f"  Break-even cells: {len(be_df)}")

    # 打印主终点
    print(f"\n=== {name} PRIMARY ENDPOINT (r=3, C_rev=0.5) ===")
    primary = stat_df[(stat_df['cost_ratio'] == 3) & (stat_df['C_rev'] == 0.5)]
    print(primary[['eps_hum', 'n_units', 'mean_delta', 'ci_2.5', 'ci_97.5',
                    'permutation_p', 'cs_wins_pct']].to_string(index=False))

    print(f"\n=== {name} Singh C*_rev (r=3) ===")
    be_primary = be_df[be_df['cost_ratio'] == 3]
    print(be_primary[['n_units', 'bayes_tau_cost', 'C_star_rev',
                       'C_star_ci_2.5', 'C_star_ci_97.5']].to_string(index=False))

    print(f"\n{name} total elapsed: {time.time()-t0:.1f}s")
    return stat_df, be_df


def main():
    t_start = time.time()

    # Open-Domain (受试者级)
    od_stat, od_be = run_external_dataset('Open-Domain', OPEN_DOMAIN_PREDS, 'subject')

    # Cross-Cultural (配对贡献单位级)
    cc_stat, cc_be = run_external_dataset('Cross-Cultural', CROSS_CULTURAL_PREDS, 'paired_unit')

    # 合并保存
    ext_stat = pd.concat([od_stat, cc_stat], ignore_index=True)
    ext_be = pd.concat([od_be, cc_be], ignore_index=True)

    ext_stat.to_csv(os.path.join(OUT_DIR, 'singh_external_eps_hum_comparison.csv'), index=False)
    ext_be.to_csv(os.path.join(OUT_DIR, 'singh_external_breakeven_vs_bayes.csv'), index=False)

    print(f"\n{'='*60}")
    print(f"External datasets total elapsed: {time.time()-t_start:.1f}s")
    print(f"Output: {OUT_DIR}")
    print(f"  singh_external_eps_hum_comparison.csv ({len(ext_stat)} rows)")
    print(f"  singh_external_breakeven_vs_bayes.csv ({len(ext_be)} rows)")

    return ext_stat, ext_be


if __name__ == '__main__':
    main()
