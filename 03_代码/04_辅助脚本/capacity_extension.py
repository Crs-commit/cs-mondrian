# -*- coding: utf-8 -*-
"""
CS-Mondrian 容量约束扩展（Capacity-Aware Extension）v2
=====================================================
依据 CAPACITY_AWARE_EXTENSION_PROTOCOL_2026-09-04.md v1.0 实现。

冻结边界（不得改写）：
- 主方法仍为 alpha_truth=a, alpha_lie=a/r（同一成本映射族内的 operating-point selection）
- 主终点仍为 a=0.10, r=3, C_FP=1, C_rev=0.5
- 现有严格版全部主结果保留，扩展不重跑、不替换

数学框架：
- 候选总失覆盖率网格 G = {0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30}
- 工作点 theta_r(a) = (alpha_truth, alpha_lie) = (a, a/r)
- 选择规则 a_hat(B) = argmin E[cost]_sel(a)  s.t.  Reject_sel(a) <= B
  - 可行集为空 -> 记 infeasible，选拒判率最低候选点，不删预算
  - 并列决胜：较低拒判率 -> 较低 lie-FN -> 较小 a

防选择偏差：
- calibration 受试者按受试者互斥 1:1 分层拆 selection / final_calibration
- selection 只选 a；final_calibration 只算最终分位点；临时分位点不得用于 test
- selection 不读 test 标签 / final_calibration 指标

统计口径：
- 推断单位为受试者；test 指标先按受试者聚合，再跨受试者 bootstrap 配对 CI
- 配对基线：固定锚点 CS-Mondrian（a=0.10, final_calibration 重校准）与
  固定锚点 Mondrian（alpha_truth=alpha_lie=0.10, final_calibration 重校准）
- 每个受试者的 delta 用该受试者自身片段计算，不做整折重复

预算：B in {0.50, 0.60, 0.70, 0.80} + unconstrained-reference（固定 a=0.10）

环境：python（numpy / pandas / scipy）
"""
import os

_HOME = os.path.expanduser("~")
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import os
import sys
import json
import time
import hashlib
import argparse
import numpy as np
import pandas as pd

# 复用严格版共形核心（同一套 qhat/指标逻辑，保证回归一致）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRICT_DIR = os.path.join(_PKG_ROOT, '03_代码', '01_核心严格重算')
sys.path.insert(0, STRICT_DIR)
from strict_recompute import conformal_quantile, build_flags, method_mondrian, compute_metrics

# ============================================================
# 0. 冻结配置
# ============================================================
DATA_ROOT = os.environ.get("CS_MONDRIAN_DATA", os.path.join(_PKG_ROOT, "04_数据"))
SUBJECT_MAP_PATH = os.path.join(DATA_ROOT, "01_中文_SEUMLD与MDPE", "subject_map.csv")
OUT_DIR = os.path.join(os.environ.get("CS_MONDRIAN_OUTPUT", os.path.join(_PKG_ROOT, "recomputed")), "capacity")

DATASETS = ['SEUMLD', 'MDPE']
CONFIGS = ['OADNet_text', 'OADNet_audio']
FOLDS = [0, 1, 2, 3, 4]
SEEDS = [7, 42, 123, 2024, 2026]
GRID = [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]
BUDGETS = [0.50, 0.60, 0.70, 0.80]
ANCHOR_A = 0.10
C_FP = 1.0
C_REV = 0.5
R_PRIMARY = 3.0

RNG_SEED = 20260904   # 受试者拆分种子（冻结，写入日志）
SUBJECT_CI_SEED = 7   # 受试者 bootstrap 种子

# ============================================================
# 1. 数据加载与受试者映射
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
    path = os.path.join(DATA_ROOT, '01_中文_SEUMLD与MDPE', dataset + '_preds', f'{config}_f{fold}_s{seed}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


# ============================================================
# 2. calibration -> selection / final_calibration 受试者互斥拆分
# ============================================================
def split_calibration_subjects(calib_subjects, calib_labels, seed=RNG_SEED):
    """
    按受试者互斥 1:1 分层拆分。
    - 以受试者为单位，不切割任何受试者的片段
    - 分层变量：该受试者的 lie 段数（类别平衡）
    - 返回 selection 掩码（与 calib 数组等长）
    """
    calib_subjects = np.asarray(calib_subjects)
    calib_labels = np.asarray(calib_labels)
    rng = np.random.default_rng(seed)
    uniq = np.unique(calib_subjects)
    lie_cnt = {int(s): int(np.sum((calib_subjects == s) & (calib_labels == 1))) for s in uniq}
    # 受试者级平衡分配。旧实现先排序后整体切半，可能把低 lie-count
    # 受试者集中到 selection、高 lie-count 受试者集中到 final-calibration，
    # 造成严重的类别比例偏差。这里采用“降序 lie-count + 负载最小子集”
    # 的确定性贪心分配，并严格固定 selection 的受试者数。
    n_sel = len(uniq) // 2
    tie_order = rng.permutation(uniq)
    tie_rank = {int(s): i for i, s in enumerate(tie_order)}
    ordered = sorted((int(s) for s in uniq),
                     key=lambda s: (-lie_cnt[s], tie_rank[s]))
    sel_subjects = set()
    fin_subjects = set()
    sel_lie = fin_lie = 0
    for s in ordered:
        if len(sel_subjects) >= n_sel:
            fin_subjects.add(s)
        elif len(fin_subjects) >= len(uniq) - n_sel:
            sel_subjects.add(s)
        elif sel_lie < fin_lie:
            sel_subjects.add(s)
            sel_lie += lie_cnt[s]
        elif fin_lie < sel_lie:
            fin_subjects.add(s)
            fin_lie += lie_cnt[s]
        else:
            # 相同负载时优先放入 selection，保持随机种子可复现
            sel_subjects.add(s)
            sel_lie += lie_cnt[s]
    mask = np.array([int(s) in sel_subjects for s in calib_subjects], dtype=bool)
    n_lie_sel = int(np.sum(calib_labels[mask] == 1))
    n_lie_fin = int(np.sum(calib_labels[~mask] == 1))
    return mask, {
        'n_sel_subjects': len(sel_subjects),
        'n_fin_subjects': len(uniq) - len(sel_subjects),
        'n_sel_segments': int(np.sum(mask)),
        'n_fin_segments': int(np.sum(~mask)),
        'lie_sel': n_lie_sel, 'lie_fin': n_lie_fin,
        'split_hash': hashlib.sha256(
            ','.join(map(str, sorted(sel_subjects))).encode()).hexdigest()[:16],
        'split_seed': seed}


# ============================================================
# 3. selection 选择器
# ============================================================
def select_alpha_on_selection(sel_probs, sel_labels, r, B, C_rev=C_REV):
    """在 selection 集上评估每个候选 a；临时分位点只用于选择，不得用于 test。"""
    C_FN = r * C_FP
    cand_rows = []
    for a in GRID:
        at, al = a, a / r
        flags, qt, ql = method_mondrian(sel_probs, sel_labels, sel_probs, at, al)
        m = compute_metrics(sel_labels, flags, C_FP, C_FN, C_rev)
        cand_rows.append({
            'a': a, 'alpha_truth': at, 'alpha_lie': al,
            'sel_reject': m['reject_rate'], 'sel_fn': m['fn_rate'],
            'sel_fp': m['fp_rate'], 'sel_expected_cost': m['expected_cost'],
            'sel_q_truth': qt, 'sel_q_lie': ql,
        })
    feasible_rows = [c for c in cand_rows if c['sel_reject'] <= B]
    if feasible_rows:
        chosen = min(feasible_rows, key=lambda c: (c['sel_expected_cost'],
                                                   c['sel_reject'], c['sel_fn'], c['a']))
        feasible = True
    else:
        chosen = min(cand_rows, key=lambda c: c['sel_reject'])
        feasible = False
    return chosen, cand_rows, feasible


# ============================================================
# 4. final_calibration 独立重校准 + test 评估（run 级 + subject 级）
# ============================================================
def evaluate_workpoint(fin_probs, fin_labels, test_probs, test_labels,
                       test_subjects, alpha_truth, alpha_lie, r, C_rev=C_REV):
    """在 final_calibration 重算分位点，评估 run 级与受试者级 test 指标。"""
    C_FN = r * C_FP
    flags, qt, ql = method_mondrian(fin_probs, fin_labels, test_probs,
                                    alpha_truth, alpha_lie)
    m = compute_metrics(test_labels, flags, C_FP, C_FN, C_rev)
    subj_rows = []
    for subj in np.unique(test_subjects):
        mask = test_subjects == subj
        sm = compute_metrics(test_labels[mask],
                             {k: flags[k][mask] for k in flags},
                             C_FP, C_FN, C_rev)
        subj_rows.append({
            'subject_id': int(subj), 'n_segments': int(np.sum(mask)),
            'expected_cost': sm['expected_cost'],
            'reject_rate': sm['reject_rate'],
            'fn_rate': sm['fn_rate'],
            'label_coverage_lie': sm['label_coverage_lie'],
        })
    return flags, m, qt, ql, subj_rows


# ============================================================
# 5. 单单元流程（dataset-config-fold-seed-r）
# ============================================================
def run_unit(dataset, config, fold, seed, r, verbose=False):
    d = load_npz(dataset, config, fold, seed)
    calib_probs = np.asarray(d['calib_probs'], dtype=float)
    calib_labels = np.asarray(d['calib_labels'])
    test_probs = np.asarray(d['probs'], dtype=float)
    test_labels = np.asarray(d['labels'])
    test_subjects = get_subject_ids(dataset, fold, 'test', len(test_labels))
    calib_subjects = get_subject_ids(dataset, fold, 'calib', len(calib_labels))

    sel_mask, split_info = split_calibration_subjects(calib_subjects, calib_labels)
    sel_probs, sel_labels = calib_probs[sel_mask], calib_labels[sel_mask]
    fin_probs, fin_labels = calib_probs[~sel_mask], calib_labels[~sel_mask]

    run_rows = []
    subj_rows = []
    sel_diag = []
    for B in BUDGETS:
        chosen, cands, feasible = select_alpha_on_selection(sel_probs, sel_labels, r, B)
        _, m, qt, ql, srows = evaluate_workpoint(
            fin_probs, fin_labels, test_probs, test_labels, test_subjects,
            chosen['alpha_truth'], chosen['alpha_lie'], r)
        base = {'dataset': dataset, 'config': config, 'fold': fold, 'seed': seed,
                'r': r, 'B': B, 'feasible': feasible,
                'selected_alpha_total': chosen['a'], 'selected_alpha_lie': chosen['a'] / r,
                'test_q_truth': qt, 'test_q_lie': ql}
        run_rows.append({**base, 'test_n': m['n_test'], 'test_n_truth': m['n_truth'],
                         'test_n_lie': m['n_lie'], 'test_reject': m['reject_rate'],
                         'test_n_reject': m['n_reject'],
                         'test_decision_coverage': m['decision_coverage'],
                         'test_fp': m['fp_rate'], 'test_fn': m['fn_rate'],
                         'test_label_coverage': m['label_coverage'],
                         'test_label_coverage_truth': m['label_coverage_truth'],
                         'test_label_coverage_lie': m['label_coverage_lie'],
                         'test_expected_cost': m['expected_cost']})
        for sr in srows:
            subj_rows.append({**base, **sr})
        for c in cands:
            sel_diag.append({**base, 'a': c['a'], 'alpha_truth': c['alpha_truth'],
                             'alpha_lie': c['alpha_lie'], 'sel_reject': c['sel_reject'],
                             'sel_fn': c['sel_fn'], 'sel_fp': c['sel_fp'],
                             'sel_expected_cost': c['sel_expected_cost'],
                             'sel_q_truth': c['sel_q_truth'], 'sel_q_lie': c['sel_q_lie'],
                             'selected': int(c['a'] == chosen['a'])})

    # --- unconstrained-reference：固定锚点 CS-Mondrian（a=0.10）---
    at_ref, al_ref = ANCHOR_A, ANCHOR_A / r
    _, m_ref, qt_ref, ql_ref, srows_ref = evaluate_workpoint(
        fin_probs, fin_labels, test_probs, test_labels, test_subjects,
        at_ref, al_ref, r)
    ref_base = {'dataset': dataset, 'config': config, 'fold': fold, 'seed': seed,
                'r': r, 'B': 'ref', 'feasible': True,
                'selected_alpha_total': ANCHOR_A, 'selected_alpha_lie': ANCHOR_A / r,
                'test_q_truth': qt_ref, 'test_q_lie': ql_ref}
    run_rows.append({**ref_base, 'test_n': m_ref['n_test'],
                     'test_n_truth': m_ref['n_truth'], 'test_n_lie': m_ref['n_lie'],
                     'test_reject': m_ref['reject_rate'],
                     'test_n_reject': m_ref['n_reject'],
                     'test_decision_coverage': m_ref['decision_coverage'],
                     'test_fp': m_ref['fp_rate'], 'test_fn': m_ref['fn_rate'],
                     'test_label_coverage': m_ref['label_coverage'],
                     'test_label_coverage_truth': m_ref['label_coverage_truth'],
                     'test_label_coverage_lie': m_ref['label_coverage_lie'],
                     'test_expected_cost': m_ref['expected_cost']})
    for sr in srows_ref:
        subj_rows.append({**ref_base, **sr})

    # --- 固定锚点 Mondrian（alpha_truth=alpha_lie=0.10）---
    _, m_mo, qt_mo, ql_mo, srows_mo = evaluate_workpoint(
        fin_probs, fin_labels, test_probs, test_labels, test_subjects,
        ANCHOR_A, ANCHOR_A, r)
    mo_base = {'dataset': dataset, 'config': config, 'fold': fold, 'seed': seed,
               'r': r, 'B': 'mondrian', 'feasible': True,
               'selected_alpha_total': ANCHOR_A, 'selected_alpha_lie': ANCHOR_A,
               'test_q_truth': qt_mo, 'test_q_lie': ql_mo}
    run_rows.append({**mo_base, 'test_n': m_mo['n_test'],
                     'test_n_truth': m_mo['n_truth'], 'test_n_lie': m_mo['n_lie'],
                     'test_reject': m_mo['reject_rate'],
                     'test_n_reject': m_mo['n_reject'],
                     'test_decision_coverage': m_mo['decision_coverage'],
                     'test_fp': m_mo['fp_rate'], 'test_fn': m_mo['fn_rate'],
                     'test_label_coverage': m_mo['label_coverage'],
                     'test_label_coverage_truth': m_mo['label_coverage_truth'],
                     'test_label_coverage_lie': m_mo['label_coverage_lie'],
                     'test_expected_cost': m_mo['expected_cost']})
    for sr in srows_mo:
        subj_rows.append({**mo_base, **sr})

    return run_rows, subj_rows, sel_diag, split_info


# ============================================================
# 6. 回归测试：固定 a=0.10 与严格版一致（不拆分，全 calib）
# ============================================================
def regression_check(r):
    strict_csv = os.environ.get(
        "STRICT_CORE_RUN_LEVEL",
        os.path.join(_HOME, "Desktop", "<COST_REPO>", "09_supplemental_validation", "10_strict_recompute", "strict_core_run_level.csv"))
    sdf = pd.read_csv(strict_csv)
    sdf = sdf[(sdf['method'] == 'cs_mondrian') & (sdf['cost_ratio'] == r) &
              (sdf['C_rev'] == C_REV)]
    mismatches = []
    for dataset in DATASETS:
        for config in CONFIGS:
            for fold in FOLDS:
                for seed in SEEDS:
                    d = load_npz(dataset, config, fold, seed)
                    calib_p = np.asarray(d['calib_probs'], dtype=float)
                    calib_l = np.asarray(d['calib_labels'])
                    test_p = np.asarray(d['probs'], dtype=float)
                    test_l = np.asarray(d['labels'])
                    at, al = ANCHOR_A, ANCHOR_A / r
                    flags, qt, ql = method_mondrian(calib_p, calib_l, test_p, at, al)
                    m = compute_metrics(test_l, flags, C_FP, r * C_FP, C_REV)
                    key = (sdf['dataset'] == dataset) & (sdf['config'] == config) & \
                          (sdf['fold'] == fold) & (sdf['seed'] == seed)
                    ref = sdf[key].iloc[0]
                    tol = 1e-8
                    if (abs(qt - ref['q_truth']) > tol or abs(ql - ref['q_lie']) > tol or
                            abs(m['reject_rate'] - ref['reject_rate']) > tol or
                            abs(m['expected_cost'] - ref['expected_cost']) > tol or
                            abs(m['fn_rate'] - ref['fn_rate']) > tol):
                        mismatches.append({
                            'dataset': dataset, 'config': config, 'fold': fold,
                            'seed': seed, 'q_truth': qt, 'ref_q_truth': ref['q_truth'],
                            'q_lie': ql, 'ref_q_lie': ref['q_lie'],
                            'reject': m['reject_rate'], 'ref_reject': ref['reject_rate'],
                            'cost': m['expected_cost'], 'ref_cost': ref['expected_cost'],
                            'fn': m['fn_rate'], 'ref_fn': ref['fn_rate'],
                        })
    return mismatches


# ============================================================
# 7. 受试者级配对 delta + 聚合 CI
# ============================================================
def build_subject_deltas(subj_df):
    """
    以受试者为配对单位：
    - 同一 fold-seed 内，对每个受试者计算选定工作点相对锚点 CS / 锚点 Mondrian 的 delta
    - 跨 fold-seed 按受试者平均，得到每受试者一个 delta
    输出列：dataset/config/r/B/subject_id/delta_cost_vs_cs/delta_reject_vs_cs/delta_fn_vs_cs/
            delta_lc_lie_vs_cs/delta_cost_vs_mo/...
    """
    recs = []
    keys = ['dataset', 'config', 'fold', 'seed', 'r', 'subject_id']
    for (ds, cf, fo, se, r, subj), g in subj_df.groupby(keys):
        ref_cs = g[g['B'] == 'ref']
        ref_mo = g[g['B'] == 'mondrian']
        if len(ref_cs) == 0 or len(ref_mo) == 0:
            continue
        rcs = ref_cs.iloc[0]
        rmo = ref_mo.iloc[0]
        for _, row in g[~g['B'].isin(['ref', 'mondrian'])].iterrows():
            recs.append({
                'dataset': ds, 'config': cf, 'fold': fo, 'seed': se,
                'r': r, 'B': row['B'], 'subject_id': subj,
                'expected_cost': row['expected_cost'],
                'reject_rate': row['reject_rate'], 'fn_rate': row['fn_rate'],
                'label_coverage_lie': row['label_coverage_lie'],
                'delta_cost_vs_cs': row['expected_cost'] - rcs['expected_cost'],
                'delta_reject_vs_cs': row['reject_rate'] - rcs['reject_rate'],
                'delta_fn_vs_cs': row['fn_rate'] - rcs['fn_rate'],
                'delta_lc_lie_vs_cs': (row['label_coverage_lie'] -
                                       rcs['label_coverage_lie']),
                'delta_cost_vs_mo': row['expected_cost'] - rmo['expected_cost'],
                'delta_reject_vs_mo': row['reject_rate'] - rmo['reject_rate'],
                'delta_fn_vs_mo': row['fn_rate'] - rmo['fn_rate'],
                'delta_lc_lie_vs_mo': (row['label_coverage_lie'] -
                                       rmo['label_coverage_lie']),
            })
    return recs


def aggregate_subject_ci(delta_df, n_boot=10000, rng_seed=SUBJECT_CI_SEED):
    """
    对每个 (dataset, config, r, B)：
    - 按受试者跨 fold-seed 平均 delta（独立受试者为推断单位）
    - 受试者重抽样 bootstrap 配对 CI（相对锚点 CS 与锚点 Mondrian 各一组）
    """
    out = []
    rng = np.random.default_rng(rng_seed)
    cols = ['delta_cost_vs_cs', 'delta_reject_vs_cs', 'delta_fn_vs_cs',
            'delta_lc_lie_vs_cs', 'delta_cost_vs_mo', 'delta_reject_vs_mo',
            'delta_fn_vs_mo', 'delta_lc_lie_vs_mo']
    for (ds, cf, r, B), g in delta_df.groupby(['dataset', 'config', 'r', 'B']):
        # 按受试者跨 fold-seed 平均
        subj_agg = g.groupby('subject_id')[cols].mean().reset_index()
        if len(subj_agg) < 5:
            continue
        row = {'dataset': ds, 'config': cf, 'r': r, 'B': B,
               'n_subjects': len(subj_agg)}
        for c in cols:
            vals = subj_agg[c].values
            row[f'mean_{c}'] = float(np.mean(vals))
            boot = rng.integers(0, len(vals), (n_boot, len(vals)))
            bm = np.mean(vals[boot], axis=1)
            row[f'{c}_ci_2.5'] = float(np.percentile(bm, 2.5))
            row[f'{c}_ci_97.5'] = float(np.percentile(bm, 97.5))
        # 平均 test 指标（run 级平均，供 Pareto 图）
        out.append(row)
    return pd.DataFrame(out)


# ============================================================
# 8. 泄漏审计
# ============================================================
def leakage_audit(units):
    rows = []
    for ds, cf, fo, se in units:
        d = load_npz(ds, cf, fo, se)
        calib_subjects = get_subject_ids(ds, fo, 'calib', len(d['calib_labels']))
        test_subjects = get_subject_ids(ds, fo, 'test', len(d['labels']))
        sel_mask, info = split_calibration_subjects(
            calib_subjects, np.asarray(d['calib_labels']))
        sel_set = set(int(s) for s in calib_subjects[sel_mask])
        fin_set = set(int(s) for s in calib_subjects[~sel_mask])
        test_set = set(int(s) for s in test_subjects)
        rows.append({
            'dataset': ds, 'config': cf, 'fold': fo, 'seed': se,
            'sel_calib_disjoint': len(sel_set & fin_set) == 0,
            'sel_test_disjoint': len(sel_set & test_set) == 0,
            'fin_test_disjoint': len(fin_set & test_set) == 0,
            'n_sel_subjects': len(sel_set), 'n_fin_subjects': len(fin_set),
            'n_test_subjects': len(test_set),
            'split_hash': info['split_hash'],
            'PASS': (len(sel_set & fin_set) == 0 and
                     len(sel_set & test_set) == 0 and
                     len(fin_set & test_set) == 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'capacity_leakage_audit.csv'), index=False)
    return df


# ============================================================
# 9. 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['regression', 'single', 'full'],
                    default='regression')
    ap.add_argument('--dataset', default='SEUMLD')
    ap.add_argument('--config', default='OADNet_text')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--r', type=float, default=R_PRIMARY)
    ap.add_argument('--nboot', type=int, default=10000)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {
        'protocol': 'CAPACITY_AWARE_EXTENSION_PROTOCOL_2026-09-04.md v1.0',
        'grid': GRID, 'budgets': BUDGETS, 'anchor_a': ANCHOR_A,
        'r': args.r, 'C_FP': C_FP, 'C_rev': C_REV,
        'split_seed': RNG_SEED, 'subject_ci_seed': SUBJECT_CI_SEED,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'python': sys.version.split()[0],
        'code_sha256': hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()
                                      ).hexdigest()[:16],
    }

    # ---------- 回归测试 ----------
    print('=== Regression: a=0.10 vs strict ===')
    mismatches = regression_check(args.r)
    if mismatches:
        pd.DataFrame(mismatches).to_csv(os.path.join(OUT_DIR, 'regression_mismatch.csv'),
                                        index=False)
        print(f'  FAIL: {len(mismatches)} mismatches -> regression_mismatch.csv')
        return 1
    print(f'  PASS: all {len(DATASETS)*len(CONFIGS)*len(FOLDS)*len(SEEDS)} '
          f'runs match strict (tol 1e-8)')
    meta['regression'] = 'PASS'
    if args.mode == 'regression':
        with open(os.path.join(OUT_DIR, 'capacity_protocol.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f'Done in {time.time()-t0:.1f}s')
        return 0

    # ---------- 计算单元 ----------
    if args.mode == 'single':
        units = [(args.dataset, args.config, args.fold, args.seed)]
    else:
        units = [(ds, cf, fo, se) for ds in DATASETS for cf in CONFIGS
                 for fo in FOLDS for se in SEEDS]

    # ---------- 泄漏审计（仅本次单元）----------
    print('=== Leakage audit ===')
    aud = leakage_audit(units)
    print(f'  audit rows={len(aud)}, PASS={(aud["PASS"]).sum()}/{len(aud)}')

    all_run, all_subj, all_sel, all_split = [], [], [], []
    total = len(units)
    for i, (ds, cf, fo, se) in enumerate(units, 1):
        rows, subj_rows, sel_diag, split_info = run_unit(ds, cf, fo, se, args.r)
        all_run.extend(rows)
        all_subj.extend(subj_rows)
        all_sel.extend(sel_diag)
        all_split.append({'dataset': ds, 'config': cf, 'fold': fo, 'run_seed': se,
                          **split_info})
        if i % 10 == 0 or i == total:
            print(f'  [{i}/{total}] {ds}/{cf} f{fo} s{se} ({time.time()-t0:.0f}s)')

    run_df = pd.DataFrame(all_run)
    subj_df = pd.DataFrame(all_subj)
    sel_df = pd.DataFrame(all_sel)
    split_df = pd.DataFrame(all_split)
    run_df.to_csv(os.path.join(OUT_DIR, 'capacity_test_units.csv'), index=False)
    subj_df.to_csv(os.path.join(OUT_DIR, 'capacity_subject_level.csv'), index=False)
    sel_df.to_csv(os.path.join(OUT_DIR, 'capacity_selection_diagnostics.csv'), index=False)
    split_df.to_csv(os.path.join(OUT_DIR, 'capacity_split_info.csv'), index=False)
    print(f'  run={len(run_df)}, subject={len(subj_df)}, sel_diag={len(sel_df)}')

    # ---------- 受试者配对 delta + CI ----------
    delta_recs = build_subject_deltas(subj_df)
    delta_df = pd.DataFrame(delta_recs)
    delta_df.to_csv(os.path.join(OUT_DIR, 'capacity_subject_deltas.csv'), index=False)
    ci_df = aggregate_subject_ci(delta_df, n_boot=args.nboot)
    ci_df.to_csv(os.path.join(OUT_DIR, 'capacity_subject_ci.csv'), index=False)
    show = ci_df[['dataset', 'config', 'r', 'B', 'n_subjects',
                  'mean_delta_cost_vs_cs', 'delta_cost_vs_cs_ci_2.5',
                  'delta_cost_vs_cs_ci_97.5', 'mean_delta_cost_vs_mo',
                  'delta_cost_vs_mo_ci_2.5', 'delta_cost_vs_mo_ci_97.5',
                  'mean_delta_reject_vs_mo']].copy()
    print('\n=== Subject-level paired CI ===')
    print(show.to_string(index=False))

    # ---------- 补充表（协议 5.3 建议字段）----------
    supp_cols = ['dataset', 'config', 'r', 'B', 'feasible',
                 'selected_alpha_total', 'selected_alpha_lie',
                 'test_reject', 'test_expected_cost', 'test_fn',
                 'test_label_coverage_lie']
    supp = pd.concat([run_df[run_df['B'].isin(BUDGETS)][supp_cols],
                      run_df[run_df['B'] == 'ref'][supp_cols]], ignore_index=True)
    supp.to_csv(os.path.join(OUT_DIR, 'capacity_supplement_table.csv'), index=False)

    # ---------- 日志 ----------
    meta.update({
        'elapsed_sec': round(time.time() - t0, 1),
        'unit_rows': len(run_df), 'subject_rows': len(subj_df),
        'sel_rows': len(sel_df), 'split_rows': len(split_df),
        'subject_delta_rows': len(delta_df),
        'leakage_audit_pass': int(aud['PASS'].sum()),
        'leakage_audit_total': len(aud),
    })
    with open(os.path.join(OUT_DIR, 'capacity_protocol.json'), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'\nDone in {time.time()-t0:.1f}s. Outputs -> {OUT_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
