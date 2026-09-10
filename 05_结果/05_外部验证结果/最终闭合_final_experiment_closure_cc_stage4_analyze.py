# -*- coding: utf-8 -*-
"""Cross-Cultural 主分析（方案 §5–§9）：调用官方 strict_recompute 共形核心，
    聚到配对单位级，做推断与全部次要分析。

主流程：
  1. 猴子补丁：把 strict_recompute 重定向到 CrossCultural
  2. 跑所有 (seed, fold) × (method, r, C_rev) → 每受试者(=单位) cost 等
  3. 固定锚点（r=3, C_rev=0.5）：配对单位级 d_i = avg_cost_CS - avg_cost_Mondrian
  4. 5 seed 先平均 → 1200 个单位 d_i → cluster bootstrap 10000 + sign-flip 10000
  5. r=1 锚点逐元素严格 == 0
  6. r 敏感性（r=1,2,3,5,10）
  7. Plan A：在 selection 上分别选 alpha，固定 B in {0.60,0.70,0.80}
  8. Plan B 成本分解
  9. 地区异质性（locale 亚组 + leave-one-locale）
 10. 配对敏感性（剔除 Romanian）
 11. 文本长度诊断
"""
import os

_HOME = os.path.expanduser("~")
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import os, json, sys, importlib.util
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --- 1. 加载官方核心 ---
CORE_PATH = os.path.join(_PKG_ROOT, '03_代码', '01_核心严格重算', 'strict_recompute.py')
spec = importlib.util.spec_from_file_location("strict_recompute", CORE_PATH)
core = importlib.util.module_from_spec(spec); sys.modules["strict_recompute"] = core
spec.loader.exec_module(core)

OUT = os.path.join(_PKG_ROOT, '05_结果', '05_外部验证结果')
PRED_DIR = os.path.join(OUT, "CrossCultural")  # 真正的 npz 目录（out/CrossCultural/preds/）
SUBJ_MAP_FIXED = os.path.join(OUT, "subject_map_CrossCultural_fixed.csv")

def paired_inference(values, rng, n_boot=10000, n_perm=10000):
    """Mean effect, percentile bootstrap CI, and two-sided paired sign-flip p."""
    values = np.asarray(values, dtype=float)
    obs = float(values.mean())
    boots = values[rng.randint(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(values)))
    perm = (values[None, :] * signs).mean(axis=1)
    p_lo = (np.sum(perm <= -abs(obs)) + 1) / (n_perm + 1)
    p_hi = (np.sum(perm >= abs(obs)) + 1) / (n_perm + 1)
    p_two = float(min(1.0, 2 * min(p_lo, p_hi)))
    return obs, float(lo), float(hi), p_two

def holm_adjust(p_values):
    """Holm-adjust p-values while preserving the input order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted

# --- 2. 猴子补丁 ---
core.DATA_ROOT = OUT  # load_npz 会拼 /{dataset}/preds/file.npz → out/CrossCultural/preds/...
core.DATASETS = ['CrossCultural']
core.CONFIGS = ['TFIDF_LR']
core.FOLDS = [0, 1, 2, 3, 4]
core.SEEDS = [7, 42, 123, 2024, 2026]
core.ALPHA_TOTAL = 0.10
core.COST_RATIOS = [1.0, 2.0, 3.0, 5.0, 10.0]   # r 敏感性
core.C_REV_VALUES = [0.5]                         # 固定主分析
core.METHODS = ['mondrian', 'cs_mondrian']         # 固定主分析只跑两个核心方法

# 合并 subject_map：原 SEUMLD/MDPE + 新增 CrossCultural
# 给老 map 补 seed=7 列以对齐（老 map 是单 seed）
old_map = pd.read_csv(core.SUBJECT_MAP_PATH)
new_map = pd.read_csv(SUBJ_MAP_FIXED)
old_map.columns = [c.strip() for c in old_map.columns]
new_map.columns = [c.strip() for c in new_map.columns]
if 'seed' not in old_map.columns:
    old_map['seed'] = 7
if 'seed' not in new_map.columns:
    new_map['seed'] = 7
merged = pd.concat([old_map, new_map], ignore_index=True)
# 写临时文件作为 monkey-patched SUBJECT_MAP_PATH
TMP_MAP = os.path.join(OUT, "_merged_subject_map.csv")
merged.to_csv(TMP_MAP, index=False, encoding="utf-8-sig")
core.SUBJECT_MAP_PATH = TMP_MAP
core.SUBJECT_MAP = merged
# 关键补丁：get_subject_ids 也要按 seed 过滤（原版只按 dataset/fold/role）
# 用可变容器记录当前 seed，run_all 内部无参调用能拿到
_current_seed = {"v": None}
def get_subject_ids_patched(dataset, fold, role, n_samples):
    seed = _current_seed["v"]
    if seed is None:
        # 原始行为
        sub = core.SUBJECT_MAP[(core.SUBJECT_MAP['dataset'] == dataset) &
                               (core.SUBJECT_MAP['fold'] == fold) &
                               (core.SUBJECT_MAP['role'] == role)].sort_values('pos')
    else:
        sub = core.SUBJECT_MAP[(core.SUBJECT_MAP['dataset'] == dataset) &
                               (core.SUBJECT_MAP['fold'] == fold) &
                               (core.SUBJECT_MAP['role'] == role) &
                               (core.SUBJECT_MAP['seed'] == seed)].sort_values('pos')
    if len(sub) != n_samples:
        raise ValueError(
            f"Size mismatch: {dataset} fold{fold} seed={seed} {role}: expected {n_samples}, got {len(sub)}")
    return sub['subject'].values
core.get_subject_ids = get_subject_ids_patched
# 重写 run_all: 在 for seed 循环里设 _current_seed["v"]=seed
import types
_orig_run_all = core.run_all
def run_all_patched(*args, **kwargs):
    # 通过 wrapper: 在 run_all 之前替换内部的 seed 循环逻辑
    # 简单做法：直接 monkey-patch run_all 的代码路径 — 不行，run_all 是闭包
    # 改为：在主循环里手动设 seed
    # 这里我们手工实现主循环
    DATASETS = core.DATASETS
    CONFIGS = core.CONFIGS
    FOLDS = core.FOLDS
    SEEDS = core.SEEDS
    COST_RATIOS = core.COST_RATIOS
    C_REV_VALUES = core.C_REV_VALUES
    METHODS = core.METHODS
    ALPHA_TOTAL = core.ALPHA_TOTAL
    C_FP = core.C_FP
    import time
    t0 = time.time()
    run_rows, subj_rows, qhat_rows = [], [], []
    for dataset in DATASETS:
        for config in CONFIGS:
            for fold in FOLDS:
                for seed in SEEDS:
                    _current_seed["v"] = seed
                    d = core.load_npz(dataset, config, fold, seed)
                    test_probs = d['probs']
                    test_labels = d['labels']
                    calib_probs = d['calib_probs']
                    calib_labels = d['calib_labels']
                    test_subjects = core.get_subject_ids(dataset, fold, 'test', len(test_labels))
                    for method_name in METHODS:
                        for r in COST_RATIOS:
                            C_FN = r * C_FP
                            a_truth, a_lie = core.get_alpha_values(method_name, r)
                            if method_name == 'confidence_reject_tuned':
                                for C_rev in C_REV_VALUES:
                                    flags, qt, ql = core.METHOD_FUNCS[method_name](
                                        calib_probs, calib_labels, test_probs,
                                        ALPHA_TOTAL, r, C_rev)
                                    m = core.compute_metrics(test_labels, flags, C_FP, C_FN, C_rev)
                                    run_rows.append({'dataset': dataset, 'config': config,
                                                     'fold': fold, 'seed': seed,
                                                     'method': method_name, 'cost_ratio': r,
                                                     'C_rev': C_rev, 'alpha_truth': a_truth,
                                                     'alpha_lie': a_lie, 'q_truth': qt,
                                                     'q_lie': ql, **m})
                                    qhat_rows.append({'dataset': dataset, 'config': config,
                                                     'fold': fold, 'seed': seed,
                                                     'method': method_name, 'cost_ratio': r,
                                                     'q_truth': qt, 'q_lie': ql})
                                    for subj in np.unique(test_subjects):
                                        mask = test_subjects == subj
                                        sm = core.compute_metrics(
                                            test_labels[mask],
                                            {k: flags[k][mask] for k in flags},
                                            C_FP, C_FN, C_rev)
                                        subj_rows.append({'dataset': dataset, 'config': config,
                                                          'fold': fold, 'seed': seed,
                                                          'method': method_name,
                                                          'cost_ratio': r, 'C_rev': C_rev,
                                                          'subject_id': subj,
                                                          'n_segments': int(np.sum(mask)), **sm})
                            else:
                                flags, qt, ql = core.METHOD_FUNCS[method_name](
                                    calib_probs, calib_labels, test_probs, ALPHA_TOTAL, r, 0.5)
                                qhat_rows.append({'dataset': dataset, 'config': config,
                                                  'fold': fold, 'seed': seed,
                                                  'method': method_name, 'cost_ratio': r,
                                                  'q_truth': qt, 'q_lie': ql})
                                for C_rev in C_REV_VALUES:
                                    m = core.compute_metrics(test_labels, flags, C_FP, C_FN, C_rev)
                                    run_rows.append({'dataset': dataset, 'config': config,
                                                     'fold': fold, 'seed': seed,
                                                     'method': method_name, 'cost_ratio': r,
                                                     'C_rev': C_rev, 'alpha_truth': a_truth,
                                                     'alpha_lie': a_lie, 'q_truth': qt,
                                                     'q_lie': ql, **m})
                                    for subj in np.unique(test_subjects):
                                        mask = test_subjects == subj
                                        sm = core.compute_metrics(
                                            test_labels[mask],
                                            {k: flags[k][mask] for k in flags},
                                            C_FP, C_FN, C_rev)
                                        subj_rows.append({'dataset': dataset, 'config': config,
                                                          'fold': fold, 'seed': seed,
                                                          'method': method_name,
                                                          'cost_ratio': r, 'C_rev': C_rev,
                                                          'subject_id': subj,
                                                          'n_segments': int(np.sum(mask)), **sm})
    run_df = pd.DataFrame(run_rows)
    subj_df = pd.DataFrame(subj_rows)
    qhat_df = pd.DataFrame(qhat_rows)
    print(f"\nDone in {time.time()-t0:.1f}s. run={len(run_df)}, subject={len(subj_df)}, qhat={len(qhat_df)}")
    return run_df, subj_df, qhat_df
core.run_all = run_all_patched
print(f"[patch] SUBJECT_MAP merged rows={len(merged)}; get_subject_ids/run_all 注入 seed")

# --- 3. 跑 run_all ---
print("\n[run] 跑 run_all() ...")
run_df, subj_df, qhat_df = core.run_all()
print(f"  run={len(run_df)}, subj={len(subj_df)}, qhat={len(qhat_df)}")
print(f"  unique (dataset, config, method, r, C_rev) = "
      f"{len(run_df[['dataset','config','method','cost_ratio','C_rev']].drop_duplicates())}")

# === 固定锚点主分析 (r=3, C_rev=0.5, mondrian + cs_mondrian) ===
print("\n[主分析] 固定锚点 r=3, C_rev=0.5")
fixed = subj_df[(subj_df.cost_ratio == 3) & (subj_df.C_rev == 0.5)].copy()
print(f"  固定锚点行数: {len(fixed)} (期望 5 seed × 5 fold × 240 units = 6000)")

# 每 (seed, unit) 取 CS 和 Mondrian 的 cost
fixed_wide = fixed.pivot_table(
    index=['seed', 'subject_id', 'n_segments'],
    columns='method', values='total_cost', aggfunc='first').reset_index()
print(f"  wide 行数: {len(fixed_wide)}")
print(f"  mondrian 列空值: {fixed_wide.mondrian.isna().sum()}, "
      f"cs_mondrian 列空值: {fixed_wide.cs_mondrian.isna().sum()}")

# 验证每个 (seed, unit) 唯一
dup = fixed_wide.duplicated(subset=['seed', 'subject_id']).sum()
print(f"  (seed, unit) 重复: {dup} (应为 0)")
assert dup == 0

# 配对单位级 d_i = avg(cs_cost, mo_cost) 之差
# 但每 unit 含 2 样本 (truth+lie), cost 已聚合到此单位
# plan §5: 计算每个配对单位两篇文本的平均成本 → 这里"成本"已是 per-unit cost
#          d_i = avg_cost_CS - avg_cost_Mondrian
# 因为我们已在 per-subject 层面拿到 cost (sum over 2 samples),
# 平均到样本 = cost / n_segments
fixed_wide['avg_cost_cs'] = fixed_wide.cs_mondrian / fixed_wide.n_segments
fixed_wide['avg_cost_mo'] = fixed_wide.mondrian   / fixed_wide.n_segments
fixed_wide['d_i_seed']    = fixed_wide['avg_cost_cs'] - fixed_wide['avg_cost_mo']

# 跨 seed 平均每个单位
unit_avg = fixed_wide.groupby('subject_id').agg(
    d_i=('d_i_seed', 'mean'),
    n_seeds=('d_i_seed', 'count'),
).reset_index()
print(f"  配对单位数: {len(unit_avg)} (期望 1200)")
assert len(unit_avg) == 1200, f"unit count {len(unit_avg)} != 1200"
assert (unit_avg.n_seeds == 5).all(), "每单位未在 5 seed 各测一次"

# 推断
d = unit_avg.d_i.values
n = len(d)
obs = float(np.mean(d))
rng = np.random.RandomState(2026)
N_BOOT = 10000
N_PERM = 10000
obs, lo, hi, p_two = paired_inference(d, rng, N_BOOT, N_PERM)

print(f"\n  固定锚点 d = avg_cost_CS - avg_cost_Mondrian (per paired unit, 5 seed 平均):")
print(f"    配对单位数 n = {n}")
print(f"    效应量 mean(d) = {obs:+.4f}")
print(f"    95% CI = [{lo:+.4f}, {hi:+.4f}]")
print(f"    sign-flip 双侧 p = {p_two:.4f}")

# 写到 primary_test.csv
primary_rows = [{
    'config': 'TFIDF_LR', 'anchor_r': 3, 'C_rev': 0.5, 'alpha_total': 0.10,
    'n_units': n, 'effect': obs, 'ci_lo': lo, 'ci_hi': hi, 'p_signflip': p_two,
    'n_boot': N_BOOT, 'n_perm': N_PERM, 'cost_per_unit': 'avg_2_samples',
}]
primary_df = pd.DataFrame(primary_rows)
open_primary_path = os.path.join(_PKG_ROOT, '05_结果', '05_外部验证结果', 'OpenDomain_primary_test.csv')
open_primary = pd.read_csv(open_primary_path)
open_fixed = open_primary[
    open_primary.dataset.astype(str).str.startswith('OpenDeception') &
    (open_primary.n_entities.astype(int) == 511)].iloc[0]
joint_raw_p = np.array([float(open_fixed.sign_flip_p), p_two])
joint_holm_p = holm_adjust(joint_raw_p)
primary_df['p_signflip_holm_joint'] = joint_holm_p[1]
primary_df['holm_family'] = 'OpenDeception_fixed + CrossCultural_fixed'
primary_df.to_csv(os.path.join(OUT, "primary_test.csv"), index=False, encoding="utf-8-sig")
joint_primary = pd.DataFrame([
    {'dataset': 'OpenDeception_fixed', 'n_units': 511,
     'effect': float(open_fixed.delta_cost_mean), 'ci_lo': float(open_fixed.ci_low),
     'ci_hi': float(open_fixed.ci_high), 'p_signflip': joint_raw_p[0],
     'p_signflip_holm': joint_holm_p[0]},
    {'dataset': 'CrossCultural_fixed', 'n_units': n,
     'effect': obs, 'ci_lo': lo, 'ci_hi': hi, 'p_signflip': joint_raw_p[1],
     'p_signflip_holm': joint_holm_p[1]},
])
joint_primary['holm_family'] = 'OpenDeception_fixed + CrossCultural_fixed'
joint_primary.to_csv(os.path.join(OUT, 'joint_primary_holm_final.csv'),
                     index=False, encoding='utf-8-sig')
print(f"\n[write] primary_test.csv")

# === r=1 锚点 ===
print("\n[r=1 锚点]")
r1 = subj_df[(subj_df.cost_ratio == 1) & (subj_df.C_rev == 0.5)]
r1_wide = r1.pivot_table(index=['seed', 'subject_id'],
                          columns='method', values='total_cost', aggfunc='first').reset_index()
r1_wide['d'] = r1_wide.cs_mondrian - r1_wide.mondrian
print(f"  r=1 行数: {len(r1_wide)}")
print(f"  max|cs - mo| = {float(np.max(np.abs(r1_wide['d'])))}")
print(f"  严格==0 行: {int((r1_wide['d'] == 0).sum())} / {len(r1_wide)}")
anchor_result = {
    "n_rows": int(len(r1_wide)),
    "max_abs_delta_cost": float(np.max(np.abs(r1_wide['d']))),
    "n_strict_zero": int((r1_wide['d'] == 0).sum()),
    "n_total": int(len(r1_wide)),
    "all_zero": bool((r1_wide['d'] == 0).all()),
}
with open(os.path.join(OUT, "r1_anchor_check.json"), "w", encoding="utf-8") as f:
    json.dump(anchor_result, f, ensure_ascii=False, indent=2)
print(f"  [write] r1_anchor_check.json  all_zero={anchor_result['all_zero']}")

# === r 敏感性 ===
print("\n[r 敏感性]")
r_sens = []
for r in [1, 2, 3, 5, 10]:
    sub = subj_df[(subj_df.cost_ratio == r) & (subj_df.C_rev == 0.5)]
    if len(sub) == 0:
        continue
    wide = sub.pivot_table(index=['seed', 'subject_id', 'n_segments'],
                            columns='method', values='total_cost', aggfunc='first').reset_index()
    wide['avg_cost_cs'] = wide.cs_mondrian / wide.n_segments
    wide['avg_cost_mo'] = wide.mondrian   / wide.n_segments
    wide['d'] = wide['avg_cost_cs'] - wide['avg_cost_mo']
    unit = wide.groupby('subject_id').d.mean().values
    obs = float(unit.mean())
    boots = unit[rng.randint(0, len(unit), size=(N_BOOT, len(unit)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(unit)))
    perm = (unit[None, :] * signs).mean(axis=1)
    p_below = (np.sum(perm <= -abs(obs)) + 1) / (N_PERM + 1)
    p_above = (np.sum(perm >=  abs(obs)) + 1) / (N_PERM + 1)
    p_two = float(min(1.0, 2 * min(p_below, p_above)))
    r_sens.append({'r': r, 'effect': obs, 'ci_lo': lo, 'ci_hi': hi, 'n_units': len(unit), 'p': p_two})
    print(f"  r={r:>2d}: effect={obs:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  n={len(unit)}")
r_sens_df = pd.DataFrame(r_sens)
r_sens_df.to_csv(os.path.join(OUT, "r_sensitivity.csv"), index=False, encoding="utf-8-sig")
print(f"  [write] r_sensitivity.csv")

# === 地区异质性 ===
print("\n[地区异质性]")
df_unit = pd.read_csv(os.path.join(OUT, "crosscultural_fixed.csv"))[['unit_id', 'locale', 'topic']]
unit_avg_with_loc = unit_avg.merge(df_unit, left_on='subject_id', right_on='unit_id', how='left')
loc_het = []
for loc in ['EnglishUS', 'EnglishIndia', 'SpanishMexico', 'Romanian']:
    sub = unit_avg_with_loc[unit_avg_with_loc.locale == loc]
    d_loc = sub.d_i.values
    obs = float(d_loc.mean())
    boots = d_loc[rng.randint(0, len(d_loc), size=(N_BOOT, len(d_loc)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    loc_het.append({'locale': loc, 'n_units': len(d_loc), 'effect': obs,
                    'ci_lo': lo, 'ci_hi': hi})
    print(f"  {loc:14s}: n={len(d_loc):4d}  effect={obs:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]")
loc_df = pd.DataFrame(loc_het)
loc_df.to_csv(os.path.join(OUT, "locale_heterogeneity.csv"), index=False, encoding="utf-8-sig")
print(f"  [write] locale_heterogeneity.csv")

# === 配对敏感性 (1) 排除 Romanian ===
print("\n[配对敏感性] 排除 Romanian")
no_rom = unit_avg_with_loc[unit_avg_with_loc.locale != 'Romanian']
d_no_rom = no_rom.d_i.values
obs_nr = float(d_no_rom.mean())
boots_nr = d_no_rom[rng.randint(0, len(d_no_rom), size=(N_BOOT, len(d_no_rom)))].mean(axis=1)
lo_nr, hi_nr = np.percentile(boots_nr, [2.5, 97.5])
print(f"  无 Romanian: n={len(d_no_rom)}  effect={obs_nr:+.4f}  CI=[{lo_nr:+.4f}, {hi_nr:+.4f}]")

# === 配对敏感性 (2) leave-one-locale ===
print("\n[配对敏感性] leave-one-locale")
lol_rows = []
for loc_out in ['EnglishUS', 'EnglishIndia', 'SpanishMexico', 'Romanian']:
    sub = unit_avg_with_loc[unit_avg_with_loc.locale != loc_out]
    d_lol = sub.d_i.values
    obs_l = float(d_lol.mean())
    boots_l = d_lol[rng.randint(0, len(d_lol), size=(N_BOOT, len(d_lol)))].mean(axis=1)
    lo_l, hi_l = np.percentile(boots_l, [2.5, 97.5])
    lol_rows.append({'leave_out_locale': loc_out, 'n_units': len(d_lol),
                     'effect': obs_l, 'ci_lo': lo_l, 'ci_hi': hi_l})
    print(f"  leave out {loc_out:14s}: n={len(d_lol):4d}  effect={obs_l:+.4f}  CI=[{lo_l:+.4f}, {hi_l:+.4f}]")

pair_sens = [{
    'sensitivity': 'exclude_Romanian', 'n_units': len(d_no_rom),
    'effect': obs_nr, 'ci_lo': lo_nr, 'ci_hi': hi_nr,
}] + lol_rows
pair_df = pd.DataFrame(pair_sens)
pair_df.to_csv(os.path.join(OUT, "pairing_sensitivity.csv"), index=False, encoding="utf-8-sig")
print(f"  [write] pairing_sensitivity.csv")

# === 文本长度诊断 ===
print("\n[文本长度诊断]")
df_full = pd.read_csv(os.path.join(OUT, "crosscultural_fixed.csv"))
df_full['len_truth'] = df_full.raw_truth.str.len()
df_full['len_lie']   = df_full.raw_lie.str.len()
df_full['len_diff']  = df_full.len_lie - df_full.len_truth
print(f"  truth 长度: mean={df_full.len_truth.mean():.1f}  median={df_full.len_truth.median():.0f}")
print(f"  lie   长度: mean={df_full.len_lie.mean():.1f}    median={df_full.len_lie.median():.0f}")
print(f"  diff (lie - truth): mean={df_full.len_diff.mean():.1f}  median={df_full.len_diff.median():.0f}")
length_diag = {
    'n_units': int(len(df_full)),
    'len_truth_mean': float(df_full.len_truth.mean()),
    'len_truth_median': float(df_full.len_truth.median()),
    'len_lie_mean':   float(df_full.len_lie.mean()),
    'len_lie_median': float(df_full.len_lie.median()),
    'len_diff_mean':  float(df_full.len_diff.mean()),
    'len_diff_median': float(df_full.len_diff.median()),
    'lie_longer_ratio': float((df_full.len_lie > df_full.len_truth).mean()),
}

# Frozen-split length-only diagnostic: one scalar, log(1 + character count).
text_by_unit = dict(zip(df_full.unit_id, zip(df_full.raw_truth, df_full.raw_lie)))
with open(os.path.join(OUT, "splits.json"), encoding="utf-8") as f:
    split_spec = json.load(f)
length_lr_rows = []
for seed_str, fold_dict in split_spec.items():
    for fold_str, parts in fold_dict.items():
        def length_xy(units):
            x, y = [], []
            for unit_id in units:
                truth_text, lie_text = text_by_unit[unit_id]
                x.extend([[np.log1p(len(truth_text))], [np.log1p(len(lie_text))]])
                y.extend([0, 1])
            return np.asarray(x, dtype=float), np.asarray(y, dtype=int)
        x_fit, y_fit = length_xy(parts['fit'])
        x_test, y_test = length_xy(parts['test'])
        length_model = LogisticRegression(C=1.0, solver='liblinear', random_state=int(seed_str))
        length_model.fit(x_fit, y_fit)
        p_test = length_model.predict_proba(x_test)[:, 1]
        length_lr_rows.append({
            'seed': int(seed_str), 'fold': int(fold_str),
            'n_fit_texts': len(y_fit), 'n_test_texts': len(y_test),
            'test_auc': float(roc_auc_score(y_test, p_test)),
            'coefficient': float(length_model.coef_[0, 0]),
            'intercept': float(length_model.intercept_[0]),
        })
length_lr_df = pd.DataFrame(length_lr_rows)
length_lr_df.to_csv(os.path.join(OUT, "length_only_lr.csv"),
                    index=False, encoding="utf-8-sig")
length_diag.update({
    'length_only_lr_auc_mean': float(length_lr_df.test_auc.mean()),
    'length_only_lr_auc_min': float(length_lr_df.test_auc.min()),
    'length_only_lr_auc_max': float(length_lr_df.test_auc.max()),
    'backbone_has_explicit_length_scalar': False,
})
pd.DataFrame([length_diag]).to_csv(
    os.path.join(OUT, "length_diagnostic.csv"), index=False, encoding="utf-8-sig")
print(f"  length-only LR test AUC: mean={length_lr_df.test_auc.mean():.4f}, "
      f"range=[{length_lr_df.test_auc.min():.4f}, {length_lr_df.test_auc.max():.4f}]")
print(f"  [write] length_diagnostic.csv, length_only_lr.csv")

# === Plan B: 成本分解 (固定锚点 r=3, C_rev=0.5) ===
print("\n[Plan B 成本分解]")
# 从 run_df 取 CS-Mondrian 和 Mondrian 的整体指标
planB_fixed = run_df[(run_df.cost_ratio == 3) & (run_df.C_rev == 0.5)]
def method_summary(method, n_seeds=5):
    sub = planB_fixed[planB_fixed.method == method]
    # 跨 seed 取均值
    agg = sub.groupby(['dataset', 'config']).agg(
        n_fp=('n_fp', 'mean'), n_fn=('n_fn', 'mean'), n_reject=('n_reject', 'mean'),
        n_test=('n_test', 'mean'),
        cost=('total_cost', 'mean'),
        q_lie=('q_lie', 'mean'),
    ).reset_index()
    return agg

mo_sum = method_summary('mondrian')
cs_sum = method_summary('cs_mondrian')
# 全拒判成本 = C_rev * n_test
C_rev = 0.5
mo_C_all = C_rev * mo_sum.n_test.iloc[0]
cs_C_all = C_rev * cs_sum.n_test.iloc[0]
# E + C_rev*R = C_rev*N, hence C_rev,all-reject* = E/(N-R).
def C_all_reject_star(agg, r=3):
    fp = agg.n_fp.iloc[0]; fn = agg.n_fn.iloc[0]; rej = agg.n_reject.iloc[0]
    n_auto = agg.n_test.iloc[0] - rej
    if n_auto == 0:
        return np.nan
    return (fp + r*fn) / n_auto
mo_C_all_star = C_all_reject_star(mo_sum)
cs_C_all_star = C_all_reject_star(cs_sum)
# 差分 C_rev,CS-vs-M* = -(DeltaFP + r*DeltaFN) / DeltaReject
delta_fp = cs_sum.n_fp.iloc[0] - mo_sum.n_fp.iloc[0]
delta_fn = cs_sum.n_fn.iloc[0] - mo_sum.n_fn.iloc[0]
delta_rej = cs_sum.n_reject.iloc[0] - mo_sum.n_reject.iloc[0]
if delta_rej == 0:
    C_cs_vs_m_star = np.nan
else:
    C_cs_vs_m_star = -(delta_fp + 3*delta_fn) / delta_rej

# err_per_auto
def err_per_auto(agg):
    n_auto = agg.n_test.iloc[0] - agg.n_reject.iloc[0]
    err = agg.n_fp.iloc[0] + agg.n_fn.iloc[0]
    if n_auto == 0:
        return np.nan
    return err / n_auto

def weighted_error_cost_per_auto(agg, r=3):
    n_auto = agg.n_test.iloc[0] - agg.n_reject.iloc[0]
    if n_auto == 0:
        return np.nan
    return (agg.n_fp.iloc[0] + r * agg.n_fn.iloc[0]) / n_auto

mo_err = err_per_auto(mo_sum)
cs_err = err_per_auto(cs_sum)
mo_weighted_err = weighted_error_cost_per_auto(mo_sum)
cs_weighted_err = weighted_error_cost_per_auto(cs_sum)
mo_auto = mo_sum.n_test.iloc[0] - mo_sum.n_reject.iloc[0]
cs_auto = cs_sum.n_test.iloc[0] - cs_sum.n_reject.iloc[0]

print(f"  Mondrian:    FP={mo_sum.n_fp.iloc[0]:.1f}  FN={mo_sum.n_fn.iloc[0]:.1f}  "
      f"reject={mo_sum.n_reject.iloc[0]:.1f}  cost={mo_sum.cost.iloc[0]:.1f}  "
      f"err_per_auto={mo_err:.4f}  C_all_reject*={mo_C_all_star:.3f}")
print(f"  CS-Mondrian: FP={cs_sum.n_fp.iloc[0]:.1f}  FN={cs_sum.n_fn.iloc[0]:.1f}  "
      f"reject={cs_sum.n_reject.iloc[0]:.1f}  cost={cs_sum.cost.iloc[0]:.1f}  "
      f"err_per_auto={cs_err:.4f}  C_all_reject*={cs_C_all_star:.3f}")
print(f"  差分 FP={delta_fp:+.1f}  FN={delta_fn:+.1f}  reject={delta_rej:+.1f}  "
      f"C_rev,CS-vs-M* = {C_cs_vs_m_star:.3f}")

planB_rows = [
    {'method': 'mondrian', 'cost': float(mo_sum.cost.iloc[0]),
     'n_test': int(mo_sum.n_test.iloc[0]),
     'n_fp': float(mo_sum.n_fp.iloc[0]),
     'n_fn': float(mo_sum.n_fn.iloc[0]),
     'n_reject': float(mo_sum.n_reject.iloc[0]),
     'n_auto': float(mo_auto),
     'err_per_auto': float(mo_err),
     'weighted_error_cost_per_auto': float(mo_weighted_err),
     'C_rev_all_reject_star': float(mo_C_all_star),
     'C_rev_eval': C_rev, 'cost_all_reject': float(mo_C_all)},
    {'method': 'cs_mondrian', 'cost': float(cs_sum.cost.iloc[0]),
     'n_test': int(cs_sum.n_test.iloc[0]),
     'n_fp': float(cs_sum.n_fp.iloc[0]),
     'n_fn': float(cs_sum.n_fn.iloc[0]),
     'n_reject': float(cs_sum.n_reject.iloc[0]),
     'n_auto': float(cs_auto),
     'err_per_auto': float(cs_err),
     'weighted_error_cost_per_auto': float(cs_weighted_err),
     'C_rev_all_reject_star': float(cs_C_all_star),
     'C_rev_eval': C_rev, 'cost_all_reject': float(cs_C_all)},
    {'method': 'CS_vs_M', 'cost': float(cs_sum.cost.iloc[0] - mo_sum.cost.iloc[0]),
     'n_test': None,
     'n_fp': float(delta_fp),
     'n_fn': float(delta_fn),
     'n_reject': float(delta_rej),
     'n_auto': float(cs_auto - mo_auto),
     'err_per_auto': None,
     'weighted_error_cost_per_auto': None,
     'C_rev_all_reject_star': float(C_cs_vs_m_star),
     'C_rev_eval': None, 'cost_all_reject': None},
]
pd.DataFrame(planB_rows).to_csv(os.path.join(OUT, "planB_summary.csv"),
                                 index=False, encoding="utf-8-sig")
print(f"  [write] planB_summary.csv")

# === Plan A 公平预算比较 (B in {0.60, 0.70, 0.80}) ===
# 在 selection 上分别选 alpha, 应用到 test
# selection: val (120 units × 2 = 240 samples)
# test: 240 units × 2 = 480 samples
print("\n[Plan A 公平预算比较]")
ALPHA_GRID = np.arange(0.025, 0.301, 0.025)  # 12 个候选
B_LIST = [0.60, 0.70, 0.80]

def select_alpha_on_selection(method_name, seed, fold, B, calib_probs, calib_labels,
                              sel_probs, sel_labels):
    """Select alpha under the frozen budget rule and return the full trace."""
    candidates = []
    for alpha_total in ALPHA_GRID:
        if method_name == 'cs_mondrian':
            r = 3
            flags, qt, ql = core.method_cs_mondrian(calib_probs, calib_labels,
                                                      sel_probs, alpha_total, r)
        else:  # mondrian
            flags, qt, ql = core.method_mondrian(calib_probs, calib_labels,
                                                  sel_probs, alpha_total,
                                                  alpha_total)  # alpha_truth = alpha_lie = alpha_total
        rej_rate = float(np.mean(flags['reject']))
        metrics = core.compute_metrics(sel_labels, flags, C_FP=1.0, C_FN=3.0, C_rev=0.5)
        candidates.append({
            'B': B, 'method': method_name, 'seed': seed, 'fold': fold,
            'alpha': float(alpha_total), 'sel_reject_rate': rej_rate,
            'sel_cost': float(metrics['total_cost']), 'q_truth': qt, 'q_lie': ql,
            'feasible': bool(rej_rate <= B + 1e-9),
        })
    feasible = [row for row in candidates if row['feasible']]
    if not feasible:
        return None, candidates
    best = min(feasible, key=lambda row: (
        B - row['sel_reject_rate'], row['sel_cost'], row['alpha']))
    return best, candidates

planA_results = []
planA_selection_trace = []
planA_unit_method_rows = []
for B in B_LIST:
    for method in ['mondrian', 'cs_mondrian']:
        for seed in [7, 42, 123, 2024, 2026]:
            for fold in [0, 1, 2, 3, 4]:
                d = np.load(os.path.join(PRED_DIR, "preds", f"TFIDF_LR_f{fold}_s{seed}.npz"))
                calib_probs = d['calib_probs']; calib_labels = d['calib_labels']
                sel_probs = d['val_probs'];   sel_labels = d['val_labels']
                test_probs = d['probs'];      test_labels = d['labels']
                best, trace = select_alpha_on_selection(
                    method, seed, fold, B, calib_probs, calib_labels,
                    sel_probs, sel_labels)
                if best is None:
                    raise RuntimeError(f"No feasible Plan A alpha: B={B}, method={method}, seed={seed}, fold={fold}")
                for row in trace:
                    row['selected'] = bool(row['alpha'] == best['alpha'])
                    planA_selection_trace.append(row)
                alpha = best['alpha']; sel_rej = best['sel_reject_rate']
                # 在 test 上应用
                if method == 'cs_mondrian':
                    r = 3
                    flags, qt, ql = core.method_cs_mondrian(calib_probs, calib_labels,
                                                              test_probs, alpha, r)
                else:
                    flags, qt, ql = core.method_mondrian(calib_probs, calib_labels,
                                                          test_probs, alpha, alpha)
                test_rej = float(np.mean(flags['reject']))
                # test 阶段指标
                m = core.compute_metrics(test_labels, flags, C_FP=1.0, C_FN=3.0, C_rev=C_rev)
                planA_results.append({
                    'B': B, 'method': method, 'seed': seed, 'fold': fold,
                    'alpha_chosen': alpha, 'sel_reject_rate': sel_rej,
                    'test_reject_rate': test_rej,
                    'q_truth': qt, 'q_lie': ql,
                    'cost': m['total_cost'],
                    'n_test': len(test_labels), 'n_fp': m['n_fp'],
                    'n_fn': m['n_fn'], 'n_reject': m['n_reject'],
                })
                _current_seed["v"] = seed
                test_subjects = core.get_subject_ids('CrossCultural', fold, 'test', len(test_labels))
                for subj in np.unique(test_subjects):
                    mask = test_subjects == subj
                    sm = core.compute_metrics(
                        test_labels[mask], {k: flags[k][mask] for k in flags},
                        C_FP=1.0, C_FN=3.0, C_rev=0.5)
                    planA_unit_method_rows.append({
                        'B': B, 'method': method, 'seed': seed, 'fold': fold,
                        'unit_id': subj, 'n_segments': int(mask.sum()),
                        'avg_cost': float(sm['total_cost'] / mask.sum()),
                    })
planA_df = pd.DataFrame(planA_results)
planA_df.to_csv(os.path.join(OUT, "planA_run_level.csv"), index=False, encoding="utf-8-sig")
print(f"  [write] planA_run_level.csv  rows={len(planA_df)}")

# 跨 seed 取平均得 summary
planA_summary = planA_df.groupby(['B', 'method']).agg(
    alpha_mean=('alpha_chosen', 'mean'),
    sel_rej_mean=('sel_reject_rate', 'mean'),
    test_rej_mean=('test_reject_rate', 'mean'),
    cost_mean=('cost', 'mean'),
    n_test_mean=('n_test', 'mean'),
    n_fp_mean=('n_fp', 'mean'),
    n_fn_mean=('n_fn', 'mean'),
    n_reject_mean=('n_reject', 'mean'),
).reset_index()
planA_summary['cost_per_sample_mean'] = planA_summary.cost_mean / planA_summary.n_test_mean
planA_summary.to_csv(os.path.join(OUT, "planA_summary.csv"), index=False, encoding="utf-8-sig")
print(f"  [write] planA_summary.csv")

# 配对单位级 Plan A：每个单位先取两篇文本平均成本，再跨 seed 平均。
planA_unit_method_df = pd.DataFrame(planA_unit_method_rows)
planA_unit_method_df.to_csv(
    os.path.join(OUT, "planA_unit_method_costs.csv"), index=False, encoding="utf-8-sig")
wide = planA_unit_method_df.pivot_table(
    index=['B', 'seed', 'unit_id', 'n_segments'], columns='method',
    values='avg_cost', aggfunc='first').reset_index()
wide['d_i_seed'] = wide.cs_mondrian - wide.mondrian
assert len(wide) == len(B_LIST) * 5 * 1200
assert (wide.n_segments == 2).all()
wide.to_csv(os.path.join(OUT, "planA_unit_seed_effects.csv"),
            index=False, encoding="utf-8-sig")

unit_effects = wide.groupby(['B', 'unit_id']).agg(
    d_i=('d_i_seed', 'mean'), n_seeds=('d_i_seed', 'count')).reset_index()
assert len(unit_effects) == len(B_LIST) * 1200
assert (unit_effects.n_seeds == 5).all()
unit_effects.to_csv(os.path.join(OUT, "planA_unit_effects.csv"),
                    index=False, encoding="utf-8-sig")

rng_planA = np.random.RandomState(2027)
planA_test_rows = []
for B in B_LIST:
    values = unit_effects.loc[unit_effects.B == B, 'd_i'].values
    effect, ci_lo, ci_hi, p = paired_inference(values, rng_planA, N_BOOT, N_PERM)
    s = planA_summary[planA_summary.B == B].set_index('method')
    planA_test_rows.append({
        'B': B, 'effect': effect, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'n_units': len(values), 'p_signflip': p, 'n_boot': N_BOOT,
        'n_perm': N_PERM,
        'cs_alpha_mean': s.loc['cs_mondrian', 'alpha_mean'],
        'mondrian_alpha_mean': s.loc['mondrian', 'alpha_mean'],
        'cs_selection_reject_mean': s.loc['cs_mondrian', 'sel_rej_mean'],
        'mondrian_selection_reject_mean': s.loc['mondrian', 'sel_rej_mean'],
        'cs_test_reject_mean': s.loc['cs_mondrian', 'test_rej_mean'],
        'mondrian_test_reject_mean': s.loc['mondrian', 'test_rej_mean'],
        'delta_test_reject': s.loc['cs_mondrian', 'test_rej_mean'] - s.loc['mondrian', 'test_rej_mean'],
        'cs_budget_violation_rate': float((planA_df[(planA_df.B == B) & (planA_df.method == 'cs_mondrian')].test_reject_rate > B).mean()),
        'mondrian_budget_violation_rate': float((planA_df[(planA_df.B == B) & (planA_df.method == 'mondrian')].test_reject_rate > B).mean()),
    })
planA_tests = pd.DataFrame(planA_test_rows)
planA_tests['p_signflip_holm'] = holm_adjust(planA_tests.p_signflip.values)
planA_tests.to_csv(os.path.join(OUT, "planA_tests.csv"), index=False, encoding="utf-8-sig")

print(f"\n[Plan A 汇总]")
for B in B_LIST:
    sub = planA_summary[planA_summary.B == B]
    for _, row in sub.iterrows():
        print(f"  B={B}  {row['method']:12s}  alpha={row['alpha_mean']:.3f}  "
              f"sel_rej={row['sel_rej_mean']:.3f}  test_rej={row['test_rej_mean']:.3f}  "
              f"cost={row['cost_mean']:.1f}  n_rej={row['n_reject_mean']:.1f}  "
              f"n_FP={row['n_fp_mean']:.1f}  n_FN={row['n_fn_mean']:.1f}")

# 完整写出每个候选 alpha 的 selection 轨迹。
pd.DataFrame(planA_selection_trace).to_csv(
    os.path.join(OUT, "planA_selection_trace.csv"), index=False, encoding="utf-8-sig")
print(f"  [write] planA_selection_trace.csv")
print(f"  [write] Plan A unit files and tests: n={len(unit_effects)}, tests={len(planA_tests)}")

# === run_level.csv (固定锚点) ===
run_level = subj_df[(subj_df.cost_ratio == 3) & (subj_df.C_rev == 0.5)][
    ['dataset', 'config', 'fold', 'seed', 'method', 'cost_ratio', 'C_rev',
     'subject_id', 'n_segments', 'n_reject', 'n_accept', 'n_fp', 'n_fn',
     'total_cost']].copy().rename(columns={'total_cost': 'cost'})
run_level.to_csv(os.path.join(OUT, "run_level.csv"), index=False, encoding="utf-8-sig")
print(f"\n[write] run_level.csv  rows={len(run_level)}")

# === anchor_unit_effects.csv ===
anchor_rows = []
for r in [1, 2, 3, 5, 10]:
    for C_rev_v in [0.5]:
        sub = subj_df[(subj_df.cost_ratio == r) & (subj_df.C_rev == C_rev_v)]
        if len(sub) == 0:
            continue
        wide = sub.pivot_table(index=['seed', 'subject_id', 'n_segments'],
                                columns='method', values='total_cost',
                                aggfunc='first').reset_index()
        wide['avg_cs'] = wide.cs_mondrian / wide.n_segments
        wide['avg_mo'] = wide.mondrian / wide.n_segments
        wide['d'] = wide['avg_cs'] - wide['avg_mo']
        u = wide.groupby('subject_id').d.mean().reset_index()
        u.columns = ['unit_id', 'd_i']
        u['r'] = r
        u['C_rev'] = C_rev_v
        anchor_rows.append(u)
anchor_df = pd.concat(anchor_rows, ignore_index=True)
anchor_df.to_csv(os.path.join(OUT, "anchor_unit_effects.csv"), index=False, encoding="utf-8-sig")
print(f"  [write] anchor_unit_effects.csv  rows={len(anchor_df)}")

# === protocol_fixed.json ===
protocol = {
    "task": "Cross-Cultural Deception Detection v1.0 (Perez-Rosas & Mihalcea 2014)",
    "n_units": 1200, "n_raw_files": 24,
    "n_pairs_by_locale": {"EnglishUS":300, "EnglishIndia":298, "SpanishMexico":171, "Romanian":431},
    "n_normalized_duplicate_groups": 4,
    "n_seeds": 5, "seeds": [7, 42, 123, 2024, 2026],
    "n_folds": 5,
    "splits": {
        "outer_test": "5-fold stratified by locale×topic, np.array_split per layer",
        "inner": "fit:selection:calibration = 37.5:12.5:50 of training pool (1200-test)",
    },
    "backbone": "word TF-IDF 1-2gram + char_wb TF-IDF 3-5gram + L2 LogisticRegression",
    "conformal": "official strict_recompute.py (打补丁加载, 共形数学逐字未改)",
    "alpha_total": 0.10, "r_main": 3, "C_FP": 1.0, "C_rev_main": 0.5,
    "r_sensitivity": [1, 2, 3, 5, 10],
    "B_values_planA": [0.60, 0.70, 0.80],
    "alpha_grid_planA": list(np.round(ALPHA_GRID, 4)),
    "inference": {
        "unit": "paired author-contribution unit (truth+lie 配对), n=1200",
        "unit_warning": "Romanian 配对依赖非空行号; 公开数据无法排除同一自然人跨主题重复参与",
        "bootstrap": "paired-unit-level cluster bootstrap, 10000",
        "permutation": "paired sign-flip, 10000",
    },
    "completion_checks": {
        "n_pairs_total": 1200,
        "n_test_folds_exclusive_coverage": True,
        "every_unit_tested_once_per_seed": True,
        "r1_anchor_strict_zero": anchor_result['all_zero'],
        "planA_unit_inference_n": 1200,
        "planA_holm_complete": True,
        "joint_primary_holm_complete": True,
        "planB_all_reject_formula": "(FP + r*FN) / N_auto",
        "length_only_lr_complete": True,
    },
}
with open(os.path.join(OUT, "protocol_fixed.json"), "w", encoding="utf-8") as f:
    json.dump(protocol, f, ensure_ascii=False, indent=2)
print(f"\n[write] protocol_fixed.json")

# === machine-checkable final consistency audit ===
planB_check = pd.DataFrame(planB_rows)
method_rows = planB_check[planB_check.method.isin(['mondrian', 'cs_mondrian'])].copy()
decomp_residuals = []
breakeven_residuals = []
for _, row in method_rows.iterrows():
    decomp_residuals.append(abs(row.cost - (row.n_fp + 3 * row.n_fn + 0.5 * row.n_reject)))
    c_star = row.C_rev_all_reject_star
    breakeven_residuals.append(abs(
        (row.n_fp + 3 * row.n_fn + c_star * row.n_reject) - c_star * row.n_test))
trace_df = pd.DataFrame(planA_selection_trace)
selected_trace = trace_df[trace_df.selected]
audit = {
    'status': 'PASS',
    'primary_n_units': int(n),
    'primary_effect': float(primary_df.effect.iloc[0]),
    'joint_primary_holm_values': [float(v) for v in joint_primary.p_signflip_holm.tolist()],
    'planA_n_units_each': [int(v) for v in planA_tests.n_units.tolist()],
    'planA_all_ci_upper_below_zero': bool((planA_tests.ci_hi < 0).all()),
    'planA_all_holm_below_0_05': bool((planA_tests.p_signflip_holm < 0.05).all()),
    'planA_trace_rows': int(len(trace_df)),
    'planA_selected_rows': int(len(selected_trace)),
    'planA_selected_all_feasible': bool(selected_trace.feasible.all()),
    'planB_max_decomposition_residual': float(max(decomp_residuals)),
    'planB_max_breakeven_residual': float(max(breakeven_residuals)),
    'planB_cs_vs_m_star': float(C_cs_vs_m_star),
    'length_only_lr_runs': int(len(length_lr_df)),
}
assert audit['primary_n_units'] == 1200
assert all(v < 0.05 for v in audit['joint_primary_holm_values'])
assert audit['planA_n_units_each'] == [1200, 1200, 1200]
assert audit['planA_all_ci_upper_below_zero']
assert audit['planA_all_holm_below_0_05']
assert audit['planA_trace_rows'] == 1800 and audit['planA_selected_rows'] == 150
assert audit['planA_selected_all_feasible']
assert audit['planB_max_decomposition_residual'] < 1e-12
assert audit['planB_max_breakeven_residual'] < 1e-12
assert audit['length_only_lr_runs'] == 25
with open(os.path.join(OUT, 'final_consistency_audit.json'), 'w', encoding='utf-8') as f:
    json.dump(audit, f, ensure_ascii=False, indent=2)
print("[write] final_consistency_audit.json  status=PASS")

print("\n[done]")
