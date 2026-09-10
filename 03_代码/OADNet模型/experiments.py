# -*- coding: utf-8 -*-
"""实验编排：双数据集 × 基线矩阵 × 多种子 × 受试者级折 → 结果表 + 显著性。

用法（服务器上，rerun/ 目录内）：
    python experiments.py --datasets SEUMLD,MDPE            # 全量
    python experiments.py --datasets SEUMLD --losses CE,Focal --folds 1 --seeds 42
    python experiments.py --datasets MDPE --main-only        # 只跑主模型+先验消融

输出到 results/<dataset>/：
    per_run_<config>.csv     逐 (fold, seed) 运行结果
    summary.csv              每实验 mean±std（整体 + 按折）
    significance.csv         主模型 vs 基线（McNemar / Wilcoxon）
    report.md                汇总报告
"""
import argparse
import glob
import os
import sys
import time
import numpy as np
import pandas as pd
import re

import config
from data import load_index, build_subject_folds, split_train_val, split_train_val_calib, integrity_check, preload_features
from train import train_one_run
from metrics import (mcnemar_per_fold, wilcoxon_paired, format_mean_std)

# ---------------------------------------------------------------------
# 实验矩阵
# ---------------------------------------------------------------------
def experiment_matrix(dataset):
    """返回配置列表，每项 (config, loss, zero_prior, ablate, sub_dataset)。

    - sub_dataset：数据加载用的 dataset 名（单模态时是 SEUMLD_text / MDPE_vision 等）
    - ablate：组件消融关掉的组件（orthogonal/cross_attn/adaptive_mask/pseudo_cnn）；None 为完整模型
    - config 名以 "MLP" 开头 → 走 SimpleMLP 基线
    """
    base_losses = [
        ("OADNet_CE",     "CE",    False, None, dataset),
        ("OADNet_Focal",  "Focal", False, None, dataset),
        ("OADNet_CB",     "CB",    False, None, dataset),
        ("OADNet_LDAM",   "LDAM",  False, None, dataset),
        ("OADNet_Kang",   "Kang",  False, None, dataset),
        ("OADNet_OGM_GE", "OGM_GE", False, None, dataset),
    ]
    if dataset == "MDPE":
        # 首位是主模型（真实先验）；再加一个"先验消融"（先验置零，架构不变）
        matrix = [("OADNet_CE_prior", "CE", False, None, dataset),
                  ("OADNet_CE_no_prior", "CE", True, None, dataset)] + base_losses[1:]
    else:
        matrix = base_losses

    # 组件消融（关创新点 1-4，主损失 CE，完整多模态）
    ablate_map = {
        "orthogonal": "OADNet_woOrthogonal",
        "cross_attn": "OADNet_woCrossAttn",
        "adaptive_mask": "OADNet_woAdaptiveMask",
        "pseudo_cnn": "OADNet_woPseudoCNN",
    }
    for ablate, name in ablate_map.items():
        matrix.append((name, "CE", False, ablate, dataset))

    # 单模态基线（text/vision/audio 各一，主损失 CE）
    for mod, name in [("text", "OADNet_text"), ("vision", "OADNet_vision"), ("audio", "OADNet_audio")]:
        matrix.append((name, "CE", False, None, f"{dataset}_{mod}"))

    # 文本主导门控融合（针对欺骗低信噪比的核心方案，use_text_gated 触发）
    matrix.append(("OADNet_TGated", "CE", False, None, dataset))

    # Text-Safe Behavioral Residual Learning：文本 logits 保底，音视频只做受限残差纠错。
    matrix.append(("TS_BRL", "CE", False, None, dataset))
    matrix.append(("TS_BRL_woGate", "CE", False, "no_gate", dataset))
    matrix.append(("TS_BRL_woResidualBound", "CE", False, "no_residual_bound", dataset))

    # 文本教师蒸馏：教师只使用当前训练折的文本数据，避免测试信息泄漏
    matrix.append(("OADNet_KD", "KD", False, None, dataset))
    matrix.append(("OADNet_ConfKD", "ConfKD", False, None, dataset))

    # MLP 简单融合基线（证明复杂架构的价值）
    matrix.append(("MLP_fusion", "CE", False, None, dataset))
    return matrix


# ---------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------
def run_dataset(dataset, args, df=None, folds=None):
    print("\n" + "=" * 78)
    print(f"▶▶ 数据集 {dataset}  受试者级 {args.folds} 折 × {len(args.seeds)} 种子")
    print("=" * 78)

    if df is None:
        integrity_check(dataset)
        df = load_index(dataset)
        # 全量特征预载内存（一次性，之后 25×run 的 DataLoader 零磁盘 IO）
        feat_root = (config.FEAT_SEUMLD if config.base_dataset(dataset) == "SEUMLD" else config.FEAT_MDPE)
        preload_features(dataset, df, feat_root)
    if folds is None:
        folds = build_subject_folds(df, n_folds=args.folds)

    res_dir = os.path.join(config.RESULTS_DIR, dataset)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(os.path.join(res_dir, "preds"), exist_ok=True)

    matrix = experiment_matrix(dataset)
    if args.losses:
        keep = set(args.losses)
        matrix = [m for m in matrix if m[1] in keep or m[0] in args.losses]
    if args.main_only:
        matrix = [m for m in matrix if m[0].endswith("_CE_prior") or m[0].startswith("OADNet_CE")]
    if args.configs:
        keep = set(args.configs)
        matrix = [m for m in matrix if m[0] in keep]

    # 运行进度表
    all_rows = []
    main_name = matrix[0][0] if matrix else None
    preds_store = {}   # (config_name, fold) -> (preds, labels)
    teacher_cache = {}  # (fold, seed) -> 文本教师参数

    for exp_name, loss, zero_prior, ablate, sub_dataset in matrix:
        csv_path = os.path.join(res_dir, f"per_run_{exp_name}.csv")
        if args.force and os.path.exists(csv_path):
            os.remove(csv_path)   # force 重跑：清空旧 csv，逐 run 重新累加
        done = set()
        if os.path.exists(csv_path) and not args.force:
            old = pd.read_csv(csv_path)
            done = {(int(r.fold), int(r.seed)) for _, r in old.iterrows()}
        rows = []
        for fold, (tr, te) in enumerate(folds):
            train_idx, val_idx, calib_idx = split_train_val_calib(df, tr, random_state=config.FOLD_RANDOM_STATE)
            for seed in args.seeds:
                if (fold, seed) in done:
                    continue
                t0 = time.time()
                teacher_state = None
                if loss in ("KD", "ConfKD"):
                    teacher_key = (fold, seed)
                    if teacher_key not in teacher_cache:
                        teacher_dataset = f"{config.base_dataset(dataset)}_text"
                        teacher_res = train_one_run(
                            teacher_dataset, df, train_idx, val_idx, te, seed,
                            loss_type="CE", zero_prior=True,
                            feat_root=(config.FEAT_SEUMLD if config.base_dataset(dataset) == "SEUMLD" else config.FEAT_MDPE),
                            prior_csv=config.CSV_PERS,
                            config_name="TextTeacher",
                            fold=fold, return_state=True)
                        teacher_cache[teacher_key] = teacher_res.model_state
                    teacher_state = teacher_cache[teacher_key]
                res = train_one_run(
                    sub_dataset, df, train_idx, val_idx, te, seed,
                    calib_idx=calib_idx,
                    loss_type=loss, zero_prior=zero_prior,
                    feat_root=(config.FEAT_SEUMLD if config.base_dataset(sub_dataset) == "SEUMLD" else config.FEAT_MDPE),
                    prior_csv=config.CSV_PERS,
                    config_name=exp_name,
                    ablate=None if exp_name.startswith("TS_BRL") else ablate,
                    use_mlp=exp_name.startswith("MLP"),
                    use_text_gated=("TGated" in exp_name),
                    use_text_safe=exp_name.startswith("TS_BRL"),
                    text_safe_ablate=ablate if exp_name.startswith("TS_BRL") else None,
                    fold=fold, teacher_state=teacher_state)
                res.fold = fold
                m = res.metrics
                rows.append({"config": exp_name, "loss": loss, "fold": fold, "seed": seed,
                             "Acc": m["Acc"], "Macro_F1": m["Macro_F1"], "Macro_F2": m["Macro_F2"],
                             "P0": m["P0"], "R0": m["R0"], "P1": m["P1"], "R1": m["R1"],
                             "AUC": m["AUC"], "threshold": res.threshold, "epochs": res.epochs})
                preds_store.setdefault((exp_name, fold), ([], []))
                preds_store[(exp_name, fold)][0].extend(res.test_preds.tolist())
                preds_store[(exp_name, fold)][1].extend(res.test_labels.tolist())
                # 预测落盘：preds + probs + labels + val_probs + val_labels
                # 这样后续可直接做 split / Mondrian conformal，不必重新跑训练。
                np.savez(os.path.join(res_dir, "preds", f"{exp_name}_f{fold}_s{seed}.npz"),
                         preds=res.test_preds, probs=res.test_probs, labels=res.test_labels,
                         val_preds=res.val_preds, val_probs=res.val_probs, val_labels=res.val_labels,
                         calib_probs=res.calib_probs, calib_labels=res.calib_labels)
                print(f"  [{exp_name}] fold={fold} seed={seed}  F2={m['Macro_F2']:.4f} "
                      f"({time.time()-t0:.0f}s)")
                # 每 run 完成即落盘（追加当前行，读盘防重复）：断电/杀进程只丢未完成的 run
                ink = pd.DataFrame(rows[-1:])
                if os.path.exists(csv_path):
                    try:
                        prev = pd.read_csv(csv_path)
                        ink = pd.concat([prev, ink], ignore_index=True)
                    except pd.errors.EmptyDataError:
                        pass
                ink.to_csv(csv_path, index=False, encoding="utf-8-sig")
        all_rows.extend(pd.read_csv(csv_path).to_dict("records"))

    # ---- 汇总 ----
    per_run = pd.DataFrame(all_rows)
    duplicate_mask = per_run.duplicated(subset=["config", "fold", "seed"], keep=False)
    if duplicate_mask.any():
        dup_keys = per_run.loc[duplicate_mask, ["config", "fold", "seed"]].drop_duplicates().to_dict("records")
        raise ValueError(f"发现重复逻辑 run，拒绝汇总：{dup_keys[:10]}")
    # 兼容历史调用；当前正式路径在上面已拒绝重复。
    per_run = per_run.drop_duplicates(
        subset=["config", "fold", "seed"], keep="last").reset_index(drop=True)
    summarize(dataset, per_run, matrix, preds_store, res_dir,
              expected_folds=args.folds, expected_seeds=args.seeds)
    return per_run


def load_preds(res_dir, exp_name, fold):
    """从磁盘加载某配置某折的所有种子测试预测/标签（合并）；无则返回 (None, None)。"""
    files = sorted(glob.glob(os.path.join(res_dir, "preds", f"{exp_name}_f{fold}_s*.npz")))
    if not files:
        return None, None
    ps, ls = [], []
    for fp in files:
        d = np.load(fp)
        ps.append(d["preds"])
        ls.append(d["labels"])
    return np.concatenate(ps), np.concatenate(ls)


def load_preds_by_seed(res_dir, exp_name, fold):
    """按 seed 加载某配置某折的测试预测/标签，返回 {seed: (preds, labels)}。

    不做跨种子拼接，供 McNemar 逐 (fold, seed) 检验，避免多种子拼接造成的伪重复。
    """
    out = {}
    for fp in sorted(glob.glob(os.path.join(res_dir, "preds", f"{exp_name}_f{fold}_s*.npz"))):
        m = re.match(rf"{re.escape(exp_name)}_f{fold}_s(\d+)\.npz", os.path.basename(fp))
        if m:
            d = np.load(fp)
            out[int(m.group(1))] = (np.asarray(d["preds"]), np.asarray(d["labels"]))
    return out


def summarize(dataset, per_run, matrix, preds_store, res_dir,
              expected_folds=None, expected_seeds=None):
    """生成 summary.csv / significance.csv / report.md。"""
    lines = [f"# {dataset} 二区重跑结果汇总", ""]
    actual_folds = int(per_run["fold"].nunique()) if not per_run.empty else config.N_FOLDS
    actual_seeds = sorted(per_run["seed"].drop_duplicates().astype(int).tolist()) if not per_run.empty else list(config.SEEDS)
    lines.append(f"协议：受试者级 {actual_folds} 折 × 种子 {actual_seeds}，"
                 f"阈值验证集最大化 Macro-F2，折叠划分固定(rs={config.FOLD_RANDOM_STATE})")
    lines.append("")

    # 整体 mean±std（跨 fold×seed）
    rows = []
    for exp_name, loss, _, _, _ in matrix:
        sub = per_run[per_run["config"] == exp_name]
        if sub.empty:
            continue
        row = {"config": exp_name, "loss": loss, "n_runs": len(sub)}
        for col in ["Acc", "Macro_F1", "Macro_F2", "P0", "R0", "P1", "R1", "AUC"]:
            row[col] = format_mean_std(sub[col].values)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(res_dir, "summary.csv"), index=False, encoding="utf-8-sig")
    lines.append("## 整体 mean ± std（跨所有 fold×seed 运行）")
    lines.append("")
    lines.append(summary.to_string(index=False))
    lines.append("")

    # 按折 mean±std（跨种子）→ 经典 k 折报告
    fold_rows = []
    for exp_name, loss, _, _, _ in matrix:
        sub = per_run[per_run["config"] == exp_name]
        if sub.empty:
            continue
        fold_means = sub.groupby("fold")["Macro_F2"].mean().values
        fold_rows.append({"config": exp_name, "loss": loss,
                          "fold_F2_mean±std": format_mean_std(fold_means)})
    if fold_rows:
        lines.append("## 按折 F2（每折跨种子取均值，再跨折 mean±std）")
        lines.append("")
        lines.append(pd.DataFrame(fold_rows).to_string(index=False))
        lines.append("")

    # ---- 显著性：主模型 vs 各基线 ----
    sig_rows = []
    expected_pairs = None
    if expected_folds is not None and expected_seeds is not None:
        expected_pairs = {(int(f), int(s)) for f in range(int(expected_folds)) for s in expected_seeds}
        for exp_name, *_ in matrix:
            got = set(zip(
                per_run.loc[per_run["config"] == exp_name, "fold"].astype(int),
                per_run.loc[per_run["config"] == exp_name, "seed"].astype(int),
            ))
            if got != expected_pairs:
                raise ValueError(
                    f"{exp_name} 运行网格不完整：got={len(got)}/{len(expected_pairs)}"
                )
    if matrix:
        main_name = matrix[0][0]
        for exp_name, loss, _, _, _ in matrix[1:]:
            sub = per_run[per_run["config"] == exp_name]
            msub = per_run[per_run["config"] == main_name]
            if sub.empty or msub.empty:
                raise ValueError(f"显著性检验缺少主模型或基线运行：{main_name} vs {exp_name}")
            main_pairs = set(zip(msub["fold"].astype(int), msub["seed"].astype(int)))
            base_pairs = set(zip(sub["fold"].astype(int), sub["seed"].astype(int)))
            if expected_pairs is not None and (main_pairs != expected_pairs or base_pairs != expected_pairs):
                raise ValueError(
                    f"显著性检验要求完整配对网格：main={len(main_pairs)}/{len(expected_pairs)}, "
                    f"baseline={len(base_pairs)}/{len(expected_pairs)} ({main_name} vs {exp_name})"
                )
            if main_pairs != base_pairs:
                raise ValueError(f"主模型与基线的 (fold, seed) 集合不一致：{main_name} vs {exp_name}")
            # McNemar（逐 (fold, seed) 单独检验，避免多种子拼接造成的伪重复）
            # 一律从磁盘按 seed 加载完整集合，避免断点续跑只检验本轮内存子集
            mcnemar_ps, mcnemar_delta = [], []
            folds_to_check = sorted({p[0] for p in (expected_pairs or main_pairs)})
            for fold in folds_to_check:
                mp_seeds = load_preds_by_seed(res_dir, main_name, fold)
                bp_seeds = load_preds_by_seed(res_dir, exp_name, fold)
                seeds_to_check = sorted({p[1] for p in (expected_pairs or main_pairs) if p[0] == fold})
                for seed in seeds_to_check:
                    if seed not in mp_seeds or seed not in bp_seeds:
                        raise ValueError(f"McNemar 缺少配对预测：fold={fold}, seed={seed}")
                    mp, ml = mp_seeds[seed]
                    bp, bl = bp_seeds[seed]
                    if len(mp) != len(bp) or len(mp) != len(ml) or len(mp) != len(bl):
                        raise ValueError(f"McNemar 长度不一致：fold={fold}, seed={seed}")
                    if not np.array_equal(ml, bl):
                        raise ValueError(f"McNemar 主/基线标签顺序不一致：fold={fold}, seed={seed}")
                    results = mcnemar_per_fold([mp], [bp], [ml])
                    _, p, b, c, _ = results[0]
                    mcnemar_ps.append(p)
                    mcnemar_delta.append((c - b) / max(len(ml), 1))   # 主模型净优势（>0 表示主更优）
            # Wilcoxon（逐 run 配对 Macro-F2，按 (fold,seed) 对齐）
            mf2_main = msub.set_index(["fold", "seed"])["Macro_F2"].astype(float)
            mf2_base = sub.set_index(["fold", "seed"])["Macro_F2"].astype(float)
            common = mf2_main.index.intersection(mf2_base.index)
            if len(common) != len(mf2_main) or len(common) != len(mf2_base):
                print(f"[warn] {main_name} vs {exp_name} 配对不完整：主 {len(mf2_main)}、基 {len(mf2_base)}、共同 {len(common)}")
            if len(common) >= 2:
                w_stat, w_p = wilcoxon_paired(mf2_main[common].values, mf2_base[common].values)
            else:
                w_stat, w_p = np.nan, 1.0
            mean_delta = mf2_main[common].mean() - mf2_base[common].mean()   # 与 Wilcoxon 同一配对集合
            sig_rows.append({
                "main": main_name, "baseline": exp_name,
                "ΔMacro_F2(主-基)": f"{mean_delta:+.4f}",
                "McNemar_run_pairs_sig": f"{sum(p < 0.05 for p in mcnemar_ps)}/{len(mcnemar_ps)}",
                "McNemar_p_med": np.median(mcnemar_ps) if mcnemar_ps else np.nan,
                "主模型净优势_中位": np.median(mcnemar_delta) if mcnemar_delta else np.nan,
                "Wilcoxon_n_pairs": int(len(common)),
                "Wilcoxon_stat": w_stat,
                "Wilcoxon_p(逐run)": w_p,
            })
    if sig_rows:
        sig = pd.DataFrame(sig_rows)
        sig.to_csv(os.path.join(res_dir, "significance.csv"), index=False, encoding="utf-8-sig")
        lines.append("## 显著性检验（主模型 = " + main_name + "）")
        lines.append("")
        lines.append(sig.to_string(index=False))
        lines.append("")

    with open(os.path.join(res_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\n[results] 已写入 {res_dir}/  (report.md / summary.csv / significance.csv)")

    # 可视化（训练曲线/混淆矩阵/t-SNE/消融对比/单模态对比）；matplotlib 缺失时静默跳过
    try:
        import visualize
        visualize.summarize_figures(dataset, per_run, res_dir)
    except Exception as e:
        print(f"[visualize] 跳过（{e}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="SEUMLD", help="逗号分隔: SEUMLD,MDPE")
    ap.add_argument("--losses", default="", help="只跑指定损失: CE,Focal,CB,LDAM,Kang,OGM_GE")
    ap.add_argument("--configs", default="", help="只跑指定配置名（逗号分隔）")
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--seeds", default=",".join(map(str, config.SEEDS)))
    ap.add_argument("--main-only", action="store_true", help="只跑主模型+先验消融")
    ap.add_argument("--force", action="store_true", help="强制重跑（覆盖已有结果）")
    args = ap.parse_args()

    args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.losses = [x.strip() for x in args.losses.split(",") if x.strip()]
    args.configs = [x.strip() for x in args.configs.split(",") if x.strip()]
    args.seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    for ds in args.datasets:
        run_dataset(ds, args)


if __name__ == "__main__":
    main()
