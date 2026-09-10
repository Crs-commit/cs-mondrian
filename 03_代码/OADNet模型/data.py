# -*- coding: utf-8 -*-
"""数据层：双数据集加载、受试者级分组、受试者级划分、特征读取、完整性检查。

设计要点（二区标准的诚实性要求）：
  1. 受试者级划分（StratifiedGroupKFold，按 subject 分组），杜绝被试泄漏；
  2. 特征文件缺失 → 显式报错（绝不静默填充随机噪声，旧代码 `_safe_load` 的 1e-4 噪声是隐患）；
  3. 样本级掩码：pad 到统一序列长度，零填充帧不参与注意力/池化。
"""
import glob
import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

import config

# ---------------------------------------------------------------------
# 受试者 ID 派生
# ---------------------------------------------------------------------
def subject_of(video_id: str, dataset: str) -> str:
    """按数据集约定从 video_id 提取受试者 ID。
    SEU-MLD: '005_01' -> '005'（下划线前）
    MDPE   : '2-1-01' -> '2'（第一个连字符前）
    """
    v = str(video_id).strip()
    if config.base_dataset(dataset) == "SEUMLD":
        return v.split("_")[0]
    elif config.base_dataset(dataset) == "MDPE":
        return v.split("-")[0]
    raise ValueError(f"unknown dataset {dataset}")


def load_index(dataset: str):
    """加载索引 CSV，附加 subject 列，返回 DataFrame。

    MDPE：剔除视觉(clipVIT)缺失的样本（受试者69整人24段 + 109/131/84各1段 = 27段），
          保证每样本三模态完整。净样本 4581。
    """
    base = config.base_dataset(dataset)
    csv_path = config.CSV_SEUMLD if base == "SEUMLD" else config.CSV_MDPE
    df = pd.read_csv(csv_path)
    df["subject"] = df["video_id"].map(lambda v: subject_of(v, dataset))
    if base == "MDPE":
        df = df[df["video_id"].map(lambda v: os.path.exists(
            feature_path("MDPE", str(v).strip(), "vision")))]
    return df


def build_subject_folds(df: pd.DataFrame, n_folds: int = config.N_FOLDS,
                        random_state: int = config.FOLD_RANDOM_STATE):
    """受试者级 K 折划分，返回 [(train_idx, test_idx), ...]，每组受试者不跨集。"""
    groups = df["subject"].values
    y = df["label"].values
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []
    for train_idx, test_idx in sgkf.split(df, y, groups):
        folds.append((train_idx, test_idx))
    # 折叠随机状态固定，保证跨种子折叠划分一致（配对检验的前提）
    return folds


def split_train_val(df: pd.DataFrame, train_idx, val_fraction: float = config.VAL_FRACTION,
                    random_state: int = config.FOLD_RANDOM_STATE):
    """在训练受试者内部再按受试者级划出验证集。返回 (train_idx, val_idx)。"""
    train_df = df.iloc[train_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    tr, va = next(gss.split(train_df, train_df["label"].values, train_df["subject"].values))
    return train_idx[tr], train_idx[va]


def split_train_val_calib(df: pd.DataFrame, train_idx,
                          val_fraction: float = config.VAL_FRACTION,
                          calib_fraction: float = config.CALIB_FRACTION,
                          random_state: int = config.FOLD_RANDOM_STATE):
    """在训练受试者内部划出【模型选择验证集】+【conformal 校准集】两路受试者不相交集合。

    返回 (train_idx, val_idx, calib_idx)。关键：校准集独立于模型选择（早停/阈值）。
    由于同一受试者可含多个相关片段，覆盖结果应按经验片段级解释。
    """
    train_df = df.iloc[train_idx]
    y = train_df["label"].values
    groups = train_df["subject"].values
    # 第一步：划校准集（占 train_df 的 calib_fraction）
    gss_calib = GroupShuffleSplit(n_splits=1, test_size=calib_fraction, random_state=random_state)
    tr_rest_pos, calib_pos = next(gss_calib.split(train_df, y, groups))
    # 第二步：从剩余 train 中划验证集（保持 val 占总体比例 val_fraction）
    rest_df = train_df.iloc[tr_rest_pos]
    val_frac_of_rest = val_fraction / (1.0 - calib_fraction)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_rest, random_state=random_state + 1)
    tr_pos, va_pos = next(gss_val.split(rest_df, rest_df["label"].values, rest_df["subject"].values))
    return (train_idx[tr_rest_pos[tr_pos]], train_idx[tr_rest_pos[va_pos]], train_idx[calib_pos])


# ---------------------------------------------------------------------
# 特征路径
# ---------------------------------------------------------------------
def mdpe_filename(vid: str, modality: str) -> str:
    """把索引 video_id（'2-1-01'）转成 MDPE 某模态的特征文件名。

    MDPE 命名混乱：clipVIT 用下划线（001_10_8.npy）、文本/音频用连字符（001-1-1.csv）。
    规则：ID 补零 3 位 + 题序 + 题号去前导0，分隔符按模态取（config.MDPE_NAME_SEP）。
    例：video_id='2-1-01', vision → '002_1_1.npy'；text_llm → '002-1-1.csv'
    """
    parts = str(vid).split("-")           # 索引 video_id 统一用连字符 'ID-X-Y'
    assert len(parts) == 3, f"MDPE video_id 格式异常: {vid}"
    ID, X, Y = parts[0].zfill(3), parts[1], str(int(parts[2]))   # ID补零3位, 题号去前导0
    sep = config.MDPE_NAME_SEP[modality]
    fname = f"{ID}{sep}{X}{sep}{Y}{config.MDPE_MODALITY_EXT[modality]}"
    return fname


def feature_path(dataset: str, vid: str, modality: str, feat_root: str = None) -> str:
    """按数据集约定的文件名/目录定位单个特征文件。

    SEU-MLD: 提取脚本输出到单一目录，命名 {vid}_au/text/mfcc.npy
    MDPE   : 按模态子目录（baichuan13B-base/ 等）+ mdpe_filename() 构造文件名
    """
    if config.base_dataset(dataset) == "SEUMLD":
        root = feat_root or config.FEAT_SEUMLD
        suffix = {"vision": "au", "text_classic": "text", "audio_classic": "mfcc"}.get(modality)
        assert suffix is not None, f"SEU-MLD 无模态 {modality}"
        return os.path.join(root, f"{vid}_{suffix}.npy")
    else:  # MDPE —— 特征根固定用 mdpe_raw（各自解压目录）
        root = config.MDPE_RAW
        subdir = config.MDPE_MODALITY_SUBDIRS[modality]
        ID3 = str(vid).split("-")[0].zfill(3)   # 受试者 ID 补零 3 位子目录
        fname = mdpe_filename(vid, modality)
        return os.path.join(root, subdir, ID3, fname)


# ---------------------------------------------------------------------
# 特征完整性检查（训练前必须全绿）
# ---------------------------------------------------------------------
def integrity_check(dataset: str, feat_root: str = None):
    """校验：每个样本所需的每个模态特征文件都存在，且维度正确。缺一个即抛错。"""
    cfg = config.get_dataset_cfg(dataset)
    mods = [m for m in cfg if m.endswith(("vision", "text_llm", "text_classic", "audio_llm", "audio_classic"))]
    df = load_index(dataset)
    missing, bad_dim = [], []
    for _, row in df.iterrows():
        vid = str(row["video_id"]).strip()
        for m in mods:
            p = feature_path(dataset, vid, m, feat_root)
            if not os.path.exists(p):
                missing.append(p)
            else:
                if config.base_dataset(dataset) == "MDPE" and config.MDPE_MODALITY_EXT.get(m) == ".csv":
                    arr = pd.read_csv(p, header=None).values
                else:
                    arr = np.load(p, allow_pickle=True)
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.shape[-1] != cfg[m]:
                    bad_dim.append(f"{p}: got {arr.shape[-1]}, expect {cfg[m]}")
    if config.base_dataset(dataset) == "MDPE" and cfg.get("use_prior"):
        emo = load_emotion_labels()
        subs = set(df["subject"].astype(str))
        lacking = subs - set(emo)
        if lacking:
            raise RuntimeError(f"MDPE 情绪标注缺失受试者 {sorted(lacking)[:5]}（共 {len(lacking)}）")
        print(f"[integrity] MDPE 情绪先验 ✓ {len(emo)} 受试者 × 8 类强度（真实标注，非零占位）")
    n = len(df)
    n_missing, n_bad = len(missing), len(bad_dim)
    print(f"[integrity] {dataset}: {n} 样本 × {len(mods)} 模态 → 缺失 {n_missing}，维度错误 {n_bad}")
    if missing:
        print("  缺失示例:", missing[:5])
    if bad_dim:
        print("  维度错误示例:", bad_dim[:5])
    if n_missing or n_bad:
        raise RuntimeError(f"{dataset} 特征不完整，请先运行特征提取/下载")
    print(f"[integrity] {dataset} ✓ 全部特征完整，维度正确")
    return True


# ---------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------
# 特征内存缓存：预载全部特征到内存（SEU-MLD ≈300MB / MDPE ≈1GB，服务器 752GB 内存无压力）。
#   * 在 experiments.run_dataset 预载一次 → 之后所有 (fold×seed) run 的 DataLoader 零磁盘 IO；
#   * Linux fork 的 DataLoader worker 通过 COW 共享同一份缓存（零复制）；
#   * 模块级 dict 本质是"路径→numpy"，按数据集/数据集划分天然复用（train/val/test 不重叠 → 全量恰好加载一次）。
# ---------------------------------------------------------------------
_MEM_STORE = {}          # path -> np.ndarray(特征数组)
_MEM_PRELOADED = set()   # (dataset, feat_root) 已预载标记


def _read_feature(dataset: str, path: str, m: str, cfg: dict):
    """读单个特征文件为 np.float32（MDPE text_llm 是 csv；clipVIT 为 float16 转 float32），带维度断言。"""
    if config.base_dataset(dataset) == "MDPE" and config.MDPE_MODALITY_EXT.get(m) == ".csv":
        arr = pd.read_csv(path, header=None).values
    else:
        arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    assert arr.shape[-1] == cfg[m], (
        f"{path}: dim {arr.shape[-1]} != 期望 {cfg[m]}")
    return arr


def preload_features(dataset: str, df: pd.DataFrame, feat_root: str = None):
    """把 df 全部样本 × 全部模态一次性读入 _MEM_STORE（重复调用幂等）。"""
    key = (dataset, feat_root)
    if key in _MEM_PRELOADED:
        return
    cfg = config.get_dataset_cfg(dataset)
    mods = [m for m in cfg
            if m.endswith(("vision", "text_llm", "text_classic", "audio_llm", "audio_classic"))]
    n = 0
    for vid in df["video_id"].astype(str).str.strip().unique():
        for m in mods:
            p = feature_path(dataset, vid, m, feat_root)
            if p not in _MEM_STORE:
                _MEM_STORE[p] = _read_feature(dataset, p, m, cfg)
                n += 1
    _MEM_PRELOADED.add(key)
    print(f"[cache] {dataset} 预载 {n} 特征文件 → 内存（之后所有 run 零磁盘 IO）")


# ---------------------------------------------------------------------
# 情绪先验（MDPE）：labels/ 下 {ID3}.csv，193 受试者 × 16 情绪视频 × 8 类强度(1-5)
# 每受试者聚合为 8 维向量（8 类列均值，受试者级画像先验）
# ---------------------------------------------------------------------
def load_emotion_labels(labels_dir: str = None):
    """返回 {subj_id_str: 8维np.float32}；目录缺失或空 → 报错（绝不静默零占位）。"""
    # 运行时拼接（相对 config.MDPE_RAW），保证 smoke 覆盖/移动数据根后仍正确
    if labels_dir is None:
        labels_dir = os.path.join(config.MDPE_RAW, "emotion", "labels")
    files = sorted(glob.glob(os.path.join(labels_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"未找到情绪标注 {labels_dir}/*.csv —— 需先用 labels.zip 解压（密码见申请邮件）")
    data = {}
    for f in files:
        df = pd.read_csv(f)
        # 8 类列（sad..neutral 顺序不定，按列名取值），16 行均值 → 8 维
        cols = ["sad", "relax", "happy", "surprise", "fear", "anger", "disgust", "neutral"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{f}: 缺少情绪列 {missing}")
        subj = str(int(os.path.basename(f).split(".")[0]))   # '001.csv' -> '1'（与 personality id 对齐）
        data[subj] = df[cols].values.astype(np.float32).mean(axis=0)
    return data


# ---------------------------------------------------------------------
class DeceptionDataset(Dataset):
    """单数据集特征加载。模态按 config.DATASET_MODALITIES 决定。

    返回 dict：
      vision/text_llm/text_classic/audio_llm/audio_classic : (T, D) 张量（文本为 (1,768)）
      personality/emotion : 先验向量（仅 MDPE 有；SEU-MLD 返回空向量但 use_prior=False 时不使用）
      label : 二元标签
      index : 原始行号（用于配对检验/回溯）
    """

    def __init__(self, dataset: str, df: pd.DataFrame, idx: np.ndarray,
                 feat_root: str = None, prior_csv: str = None):
        self.dataset = dataset
        self.df = df
        self.idx = np.asarray(idx)
        self.cfg = config.get_dataset_cfg(dataset)
        self.feat_root = feat_root
        self.mods = [m for m in self.cfg
                     if m.endswith(("vision", "text_llm", "text_classic", "audio_llm", "audio_classic"))]
        self.use_prior = bool(self.cfg.get("use_prior", False))
        self.personality = None
        self.emotion = None
        if self.use_prior and prior_csv:
            pers = pd.read_csv(prior_csv)
            id_col = "id" if "id" in pers.columns else "ID"
            pers[id_col] = pers[id_col].astype(str).str.strip()
            pers.set_index(id_col, inplace=True)
            self.personality = pers
            # 情绪先验：8 类强度标注（受试者级，真实数据；缺失即报错，不虚构）
            self.emotion = load_emotion_labels()

    def __len__(self):
        return len(self.idx)

    def _load(self, path: str, m: str):
        # 命中模块级内存缓存 → 零磁盘 IO；miss（如在 Windows spawn 的 worker 内）则读盘并回填
        if path in _MEM_STORE:
            arr = _MEM_STORE[path]
        else:
            arr = _read_feature(self.dataset, path, m, self.cfg)
            _MEM_STORE[path] = arr
        return torch.from_numpy(arr)

    def __getitem__(self, i):
        row = self.df.iloc[self.idx[i]]
        vid = str(row["video_id"]).strip()
        out = {}
        for m in self.mods:
            p = feature_path(self.dataset, vid, m, self.feat_root)
            out[m] = self._load(p, m)
        out["label"] = int(row["label"])
        out["video_id"] = vid
        out["index"] = self.idx[i]
        # 先验（MDPE）：受试者 = 第一个连字符前 ID
        #   personality = BFI-2 前 60 题原始得分（真实人格标注，personality.csv）
        #   emotion     = 8 类情绪强度自评 1-5 的 16 视频均值（真实标注，labels.zip）
        if self.use_prior and self.personality is not None:
            subj = str(row["subject"])
            if subj in self.personality.index:
                vals = self.personality.loc[subj].values.astype(np.float32)
                pers = torch.from_numpy(vals[:config.PERSONALITY_DIM])
            else:
                pers = torch.zeros(config.PERSONALITY_DIM)
            if subj in self.emotion:
                emo = torch.from_numpy(self.emotion[subj])
            else:
                # 情绪标注缺该受试者 → 零向量并警告（当前 193 文件全覆盖 192 受试者，不应触发）
                warnings.warn(f"MDPE 情绪标注缺失受试者 {subj}，该样本情绪先验置零")
                emo = torch.zeros(config.PRIOR_EMB_DIM)
            out["personality"] = pers
            out["emotion"] = emo
        return out


MAX_SEQ_LEN = 300

def collate_fn(batch):
    """统一序列长度 + 真值掩码。文本 (1,768) 广播到序列长度。"""
    first = batch[0]
    feat_keys = [k for k in first if k.endswith(("vision", "text_llm", "text_classic",
                                                 "audio_llm", "audio_classic"))]
    res = {}
    for k in feat_keys:
        tensors = [b[k] for b in batch]
        tensors = [t[:MAX_SEQ_LEN] if t.size(0) > MAX_SEQ_LEN else t for t in tensors]
        res[k] = pad_sequence(tensors, batch_first=True)
        res[f"{k}_mask"] = torch.zeros(res[k].shape[:2], dtype=torch.bool)
        for j, t in enumerate(tensors):
            res[f"{k}_mask"][j, :t.size(0)] = True
    if "personality" in first:
        res["personality"] = torch.stack([b["personality"] for b in batch])
        res["emotion"] = torch.stack([b["emotion"] for b in batch])
    res["labels"] = torch.LongTensor([b["label"] for b in batch])
    res["video_ids"] = [b["video_id"] for b in batch]
    res["indexes"] = torch.LongTensor([b["index"] for b in batch])
    return res
