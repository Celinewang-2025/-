# -*- coding: utf-8 -*-
"""
风扇宽表数据读取与列识别工具。

宽表约定（每台风扇一行）：
  - 命名列（不参与训练）：风扇ID / 风扇名称 / 公司
  - 基本+试验参数 与 截面参数：作为特征
  - 性能参数：η_max、φ_at_ηmax，以及 12 个工况点 φ_k / ψst_k / λ_k（η_k 可由 ψst·φ/λ 计算）
"""

import re
import numpy as np
import pandas as pd

ID_COLS = ["风扇ID", "风扇名称", "公司"]


def _trailing_num(name):
    m = re.search(r"(\d+)\s*$", str(name))
    return int(m.group(1)) if m else -1


def load_table(path, sheet=0):
    """读取 Excel 或 CSV，并去除表头首尾空格（你的 'φ_01' 可能带前导空格）。"""
    path = str(path)
    low = path.lower()
    if low.endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        # 中文 CSV 默认按 utf-8-sig 读，兼容带 BOM 的导出
        df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def list_sheets(path):
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls", ".xlsm")):
        try:
            xls = pd.ExcelFile(path)
            return list(xls.sheet_names)
        except Exception:
            return [0]
    return [0]


def detect_columns(df):
    """根据列名识别命名列、特征列和各类性能目标列。"""
    cols = list(df.columns)

    def match(pat):
        return [c for c in cols if re.fullmatch(pat, c.strip())]

    phi_pts = sorted(match(r"φ_\d+"), key=_trailing_num)
    psi_pts = sorted(match(r"ψst_\d+"), key=_trailing_num)
    lam_pts = sorted(match(r"λ_\d+"), key=_trailing_num)
    eta_pts = sorted(match(r"η_\d+"), key=_trailing_num)
    eta_max = [c for c in cols if c.strip() == "η_max"]
    phi_at = [c for c in cols if c.strip().startswith("φ_at")]

    perf = set(phi_pts + psi_pts + lam_pts + eta_pts + eta_max + phi_at)
    id_cols = [c for c in cols if c in ID_COLS]
    feature_cols = [c for c in cols if c not in perf and c not in id_cols]

    return {
        "id_cols": id_cols,
        "feature_cols": feature_cols,
        "phi_pts": phi_pts,
        "psi_pts": psi_pts,
        "lam_pts": lam_pts,
        "eta_pts": eta_pts,
        "eta_max": eta_max[0] if eta_max else None,
        "phi_at_etamax": phi_at[0] if phi_at else None,
    }


def engineer_features(df, feature_cols, nondim_insert=True, drop_diameter=False):
    """
    物理特征工程：
      - 把绝对量 插入深度(mm) 除以 风扇直径(mm) 变成无量纲比（TabPFN 外推弱，无量纲化可把外推变插值）。
      - 可选删除绝对直径列（其余特征大多已无量纲）。
      - 条件缺失的 one-hot 关联参数（如无环形时的 ring_ratio）以 0 视为“无该结构”。
    """
    df = df.copy()
    feats = list(feature_cols)
    dia = "风扇直径(mm)"
    ins = "插入深度(mm)"

    if nondim_insert and dia in df.columns and ins in df.columns:
        df["插入深度_比D"] = df[ins] / df[dia].replace(0, np.nan)
        if ins in feats:
            feats.remove(ins)
        if "插入深度_比D" not in feats:
            feats.append("插入深度_比D")

    if drop_diameter and dia in feats:
        feats.remove(dia)

    # one-hot 关联参数的条件缺失按 0 处理（表示“无此结构”）
    for c in feats:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, feats


def build_long(df, feature_cols, det):
    """把宽表展开成长表：每台风扇的每个有效工况点一行。"""
    phi_pts = det["phi_pts"]
    psi_pts = det["psi_pts"]
    lam_pts = det["lam_pts"]
    eta_pts = det["eta_pts"]
    n = len(phi_pts)
    records = []
    for ridx, r in df.iterrows():
        fan = r["风扇ID"] if "风扇ID" in df.columns else ridx
        for k in range(n):
            phi = r[phi_pts[k]]
            if pd.isna(phi):
                continue
            rec = {c: r[c] for c in feature_cols}
            rec["__fan__"] = fan
            rec["point"] = k + 1
            rec["phi"] = float(phi)
            rec["psi"] = float(r[psi_pts[k]]) if k < len(psi_pts) and not pd.isna(r[psi_pts[k]]) else np.nan
            rec["lam"] = float(r[lam_pts[k]]) if k < len(lam_pts) and not pd.isna(r[lam_pts[k]]) else np.nan
            if eta_pts and k < len(eta_pts) and not pd.isna(r[eta_pts[k]]):
                rec["eta"] = float(r[eta_pts[k]])
            else:
                rec["eta"] = np.nan
            records.append(rec)
    return pd.DataFrame(records)
