# -*- coding: utf-8 -*-
"""
TabPFN v2 训练与留一台风扇（leave-one-fan-out）验证。

- 曲线模型：特征 = 结构参数 + φ，目标分别为 ψst 和 λ。
- 标量模型：特征 = 结构参数，目标为 η_max 和 φ_at_ηmax。
- η 由 ψst·φ/λ 计算（静效率定义恒等式）。
"""

import numpy as np
import pandas as pd


def detect_device(prefer="auto"):
    if prefer in ("cpu", "cuda"):
        return prefer
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def make_regressor(device="cpu", random_state=42):
    """构造 TabPFNRegressor，兼容不同小版本的构造参数差异。"""
    from tabpfn import TabPFNRegressor
    for kwargs in (
        dict(device=device, ignore_pretraining_limits=True, random_state=random_state),
        dict(device=device, random_state=random_state),
        dict(device=device),
    ):
        try:
            return TabPFNRegressor(**kwargs)
        except TypeError:
            continue
    return TabPFNRegressor()


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[m], y_pred[m]
    if len(yt) == 0:
        return dict(n=0, rmse=float("nan"), mae=float("nan"), r2=float("nan"), mape=float("nan"))
    err = yt - yp
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    denom = np.where(np.abs(yt) > 1e-9, np.abs(yt), np.nan)
    mape = float(np.nanmean(np.abs(err) / denom) * 100.0)
    return dict(n=int(len(yt)), rmse=rmse, mae=mae, r2=r2, mape=mape)


def compute_eta(psi, phi, lam):
    """静效率 η = ψst·φ/λ（无量纲一致定义）。"""
    psi = np.asarray(psi, dtype=float)
    phi = np.asarray(phi, dtype=float)
    lam = np.asarray(lam, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = psi * phi / lam
    return eta


def peak_from_curve(phi, eta):
    """在离散 (φ, η) 曲线上求峰：最大点附近局部二次拟合取顶点，失败则退回离散最大点。"""
    phi = np.asarray(phi, dtype=float)
    eta = np.asarray(eta, dtype=float)
    m = np.isfinite(phi) & np.isfinite(eta)
    phi, eta = phi[m], eta[m]
    if len(phi) == 0:
        return float("nan"), float("nan")
    order = np.argsort(phi)
    phi, eta = phi[order], eta[order]
    i = int(np.argmax(eta))
    lo = max(0, i - 1)
    hi = min(len(phi), i + 2)
    pw, ew = phi[lo:hi], eta[lo:hi]
    if len(pw) >= 3:
        try:
            a, b, c = np.polyfit(pw, ew, 2)
            if a < 0:
                phi_star = -b / (2.0 * a)
                if pw.min() <= phi_star <= pw.max():
                    eta_star = a * phi_star ** 2 + b * phi_star + c
                    return float(eta_star), float(phi_star)
        except Exception:
            pass
    return float(eta[i]), float(phi[i])


def derive_scalars_from_curve(eta_curve, df, det):
    """从预测 η 曲线对每台风扇求峰得到 η_max / φ_at_ηmax，并对齐真实列（自动处理百分数/分数单位）。"""
    fans = list(pd.unique(eta_curve["__fan__"]))
    etamax_col = det.get("eta_max")
    phiat_col = det.get("phi_at_etamax")
    has_id = "风扇ID" in df.columns
    true_etamax = dict(zip(df["风扇ID"], df[etamax_col])) if (etamax_col and has_id) else {}
    true_phiat = dict(zip(df["风扇ID"], df[phiat_col])) if (phiat_col and has_id) else {}
    recs = []
    for f in fans:
        sub = eta_curve[eta_curve["__fan__"] == f]
        eta_peak, phi_peak = peak_from_curve(sub["phi"].to_numpy(), sub["eta_pred"].to_numpy())
        recs.append(dict(__fan__=f, etamax_pred_frac=eta_peak, phiat_pred=phi_peak,
                         etamax_true=float(true_etamax.get(f, np.nan)) if etamax_col else np.nan,
                         phiat_true=float(true_phiat.get(f, np.nan)) if phiat_col else np.nan))
    der = pd.DataFrame(recs)
    scale = 1.0
    if etamax_col is not None and len(der) and np.isfinite(der["etamax_true"].to_numpy(dtype=float)).any():
        if np.nanmedian(der["etamax_true"].to_numpy(dtype=float)) > 1.5:
            scale = 100.0
    der["etamax_pred"] = der["etamax_pred_frac"] * scale
    return der


def lofo_curve(long_df, feature_cols, target, device, log=print, cancel=None):
    """留一台风扇验证曲线目标（'psi' 或 'lam'），返回带预测列的结果表。"""
    if cancel is None:
        cancel = lambda: False
    fans = list(pd.unique(long_df["__fan__"]))
    feats = list(feature_cols) + ["phi"]
    out = long_df[["__fan__", "point", "phi", target]].copy()
    out["pred"] = np.nan
    total = len(fans)
    for fi, f in enumerate(fans):
        if cancel():
            log("已取消。")
            break
        test_mask = long_df["__fan__"] == f
        train = long_df[~test_mask].dropna(subset=[target])
        if len(train) < 5:
            log("[%s] 跳过风扇 %s：训练样本不足（%d）" % (target, f, len(train)))
            continue
        Xtr = train[feats].to_numpy(dtype=float)
        ytr = train[target].to_numpy(dtype=float)
        Xte = long_df.loc[test_mask, feats].to_numpy(dtype=float)
        reg = make_regressor(device)
        reg.fit(Xtr, ytr)
        pred = np.asarray(reg.predict(Xte), dtype=float).ravel()
        out.loc[test_mask, "pred"] = pred
        log("[%s] 留一验证 %d/%d 完成（风扇 %s）" % (target, fi + 1, total, f))
    return out


def lofo_scalar(df, feature_cols, target_col, device, log=print, cancel=None):
    """留一台风扇验证标量目标（η_max 或 φ_at_ηmax）。"""
    if cancel is None:
        cancel = lambda: False
    sub = df.dropna(subset=[target_col]).reset_index(drop=True)
    X = sub[feature_cols].to_numpy(dtype=float)
    y = sub[target_col].to_numpy(dtype=float)
    n = len(y)
    preds = np.full(n, np.nan)
    for i in range(n):
        if cancel():
            log("已取消。")
            break
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if mask.sum() < 5:
            continue
        reg = make_regressor(device)
        reg.fit(X[mask], y[mask])
        preds[i] = float(np.asarray(reg.predict(X[i:i + 1]), dtype=float).ravel()[0])
        log("[%s] 留一验证 %d/%d 完成" % (target_col, i + 1, n))
    fan_ids = sub["风扇ID"] if "风扇ID" in sub.columns else pd.Series(range(n))
    return fan_ids.reset_index(drop=True), y, preds


def fit_curve_final(long_df, feature_cols, target, device):
    feats = list(feature_cols) + ["phi"]
    tr = long_df.dropna(subset=[target])
    reg = make_regressor(device)
    reg.fit(tr[feats].to_numpy(dtype=float), tr[target].to_numpy(dtype=float))
    return reg


def fit_scalar_final(df, feature_cols, target_col, device):
    sub = df.dropna(subset=[target_col])
    reg = make_regressor(device)
    reg.fit(sub[feature_cols].to_numpy(dtype=float), sub[target_col].to_numpy(dtype=float))
    return reg
