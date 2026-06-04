# -*- coding: utf-8 -*-
"""
冷却轴流风扇性能预测套件（单文件双页版，PyQt5）

页1 训练与验证：
  - 读取风扇结构+性能宽表（Excel/CSV），自动识别列、物理特征工程。
  - 可"增加训练数据"（把更多宽表追加进当前训练集）。
  - TabPFN v2 训练 ψst、λ 曲线模型；η = ψst·φ/λ 计算；η_max、φ_at_ηmax 由预测 η 曲线求峰。
  - 留一台风扇（leave-one-fan-out）验证，输出 RMSE/MAE/R²/MAPE 并画曲线。
  - 用全量数据训练最终模型并保存 .joblib。

页2 预测对比：
  - 加载 .joblib 模型（或直接用训练页刚得到的模型），读入新宽表预测。
  - 预测 vs 实测 画在同一图（ψst、λ、η 三子图），给出误差指标，可导出 CSV。
  - "一键并入训练并重训"：把预测页的新数据并入训练集并重训最终模型，更新当前模型。

运行：
  conda activate AIfan
  python fan_tabpfn_suite.py
"""

import os
import sys
import re
import traceback

import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtWidgets

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

try:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


# =====================================================================
# 数据读取与列识别
# =====================================================================
ID_COLS = ["风扇ID", "风扇名称", "公司"]


def _trailing_num(name):
    m = re.search(r"(\d+)\s*$", str(name))
    return int(m.group(1)) if m else -1


def load_table(path, sheet=0):
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def list_sheets(path):
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls", ".xlsm")):
        try:
            return list(pd.ExcelFile(path).sheet_names)
        except Exception:
            return [0]
    return [0]


def detect_columns(df):
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
        "id_cols": id_cols, "feature_cols": feature_cols,
        "phi_pts": phi_pts, "psi_pts": psi_pts, "lam_pts": lam_pts, "eta_pts": eta_pts,
        "eta_max": eta_max[0] if eta_max else None,
        "phi_at_etamax": phi_at[0] if phi_at else None,
    }


def engineer_features(df, feature_cols, nondim_insert=True, drop_diameter=False):
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
    for c in feats:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, feats


def build_long(df, feature_cols, det):
    phi_pts, psi_pts, lam_pts, eta_pts = det["phi_pts"], det["psi_pts"], det["lam_pts"], det["eta_pts"]
    use_feats = [c for c in feature_cols if c in df.columns]
    n = len(phi_pts)
    records = []
    for ridx, r in df.iterrows():
        fan = r["风扇ID"] if "风扇ID" in df.columns else ridx
        for k in range(n):
            phi = r[phi_pts[k]]
            if pd.isna(phi):
                continue
            rec = {c: r[c] for c in use_feats}
            rec["__fan__"] = fan
            rec["phi"] = float(phi)
            rec["psi"] = float(r[psi_pts[k]]) if k < len(psi_pts) and not pd.isna(r[psi_pts[k]]) else np.nan
            rec["lam"] = float(r[lam_pts[k]]) if k < len(lam_pts) and not pd.isna(r[lam_pts[k]]) else np.nan
            records.append(rec)
    return pd.DataFrame(records)


# =====================================================================
# 训练 / 验证 / 预测核心
# =====================================================================
def detect_device(prefer="auto"):
    if prefer in ("cpu", "cuda"):
        return prefer
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def make_regressor(device="cpu", random_state=42):
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


def _fmt_metrics(m):
    return "n=%d  RMSE=%.4g  MAE=%.4g  R2=%.4f  MAPE=%.2f%%" % (
        m["n"], m["rmse"], m["mae"], m["r2"], m["mape"])


def compute_eta(psi, phi, lam):
    psi = np.asarray(psi, dtype=float)
    phi = np.asarray(phi, dtype=float)
    lam = np.asarray(lam, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return psi * phi / lam


def peak_from_curve(phi, eta):
    phi = np.asarray(phi, dtype=float)
    eta = np.asarray(eta, dtype=float)
    m = np.isfinite(phi) & np.isfinite(eta)
    phi, eta = phi[m], eta[m]
    if len(phi) == 0:
        return np.nan, np.nan
    order = np.argsort(phi)
    phi, eta = phi[order], eta[order]
    i = int(np.argmax(eta))
    lo, hi = max(0, i - 1), min(len(phi), i + 2)
    pw, ew = phi[lo:hi], eta[lo:hi]
    if len(pw) >= 3:
        try:
            a, b, c = np.polyfit(pw, ew, 2)
            if a < 0:
                phi_star = -b / (2.0 * a)
                if pw.min() <= phi_star <= pw.max():
                    return float(a * phi_star ** 2 + b * phi_star + c), float(phi_star)
        except Exception:
            pass
    return float(eta[i]), float(phi[i])


def derive_scalars_from_curve(eta_curve, df, det):
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
    if cancel is None:
        cancel = lambda: False
    fans = list(pd.unique(long_df["__fan__"]))
    feats = list(feature_cols) + ["phi"]
    out = long_df[["__fan__", "phi", target]].copy()
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
        out.loc[test_mask, "pred"] = np.asarray(reg.predict(Xte), dtype=float).ravel()
        log("[%s] 留一验证 %d/%d 完成（风扇 %s）" % (target, fi + 1, total, f))
    return out


def fit_curve_final(long_df, feature_cols, target, device):
    feats = list(feature_cols) + ["phi"]
    tr_df = long_df.dropna(subset=[target])
    reg = make_regressor(device)
    reg.fit(tr_df[feats].to_numpy(dtype=float), tr_df[target].to_numpy(dtype=float))
    return reg


def predict_with_bundle(bundle, long_df):
    feats = list(bundle["feature_cols"]) + ["phi"]
    X = long_df.reindex(columns=feats).to_numpy(dtype=float)
    out = long_df.copy()
    models = bundle.get("models", {})
    if "psi" in models:
        out["pred_psi"] = np.asarray(models["psi"].predict(X), dtype=float).ravel()
    if "lam" in models:
        out["pred_lam"] = np.asarray(models["lam"].predict(X), dtype=float).ravel()
    if "pred_psi" in out.columns and "pred_lam" in out.columns:
        out["eta_pred"] = compute_eta(out["pred_psi"].to_numpy(), out["phi"].to_numpy(), out["pred_lam"].to_numpy())
    if "psi" in out.columns and "lam" in out.columns:
        out["eta_actual"] = compute_eta(out["psi"].to_numpy(), out["phi"].to_numpy(), out["lam"].to_numpy())
    return out


def merge_tables(df_a, df_b, id_col="风扇ID"):
    """纵向合并两张宽表，按风扇ID去重（保留后者）。"""
    merged = pd.concat([df_a, df_b], ignore_index=True, sort=False)
    if id_col in merged.columns:
        merged = merged.drop_duplicates(subset=[id_col], keep="last").reset_index(drop=True)
    return merged


# =====================================================================
# 后台线程
# =====================================================================
class TrainWorker(QtCore.QThread):
    log = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, df, det, feats, device, targets, parent=None):
        super().__init__(parent)
        self.df, self.det, self.feats = df, det, feats
        self.device, self.targets = device, targets
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _cancelled(self):
        return self._cancel

    def run(self):
        try:
            results = {}
            long_df = build_long(self.df, self.feats, self.det)
            self.log.emit("长表构建完成：%d 行（%d 台风扇）" % (len(long_df), long_df["__fan__"].nunique()))
            results["long_df"] = long_df

            if self.targets.get("psi"):
                self.log.emit("===== 训练/验证 静压系数 ψst =====")
                out = lofo_curve(long_df, self.feats, "psi", self.device, log=self.log.emit, cancel=self._cancelled)
                results["psi"] = out
                results["psi_metrics"] = regression_metrics(out["psi"], out["pred"])
                self.log.emit("ψst 指标: %s" % _fmt_metrics(results["psi_metrics"]))

            if self.targets.get("lam") and not self._cancel:
                self.log.emit("===== 训练/验证 功率系数 λ =====")
                out = lofo_curve(long_df, self.feats, "lam", self.device, log=self.log.emit, cancel=self._cancelled)
                results["lam"] = out
                results["lam_metrics"] = regression_metrics(out["lam"], out["pred"])
                self.log.emit("λ 指标: %s" % _fmt_metrics(results["lam_metrics"]))

            if results.get("psi") is not None and results.get("lam") is not None and not self._cancel:
                p, l = results["psi"], results["lam"]
                merged = p.merge(l, on=["__fan__", "phi"], suffixes=("_psi", "_lam"))
                merged["eta_true"] = compute_eta(merged["psi"], merged["phi"], merged["lam"])
                merged["eta_pred"] = compute_eta(merged["pred_psi"], merged["phi"], merged["pred_lam"])
                results["eta_curve"] = merged
                results["eta_metrics"] = regression_metrics(merged["eta_true"], merged["eta_pred"])
                self.log.emit("η(由ψst·φ/λ计算) 指标: %s" % _fmt_metrics(results["eta_metrics"]))

            need_scalar = self.targets.get("eta_max") or self.targets.get("phi_at")
            if need_scalar and not self._cancel:
                if results.get("eta_curve") is None:
                    self.log.emit("提示：η_max / φ_at_ηmax 由预测 η 曲线求峰得到，需要同时勾选 ψst 和 λ。已跳过。")
                else:
                    der = derive_scalars_from_curve(results["eta_curve"], self.df, self.det)
                    results["scalar_from_curve"] = der
                    if self.targets.get("eta_max") and self.det.get("eta_max"):
                        results["eta_max_metrics"] = regression_metrics(
                            der["etamax_true"].to_numpy(dtype=float), der["etamax_pred"].to_numpy(dtype=float))
                        self.log.emit("η_max(由η曲线求峰) 指标: %s" % _fmt_metrics(results["eta_max_metrics"]))
                    if self.targets.get("phi_at") and self.det.get("phi_at_etamax"):
                        results["phi_at_metrics"] = regression_metrics(
                            der["phiat_true"].to_numpy(dtype=float), der["phiat_pred"].to_numpy(dtype=float))
                        self.log.emit("φ_at_ηmax(由η曲线求峰) 指标: %s" % _fmt_metrics(results["phi_at_metrics"]))

            self.finished_ok.emit(results)
        except Exception:
            self.failed.emit(traceback.format_exc())


class RetrainWorker(QtCore.QThread):
    """用全量数据训练最终 ψst、λ 模型，产出可保存/可预测的 bundle。"""
    log = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, df, det, feats, device, parent=None):
        super().__init__(parent)
        self.df, self.det, self.feats, self.device = df, det, feats, device

    def run(self):
        try:
            long_df = build_long(self.df, self.feats, self.det)
            self.log.emit("全量长表：%d 行（%d 台风扇）" % (len(long_df), long_df["__fan__"].nunique()))
            bundle = {"feature_cols": self.feats, "detection": self.det, "models": {}, "scalar_from_curve": True}
            self.log.emit("训练 ψst 最终模型…")
            bundle["models"]["psi"] = fit_curve_final(long_df, self.feats, "psi", self.device)
            self.log.emit("训练 λ 最终模型…")
            bundle["models"]["lam"] = fit_curve_final(long_df, self.feats, "lam", self.device)
            self.log.emit("最终模型训练完成。")
            self.finished_ok.emit(bundle)
        except Exception:
            self.failed.emit(traceback.format_exc())


class PredictWorker(QtCore.QThread):
    log = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, bundle, df, det, parent=None):
        super().__init__(parent)
        self.bundle, self.df, self.det = bundle, df, det

    def run(self):
        try:
            feats_used = self.bundle.get("feature_cols", [])
            long_df = build_long(self.df, feats_used, self.det)
            self.log.emit("长表构建完成：%d 行（%d 台风扇）" % (len(long_df), long_df["__fan__"].nunique()))
            missing = [c for c in feats_used if c not in long_df.columns]
            if missing:
                self.log.emit("⚠️ 数据缺少 %d 个训练特征列（按缺失处理）：%s" % (len(missing), ", ".join(map(str, missing))))
            out = predict_with_bundle(self.bundle, long_df)
            self.log.emit("预测完成。")
            self.finished_ok.emit(out)
        except Exception:
            self.failed.emit(traceback.format_exc())


# =====================================================================
# 主窗口（双页）
# =====================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("冷却轴流风扇性能预测套件 - TabPFN v2（训练 / 预测）")
        self.resize(1360, 860)

        # 共享状态
        self.train_df = None
        self.train_det = None
        self.train_feats = None
        self.train_results = None
        self.bundle = None
        self.pred_df = None
        self.pred_det = None
        self.pred_out = None
        self.train_path = None
        self.pred_path = None
        self.train_worker = None
        self.retrain_worker = None
        self.pred_worker = None

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)
        tabs.addTab(self._build_train_tab(), "训练与验证")
        tabs.addTab(self._build_predict_tab(), "预测对比")
        self._refresh_device_label()

    # ----------------------------------------------------------------
    # 训练页
    # ----------------------------------------------------------------
    def _build_train_tab(self):
        w = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(w)
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)

        gb1 = QtWidgets.QGroupBox("一、训练数据")
        v = QtWidgets.QVBoxLayout(gb1)
        b1 = QtWidgets.QPushButton("选择训练宽表 (Excel/CSV)")
        b1.clicked.connect(self.on_train_open)
        v.addWidget(b1)
        b_add = QtWidgets.QPushButton("增加训练数据（追加另一张宽表）")
        b_add.clicked.connect(self.on_train_add)
        v.addWidget(b_add)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("工作表:"))
        self.tr_sheet = QtWidgets.QComboBox()
        self.tr_sheet.currentIndexChanged.connect(self.on_train_sheet_changed)
        h.addWidget(self.tr_sheet, 1)
        v.addLayout(h)
        self.tr_lbl = QtWidgets.QLabel("尚未加载训练数据")
        self.tr_lbl.setWordWrap(True)
        v.addWidget(self.tr_lbl)
        left.addWidget(gb1)

        gb2 = QtWidgets.QGroupBox("二、特征工程")
        v2 = QtWidgets.QVBoxLayout(gb2)
        self.chk_nondim = QtWidgets.QCheckBox("插入深度 ÷ 直径 做无量纲化")
        self.chk_nondim.setChecked(True)
        v2.addWidget(self.chk_nondim)
        self.chk_drop_d = QtWidgets.QCheckBox("删除绝对直径列")
        v2.addWidget(self.chk_drop_d)
        left.addWidget(gb2)

        gb3 = QtWidgets.QGroupBox("三、训练目标")
        v3 = QtWidgets.QVBoxLayout(gb3)
        self.chk_psi = QtWidgets.QCheckBox("静压系数曲线 ψst"); self.chk_psi.setChecked(True)
        self.chk_lam = QtWidgets.QCheckBox("功率系数曲线 λ"); self.chk_lam.setChecked(True)
        self.chk_etamax = QtWidgets.QCheckBox("最大效率 η_max（由曲线求峰）"); self.chk_etamax.setChecked(True)
        self.chk_phiat = QtWidgets.QCheckBox("最大效率点 φ_at_ηmax（由曲线求峰）"); self.chk_phiat.setChecked(True)
        for x in (self.chk_psi, self.chk_lam, self.chk_etamax, self.chk_phiat):
            v3.addWidget(x)
        left.addWidget(gb3)

        gb4 = QtWidgets.QGroupBox("四、计算设备")
        v4 = QtWidgets.QVBoxLayout(gb4)
        self.cmb_dev = QtWidgets.QComboBox()
        self.cmb_dev.addItems(["auto", "cuda", "cpu"])
        self.cmb_dev.currentIndexChanged.connect(self._refresh_device_label)
        v4.addWidget(self.cmb_dev)
        self.lbl_dev = QtWidgets.QLabel("")
        v4.addWidget(self.lbl_dev)
        left.addWidget(gb4)

        gb5 = QtWidgets.QGroupBox("五、运行")
        v5 = QtWidgets.QVBoxLayout(gb5)
        self.btn_train = QtWidgets.QPushButton("开始训练 + 留一验证")
        self.btn_train.clicked.connect(self.on_train_start)
        v5.addWidget(self.btn_train)
        self.btn_cancel = QtWidgets.QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.on_train_cancel)
        v5.addWidget(self.btn_cancel)
        self.btn_final = QtWidgets.QPushButton("用全量数据训练最终模型")
        self.btn_final.clicked.connect(self.on_train_final)
        v5.addWidget(self.btn_final)
        self.btn_save = QtWidgets.QPushButton("保存当前模型到 .joblib")
        self.btn_save.clicked.connect(self.on_save_model)
        v5.addWidget(self.btn_save)
        left.addWidget(gb5)

        gb6 = QtWidgets.QGroupBox("六、曲线查看")
        v6 = QtWidgets.QVBoxLayout(gb6)
        hh = QtWidgets.QHBoxLayout()
        hh.addWidget(QtWidgets.QLabel("风扇:"))
        self.tr_fan = QtWidgets.QComboBox()
        self.tr_fan.currentIndexChanged.connect(self.on_train_plot)
        hh.addWidget(self.tr_fan, 1)
        v6.addLayout(hh)
        left.addWidget(gb6)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)
        self.tr_log = QtWidgets.QPlainTextEdit(); self.tr_log.setReadOnly(True); self.tr_log.setMaximumBlockCount(5000)
        right.addWidget(self.tr_log, 1)
        self.tr_fig = Figure(figsize=(9, 4)); self.tr_canvas = FigureCanvas(self.tr_fig)
        right.addWidget(self.tr_canvas, 2)
        return w

    # ----------------------------------------------------------------
    # 预测页
    # ----------------------------------------------------------------
    def _build_predict_tab(self):
        w = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(w)
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)

        gb1 = QtWidgets.QGroupBox("一、模型")
        v = QtWidgets.QVBoxLayout(gb1)
        b_use = QtWidgets.QPushButton("使用训练页当前模型")
        b_use.clicked.connect(self.on_pred_use_current)
        v.addWidget(b_use)
        b_load = QtWidgets.QPushButton("加载模型 (.joblib)")
        b_load.clicked.connect(self.on_pred_load_model)
        v.addWidget(b_load)
        self.pr_model_lbl = QtWidgets.QLabel("尚未指定模型")
        self.pr_model_lbl.setWordWrap(True)
        v.addWidget(self.pr_model_lbl)
        left.addWidget(gb1)

        gb2 = QtWidgets.QGroupBox("二、验证/预测数据")
        v2 = QtWidgets.QVBoxLayout(gb2)
        b_data = QtWidgets.QPushButton("加载宽表 (Excel/CSV)")
        b_data.clicked.connect(self.on_pred_open)
        v2.addWidget(b_data)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("工作表:"))
        self.pr_sheet = QtWidgets.QComboBox()
        self.pr_sheet.currentIndexChanged.connect(self.on_pred_sheet_changed)
        h.addWidget(self.pr_sheet, 1)
        v2.addLayout(h)
        self.pr_data_lbl = QtWidgets.QLabel("尚未加载数据")
        self.pr_data_lbl.setWordWrap(True)
        v2.addWidget(self.pr_data_lbl)
        left.addWidget(gb2)

        gb3 = QtWidgets.QGroupBox("三、预测")
        v3 = QtWidgets.QVBoxLayout(gb3)
        self.btn_predict = QtWidgets.QPushButton("开始预测并对比")
        self.btn_predict.clicked.connect(self.on_pred_start)
        v3.addWidget(self.btn_predict)
        self.btn_export = QtWidgets.QPushButton("导出预测结果 CSV")
        self.btn_export.clicked.connect(self.on_pred_export)
        v3.addWidget(self.btn_export)
        left.addWidget(gb3)

        gb4 = QtWidgets.QGroupBox("四、并入训练")
        v4 = QtWidgets.QVBoxLayout(gb4)
        self.btn_merge = QtWidgets.QPushButton("一键并入训练并重训")
        self.btn_merge.clicked.connect(self.on_merge_retrain)
        v4.addWidget(self.btn_merge)
        v4.addWidget(QtWidgets.QLabel("说明：把本页数据并入\n训练页数据集后重训最终模型，\n并更新当前模型。"))
        left.addWidget(gb4)

        gb5 = QtWidgets.QGroupBox("五、查看风扇")
        v5 = QtWidgets.QVBoxLayout(gb5)
        hh = QtWidgets.QHBoxLayout()
        hh.addWidget(QtWidgets.QLabel("风扇:"))
        self.pr_fan = QtWidgets.QComboBox()
        self.pr_fan.currentIndexChanged.connect(self.on_pred_plot)
        hh.addWidget(self.pr_fan, 1)
        v5.addLayout(hh)
        left.addWidget(gb5)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)
        self.pr_log = QtWidgets.QPlainTextEdit(); self.pr_log.setReadOnly(True); self.pr_log.setMaximumBlockCount(5000)
        right.addWidget(self.pr_log, 1)
        self.pr_fig = Figure(figsize=(9, 4)); self.pr_canvas = FigureCanvas(self.pr_fig)
        right.addWidget(self.pr_canvas, 2)
        return w

    # ----------------------------------------------------------------
    # 通用
    # ----------------------------------------------------------------
    def _refresh_device_label(self):
        self.lbl_dev.setText("实际使用设备: %s" % detect_device(self.cmb_dev.currentText()))

    def tlog(self, msg):
        self.tr_log.appendPlainText(str(msg))

    def plog(self, msg):
        self.pr_log.appendPlainText(str(msg))

    def _device(self):
        return detect_device(self.cmb_dev.currentText())

    # ----------------------------------------------------------------
    # 训练页：数据加载
    # ----------------------------------------------------------------
    def on_train_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择训练宽表", "", "数据文件 (*.xlsx *.xls *.xlsm *.csv)")
        if not path:
            return
        self.train_path = path
        self.tr_sheet.blockSignals(True)
        self.tr_sheet.clear()
        for s in list_sheets(path):
            self.tr_sheet.addItem(str(s))
        self.tr_sheet.blockSignals(False)
        self._train_load_current(replace=True)

    def on_train_sheet_changed(self, _):
        if self.train_path:
            self._train_load_current(replace=True)

    def _read_with_sheet(self, path, combo):
        sheet = combo.currentText()
        try:
            sheet_arg = int(sheet)
        except (ValueError, TypeError):
            sheet_arg = sheet
        return load_table(path, sheet=sheet_arg)

    def _train_load_current(self, replace=True):
        try:
            df = self._read_with_sheet(self.train_path, self.tr_sheet)
            if replace or self.train_df is None:
                base = df
            else:
                base = merge_tables(self.train_df, df)
            self.train_det = detect_columns(base)
            self.train_df, self.train_feats = engineer_features(
                base, self.train_det["feature_cols"],
                nondim_insert=self.chk_nondim.isChecked(), drop_diameter=self.chk_drop_d.isChecked())
            self._train_report()
        except Exception:
            self.tlog("读取失败：\n" + traceback.format_exc())

    def on_train_add(self):
        if self.train_df is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先用上面的按钮加载初始训练宽表，再追加。")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择要追加的训练宽表", "", "数据文件 (*.xlsx *.xls *.xlsm *.csv)")
        if not path:
            return
        try:
            sheets = list_sheets(path)
            sheet_arg = sheets[0]
            try:
                sheet_arg = int(sheet_arg)
            except (ValueError, TypeError):
                pass
            add_df = load_table(path, sheet=sheet_arg)
            before = len(self.train_df)
            base = merge_tables(self.train_df, add_df)
            self.train_det = detect_columns(base)
            self.train_df, self.train_feats = engineer_features(
                base, self.train_det["feature_cols"],
                nondim_insert=self.chk_nondim.isChecked(), drop_diameter=self.chk_drop_d.isChecked())
            self.tlog("已追加训练数据：%d → %d 台风扇" % (before, len(self.train_df)))
            self._train_report()
        except Exception:
            self.tlog("追加失败：\n" + traceback.format_exc())

    def _train_report(self):
        d = self.train_det
        self.tr_lbl.setText("当前训练集：%d 台风扇，%d 个特征" % (len(self.train_df), len(self.train_feats)))
        self.tlog("=" * 60)
        self.tlog("当前训练集 %d 台风扇 | 特征数 %d" % (len(self.train_df), len(self.train_feats)))
        self.tlog("工况点 φ:%d ψst:%d λ:%d | η_max:%s φ_at:%s" %
                  (len(d["phi_pts"]), len(d["psi_pts"]), len(d["lam_pts"]), d["eta_max"], d["phi_at_etamax"]))

    # ----------------------------------------------------------------
    # 训练页：训练/验证
    # ----------------------------------------------------------------
    def on_train_start(self):
        if self.train_df is None:
            QtWidgets.QMessageBox.warning(self, "无数据", "请先加载训练数据。")
            return
        self.train_df, self.train_feats = engineer_features(
            self.train_df, self.train_det["feature_cols"],
            nondim_insert=self.chk_nondim.isChecked(), drop_diameter=self.chk_drop_d.isChecked())
        targets = dict(psi=self.chk_psi.isChecked(), lam=self.chk_lam.isChecked(),
                       eta_max=self.chk_etamax.isChecked(), phi_at=self.chk_phiat.isChecked())
        self.tlog("使用设备: %s" % self._device())
        self.btn_train.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.train_worker = TrainWorker(self.train_df, self.train_det, self.train_feats, self._device(), targets)
        self.train_worker.log.connect(self.tlog)
        self.train_worker.finished_ok.connect(self.on_train_done)
        self.train_worker.failed.connect(self.on_train_failed)
        self.train_worker.start()

    def on_train_cancel(self):
        if self.train_worker is not None:
            self.train_worker.cancel()
            self.tlog("已请求取消，当前折结束后停止…")

    def on_train_done(self, results):
        self.train_results = results
        self.btn_train.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.tlog("=" * 60)
        self.tlog("训练与留一验证完成。指标汇总：")
        for key, label in (("psi_metrics", "静压系数 ψst"), ("lam_metrics", "功率系数 λ"),
                           ("eta_metrics", "效率 η(计算)"), ("eta_max_metrics", "最大效率 η_max"),
                           ("phi_at_metrics", "最大效率点 φ_at_ηmax")):
            if key in results:
                self.tlog("  %-16s %s" % (label, _fmt_metrics(results[key])))
        self._train_populate_fan()

    def on_train_failed(self, tb):
        self.btn_train.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.tlog("训练失败：\n" + tb)

    def _train_populate_fan(self):
        self.tr_fan.blockSignals(True)
        self.tr_fan.clear()
        if self.train_results and "long_df" in self.train_results:
            for f in pd.unique(self.train_results["long_df"]["__fan__"]):
                self.tr_fan.addItem(str(f))
        self.tr_fan.blockSignals(False)
        if self.tr_fan.count() > 0:
            self.on_train_plot()

    def on_train_plot(self):
        if not self.train_results or self.tr_fan.count() == 0:
            return
        fan = self.tr_fan.currentText()
        self.tr_fig.clear()
        axes = self.tr_fig.subplots(1, 3)
        for ax, t in zip(axes, ["静压系数 ψst - φ", "功率系数 λ - φ", "效率 η - φ"]):
            ax.set_title(t); ax.set_xlabel("流量系数 φ"); ax.grid(True, alpha=0.3)

        def rows(df):
            if df is None:
                return None
            s = df[df["__fan__"].astype(str) == fan].sort_values("phi")
            return s if len(s) else None

        sp = rows(self.train_results.get("psi"))
        if sp is not None:
            axes[0].scatter(sp["phi"], sp["psi"], c="tab:blue", s=28, label="真实")
            axes[0].plot(sp["phi"], sp["pred"], "r-", lw=2, label="预测"); axes[0].legend(fontsize=8)
        sl = rows(self.train_results.get("lam"))
        if sl is not None:
            axes[1].scatter(sl["phi"], sl["lam"], c="tab:blue", s=28, label="真实")
            axes[1].plot(sl["phi"], sl["pred"], "r-", lw=2, label="预测"); axes[1].legend(fontsize=8)
        se = self.train_results.get("eta_curve")
        if se is not None:
            s = se[se["__fan__"].astype(str) == fan].sort_values("phi")
            if len(s):
                axes[2].scatter(s["phi"], s["eta_true"], c="tab:blue", s=28, label="真实(计算)")
                axes[2].plot(s["phi"], s["eta_pred"], "r-", lw=2, label="预测(计算)"); axes[2].legend(fontsize=8)
        self.tr_fig.suptitle("风扇 %s 留一验证曲线" % fan)
        self.tr_fig.tight_layout(); self.tr_canvas.draw()

    # ----------------------------------------------------------------
    # 训练页：最终模型 / 保存
    # ----------------------------------------------------------------
    def on_train_final(self):
        if self.train_df is None:
            QtWidgets.QMessageBox.warning(self, "无数据", "请先加载训练数据。")
            return
        self.train_df, self.train_feats = engineer_features(
            self.train_df, self.train_det["feature_cols"],
            nondim_insert=self.chk_nondim.isChecked(), drop_diameter=self.chk_drop_d.isChecked())
        self.tlog("用全量数据训练最终模型（设备 %s）…" % self._device())
        self.btn_final.setEnabled(False)
        self.retrain_worker = RetrainWorker(self.train_df, self.train_det, self.train_feats, self._device())
        self.retrain_worker.log.connect(self.tlog)
        self.retrain_worker.finished_ok.connect(self._on_final_done)
        self.retrain_worker.failed.connect(lambda tb: (self.btn_final.setEnabled(True), self.tlog("失败：\n" + tb)))
        self.retrain_worker.start()

    def _on_final_done(self, bundle):
        self.bundle = bundle
        self.btn_final.setEnabled(True)
        self.tlog("最终模型已就绪，可在『预测对比』页直接使用，或点『保存当前模型』。")
        self.pr_model_lbl.setText("已就绪：训练页最终模型（特征 %d）" % len(bundle.get("feature_cols", [])))

    def on_save_model(self):
        if self.bundle is None:
            QtWidgets.QMessageBox.information(self, "无模型", "请先点『用全量数据训练最终模型』。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存模型", "fan_tabpfn_models.joblib", "Joblib (*.joblib)")
        if not path:
            return
        try:
            import joblib
            joblib.dump(self.bundle, path)
            self.tlog("模型已保存：%s" % path)
        except Exception:
            self.tlog("保存失败：\n" + traceback.format_exc())

    # ----------------------------------------------------------------
    # 预测页
    # ----------------------------------------------------------------
    def on_pred_use_current(self):
        if self.bundle is None:
            QtWidgets.QMessageBox.information(self, "无模型", "训练页还没有最终模型，请先在训练页点『用全量数据训练最终模型』。")
            return
        models = list(self.bundle.get("models", {}).keys())
        self.pr_model_lbl.setText("使用训练页当前模型（特征 %d，含 %s）" %
                                  (len(self.bundle.get("feature_cols", [])), ", ".join(models)))
        self.plog("已切换为使用训练页当前模型。")

    def on_pred_load_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "加载模型", "", "Joblib (*.joblib)")
        if not path:
            return
        try:
            import joblib
            self.bundle = joblib.load(path)
            feats = self.bundle.get("feature_cols", [])
            models = list(self.bundle.get("models", {}).keys())
            self.pr_model_lbl.setText("%s（特征 %d，含 %s）" % (os.path.basename(path), len(feats), ", ".join(models)))
            self.plog("已加载模型：%s" % path)
            if "psi" not in models or "lam" not in models:
                self.plog("⚠️ 模型缺少 psi 或 lam，无法预测完整曲线。")
        except Exception:
            self.plog("加载失败：\n" + traceback.format_exc())

    def on_pred_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "加载宽表", "", "数据文件 (*.xlsx *.xls *.xlsm *.csv)")
        if not path:
            return
        self.pred_path = path
        self.pr_sheet.blockSignals(True)
        self.pr_sheet.clear()
        for s in list_sheets(path):
            self.pr_sheet.addItem(str(s))
        self.pr_sheet.blockSignals(False)
        self._pred_load_current()

    def on_pred_sheet_changed(self, _):
        if self.pred_path:
            self._pred_load_current()

    def _pred_load_current(self):
        try:
            df = self._read_with_sheet(self.pred_path, self.pr_sheet)
            self.pred_det = detect_columns(df)
            self.pred_df, _ = engineer_features(df, self.pred_det["feature_cols"], nondim_insert=True)
            has_perf = len(self.pred_det["psi_pts"]) > 0 and len(self.pred_det["lam_pts"]) > 0
            self.pr_data_lbl.setText("%d 台风扇 | 含实测可对比: %s" % (len(self.pred_df), "是" if has_perf else "否"))
            self.plog("=" * 60)
            self.plog("已加载数据 %d 台风扇 | 工况点 φ:%d ψst:%d λ:%d" %
                      (len(self.pred_df), len(self.pred_det["phi_pts"]),
                       len(self.pred_det["psi_pts"]), len(self.pred_det["lam_pts"])))
        except Exception:
            self.plog("读取失败：\n" + traceback.format_exc())

    def on_pred_start(self):
        if self.bundle is None:
            QtWidgets.QMessageBox.warning(self, "无模型", "请先指定模型（使用训练页模型或加载 .joblib）。")
            return
        if self.pred_df is None:
            QtWidgets.QMessageBox.warning(self, "无数据", "请先加载验证/预测宽表。")
            return
        self.btn_predict.setEnabled(False)
        self.pred_worker = PredictWorker(self.bundle, self.pred_df, self.pred_det)
        self.pred_worker.log.connect(self.plog)
        self.pred_worker.finished_ok.connect(self.on_pred_done)
        self.pred_worker.failed.connect(self.on_pred_failed)
        self.pred_worker.start()

    def on_pred_done(self, out):
        self.pred_out = out
        self.btn_predict.setEnabled(True)
        if "psi" in out.columns and "pred_psi" in out.columns:
            self.plog("ψst 对比: %s" % _fmt_metrics(regression_metrics(out["psi"], out["pred_psi"])))
        if "lam" in out.columns and "pred_lam" in out.columns:
            self.plog("λ  对比: %s" % _fmt_metrics(regression_metrics(out["lam"], out["pred_lam"])))
        if "eta_actual" in out.columns and "eta_pred" in out.columns:
            self.plog("η  对比: %s" % _fmt_metrics(regression_metrics(out["eta_actual"], out["eta_pred"])))
        if "eta_pred" in out.columns:
            self.plog("由预测 η 曲线求峰（η_max 为分数，×100 为百分数）:")
            for f in pd.unique(out["__fan__"]):
                sub = out[out["__fan__"] == f]
                ep, pp = peak_from_curve(sub["phi"].to_numpy(), sub["eta_pred"].to_numpy())
                self.plog("  风扇 %s: η_max=%.4f, φ_at_ηmax=%.4f" % (str(f), ep, pp))
        self._pred_populate_fan()

    def on_pred_failed(self, tb):
        self.btn_predict.setEnabled(True)
        self.plog("预测失败：\n" + tb)

    def _pred_populate_fan(self):
        self.pr_fan.blockSignals(True)
        self.pr_fan.clear()
        if self.pred_out is not None:
            for f in pd.unique(self.pred_out["__fan__"]):
                self.pr_fan.addItem(str(f))
        self.pr_fan.blockSignals(False)
        if self.pr_fan.count() > 0:
            self.on_pred_plot()

    def on_pred_plot(self):
        if self.pred_out is None or self.pr_fan.count() == 0:
            return
        fan = self.pr_fan.currentText()
        sub = self.pred_out[self.pred_out["__fan__"].astype(str) == fan].sort_values("phi")
        if len(sub) == 0:
            return
        self.pr_fig.clear()
        axes = self.pr_fig.subplots(1, 3)
        cfg = [("静压系数 ψst - φ", "psi", "pred_psi"),
               ("功率系数 λ - φ", "lam", "pred_lam"),
               ("效率 η - φ", "eta_actual", "eta_pred")]
        for ax, (title, ac, pc) in zip(axes, cfg):
            ax.set_title(title); ax.set_xlabel("流量系数 φ"); ax.grid(True, alpha=0.3)
            if ac in sub.columns and np.isfinite(sub[ac].to_numpy(dtype=float)).any():
                ax.scatter(sub["phi"], sub[ac], c="tab:blue", s=30, label="实测")
            if pc in sub.columns:
                ax.plot(sub["phi"], sub[pc], "r-", lw=2, marker="o", ms=3, label="预测")
            ax.legend(fontsize=8)
        self.pr_fig.suptitle("风扇 %s：预测 vs 实测" % fan)
        self.pr_fig.tight_layout(); self.pr_canvas.draw()

    def on_pred_export(self):
        if self.pred_out is None:
            QtWidgets.QMessageBox.warning(self, "无结果", "请先预测。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出预测结果", "prediction_compare.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            self.pred_out.to_csv(path, index=False, encoding="utf-8-sig")
            self.plog("已导出：%s" % path)
        except Exception:
            self.plog("导出失败：\n" + traceback.format_exc())

    # ----------------------------------------------------------------
    # 一键并入训练并重训
    # ----------------------------------------------------------------
    def on_merge_retrain(self):
        if self.train_df is None:
            QtWidgets.QMessageBox.warning(self, "无训练数据", "请先在『训练与验证』页加载训练数据。")
            return
        if self.pred_df is None:
            QtWidgets.QMessageBox.warning(self, "无新数据", "请先在本页加载要并入的新数据宽表。")
            return
        try:
            before = len(self.train_df)
            base = merge_tables(self.train_df, self.pred_df)
            self.train_det = detect_columns(base)
            self.train_df, self.train_feats = engineer_features(
                base, self.train_det["feature_cols"],
                nondim_insert=self.chk_nondim.isChecked(), drop_diameter=self.chk_drop_d.isChecked())
            self.plog("已并入新数据：训练集 %d → %d 台风扇。开始重训最终模型…" % (before, len(self.train_df)))
            self.tlog("『预测页』并入新数据：训练集 %d → %d 台风扇。" % (before, len(self.train_df)))
            self._train_report()
            self.btn_merge.setEnabled(False)
            self.retrain_worker = RetrainWorker(self.train_df, self.train_det, self.train_feats, self._device())
            self.retrain_worker.log.connect(self.plog)
            self.retrain_worker.finished_ok.connect(self._on_merge_done)
            self.retrain_worker.failed.connect(lambda tb: (self.btn_merge.setEnabled(True), self.plog("重训失败：\n" + tb)))
            self.retrain_worker.start()
        except Exception:
            self.plog("并入失败：\n" + traceback.format_exc())

    def _on_merge_done(self, bundle):
        self.bundle = bundle
        self.btn_merge.setEnabled(True)
        models = list(bundle.get("models", {}).keys())
        self.pr_model_lbl.setText("已更新：并入新数据后的最终模型（特征 %d，含 %s）" %
                                  (len(bundle.get("feature_cols", [])), ", ".join(models)))
        self.plog("已并入并重训完成，当前模型已更新。可点『保存当前模型』持久化，或重新预测。")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
