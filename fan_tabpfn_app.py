# -*- coding: utf-8 -*-
"""
冷却轴流风扇性能预测 - TabPFN v2 训练界面（单文件版，PyQt5）

功能：
  1. 读取风扇结构+性能宽表（Excel/CSV），每台风扇一行。
  2. 自动识别命名列、特征列与性能目标列。
  3. 物理特征工程（插入深度/直径无量纲化等）。
  4. 用 TabPFN v2 训练 ψst、λ 曲线模型与 η_max、φ_at_ηmax 标量模型。
  5. 留一台风扇（leave-one-fan-out）验证并输出 RMSE/MAE/R²/MAPE。
  6. 选择某台风扇绘制真实 vs 预测的 静压/功率/效率 曲线。
  7. 用全量数据训练最终模型并保存。

运行：
  conda activate AIfan
  python fan_tabpfn_app.py
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

# 中文字体，避免坐标轴中文乱码
try:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


# =====================================================================
# 一、数据读取与列识别
# =====================================================================
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
      - 条件缺失的 one-hot 关联参数（如无环形时的 ring_ratio）以 0 视为"无该结构"。
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


# =====================================================================
# 二、TabPFN 训练与留一验证
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
    tr_df = long_df.dropna(subset=[target])
    reg = make_regressor(device)
    reg.fit(tr_df[feats].to_numpy(dtype=float), tr_df[target].to_numpy(dtype=float))
    return reg


def fit_scalar_final(df, feature_cols, target_col, device):
    sub = df.dropna(subset=[target_col])
    reg = make_regressor(device)
    reg.fit(sub[feature_cols].to_numpy(dtype=float), sub[target_col].to_numpy(dtype=float))
    return reg


def _fmt_metrics(m):
    return "n=%d  RMSE=%.4g  MAE=%.4g  R2=%.4f  MAPE=%.2f%%" % (
        m["n"], m["rmse"], m["mae"], m["r2"], m["mape"])


# =====================================================================
# 三、后台训练线程
# =====================================================================
class TrainWorker(QtCore.QThread):
    """后台训练线程，避免界面卡死。"""
    log = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, df, det, feats, device, targets, parent=None):
        super().__init__(parent)
        self.df = df
        self.det = det
        self.feats = feats
        self.device = device
        self.targets = targets
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _cancelled(self):
        return self._cancel

    def run(self):
        try:
            results = {}
            long_df = build_long(self.df, self.feats, self.det)
            self.log.emit("长表构建完成：%d 行（%d 台风扇）" %
                          (len(long_df), long_df["__fan__"].nunique()))
            results["long_df"] = long_df

            if self.targets.get("psi"):
                self.log.emit("===== 训练/验证 静压系数 ψst =====")
                out = lofo_curve(long_df, self.feats, "psi", self.device,
                                 log=self.log.emit, cancel=self._cancelled)
                results["psi"] = out
                m = regression_metrics(out["psi"], out["pred"])
                results["psi_metrics"] = m
                self.log.emit("ψst 指标: %s" % _fmt_metrics(m))

            if self.targets.get("lam") and not self._cancel:
                self.log.emit("===== 训练/验证 功率系数 λ =====")
                out = lofo_curve(long_df, self.feats, "lam", self.device,
                                 log=self.log.emit, cancel=self._cancelled)
                results["lam"] = out
                m = regression_metrics(out["lam"], out["pred"])
                results["lam_metrics"] = m
                self.log.emit("λ 指标: %s" % _fmt_metrics(m))

            if results.get("psi") is not None and results.get("lam") is not None and not self._cancel:
                p, l = results["psi"], results["lam"]
                merged = p.merge(l, on=["__fan__", "point", "phi"], suffixes=("_psi", "_lam"))
                eta_true = compute_eta(merged["psi"], merged["phi"], merged["lam"])
                eta_pred = compute_eta(merged["pred_psi"], merged["phi"], merged["pred_lam"])
                merged["eta_true"] = eta_true
                merged["eta_pred"] = eta_pred
                results["eta_curve"] = merged
                m = regression_metrics(eta_true, eta_pred)
                results["eta_metrics"] = m
                self.log.emit("η(由ψst·φ/λ计算) 指标: %s" % _fmt_metrics(m))

            if self.targets.get("eta_max") and self.det.get("eta_max") and not self._cancel:
                self.log.emit("===== 训练/验证 最大效率 η_max =====")
                fan_ids, y, pred = lofo_scalar(self.df, self.feats, self.det["eta_max"],
                                               self.device, log=self.log.emit, cancel=self._cancelled)
                results["eta_max"] = dict(fan=fan_ids, y=y, pred=pred)
                m = regression_metrics(y, pred)
                results["eta_max_metrics"] = m
                self.log.emit("η_max 指标: %s" % _fmt_metrics(m))

            if self.targets.get("phi_at") and self.det.get("phi_at_etamax") and not self._cancel:
                self.log.emit("===== 训练/验证 最大效率点流量系数 φ_at_ηmax =====")
                fan_ids, y, pred = lofo_scalar(self.df, self.feats, self.det["phi_at_etamax"],
                                               self.device, log=self.log.emit, cancel=self._cancelled)
                results["phi_at"] = dict(fan=fan_ids, y=y, pred=pred)
                m = regression_metrics(y, pred)
                results["phi_at_metrics"] = m
                self.log.emit("φ_at_ηmax 指标: %s" % _fmt_metrics(m))

            self.finished_ok.emit(results)
        except Exception:
            self.failed.emit(traceback.format_exc())


# =====================================================================
# 四、主界面
# =====================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("冷却轴流风扇性能预测 - TabPFN v2 训练界面")
        self.resize(1320, 820)

        self.df = None
        self.det = None
        self.feats = None
        self.results = None
        self.worker = None
        self.current_path = None

        self._build_ui()
        self._refresh_device_label()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)

        gb_data = QtWidgets.QGroupBox("一、数据")
        v = QtWidgets.QVBoxLayout(gb_data)
        self.btn_open = QtWidgets.QPushButton("选择 Excel/CSV 宽表")
        self.btn_open.clicked.connect(self.on_open)
        v.addWidget(self.btn_open)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("工作表:"))
        self.cmb_sheet = QtWidgets.QComboBox()
        self.cmb_sheet.currentIndexChanged.connect(self.on_sheet_changed)
        h.addWidget(self.cmb_sheet, 1)
        v.addLayout(h)
        self.lbl_file = QtWidgets.QLabel("尚未加载数据")
        self.lbl_file.setWordWrap(True)
        v.addWidget(self.lbl_file)
        left.addWidget(gb_data)

        gb_fe = QtWidgets.QGroupBox("二、特征工程")
        v2 = QtWidgets.QVBoxLayout(gb_fe)
        self.chk_nondim = QtWidgets.QCheckBox("插入深度 ÷ 直径 做无量纲化")
        self.chk_nondim.setChecked(True)
        v2.addWidget(self.chk_nondim)
        self.chk_drop_d = QtWidgets.QCheckBox("删除绝对直径列（其余多为无量纲）")
        self.chk_drop_d.setChecked(False)
        v2.addWidget(self.chk_drop_d)
        left.addWidget(gb_fe)

        gb_t = QtWidgets.QGroupBox("三、训练目标")
        v3 = QtWidgets.QVBoxLayout(gb_t)
        self.chk_psi = QtWidgets.QCheckBox("静压系数曲线 ψst"); self.chk_psi.setChecked(True)
        self.chk_lam = QtWidgets.QCheckBox("功率系数曲线 λ"); self.chk_lam.setChecked(True)
        self.chk_etamax = QtWidgets.QCheckBox("最大效率 η_max"); self.chk_etamax.setChecked(True)
        self.chk_phiat = QtWidgets.QCheckBox("最大效率点 φ_at_ηmax"); self.chk_phiat.setChecked(True)
        for w in (self.chk_psi, self.chk_lam, self.chk_etamax, self.chk_phiat):
            v3.addWidget(w)
        left.addWidget(gb_t)

        gb_dev = QtWidgets.QGroupBox("四、计算设备")
        v4 = QtWidgets.QVBoxLayout(gb_dev)
        self.cmb_dev = QtWidgets.QComboBox()
        self.cmb_dev.addItems(["auto", "cuda", "cpu"])
        self.cmb_dev.currentIndexChanged.connect(self._refresh_device_label)
        v4.addWidget(self.cmb_dev)
        self.lbl_dev = QtWidgets.QLabel("")
        v4.addWidget(self.lbl_dev)
        left.addWidget(gb_dev)

        gb_run = QtWidgets.QGroupBox("五、运行")
        v5 = QtWidgets.QVBoxLayout(gb_run)
        self.btn_train = QtWidgets.QPushButton("开始训练 + 留一验证")
        self.btn_train.clicked.connect(self.on_train)
        v5.addWidget(self.btn_train)
        self.btn_cancel = QtWidgets.QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.on_cancel)
        v5.addWidget(self.btn_cancel)
        self.btn_save = QtWidgets.QPushButton("用全量数据训练并保存最终模型")
        self.btn_save.clicked.connect(self.on_save_models)
        v5.addWidget(self.btn_save)
        left.addWidget(gb_run)

        gb_plot = QtWidgets.QGroupBox("六、曲线查看")
        v6 = QtWidgets.QVBoxLayout(gb_plot)
        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(QtWidgets.QLabel("风扇:"))
        self.cmb_fan = QtWidgets.QComboBox()
        self.cmb_fan.currentIndexChanged.connect(self.on_plot)
        h2.addWidget(self.cmb_fan, 1)
        v6.addLayout(h2)
        left.addWidget(gb_plot)

        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)

        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(5000)
        right.addWidget(self.txt_log, 1)

        self.fig = Figure(figsize=(9, 4))
        self.canvas = FigureCanvas(self.fig)
        right.addWidget(self.canvas, 2)

    def _refresh_device_label(self):
        dev = detect_device(self.cmb_dev.currentText())
        self.lbl_dev.setText("实际使用设备: %s" % dev)

    def log(self, msg):
        self.txt_log.appendPlainText(str(msg))

    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择宽表", "", "数据文件 (*.xlsx *.xls *.xlsm *.csv)")
        if not path:
            return
        self.current_path = path
        self.lbl_file.setText(path)
        self.cmb_sheet.blockSignals(True)
        self.cmb_sheet.clear()
        for s in list_sheets(path):
            self.cmb_sheet.addItem(str(s))
        self.cmb_sheet.blockSignals(False)
        self._load_current()

    def on_sheet_changed(self, _):
        if self.current_path:
            self._load_current()

    def _load_current(self):
        try:
            sheet = self.cmb_sheet.currentText()
            try:
                sheet_arg = int(sheet)
            except (ValueError, TypeError):
                sheet_arg = sheet
            self.df = load_table(self.current_path, sheet=sheet_arg)
            self.det = detect_columns(self.df)
            self.df, self.feats = engineer_features(
                self.df, self.det["feature_cols"],
                nondim_insert=self.chk_nondim.isChecked(),
                drop_diameter=self.chk_drop_d.isChecked())
            self._report_detection()
        except Exception:
            self.log("读取失败：\n" + traceback.format_exc())

    def _report_detection(self):
        d = self.det
        self.log("=" * 60)
        self.log("已加载 %d 行（每行一台风扇）" % len(self.df))
        self.log("命名列(不训练): %s" % ", ".join(d["id_cols"]))
        self.log("特征列数: %d" % len(self.feats))
        self.log("工况点 φ: %d 个 | ψst: %d | λ: %d | η: %d" %
                 (len(d["phi_pts"]), len(d["psi_pts"]), len(d["lam_pts"]), len(d["eta_pts"])))
        self.log("η_max 列: %s | φ_at_ηmax 列: %s" % (d["eta_max"], d["phi_at_etamax"]))
        self.log("特征列表: %s" % ", ".join(map(str, self.feats)))

    def on_train(self):
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "无数据", "请先加载宽表。")
            return
        self.df, self.feats = engineer_features(
            self.df, self.det["feature_cols"],
            nondim_insert=self.chk_nondim.isChecked(),
            drop_diameter=self.chk_drop_d.isChecked())
        targets = dict(
            psi=self.chk_psi.isChecked(),
            lam=self.chk_lam.isChecked(),
            eta_max=self.chk_etamax.isChecked(),
            phi_at=self.chk_phiat.isChecked(),
        )
        device = detect_device(self.cmb_dev.currentText())
        self.log("使用设备: %s" % device)
        self.btn_train.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.worker = TrainWorker(self.df, self.det, self.feats, device, targets)
        self.worker.log.connect(self.log)
        self.worker.finished_ok.connect(self.on_train_done)
        self.worker.failed.connect(self.on_train_failed)
        self.worker.start()

    def on_cancel(self):
        if self.worker is not None:
            self.worker.cancel()
            self.log("已请求取消，当前折结束后停止…")

    def on_train_done(self, results):
        self.results = results
        self.btn_train.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.log("=" * 60)
        self.log("训练与留一验证完成。指标汇总：")
        for key, label in (("psi_metrics", "静压系数 ψst"), ("lam_metrics", "功率系数 λ"),
                           ("eta_metrics", "效率 η(计算)"), ("eta_max_metrics", "最大效率 η_max"),
                           ("phi_at_metrics", "最大效率点 φ_at_ηmax")):
            if key in results:
                self.log("  %-16s %s" % (label, _fmt_metrics(results[key])))
        self._populate_fan_combo()

    def on_train_failed(self, tb):
        self.btn_train.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.log("训练失败：\n" + tb)

    def _populate_fan_combo(self):
        self.cmb_fan.blockSignals(True)
        self.cmb_fan.clear()
        if self.results and "long_df" in self.results:
            fans = list(pd.unique(self.results["long_df"]["__fan__"]))
            for f in fans:
                self.cmb_fan.addItem(str(f))
        self.cmb_fan.blockSignals(False)
        if self.cmb_fan.count() > 0:
            self.on_plot()

    def on_plot(self):
        if not self.results or self.cmb_fan.count() == 0:
            return
        fan_text = self.cmb_fan.currentText()
        self.fig.clear()
        axes = self.fig.subplots(1, 3)
        titles = ["静压系数 ψst - φ", "功率系数 λ - φ", "效率 η - φ"]
        for ax, t in zip(axes, titles):
            ax.set_title(t)
            ax.set_xlabel("流量系数 φ")
            ax.grid(True, alpha=0.3)

        psi = self.results.get("psi")
        lam = self.results.get("lam")
        eta = self.results.get("eta_curve")

        def fan_rows(df):
            if df is None:
                return None
            sel = df[df["__fan__"].astype(str) == fan_text].sort_values("phi")
            return sel if len(sel) else None

        sp = fan_rows(psi)
        if sp is not None:
            axes[0].scatter(sp["phi"], sp["psi"], c="tab:blue", s=28, label="真实")
            axes[0].plot(sp["phi"], sp["pred"], "r-", lw=2, label="预测")
            axes[0].legend(fontsize=8)

        sl = fan_rows(lam)
        if sl is not None:
            axes[1].scatter(sl["phi"], sl["lam"], c="tab:blue", s=28, label="真实")
            axes[1].plot(sl["phi"], sl["pred"], "r-", lw=2, label="预测")
            axes[1].legend(fontsize=8)

        if eta is not None:
            se = eta[eta["__fan__"].astype(str) == fan_text].sort_values("phi")
            if len(se):
                axes[2].scatter(se["phi"], se["eta_true"], c="tab:blue", s=28, label="真实(计算)")
                axes[2].plot(se["phi"], se["eta_pred"], "r-", lw=2, label="预测(计算)")
                axes[2].legend(fontsize=8)

        self.fig.suptitle("风扇 %s 留一验证曲线" % fan_text)
        self.fig.tight_layout()
        self.canvas.draw()

    def on_save_models(self):
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "无数据", "请先加载宽表。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存最终模型", "fan_tabpfn_models.joblib", "Joblib (*.joblib)")
        if not path:
            return
        try:
            import joblib
        except Exception:
            self.log("缺少 joblib，请先 pip install joblib")
            return
        device = detect_device(self.cmb_dev.currentText())
        self.log("用全量数据训练最终模型（设备 %s）…" % device)
        try:
            long_df = build_long(self.df, self.feats, self.det)
            bundle = {"feature_cols": self.feats, "detection": self.det, "models": {}}
            if self.chk_psi.isChecked():
                bundle["models"]["psi"] = fit_curve_final(long_df, self.feats, "psi", device)
                self.log("ψst 最终模型完成")
            if self.chk_lam.isChecked():
                bundle["models"]["lam"] = fit_curve_final(long_df, self.feats, "lam", device)
                self.log("λ 最终模型完成")
            if self.chk_etamax.isChecked() and self.det.get("eta_max"):
                bundle["models"]["eta_max"] = fit_scalar_final(self.df, self.feats, self.det["eta_max"], device)
                self.log("η_max 最终模型完成")
            if self.chk_phiat.isChecked() and self.det.get("phi_at_etamax"):
                bundle["models"]["phi_at"] = fit_scalar_final(self.df, self.feats, self.det["phi_at_etamax"], device)
                self.log("φ_at_ηmax 最终模型完成")
            joblib.dump(bundle, path)
            self.log("最终模型已保存：%s" % path)
        except Exception:
            self.log("保存失败：\n" + traceback.format_exc())


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
