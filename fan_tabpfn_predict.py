# -*- coding: utf-8 -*-
"""
冷却轴流风扇性能预测 - 预测对比界面（单文件版，PyQt5）

用途：
  加载训练阶段保存的模型（fan_tabpfn_models.joblib），读入新的验证/预测宽表
  （结构+试验参数，可含实测性能），用模型预测 ψst、λ 曲线并计算 η，
  把"预测 vs 实测"画在同一张图里对比（静压系数、功率系数、效率三张子图）。

运行：
  conda activate AIfan
  python fan_tabpfn_predict.py

说明：
  - 宽表格式需与训练时一致（同样的特征列、φ_xx 工况点列）。
  - 若宽表里含 ψst_xx / λ_xx（或 η_xx），会作为"实测"散点与预测曲线对比并给出误差指标；
    若不含性能列，则只画预测曲线（纯预测）。
  - η_max / φ_at_ηmax 由预测的 η 曲线求峰得到。
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
# 数据读取与列识别（与训练程序保持一致）
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


def engineer_features(df, feature_cols, nondim_insert=True):
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
    for c in feats:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, feats


def build_long(df, feature_cols, det):
    """展开成长表；feature_cols 只取实际存在的列，缺失列稍后由 reindex 补 NaN。"""
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


def predict_with_bundle(bundle, long_df):
    """用加载的模型预测 ψst、λ 曲线并计算 η（预测 & 实测）。"""
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


# =====================================================================
# 预测线程
# =====================================================================
class PredictWorker(QtCore.QThread):
    log = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, bundle, df, det, parent=None):
        super().__init__(parent)
        self.bundle = bundle
        self.df = df
        self.det = det

    def run(self):
        try:
            feats_used = self.bundle.get("feature_cols", [])
            long_df = build_long(self.df, feats_used, self.det)
            self.log.emit("长表构建完成：%d 行（%d 台风扇）" %
                          (len(long_df), long_df["__fan__"].nunique()))
            missing = [c for c in feats_used if c not in long_df.columns]
            if missing:
                self.log.emit("⚠️ 数据缺少 %d 个训练特征列（将按缺失值处理）：%s" %
                              (len(missing), ", ".join(map(str, missing))))
            out = predict_with_bundle(self.bundle, long_df)
            self.log.emit("预测完成。")
            self.finished_ok.emit(out)
        except Exception:
            self.failed.emit(traceback.format_exc())


# =====================================================================
# 主界面
# =====================================================================
class PredictWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("冷却轴流风扇性能预测 - 预测对比界面")
        self.resize(1320, 820)
        self.bundle = None
        self.df = None
        self.det = None
        self.pred = None
        self.current_path = None
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)

        gb_m = QtWidgets.QGroupBox("一、加载模型")
        v = QtWidgets.QVBoxLayout(gb_m)
        self.btn_model = QtWidgets.QPushButton("加载模型 (.joblib)")
        self.btn_model.clicked.connect(self.on_load_model)
        v.addWidget(self.btn_model)
        self.lbl_model = QtWidgets.QLabel("尚未加载模型")
        self.lbl_model.setWordWrap(True)
        v.addWidget(self.lbl_model)
        left.addWidget(gb_m)

        gb_d = QtWidgets.QGroupBox("二、加载验证/预测数据")
        v2 = QtWidgets.QVBoxLayout(gb_d)
        self.btn_data = QtWidgets.QPushButton("加载宽表 (Excel/CSV)")
        self.btn_data.clicked.connect(self.on_load_data)
        v2.addWidget(self.btn_data)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("工作表:"))
        self.cmb_sheet = QtWidgets.QComboBox()
        self.cmb_sheet.currentIndexChanged.connect(self.on_sheet_changed)
        h.addWidget(self.cmb_sheet, 1)
        v2.addLayout(h)
        self.lbl_data = QtWidgets.QLabel("尚未加载数据")
        self.lbl_data.setWordWrap(True)
        v2.addWidget(self.lbl_data)
        left.addWidget(gb_d)

        gb_r = QtWidgets.QGroupBox("三、预测")
        v3 = QtWidgets.QVBoxLayout(gb_r)
        self.btn_predict = QtWidgets.QPushButton("开始预测并对比")
        self.btn_predict.clicked.connect(self.on_predict)
        v3.addWidget(self.btn_predict)
        self.btn_export = QtWidgets.QPushButton("导出预测结果 CSV")
        self.btn_export.clicked.connect(self.on_export)
        v3.addWidget(self.btn_export)
        left.addWidget(gb_r)

        gb_p = QtWidgets.QGroupBox("四、查看风扇")
        v4 = QtWidgets.QVBoxLayout(gb_p)
        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(QtWidgets.QLabel("风扇:"))
        self.cmb_fan = QtWidgets.QComboBox()
        self.cmb_fan.currentIndexChanged.connect(self.on_plot)
        h2.addWidget(self.cmb_fan, 1)
        v4.addLayout(h2)
        left.addWidget(gb_p)
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

    def log(self, msg):
        self.txt_log.appendPlainText(str(msg))

    def on_load_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "加载模型", "", "Joblib (*.joblib)")
        if not path:
            return
        try:
            import joblib
            self.bundle = joblib.load(path)
            self.lbl_model.setText(path)
            feats = self.bundle.get("feature_cols", [])
            models = list(self.bundle.get("models", {}).keys())
            self.log("已加载模型：%s" % path)
            self.log("特征数: %d | 含模型: %s" % (len(feats), ", ".join(models) if models else "无"))
            if "psi" not in models or "lam" not in models:
                self.log("⚠️ 模型里缺少 psi 或 lam 曲线模型，无法预测完整曲线/效率。")
        except Exception:
            self.log("加载模型失败：\n" + traceback.format_exc())

    def on_load_data(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "加载宽表", "", "数据文件 (*.xlsx *.xls *.xlsm *.csv)")
        if not path:
            return
        self.current_path = path
        self.lbl_data.setText(path)
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
            self.df, _ = engineer_features(self.df, self.det["feature_cols"], nondim_insert=True)
            self.log("=" * 60)
            self.log("已加载数据 %d 行（每行一台风扇）" % len(self.df))
            self.log("工况点 φ: %d | ψst: %d | λ: %d" %
                     (len(self.det["phi_pts"]), len(self.det["psi_pts"]), len(self.det["lam_pts"])))
            has_perf = len(self.det["psi_pts"]) > 0 and len(self.det["lam_pts"]) > 0
            self.log("是否含实测性能(可对比): %s" % ("是" if has_perf else "否，仅做预测"))
        except Exception:
            self.log("读取失败：\n" + traceback.format_exc())

    def on_predict(self):
        if self.bundle is None:
            QtWidgets.QMessageBox.warning(self, "无模型", "请先加载 .joblib 模型。")
            return
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "无数据", "请先加载验证/预测宽表。")
            return
        self.btn_predict.setEnabled(False)
        self.worker = PredictWorker(self.bundle, self.df, self.det)
        self.worker.log.connect(self.log)
        self.worker.finished_ok.connect(self.on_predict_done)
        self.worker.failed.connect(self.on_predict_failed)
        self.worker.start()

    def on_predict_done(self, out):
        self.pred = out
        self.btn_predict.setEnabled(True)
        # 误差指标（仅当有实测时）
        if "psi" in out.columns and "pred_psi" in out.columns:
            self.log("ψst 对比: %s" % _fmt_metrics(regression_metrics(out["psi"], out["pred_psi"])))
        if "lam" in out.columns and "pred_lam" in out.columns:
            self.log("λ  对比: %s" % _fmt_metrics(regression_metrics(out["lam"], out["pred_lam"])))
        if "eta_actual" in out.columns and "eta_pred" in out.columns:
            self.log("η  对比: %s" % _fmt_metrics(regression_metrics(out["eta_actual"], out["eta_pred"])))
        self._scalar_peaks(out)
        self._populate_fan_combo()

    def on_predict_failed(self, tb):
        self.btn_predict.setEnabled(True)
        self.log("预测失败：\n" + tb)

    def _scalar_peaks(self, out):
        if "eta_pred" not in out.columns:
            return
        fans = list(pd.unique(out["__fan__"]))
        lines = []
        for f in fans:
            sub = out[out["__fan__"] == f]
            ep, pp = peak_from_curve(sub["phi"].to_numpy(), sub["eta_pred"].to_numpy())
            lines.append("  风扇 %s: 预测 η_max(分数)=%.4f, φ_at_ηmax=%.4f" % (str(f), ep, pp))
        if lines:
            self.log("由预测 η 曲线求峰得到（η_max 为分数，乘100为百分数）:")
            for ln in lines:
                self.log(ln)

    def _populate_fan_combo(self):
        self.cmb_fan.blockSignals(True)
        self.cmb_fan.clear()
        if self.pred is not None:
            for f in pd.unique(self.pred["__fan__"]):
                self.cmb_fan.addItem(str(f))
        self.cmb_fan.blockSignals(False)
        if self.cmb_fan.count() > 0:
            self.on_plot()

    def on_plot(self):
        if self.pred is None or self.cmb_fan.count() == 0:
            return
        fan_text = self.cmb_fan.currentText()
        sub = self.pred[self.pred["__fan__"].astype(str) == fan_text].sort_values("phi")
        if len(sub) == 0:
            return
        self.fig.clear()
        axes = self.fig.subplots(1, 3)
        cfg = [
            ("静压系数 ψst - φ", "psi", "pred_psi", "静压系数 ψst"),
            ("功率系数 λ - φ", "lam", "pred_lam", "功率系数 λ"),
            ("效率 η - φ", "eta_actual", "eta_pred", "效率 η"),
        ]
        for ax, (title, act_col, pred_col, ylab) in zip(axes, cfg):
            ax.set_title(title)
            ax.set_xlabel("流量系数 φ")
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.3)
            if act_col in sub.columns and np.isfinite(sub[act_col].to_numpy(dtype=float)).any():
                ax.scatter(sub["phi"], sub[act_col], c="tab:blue", s=30, label="实测")
            if pred_col in sub.columns:
                ax.plot(sub["phi"], sub[pred_col], "r-", lw=2, marker="o", ms=3, label="预测")
            ax.legend(fontsize=8)
        self.fig.suptitle("风扇 %s：预测 vs 实测" % fan_text)
        self.fig.tight_layout()
        self.canvas.draw()

    def on_export(self):
        if self.pred is None:
            QtWidgets.QMessageBox.warning(self, "无结果", "请先预测。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出预测结果", "prediction_compare.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            self.pred.to_csv(path, index=False, encoding="utf-8-sig")
            self.log("已导出：%s" % path)
        except Exception:
            self.log("导出失败：\n" + traceback.format_exc())


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PredictWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
