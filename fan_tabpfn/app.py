# -*- coding: utf-8 -*-
"""
冷却轴流风扇性能预测 - TabPFN v2 训练界面（PyQt5）

功能：
  1. 读取风扇结构+性能宽表（Excel/CSV）。
  2. 自动识别命名列、特征列与性能目标列。
  3. 物理特征工程（插入深度/直径无量纲化等）。
  4. 用 TabPFN v2 训练 ψst、λ 曲线模型与 η_max、φ_at_ηmax 标量模型。
  5. 留一台风扇（leave-one-fan-out）验证并输出 RMSE/MAE/R²/MAPE。
  6. 选择某台风扇绘制真实 vs 预测的 静压/功率/效率 曲线。
  7. 用全量数据训练最终模型并保存。

运行：
  conda activate AIfan
  python fan_tabpfn/app.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

import data_utils as du
import trainer as tr


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
        self.targets = targets  # dict of bools
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _cancelled(self):
        return self._cancel

    def run(self):
        try:
            results = {}
            long_df = du.build_long(self.df, self.feats, self.det)
            self.log.emit("长表构建完成：%d 行（%d 台风扇）" %
                          (len(long_df), long_df["__fan__"].nunique()))
            results["long_df"] = long_df

            if self.targets.get("psi"):
                self.log.emit("===== 训练/验证 静压系数 ψst =====")
                out = tr.lofo_curve(long_df, self.feats, "psi", self.device,
                                    log=self.log.emit, cancel=self._cancelled)
                results["psi"] = out
                m = tr.regression_metrics(out["psi"], out["pred"])
                results["psi_metrics"] = m
                self.log.emit("ψst 指标: %s" % _fmt_metrics(m))

            if self.targets.get("lam") and not self._cancel:
                self.log.emit("===== 训练/验证 功率系数 λ =====")
                out = tr.lofo_curve(long_df, self.feats, "lam", self.device,
                                    log=self.log.emit, cancel=self._cancelled)
                results["lam"] = out
                m = tr.regression_metrics(out["lam"], out["pred"])
                results["lam_metrics"] = m
                self.log.emit("λ 指标: %s" % _fmt_metrics(m))

            # 由 ψst、λ 预测合成 η，并评估
            if results.get("psi") is not None and results.get("lam") is not None and not self._cancel:
                p, l = results["psi"], results["lam"]
                merged = p.merge(l, on=["__fan__", "point", "phi"], suffixes=("_psi", "_lam"))
                eta_true = tr.compute_eta(merged["psi"], merged["phi"], merged["lam"])
                eta_pred = tr.compute_eta(merged["pred_psi"], merged["phi"], merged["pred_lam"])
                merged["eta_true"] = eta_true
                merged["eta_pred"] = eta_pred
                results["eta_curve"] = merged
                m = tr.regression_metrics(eta_true, eta_pred)
                results["eta_metrics"] = m
                self.log.emit("η(由ψst·φ/λ计算) 指标: %s" % _fmt_metrics(m))

            if self.targets.get("eta_max") and self.det.get("eta_max") and not self._cancel:
                self.log.emit("===== 训练/验证 最大效率 η_max =====")
                fan_ids, y, pred = tr.lofo_scalar(self.df, self.feats, self.det["eta_max"],
                                                  self.device, log=self.log.emit, cancel=self._cancelled)
                results["eta_max"] = dict(fan=fan_ids, y=y, pred=pred)
                m = tr.regression_metrics(y, pred)
                results["eta_max_metrics"] = m
                self.log.emit("η_max 指标: %s" % _fmt_metrics(m))

            if self.targets.get("phi_at") and self.det.get("phi_at_etamax") and not self._cancel:
                self.log.emit("===== 训练/验证 最大效率点流量系数 φ_at_ηmax =====")
                fan_ids, y, pred = tr.lofo_scalar(self.df, self.feats, self.det["phi_at_etamax"],
                                                  self.device, log=self.log.emit, cancel=self._cancelled)
                results["phi_at"] = dict(fan=fan_ids, y=y, pred=pred)
                m = tr.regression_metrics(y, pred)
                results["phi_at_metrics"] = m
                self.log.emit("φ_at_ηmax 指标: %s" % _fmt_metrics(m))

            self.finished_ok.emit(results)
        except Exception:
            self.failed.emit(traceback.format_exc())


def _fmt_metrics(m):
    return "n=%d  RMSE=%.4g  MAE=%.4g  R2=%.4f  MAPE=%.2f%%" % (
        m["n"], m["rmse"], m["mae"], m["r2"], m["mape"])


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

        self._build_ui()
        self._refresh_device_label()

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # 左侧控制面板
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)

        # 1. 数据
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

        # 2. 特征工程
        gb_fe = QtWidgets.QGroupBox("二、特征工程")
        v2 = QtWidgets.QVBoxLayout(gb_fe)
        self.chk_nondim = QtWidgets.QCheckBox("插入深度 ÷ 直径 做无量纲化")
        self.chk_nondim.setChecked(True)
        v2.addWidget(self.chk_nondim)
        self.chk_drop_d = QtWidgets.QCheckBox("删除绝对直径列（其余多为无量纲）")
        self.chk_drop_d.setChecked(False)
        v2.addWidget(self.chk_drop_d)
        left.addWidget(gb_fe)

        # 3. 训练目标
        gb_t = QtWidgets.QGroupBox("三、训练目标")
        v3 = QtWidgets.QVBoxLayout(gb_t)
        self.chk_psi = QtWidgets.QCheckBox("静压系数曲线 ψst"); self.chk_psi.setChecked(True)
        self.chk_lam = QtWidgets.QCheckBox("功率系数曲线 λ"); self.chk_lam.setChecked(True)
        self.chk_etamax = QtWidgets.QCheckBox("最大效率 η_max"); self.chk_etamax.setChecked(True)
        self.chk_phiat = QtWidgets.QCheckBox("最大效率点 φ_at_ηmax"); self.chk_phiat.setChecked(True)
        for w in (self.chk_psi, self.chk_lam, self.chk_etamax, self.chk_phiat):
            v3.addWidget(w)
        left.addWidget(gb_t)

        # 4. 计算设备
        gb_dev = QtWidgets.QGroupBox("四、计算设备")
        v4 = QtWidgets.QVBoxLayout(gb_dev)
        self.cmb_dev = QtWidgets.QComboBox()
        self.cmb_dev.addItems(["auto", "cuda", "cpu"])
        self.cmb_dev.currentIndexChanged.connect(self._refresh_device_label)
        v4.addWidget(self.cmb_dev)
        self.lbl_dev = QtWidgets.QLabel("")
        v4.addWidget(self.lbl_dev)
        left.addWidget(gb_dev)

        # 5. 运行
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

        # 6. 绘图选择
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

        # 右侧：日志 + 图
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
        dev = tr.detect_device(self.cmb_dev.currentText())
        self.lbl_dev.setText("实际使用设备: %s" % dev)

    def log(self, msg):
        self.txt_log.appendPlainText(str(msg))

    # ---------------- 数据加载 ----------------
    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择宽表", "", "数据文件 (*.xlsx *.xls *.xlsm *.csv)")
        if not path:
            return
        self.current_path = path
        self.lbl_file.setText(path)
        self.cmb_sheet.blockSignals(True)
        self.cmb_sheet.clear()
        for s in du.list_sheets(path):
            self.cmb_sheet.addItem(str(s))
        self.cmb_sheet.blockSignals(False)
        self._load_current()

    def on_sheet_changed(self, _):
        if getattr(self, "current_path", None):
            self._load_current()

    def _load_current(self):
        try:
            sheet = self.cmb_sheet.currentText()
            try:
                sheet_arg = int(sheet)
            except (ValueError, TypeError):
                sheet_arg = sheet
            self.df = du.load_table(self.current_path, sheet=sheet_arg)
            self.det = du.detect_columns(self.df)
            self.df, self.feats = du.engineer_features(
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
        self.log("η_max 列: %s | φ_at_ηmax 列: %s" %
                 (d["eta_max"], d["phi_at_etamax"]))
        self.log("特征列表: %s" % ", ".join(map(str, self.feats)))

    # ---------------- 训练 ----------------
    def on_train(self):
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "无数据", "请先加载宽表。")
            return
        # 每次训练前按当前特征工程选项重建特征
        self.df, self.feats = du.engineer_features(
            self.df, self.det["feature_cols"],
            nondim_insert=self.chk_nondim.isChecked(),
            drop_diameter=self.chk_drop_d.isChecked())
        targets = dict(
            psi=self.chk_psi.isChecked(),
            lam=self.chk_lam.isChecked(),
            eta_max=self.chk_etamax.isChecked(),
            phi_at=self.chk_phiat.isChecked(),
        )
        device = tr.detect_device(self.cmb_dev.currentText())
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

    # ---------------- 绘图 ----------------
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

    # ---------------- 保存最终模型 ----------------
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
        device = tr.detect_device(self.cmb_dev.currentText())
        self.log("用全量数据训练最终模型（设备 %s）…" % device)
        try:
            long_df = du.build_long(self.df, self.feats, self.det)
            bundle = {"feature_cols": self.feats, "detection": self.det, "models": {}}
            if self.chk_psi.isChecked():
                bundle["models"]["psi"] = tr.fit_curve_final(long_df, self.feats, "psi", device)
                self.log("ψst 最终模型完成")
            if self.chk_lam.isChecked():
                bundle["models"]["lam"] = tr.fit_curve_final(long_df, self.feats, "lam", device)
                self.log("λ 最终模型完成")
            if self.chk_etamax.isChecked() and self.det.get("eta_max"):
                bundle["models"]["eta_max"] = tr.fit_scalar_final(self.df, self.feats, self.det["eta_max"], device)
                self.log("η_max 最终模型完成")
            if self.chk_phiat.isChecked() and self.det.get("phi_at_etamax"):
                bundle["models"]["phi_at"] = tr.fit_scalar_final(self.df, self.feats, self.det["phi_at_etamax"], device)
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
