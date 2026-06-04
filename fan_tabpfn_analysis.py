# -*- coding: utf-8 -*-
"""
冷却轴流风扇 - 数据分析工具（单文件，PyQt5）

功能（读取原始宽表 Excel/CSV）：
  页1 数据与设置：加载训练宽表（必需）、新风扇宽表（外推风险用，可选），自动识别列。
  页2 PCA 降维分析（Q_a）：对"每类截面参数 × 5 个径向站位"做 PCA，给出可保留主成分数、
       方差解释率与物理含义解释，并汇总"50 维 → 多少维"的压缩结论。
  页3 敏感度分析（Q_b）：按"互信息 → 随机森林重要性 + 排列重要性 → ARD(GPy) →（可选）Sobol"
       依次分析各参数对所选性能目标的敏感程度，显示每种方法的大致耗时与详细解读。
  页4 外推风险（Q_d）：对每款新风扇给出"低/中/高"外推风险（最近邻距离分位 + 马氏距离 + 越界特征）。

运行：
  conda activate AIfan
  python fan_tabpfn_analysis.py
"""

import os
import sys
import re
import time
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


def to_numeric_df(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def impute_matrix(df, cols):
    """取出特征矩阵并按列中位数填补缺失（全空列填 0）。返回 (X, used_cols)。"""
    used = [c for c in cols if c in df.columns]
    X = df[used].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        if not np.isfinite(med):
            med = 0.0
        col[~np.isfinite(col)] = med
        X[:, j] = col
    return X, used


# =====================================================================
# 截面参数分组（按 "@r*=" 前缀分组）
# =====================================================================
def parse_section_groups(feature_cols):
    """把形如 'chord_pct(%)@r*=0.15' 的列按前缀分组，返回 {组名: [(径向位置, 列名), ...]}。"""
    groups = {}
    for c in feature_cols:
        m = re.match(r"^(.*?)@\s*r\*?\s*=\s*([0-9.]+)", str(c).strip())
        if m:
            key = m.group(1).strip()
            r = float(m.group(2))
            groups.setdefault(key, []).append((r, c))
    for k in groups:
        groups[k].sort(key=lambda t: t[0])
    # 只保留站位数>=3 的组（适合做径向 PCA）
    return {k: v for k, v in groups.items() if len(v) >= 3}


def interpret_pc(loadings, radii):
    """根据主成分载荷的符号模式给出物理含义的粗略解释。"""
    s = np.sign(loadings)
    s[s == 0] = 1
    changes = int(np.sum(s[1:] != s[:-1]))
    if np.all(loadings >= 0) or np.all(loadings <= 0):
        return "整体水平/大小（所有站位同向变化）"
    if changes == 1:
        return "根→尖 梯度（根部与叶尖反向，反映扭转/锥度/沿叶高的单调变化）"
    if changes == 2:
        return "中部 vs 两端（反映沿叶高的弯曲/驼峰形态）"
    return "高阶/复杂径向形态"


def run_pca_groups(df, groups, var_threshold=0.95):
    """对每个截面参数组做标准化 PCA，返回结果列表与汇总。"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    results = []
    total_stations = 0
    total_keep = 0
    for key, items in groups.items():
        cols = [c for _, c in items]
        radii = [r for r, _ in items]
        X, _ = impute_matrix(df, cols)
        if X.shape[0] < 3 or np.allclose(np.nanstd(X, axis=0), 0):
            continue
        Xs = StandardScaler().fit_transform(X)
        n_comp = min(Xs.shape[1], Xs.shape[0])
        pca = PCA(n_components=n_comp)
        pca.fit(Xs)
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)
        k95 = int(np.searchsorted(cum, var_threshold) + 1)
        k95 = max(1, min(k95, len(evr)))
        comps = pca.components_
        pc1 = comps[0]
        pc2 = comps[1] if comps.shape[0] > 1 else np.zeros_like(pc1)
        results.append(dict(
            group=key, n_stations=len(cols), radii=radii,
            evr=evr, cum=cum, k95=k95,
            pc1=pc1, pc2=pc2,
            pc1_interp=interpret_pc(pc1, radii),
            pc2_interp=interpret_pc(pc2, radii) if comps.shape[0] > 1 else "（无）",
        ))
        total_stations += len(cols)
        total_keep += k95
    return results, total_stations, total_keep


# =====================================================================
# 敏感度分析方法
# =====================================================================
def sens_mutual_info(X, y):
    from sklearn.feature_selection import mutual_info_regression
    return mutual_info_regression(X, y, random_state=42)


def sens_random_forest(X, y, n_estimators=400):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf


def sens_permutation(rf, X, y, n_repeats=10):
    from sklearn.inspection import permutation_importance
    r = permutation_importance(rf, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    return r.importances_mean


def sens_ard(X, y, max_iters=200):
    """GPy 的 ARD-Matern52 长度尺度；返回每个特征的重要性(1/长度尺度，归一化)。"""
    import GPy
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    ys = (y - np.mean(y)) / (np.std(y) + 1e-9)
    k = GPy.kern.Matern52(input_dim=Xs.shape[1], ARD=True)
    m = GPy.models.GPRegression(Xs, ys.reshape(-1, 1), k)
    m.optimize(max_iters=max_iters, messages=False)
    ls = np.asarray(m.kern.lengthscale.values, dtype=float).ravel()
    imp = 1.0 / (ls + 1e-9)
    return imp / (np.max(imp) + 1e-12)


def sens_sobol(X, y, names, N=256):
    """用随机森林作代理模型 + SALib Sobol；返回 (S1, ST)。需要 SALib。"""
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    rf = sens_random_forest(X, y)
    bounds = [[float(np.min(X[:, j])), float(np.max(X[:, j] + 1e-9))] for j in range(X.shape[1])]
    problem = {"num_vars": X.shape[1], "names": list(names), "bounds": bounds}
    param = saltelli.sample(problem, N, calc_second_order=False)
    Y = rf.predict(param)
    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
    return np.asarray(Si["S1"], dtype=float), np.asarray(Si["ST"], dtype=float)


def build_target_matrix(df, det, feature_cols, target):
    """根据目标返回 (X, y, names, note)。标量目标用每台风扇一行；曲线目标用长表(含 φ)。"""
    if target in ("η_max", "φ_at_ηmax"):
        col = det.get("eta_max") if target == "η_max" else det.get("phi_at_etamax")
        if col is None:
            return None
        sub = to_numeric_df(df, [col]).dropna(subset=[col])
        X, used = impute_matrix(sub, feature_cols)
        y = sub[col].to_numpy(dtype=float)
        return X, y, used, "标量目标：每台风扇 1 行（结构特征 → %s）" % target
    # 曲线目标 ψst / λ：构建长表，特征加入 φ
    phi_pts = det["phi_pts"]
    tgt_pts = det["psi_pts"] if target == "ψst" else det["lam_pts"]
    rows = []
    yv = []
    used = [c for c in feature_cols if c in df.columns]
    for _, r in df.iterrows():
        for k in range(min(len(phi_pts), len(tgt_pts))):
            phi = pd.to_numeric(r[phi_pts[k]], errors="coerce")
            yk = pd.to_numeric(r[tgt_pts[k]], errors="coerce")
            if not np.isfinite(phi) or not np.isfinite(yk):
                continue
            rows.append([pd.to_numeric(r[c], errors="coerce") for c in used] + [phi])
            yv.append(yk)
    X = np.array(rows, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        if not np.isfinite(med):
            med = 0.0
        col[~np.isfinite(col)] = med
        X[:, j] = col
    names = used + ["φ(流量系数)"]
    return X, np.array(yv, dtype=float), names, "曲线目标：每个工况点 1 行（结构特征 + φ → %s）" % target


# =====================================================================
# 外推风险
# =====================================================================
def compute_extrapolation_risk(train_df, new_df, feature_cols):
    from scipy.spatial.distance import cdist
    cols = [c for c in feature_cols if c in train_df.columns and c in new_df.columns]
    Xtr, _ = impute_matrix(train_df, cols)
    Xnew, _ = impute_matrix(new_df, cols)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-9
    Ztr = (Xtr - mu) / sd
    Znew = (Xnew - mu) / sd

    # 训练集内部最近邻距离分布
    Dtt = cdist(Ztr, Ztr)
    np.fill_diagonal(Dtt, np.inf)
    nn_tr = Dtt.min(axis=1)
    p50, p95 = np.percentile(nn_tr, [50, 95])

    # 新风扇到训练集最近邻
    Dnt = cdist(Znew, Ztr)
    nn_new = Dnt.min(axis=1)
    nn_idx = Dnt.argmin(axis=1)

    # 马氏距离（Ledoit-Wolf 收缩协方差，适合 D 接近/大于 N）
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(Ztr)
        md_tr = cov.mahalanobis(Ztr)
        md_new = cov.mahalanobis(Znew)
        md95 = np.percentile(md_tr, 95)
    except Exception:
        md_new = np.full(len(Znew), np.nan)
        md95 = np.nan

    lo = Xtr.min(axis=0)
    hi = Xtr.max(axis=0)
    train_ids = train_df["风扇ID"].to_numpy() if "风扇ID" in train_df.columns else np.arange(len(train_df))
    new_ids = new_df["风扇ID"].to_numpy() if "风扇ID" in new_df.columns else np.arange(len(new_df))

    rows = []
    for i in range(len(Znew)):
        oor_mask = (Xnew[i] < lo) | (Xnew[i] > hi)
        n_oor = int(np.sum(oor_mask))
        # 最越界的特征（按标准化偏离）
        z = np.abs(Znew[i])
        order = np.argsort(-z)
        top_feats = [cols[j] for j in order[:3]]
        # 风险等级：最近邻分位为主，越界比例与马氏距离辅助
        if nn_new[i] <= p50:
            risk = "低"
        elif nn_new[i] <= p95:
            risk = "中"
        else:
            risk = "高"
        if n_oor > max(1, int(0.05 * len(cols))):
            risk = "高" if risk != "低" else "中"
        if np.isfinite(md95) and md_new[i] > md95 and risk == "低":
            risk = "中"
        rows.append(dict(
            new_id=str(new_ids[i]),
            near_train=str(train_ids[nn_idx[i]]),
            nn_dist=float(nn_new[i]),
            nn_pct=float((nn_tr < nn_new[i]).mean() * 100.0),
            n_oor=n_oor,
            maha=float(md_new[i]) if np.isfinite(md_new[i]) else float("nan"),
            risk=risk,
            top_feats="; ".join(top_feats),
        ))
    summary = dict(p50=float(p50), p95=float(p95), n_features=len(cols), md95=float(md95))
    return rows, summary


# =====================================================================
# 后台线程
# =====================================================================
class SensWorker(QtCore.QThread):
    log = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, X, y, names, note, do_sobol, parent=None):
        super().__init__(parent)
        self.X, self.y, self.names, self.note = X, y, names, note
        self.do_sobol = do_sobol

    def run(self):
        try:
            res = {"names": self.names, "note": self.note}
            self.log.emit("数据规模：%d 行 × %d 特征。%s" % (self.X.shape[0], self.X.shape[1], self.note))

            self.log.emit("① 互信息（预计 1–5 秒）…")
            t = time.time()
            res["mi"] = sens_mutual_info(self.X, self.y)
            self.log.emit("   完成，用时 %.1f 秒。" % (time.time() - t))

            self.log.emit("② 随机森林重要性（预计 1–15 秒）…")
            t = time.time()
            rf = sens_random_forest(self.X, self.y)
            res["rf"] = rf.feature_importances_
            self.log.emit("   完成，用时 %.1f 秒。" % (time.time() - t))

            self.log.emit("③ 排列重要性（预计 5–40 秒）…")
            t = time.time()
            res["perm"] = sens_permutation(rf, self.X, self.y, n_repeats=10)
            self.log.emit("   完成，用时 %.1f 秒。" % (time.time() - t))

            self.log.emit("④ ARD 长度尺度 (GPy)（预计 10 秒–3 分钟，特征多/样本多更久）…")
            t = time.time()
            try:
                res["ard"] = sens_ard(self.X, self.y)
                self.log.emit("   完成，用时 %.1f 秒。" % (time.time() - t))
            except Exception as e:
                res["ard"] = None
                self.log.emit("   ARD 跳过（%s）。" % str(e))

            if self.do_sobol:
                self.log.emit("⑤ Sobol 全局敏感度 (SALib，代理模型)（预计 1–3 分钟）…")
                t = time.time()
                try:
                    s1, st = sens_sobol(self.X, self.y, self.names)
                    res["sobol_s1"] = s1
                    res["sobol_st"] = st
                    self.log.emit("   完成，用时 %.1f 秒。" % (time.time() - t))
                except Exception as e:
                    self.log.emit("   Sobol 跳过（%s）。如需启用请先 pip install SALib。" % str(e))
            self.finished_ok.emit(res)
        except Exception:
            self.failed.emit(traceback.format_exc())


# =====================================================================
# 主窗口
# =====================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("冷却轴流风扇 - 数据分析工具（PCA / 敏感度 / 外推风险）")
        self.resize(1360, 880)
        self.train_df = None
        self.train_det = None
        self.new_df = None
        self.new_det = None
        self.train_path = None
        self.new_path = None
        self.sens_worker = None

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)
        tabs.addTab(self._build_data_tab(), "1 数据与设置")
        tabs.addTab(self._build_pca_tab(), "2 PCA 降维分析")
        tabs.addTab(self._build_sens_tab(), "3 敏感度分析")
        tabs.addTab(self._build_risk_tab(), "4 外推风险")

    # ---------------- 页1 数据 ----------------
    def _build_data_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        row1 = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("加载训练宽表 (Excel/CSV)")
        b1.clicked.connect(self.on_load_train)
        row1.addWidget(b1)
        row1.addWidget(QtWidgets.QLabel("工作表:"))
        self.tr_sheet = QtWidgets.QComboBox()
        self.tr_sheet.currentIndexChanged.connect(lambda _: self._load_train_current())
        row1.addWidget(self.tr_sheet)
        row1.addStretch(1)
        v.addLayout(row1)
        self.tr_lbl = QtWidgets.QLabel("尚未加载训练数据")
        v.addWidget(self.tr_lbl)

        row2 = QtWidgets.QHBoxLayout()
        b2 = QtWidgets.QPushButton("加载新风扇宽表（外推风险用，可选）")
        b2.clicked.connect(self.on_load_new)
        row2.addWidget(b2)
        row2.addWidget(QtWidgets.QLabel("工作表:"))
        self.new_sheet = QtWidgets.QComboBox()
        self.new_sheet.currentIndexChanged.connect(lambda _: self._load_new_current())
        row2.addWidget(self.new_sheet)
        row2.addStretch(1)
        v.addLayout(row2)
        self.new_lbl = QtWidgets.QLabel("尚未加载新风扇数据")
        v.addWidget(self.new_lbl)

        self.data_log = QtWidgets.QPlainTextEdit()
        self.data_log.setReadOnly(True)
        v.addWidget(self.data_log, 1)
        return w

    # ---------------- 页2 PCA ----------------
    def _build_pca_tab(self):
        w = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(w)
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)
        gb = QtWidgets.QGroupBox("PCA 设置")
        vv = QtWidgets.QVBoxLayout(gb)
        vv.addWidget(QtWidgets.QLabel("方差保留阈值:"))
        self.pca_thr = QtWidgets.QDoubleSpinBox()
        self.pca_thr.setRange(0.80, 0.999)
        self.pca_thr.setSingleStep(0.01)
        self.pca_thr.setValue(0.95)
        vv.addWidget(self.pca_thr)
        b = QtWidgets.QPushButton("运行 PCA 分析")
        b.clicked.connect(self.on_run_pca)
        vv.addWidget(b)
        vv.addWidget(QtWidgets.QLabel("仅对'每类截面参数×径向站位'\n做 PCA（站位≥3）。"))
        left.addWidget(gb)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)
        self.pca_text = QtWidgets.QPlainTextEdit()
        self.pca_text.setReadOnly(True)
        right.addWidget(self.pca_text, 1)
        self.pca_fig = Figure(figsize=(9, 3.2))
        self.pca_canvas = FigureCanvas(self.pca_fig)
        right.addWidget(self.pca_canvas, 1)
        return w

    # ---------------- 页3 敏感度 ----------------
    def _build_sens_tab(self):
        w = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(w)
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)
        gb = QtWidgets.QGroupBox("敏感度设置")
        vv = QtWidgets.QVBoxLayout(gb)
        vv.addWidget(QtWidgets.QLabel("分析目标:"))
        self.sens_target = QtWidgets.QComboBox()
        self.sens_target.addItems(["η_max", "φ_at_ηmax", "ψst", "λ"])
        vv.addWidget(self.sens_target)
        self.chk_sobol = QtWidgets.QCheckBox("额外做 Sobol（需 SALib，较慢）")
        vv.addWidget(self.chk_sobol)
        self.btn_sens = QtWidgets.QPushButton("依次运行敏感度分析")
        self.btn_sens.clicked.connect(self.on_run_sens)
        vv.addWidget(self.btn_sens)
        vv.addWidget(QtWidgets.QLabel(
            "顺序：①互信息 ②随机森林 ③排列重要性\n④ARD(GPy) ⑤Sobol(可选)\n"
            "耗时：①1-5s ②1-15s ③5-40s\n④10s-3min ⑤1-3min"))
        left.addWidget(gb)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)
        self.sens_log = QtWidgets.QPlainTextEdit()
        self.sens_log.setReadOnly(True)
        right.addWidget(self.sens_log, 1)
        self.sens_fig = Figure(figsize=(9, 4))
        self.sens_canvas = FigureCanvas(self.sens_fig)
        right.addWidget(self.sens_canvas, 2)
        return w

    # ---------------- 页4 外推风险 ----------------
    def _build_risk_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        b = QtWidgets.QPushButton("评估新风扇外推风险")
        b.clicked.connect(self.on_run_risk)
        v.addWidget(b)
        self.risk_table = QtWidgets.QTableWidget()
        v.addWidget(self.risk_table, 1)
        self.risk_text = QtWidgets.QPlainTextEdit()
        self.risk_text.setReadOnly(True)
        self.risk_text.setMaximumHeight(220)
        v.addWidget(self.risk_text)
        return w

    # ---------------- 数据加载逻辑 ----------------
    def dlog(self, m):
        self.data_log.appendPlainText(str(m))

    def on_load_train(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "加载训练宽表", "", "数据文件 (*.xlsx *.xls *.xlsm *.csv)")
        if not path:
            return
        self.train_path = path
        self.tr_sheet.blockSignals(True)
        self.tr_sheet.clear()
        for s in list_sheets(path):
            self.tr_sheet.addItem(str(s))
        self.tr_sheet.blockSignals(False)
        self._load_train_current()

    def _sheet_arg(self, combo):
        s = combo.currentText()
        try:
            return int(s)
        except (ValueError, TypeError):
            return s

    def _load_train_current(self):
        if not self.train_path:
            return
        try:
            self.train_df = load_table(self.train_path, sheet=self._sheet_arg(self.tr_sheet))
            self.train_det = detect_columns(self.train_df)
            d = self.train_det
            self.tr_lbl.setText("训练数据：%d 台风扇，%d 个特征" % (len(self.train_df), len(d["feature_cols"])))
            groups = parse_section_groups(d["feature_cols"])
            self.dlog("=" * 60)
            self.dlog("训练数据 %d 台风扇 | 特征 %d 个" % (len(self.train_df), len(d["feature_cols"])))
            self.dlog("识别到截面参数组 %d 类（每类多个径向站位）：" % len(groups))
            for k, items in groups.items():
                self.dlog("  - %s：%d 个站位" % (k, len(items)))
            self.dlog("性能列 φ:%d ψst:%d λ:%d | η_max:%s φ_at:%s" %
                      (len(d["phi_pts"]), len(d["psi_pts"]), len(d["lam_pts"]), d["eta_max"], d["phi_at_etamax"]))
        except Exception:
            self.dlog("读取失败：\n" + traceback.format_exc())

    def on_load_new(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "加载新风扇宽表", "", "数据文件 (*.xlsx *.xls *.xlsm *.csv)")
        if not path:
            return
        self.new_path = path
        self.new_sheet.blockSignals(True)
        self.new_sheet.clear()
        for s in list_sheets(path):
            self.new_sheet.addItem(str(s))
        self.new_sheet.blockSignals(False)
        self._load_new_current()

    def _load_new_current(self):
        if not self.new_path:
            return
        try:
            self.new_df = load_table(self.new_path, sheet=self._sheet_arg(self.new_sheet))
            self.new_det = detect_columns(self.new_df)
            self.new_lbl.setText("新风扇数据：%d 台" % len(self.new_df))
            self.dlog("新风扇数据 %d 台，特征 %d 个。" % (len(self.new_df), len(self.new_det["feature_cols"])))
        except Exception:
            self.dlog("读取失败：\n" + traceback.format_exc())

    # ---------------- PCA ----------------
    def on_run_pca(self):
        if self.train_df is None:
            QtWidgets.QMessageBox.warning(self, "无数据", "请先在页1加载训练宽表。")
            return
        try:
            groups = parse_section_groups(self.train_det["feature_cols"])
            if not groups:
                self.pca_text.setPlainText("未识别到'@r*='形式的截面参数组，无法做径向 PCA。")
                return
            thr = float(self.pca_thr.value())
            results, total_st, total_keep = run_pca_groups(self.train_df, groups, var_threshold=thr)
            lines = []
            lines.append("=" * 70)
            lines.append("PCA 降维分析（方差保留阈值 = %.0f%%）" % (thr * 100))
            lines.append("说明：对每一类截面参数的多个径向站位做标准化 PCA。")
            lines.append("主成分含义：PC1 多为'整体水平'，PC2 多为'根→尖梯度'，更高阶为复杂形态。")
            lines.append("-" * 70)
            for r in results:
                evr_str = ", ".join("%.1f%%" % (x * 100) for x in r["evr"][:4])
                lines.append("【%s】%d 个站位 → 保留 %d 个主成分达 %.0f%%" %
                             (r["group"], r["n_stations"], r["k95"], thr * 100))
                lines.append("    各主成分方差解释率(前4): %s" % evr_str)
                lines.append("    PC1 含义：%s" % r["pc1_interp"])
                lines.append("    PC2 含义：%s" % r["pc2_interp"])
            lines.append("-" * 70)
            lines.append("【压缩汇总】截面参数 %d 维 → %d 维（保留 %.0f%% 方差），压缩约 %.0f%%。"
                         % (total_st, total_keep, thr * 100,
                            (1 - total_keep / max(total_st, 1)) * 100))
            lines.append("建议：每类截面参数用其前 %s 个主成分（或'根值/尖值/根尖差'）替代原始 5 个站位，"
                         % "1–2")
            lines.append("      可显著降低维度、抑制小样本过拟合，对外推更稳。")
            self.pca_text.setPlainText("\n".join(lines))
            self._plot_pca(results)
        except Exception:
            self.pca_text.setPlainText("PCA 失败：\n" + traceback.format_exc())

    def _plot_pca(self, results):
        self.pca_fig.clear()
        if not results:
            self.pca_canvas.draw()
            return
        ax = self.pca_fig.add_subplot(111)
        labels = [r["group"][:10] for r in results]
        k95 = [r["k95"] for r in results]
        nst = [r["n_stations"] for r in results]
        x = np.arange(len(results))
        ax.bar(x - 0.2, nst, width=0.4, label="原始站位数", color="tab:gray")
        ax.bar(x + 0.2, k95, width=0.4, label="保留主成分数", color="tab:blue")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("维度")
        ax.set_title("各截面参数组：原始站位数 vs 保留主成分数")
        ax.legend(fontsize=8)
        self.pca_fig.tight_layout()
        self.pca_canvas.draw()

    # ---------------- 敏感度 ----------------
    def slog(self, m):
        self.sens_log.appendPlainText(str(m))

    def on_run_sens(self):
        if self.train_df is None:
            QtWidgets.QMessageBox.warning(self, "无数据", "请先在页1加载训练宽表。")
            return
        target = self.sens_target.currentText()
        built = build_target_matrix(self.train_df, self.train_det, self.train_det["feature_cols"], target)
        if built is None:
            self.slog("数据中没有目标列 %s。" % target)
            return
        X, y, names, note = built
        if X.shape[0] < 5:
            self.slog("有效样本太少，无法分析。")
            return
        self.slog("=" * 60)
        self.slog("目标：%s" % target)
        self.btn_sens.setEnabled(False)
        self.sens_worker = SensWorker(X, y, names, note, self.chk_sobol.isChecked())
        self.sens_worker.log.connect(self.slog)
        self.sens_worker.finished_ok.connect(self.on_sens_done)
        self.sens_worker.failed.connect(lambda tb: (self.btn_sens.setEnabled(True), self.slog("失败：\n" + tb)))
        self.sens_worker.start()

    def on_sens_done(self, res):
        self.btn_sens.setEnabled(True)
        names = res["names"]

        def rank_text(title, vals, topn=15):
            if vals is None:
                return ["%s：（跳过）" % title]
            vals = np.asarray(vals, dtype=float)
            order = np.argsort(-vals)
            out = ["%s（前%d）：" % (title, topn)]
            for j in order[:topn]:
                out.append("    %-28s %.4g" % (str(names[j])[:28], vals[j]))
            return out

        lines = []
        lines.append("=" * 70)
        lines.append("敏感度分析结果解读")
        lines.append(res["note"])
        lines.append("-" * 70)
        lines.append("方法说明：")
        lines.append("  ① 互信息：单变量非线性关联（快，忽略交互作用，用于粗筛）。")
        lines.append("  ② 随机森林重要性：基于不纯度下降（考虑交互，但对相关特征有偏）。")
        lines.append("  ③ 排列重要性：打乱某列后误差上升幅度（模型无关，较可靠）。")
        lines.append("  ④ ARD：GP 反长度尺度，越大越重要（可解释，小样本下偏不稳）。")
        lines.append("  ⑤ Sobol：方差分解的一阶/总效应（含交互，工程意义最正）。")
        lines.append("-" * 70)
        lines += rank_text("① 互信息", res.get("mi"))
        lines += rank_text("② 随机森林重要性", res.get("rf"))
        lines += rank_text("③ 排列重要性", res.get("perm"))
        lines += rank_text("④ ARD 重要性", res.get("ard"))
        if "sobol_st" in res:
            lines += rank_text("⑤ Sobol 总效应 ST", res.get("sobol_st"))
        lines.append("-" * 70)
        lines.append("用法建议：综合 ③排列重要性 + ④ARD（+⑤Sobol）取交集靠前的特征作为关键参数；")
        lines.append("排名持续靠后的特征可考虑合并/删除以降维。注意小样本下排名有噪声，仅作指导。")
        self.sens_log.appendPlainText("\n".join(lines))
        self._plot_sens(res)

    def _plot_sens(self, res):
        self.sens_fig.clear()
        names = res["names"]
        # 优先画排列重要性，否则随机森林
        vals = res.get("perm")
        title = "排列重要性"
        if vals is None:
            vals = res.get("rf")
            title = "随机森林重要性"
        if vals is None:
            self.sens_canvas.draw()
            return
        vals = np.asarray(vals, dtype=float)
        order = np.argsort(-vals)[:15][::-1]
        ax = self.sens_fig.add_subplot(111)
        ax.barh([str(names[j])[:26] for j in order], vals[order], color="tab:blue")
        ax.set_title("%s（前15）" % title)
        ax.tick_params(axis="y", labelsize=7)
        self.sens_fig.tight_layout()
        self.sens_canvas.draw()

    # ---------------- 外推风险 ----------------
    def on_run_risk(self):
        if self.train_df is None:
            QtWidgets.QMessageBox.warning(self, "无训练数据", "请先在页1加载训练宽表。")
            return
        if self.new_df is None:
            QtWidgets.QMessageBox.warning(self, "无新数据", "请先在页1加载新风扇宽表。")
            return
        try:
            rows, summary = compute_extrapolation_risk(
                self.train_df, self.new_df, self.train_det["feature_cols"])
            headers = ["新风扇ID", "最近训练风扇", "最近距离", "距离分位(%)", "越界特征数", "马氏距离", "风险等级", "主要越界特征"]
            self.risk_table.clear()
            self.risk_table.setColumnCount(len(headers))
            self.risk_table.setHorizontalHeaderLabels(headers)
            self.risk_table.setRowCount(len(rows))
            for i, r in enumerate(rows):
                vals = [r["new_id"], r["near_train"], "%.3f" % r["nn_dist"], "%.0f" % r["nn_pct"],
                        str(r["n_oor"]), "%.1f" % r["maha"] if np.isfinite(r["maha"]) else "-",
                        r["risk"], r["top_feats"]]
                for j, val in enumerate(vals):
                    item = QtWidgets.QTableWidgetItem(str(val))
                    if j == 6:
                        color = {"低": QtCore.Qt.green, "中": QtCore.Qt.yellow, "高": QtCore.Qt.red}.get(r["risk"])
                        if color is not None:
                            item.setBackground(color)
                    self.risk_table.setItem(i, j, item)
            self.risk_table.resizeColumnsToContents()
            txt = []
            txt.append("外推风险判定说明：")
            txt.append("  基准：训练集内部最近邻距离 中位数=%.3f，95%%分位=%.3f（共 %d 个特征）。"
                       % (summary["p50"], summary["p95"], summary["n_features"]))
            txt.append("  规则：新风扇到训练集的最近邻距离 ≤ 中位数→低；≤95%%分位→中；>95%%分位→高。")
            txt.append("        若越界特征过多或马氏距离超 95%% 分位，风险等级上调。")
            txt.append("  含义：'高'风险=该风扇结构落在训练数据覆盖范围之外，预测基本属外推，误差可能很大，")
            txt.append("        建议优先补做实测并并入训练；'中'风险需谨慎参考；'低'风险预测相对可靠。")
            n_high = sum(1 for r in rows if r["risk"] == "高")
            n_mid = sum(1 for r in rows if r["risk"] == "中")
            txt.append("  本批：高风险 %d 台，中风险 %d 台，低风险 %d 台。"
                       % (n_high, n_mid, len(rows) - n_high - n_mid))
            self.risk_text.setPlainText("\n".join(txt))
        except Exception:
            self.risk_text.setPlainText("外推风险评估失败：\n" + traceback.format_exc())


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
