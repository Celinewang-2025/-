# 冷却轴流风扇性能预测 — TabPFN v2 训练界面

用 [TabPFN v2](https://github.com/PriorLabs/TabPFN) 在你的「风扇结构+性能宽表」上训练，预测无量纲性能曲线（静压系数 ψst、功率系数 λ）与关键标量（最大效率 η_max、最大效率点流量系数 φ_at_ηmax），并用**留一台风扇（leave-one-fan-out）**验证，对齐你 MATLAB 软件里的「定向验证」。

## 运行

```bash
conda activate AIfan
python fan_tabpfn/app.py
```

## 宽表约定（每台风扇一行）

| 分类 | 列 | 是否训练 |
|---|---|---|
| 命名 | `风扇ID`、`风扇名称`、`公司` | 否 |
| 基本+试验参数 | `风扇直径(mm)`、`叶片数`、`轮毂比`、`has_hub`、`is_ducted`、`ring_ratio`、`ring_max_ratio`、`strut_has_1/2`、`strut_r_max_pct(%)`、`strut_sub_pos_pct`、`is_large_chamber`、`insert_is_outer`、`insert_is_c_type`、`shroud_ratio`、`shroud_inner_ratio`、`shroud_outer_ratio`、`插入深度(mm)`、`密度(kg/m³)` | 是（特征） |
| 截面参数 | `chord_pct(%)@r*=...`、`安装角(°)@r*=...`、`LE_r_pct(%)`、`TE_r_pct(%)`、`最大厚度(%)`、`最大厚度位置(%)`、`最大弯度(%)`、`最大弯度位置(%)`、`axial_sweep_pct(%)`、`周向掠角度(°)`（各 5 个径向站位 r*=0.15/0.35/0.55/0.75/0.95） | 是（特征，已无量纲） |
| 性能 | `η_max`、`φ_at_ηmax`、`φ_01..N`、`ψst_01..N`、`λ_01..N`、`η_01..N` | 是（目标） |

列名按上述模式**自动识别**（`φ_\d+`、`ψst_\d+`、`λ_\d+`、`η_\d+`、`η_max`、`φ_at...`）；其余非命名列都当作特征。

## 建模设计

- **曲线模型（长表）**：把每台风扇展开为「每个工况点一行」，特征 = 结构参数 + 该点 `φ`，分别训练 **ψst** 和 **λ** 两个 `TabPFNRegressor`。
- **效率**：`η = ψst·φ/λ`（静效率定义恒等式）由预测的 ψst、λ 计算，不单独训练。
- **标量模型（每台风扇一行）**：训练 **η_max** 与 **φ_at_ηmax**。
- **验证**：留一台风扇（LOFO），输出每个目标的 RMSE / MAE / R² / MAPE，可选某台风扇画真实 vs 预测曲线。

## 关于无量纲化

- **统计标准化**：TabPFN 内部已处理，无需手动 `StandardScaler`。
- **物理无量纲化**：你的截面参数已无量纲；程序默认把 `插入深度 ÷ 直径`，可选删除绝对直径列。
  对 TabPFN（外推弱、靠相似度）这类处理收益较大。

## 文件

- `data_utils.py`：读取、列识别、特征工程、宽表→长表。
- `trainer.py`：TabPFN 训练、留一验证、指标、效率计算。
- `app.py`：PyQt5 界面。
