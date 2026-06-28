# 疏散星团潮汐尾形成的辛积分模拟及稳定性分析

本目录存放论文“疏散星团潮汐尾形成的辛积分模拟及稳定性分析——以 Coma Berenices 为观测参照”中所有数值模拟、数据分析和可视化代码。代码用于复现论文中的两体轨道积分器比较、相对能量误差测试、Velocity Verlet 步长收敛、二维软化 N 体星团演化、潮汐强度参数扫描，以及 Gaia DR3 候选成员观测参照图。

## 文件结构

当前 `2_Code/` 目录的主要结构如下：

```text
2_Code/
├── README.md                              # 本说明文件
├── requirements.txt                       # Python 环境依赖
├── src/                                   # 核心物理模型和数值算法函数
│   ├── __init__.py                        # 源码包标记文件
│   ├── integrators.py                     # Euler、RK4、Velocity Verlet 单步积分器
│   ├── two_body.py                        # 两体引力模型、轨道模拟和能量计算
│   └── nbody_cluster.py                   # 二维软化 N 体星团模型、潮汐场和星团积分器
├── experiments/                           # 可直接运行的实验脚本
│   ├── run_two_body_comparison.py         # 两体轨道积分器比较
│   ├── run_stepsize_convergence.py        # Velocity Verlet 步长收敛测试
│   ├── run_cluster_simulation.py          # 孤立星团能量测试和温和潮汐场演化
│   ├── run_tidal_strength_scan.py         # 单随机种子的潮汐强度扫描和尾部分类图
│   ├── run_tidal_strength_seed_scan.py    # 多随机种子的潮汐强度统计扫描
│   └── run_gaia_coma_berenices_query.py   # Gaia DR3 候选成员查询与绘图
└── outputs/                               # 脚本生成的图像、CSV 表格和运行结果
    ├── *.png                              # 论文图像和辅助图像
    ├── *.csv                              # 潮汐强度扫描结果表
    └── deprecated/                        # 旧版或废弃输出图，仅保留作为开发记录
```

运行 Python 脚本后，`src/__pycache__/` 和 `experiments/__pycache__/` 可能会自动出现。这些是 Python 缓存目录，不是项目源码的一部分，可以忽略。

## 核心模块说明

- `src/integrators.py`
  - 实现三种基本时间推进格式：显式 Euler、四阶 Runge--Kutta 和 Velocity Verlet。
  - 其中 Velocity Verlet 是本文长期引力系统模拟采用的主要辛积分格式。

- `src/two_body.py`
  - 定义固定中心质量下的两体引力加速度 `gravitational_acceleration`。
  - 定义两体机械能 `total_energy`。
  - 提供 `simulate_two_body`，用于比较不同积分器在轨道和能量误差上的表现。

- `src/nbody_cluster.py`
  - 生成二维疏散星团初始条件 `generate_cluster_initial_conditions`。
  - 计算软化自引力与线性外部潮汐场加速度 `compute_accelerations`。
  - 计算星团内部动能与软化自引力势能 `total_energy_cluster`。
  - 用 Velocity Verlet 推进 N 体星团 `velocity_verlet_cluster`，并保存位置、速度和能量历史。

## 实验脚本说明

- `experiments/run_two_body_comparison.py`
  - 比较 Euler、RK4 和 Velocity Verlet 在两体轨道问题中的轨道形状和能量误差。
  - 输出 `orbit_comparison.png` 和 `energy_error_comparison.png`。

- `experiments/run_stepsize_convergence.py`
  - 固定两体模型，改变 Velocity Verlet 时间步长，检验相对能量误差随步长减小的收敛趋势。
  - 输出 `verlet_stepsize_convergence.png`。

- `experiments/run_cluster_simulation.py`
  - 进行孤立星团能量守恒测试和温和潮汐场下的星团演化模拟。
  - 输出 `isolated_cluster_energy_error.png`、`cluster_tidal_evolution_comoving.png` 和尾部分类辅助图。

- `experiments/run_tidal_strength_scan.py`
  - 对多个潮汐强度进行单随机种子扫描。
  - 生成尾部分类统计、单次扫描曲线和 `tail_classification_zoom.png` 等图像。
  - 该脚本适合查看单次模拟的形态变化；论文中的统计结论以多随机种子扫描为主。

- `experiments/run_tidal_strength_seed_scan.py`
  - 对每个潮汐强度使用 `seed = 0, 1, ..., 9` 重复模拟。
  - 统计 $A_{\mathrm{tail}}$ 和 $f_{\mathrm{esc}}$ 的均值与标准差。
  - 输出 `tidal_strength_seed_scan_raw.csv`、`tidal_strength_seed_scan_summary.csv`、`A_tail_vs_tidal_strength_errorbar.png` 和 `escaped_fraction_vs_tidal_strength_errorbar.png`。

- `experiments/run_gaia_coma_berenices_query.py`
  - 使用 Gaia DR3 对 Coma Berenices 附近候选成员进行轻量盒选查询。
  - 若 `../3_Data/coma_berenices_gaia_dr3_candidates.csv` 已存在，则可读取已有 CSV 并直接绘图。
  - 输出天球分布图、自行-视差图和颜色星等图。

## 环境依赖

建议在项目根目录或 `2_Code/` 目录中安装依赖：

```bash
pip install -r 2_Code/requirements.txt
```

如果当前工作目录已经是 `2_Code/`，也可以运行：

```bash
pip install -r requirements.txt
```

主要依赖包括 `numpy`、`matplotlib`、`pandas`、`astropy` 和 `astroquery`。其中 `astroquery` 仅在需要联网查询 Gaia Archive 时使用。

## 推荐运行顺序

以下命令建议从 `2_Code/` 目录运行：

```bash
python experiments/run_two_body_comparison.py
python experiments/run_stepsize_convergence.py
python experiments/run_cluster_simulation.py
python experiments/run_tidal_strength_scan.py
python experiments/run_tidal_strength_seed_scan.py
python experiments/run_gaia_coma_berenices_query.py
```

也可以从项目根目录运行对应脚本：

```bash
python 2_Code/experiments/run_two_body_comparison.py
python 2_Code/experiments/run_stepsize_convergence.py
python 2_Code/experiments/run_cluster_simulation.py
python 2_Code/experiments/run_tidal_strength_scan.py
python 2_Code/experiments/run_tidal_strength_seed_scan.py
python 2_Code/experiments/run_gaia_coma_berenices_query.py
```

其中 `run_tidal_strength_seed_scan.py` 需要运行 40 次 N 体模拟，耗时明显长于其他脚本。`run_gaia_coma_berenices_query.py` 需要联网访问 Gaia Archive；若已有候选成员 CSV，可直接复用本地数据绘图。

## 输出结果

当前论文正文对应的主要输出文件位于 `outputs/`：

- `orbit_comparison.png`
- `energy_error_comparison.png`
- `verlet_stepsize_convergence.png`
- `isolated_cluster_energy_error.png`
- `cluster_tidal_evolution_comoving.png`
- `tail_classification_zoom.png`
- `escaped_fraction_vs_tidal_strength_errorbar.png`
- `A_tail_vs_tidal_strength_errorbar.png`
- `coma_berenices_sky_candidates.png`
- `coma_berenices_pm_parallax.png`
- `coma_berenices_cmd.png`

参数扫描还会生成以下 CSV 表格：

- `tidal_strength_seed_scan_raw.csv`
- `tidal_strength_seed_scan_summary.csv`

`outputs/deprecated/` 中包含早期开发阶段的旧版图像和单次扫描结果，例如 `cluster_initial.png`、`cluster_evolution.png`、`cluster_energy_error.png`、`A_tail_vs_tidal_strength.png`、`escaped_fraction_vs_tidal_strength.png`、`tail_classification_full.png` 和 `tidal_strength_scan.csv`。这些文件仅保留作为开发记录，不作为当前论文正文主图。

## Gaia DR3 数据说明

Gaia 脚本采用轻量候选成员查询，只查询 Coma Berenices 周围 5 度范围内满足视差、自行、星等、颜色和 RUWE 条件的候选样本。查询需要联网并依赖 Gaia Archive 的服务状态。

如果联网查询失败，可手动在 Gaia Archive 下载相同 ADQL 条件下的候选成员表，并保存为：

```text
../3_Data/coma_berenices_gaia_dr3_candidates.csv
```

再次运行 `run_gaia_coma_berenices_query.py` 时，脚本会读取已有 CSV，重新进行本地筛选并生成三张观测参照图。Gaia DR3 部分仅用于和简化 N 体模拟建立定性观测参照，不代表严格成员概率判定，也不宣称发现新的潮汐尾结构。

## 可复现性说明

- 两体轨道、步长收敛和星团演化的物理参数在脚本中显式给出。
- 星团初始条件生成函数支持 `random_seed`；潮汐强度多随机种子扫描固定使用 `seeds = range(10)`。
- 潮汐强度扫描的粒子数、时间步长、积分时长、保存间隔和软化长度均在脚本顶部集中定义。
- 图像和 CSV 表格由脚本自动生成到 `outputs/`。
- 论文中的主要数值结果和图表可通过上述运行顺序复现。

## 注意事项

- `outputs/` 中的当前主图对应论文正文；`outputs/deprecated/` 中是旧版结果，不作为论文正文图。
- 运行 `run_tidal_strength_seed_scan.py` 时间较长，建议在确认环境可用后再运行。
- 运行 Gaia 脚本可能受网络连接、Gaia Archive 服务状态和 astroquery 版本影响。
- 本项目的 Gaia 候选成员筛选是教学型盒选方法，不等价于严格天体物理成员概率分析。
