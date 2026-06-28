# 3_Data 数据目录

本目录用于存放项目中用于分析和观测对比的小型数据文件。当前保留的数据文件是 Gaia DR3 中 Coma Berenices 附近经过简单盒选得到的候选成员样本，用于论文中的观测参照图。

## 当前数据文件

### `coma_berenices_gaia_dr3_candidates.csv`

- 来源：Gaia DR3
- 对象：Coma Berenices 附近候选成员
- 查询中心：RA = 186.8110 deg，Dec = 25.8112 deg
- 搜索半径：5 deg
- 筛选条件：
  - `10.5 < parallax < 12.8` mas
  - `parallax_error < 0.5` mas
  - `-15 < pmra < 0` mas/yr
  - `-15 < pmdec < 5` mas/yr
  - `phot_g_mean_mag < 18`
  - `-0.2 < bp_rp < 4.0`
  - `ruwe < 1.4`
- 样本数量：92
- 主要字段：`source_id`、`ra`、`dec`、`parallax`、`parallax_error`、`pmra`、`pmra_error`、`pmdec`、`pmdec_error`、`phot_g_mean_mag`、`bp_rp`、`ruwe`
- 用途：生成 Gaia DR3 天球分布图、自行-视差图和颜色星等图。

该 CSV 是课程项目中的轻量候选成员盒选结果，不代表严格成员概率判定，也不构成新的 Coma Berenices 潮汐尾发现。

## 原始 Gaia 数据说明

本目录不保留大范围 Gaia raw sample，例如 `coma_berenices_gaia_dr3_raw.csv`。原始 Gaia 数据量较大，且不是论文正文图复现所必需；最终提交只需要保留轻量候选成员 CSV。

如需重新生成候选样本，可运行：

```bash
python 2_Code/experiments/run_gaia_coma_berenices_query.py
```

该脚本会优先尝试联网查询 Gaia Archive，并将候选成员保存为：

```text
3_Data/coma_berenices_gaia_dr3_candidates.csv
```

如果 Gaia 查询因网络或 Gaia Archive 服务状态失败，可以手动从 Gaia Archive 下载相同筛选条件的数据，并保存为上述 CSV 文件名。再次运行脚本时，程序会读取已有 CSV，进行本地筛选并重新生成以下观测参照图：

- `2_Code/outputs/coma_berenices_sky_candidates.png`
- `2_Code/outputs/coma_berenices_pm_parallax.png`
- `2_Code/outputs/coma_berenices_cmd.png`

## 提交注意事项

- `coma_berenices_gaia_dr3_candidates.csv` 文件较小，应保留在最终提交中。
- 大型 Gaia 原始查询结果不建议提交；如确需保留开发记录，应放入忽略目录或在 README 中说明生成方式。
- 本目录中的数据只用于计算物理期末项目的观测参照，不用于发表级别的成员概率分析。
