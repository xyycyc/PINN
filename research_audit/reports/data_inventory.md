# Data inventory

本表以原始文件树为事实源；路径均已逻辑化。磁盘量与文件数来自 `statistics/data_inventory.csv`。

| Dataset | Origin | Cases/traces | Files | Size (GB) | Waveforms | Full field | Temperature |
|---|---:|---:|---:|---:|---:|---:|---|
| metal | simulation | 200 | 6400 | 17.153 | 200 | True | 300–1500 K |
| silicon | simulation | 200 | 6007 | 4.004 | 200 | True | 300–1500 K |
| wumu | simulation | 1000 | 16883 | 215.035 | 1000 | True | 300–1500 K |
| metal_10times_dlm | physical_experiment | 375 | 391 | 1.752 | 375 | False | 12.0–210.0 unknown |
| wumu_post0_waveforms | processed_experiment | 213 | 215 | 0.540 | 213 | False | 29.0–169.0 degC |

## 关键资产判断

- `metal`、`silicon`、`wumu` 是求解器生成的 simulation case，不是物理实验；三者合计 1,400 cases。
- `REPOSITORY_ROOT/database/raw/10times` 是 10 批 DLM3000 示波器导出；有效数值文件名波形 375 条。
- `REPOSITORY_ROOT/database/raw/waveforms_by_file` 是已经做过 residual/filter/post-zero 处理的实验波形；当前原始目录有 213 条。
- 原始 `wumu` 仿真占约 215.0 GB，是总数据量主体；扫描脚本未加载 VTU 或完整场到内存。
- 现有 `wumu_exp` manifest 有 251 records，但当前 post-zero 目录只有 213 CSV，且温度上限分别为 177 °C 与 169 °C；manifest 不能代替原始目录事实。
