# Simulation data audit

## 数据集与配对

- `metal`: 200 cases，300–1500 K，求解器 receiver waveform 与 steady thermomechanical node field 齐备。
- `silicon`: 200 cases，300–1500 K，另有均匀化/宏观输入资产。
- `wumu`: 1000 cases，300–1500 K，两层结构、界面条件与六个仿真 monitor points。
- 同 case 配置引用同 case thermal CSV；1400/1400 个 case 通过 waveform↔field 目录/配置引用检查。

## Solver 与物理配置证据

- 热场文件来自 `thermomechanical_steady` 求解；配置含温度相关 `k`、`cp`、`rho`、弹性参数、Robin/Dirichlet/radiation 边界、接触热阻和 heat source。
- 超声配置读取该 case 的 thermal CSV，使用二维网格与显式时间推进；`wumu` 示例为 2.5 MHz、5 cycles、traction source、normal receiver，并包含界面刚度。
- `wumu` 网格配置记录 233,321 原节点（超声阶段界面复制后更多自由度）；处理后 fixed-node manifest 采样 10,000 点。
- solver 配置保存了旧机器路径和 executable 位置，但本机是否仍具备网格、求解器环境和许可尚未验证，不能声称可重跑。

## Waveform contract

- `metal`: 24,510–25,348 points，约 1.225–1.267 GHz，约 20 μs。
- `silicon`: 6,630–6,824 points，约 0.332–0.341 GHz，约 20 μs。
- `wumu`: 3,133–3,362 points，约 0.157–0.168 GHz，约 20 μs。
- 现有建库固定保留 native-rate 前 1,097 点，不插值；不同材料的实际物理时间窗口并不相同。

## Temperature field

字段包含 node coordinates、`T`、material IDs、热学/力学属性与界面信息。它们是 simulation full-field labels，不是实验 ground truth。
