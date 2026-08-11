# Experimental data audit

## Physical experiment assets

### metal_10times_dlm

- 375 条数值文件名的原始示波器 trace，10 个 repetition folders。
- DLM3000、CH1、125,000 samples、2.5 GHz、0.4 ns、约 50 μs、单位 V；header 保存 acquisition date/time。
- 常规温度序列 30–210，每 5 单位约 10 repetitions；另有 12/22/23 等少量异常或补充文件。
- 文件名数值的单位和物理语义没有原始日志证明：**USER CONFIRMATION REQUIRED**。

### wumu_post0_waveforms

- 当前 213 条 processed waveform CSV，75 个 filename temperature groups、6 个 acquisition dates。
- 文件名明确写 `TxxC`，因此数值单位为 °C；CSV 为 32,500 samples、1.25 GHz、-1 至 25 μs，含 raw residual 与 filtered residual。
- 这不是原始示波器导出；当前分析使用 `amplitude_filtered_residual`。sensor、gain、excitation、coupling 和 residual 生成参数未完整保存。
- `TxxC` 是炉温、设定值、热电偶值、表面温度还是其它量，仍无证据：**USER CONFIRMATION REQUIRED**。

## Labels and fields

- 真正测量到的量：示波器 voltage（metal）及由实验波形派生的 residual waveform（wumu）。
- 可用弱标签：文件名数值/名义温度；其 measurement semantics 未确认。
- 未发现 thermocouple coordinates、IR/infrared image、calibrated thermal camera、实验二维温度矩阵或其它 measured field。
- `database/data_process/wumu_exp/fields/*.npy` 是代码生成的均匀 scalar proxy；legacy metal fields 是 zero arrays。两者都不是 ground truth。

**Experimental full-field ground truth: NO**

## Artifact defects

- 当前 `wumu_exp` manifest 的 `temperature_k` 与 `meta.temperature_c` 数值相同（例如 29 与 29），单位字段错误；当前 builder source 已改为 +273.15，表明 artifact 与代码版本不一致。
- manifest 251 records 与当前目录 213 CSV 不一致；后续研究必须从 raw/processed CSV 重建 clean metadata，不能继续沿用该 manifest 作为科研标签。
