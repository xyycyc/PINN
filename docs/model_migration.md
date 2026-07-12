# 模型版本与迁移说明

- 旧 checkpoint：没有 `model_kind`，继续由原 `AIReconstructionModel` 读取并输出 `(24,24)`，行为不变。
- 固定节点 checkpoint：`checkpoint_version=2`、`model_kind=direct_point_field`，并记录点数、schema、采样指纹、采样版本和训练集温度标准化参数。
- 固定节点 manifest 与旧 checkpoint 混用时明确报版本错误；点数、采样指纹或标准化参数不一致时拒绝推理，不做静默兼容。
- `build-db` 在检测到标准 `raw/calibration_sweep/worker_*/case_*` 布局后自动使用真实 case 管线，输出到 `database/data_process/<material>_case_temperature_field`。GUI 字段、默认值和操作顺序不变。
- 预测保留 `predictions.csv` 原始节点表；每行包含 `sample_id,node_id,x_m,y_m,temperature_k,target_temperature_k,material_id,interface_side`，展示图不会覆盖该表。
