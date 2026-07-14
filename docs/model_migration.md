# 模型版本与迁移说明

- 旧 checkpoint：没有 `model_kind`，继续由原 `AIReconstructionModel` 读取并输出 `(24,24)`，行为不变。
- 固定节点 checkpoint：当前写出 `checkpoint_version=3`、`model_kind=direct_point_field`，并记录点数、schema、采样指纹、采样版本、训练集温度标准化参数和 CNN/LSTM 分支权重。读取端兼容 v2；缺失的历史分支权重按旧模型等效值 `1.0/1.0` 恢复。
- 固定节点 manifest 与旧 checkpoint 混用时明确报版本错误；点数、采样指纹或标准化参数不一致时拒绝推理，不做静默兼容。
- `build-db` 在所选材料目录检测到嵌套 `worker_*/case_*` 或扁平 `case_*` 布局后使用真实 case 管线，输出到 `database/data_process/<dataset_label>_case_temperature_field`。样本材料路由字段取材料文件夹名；热力 CSV 的材料 ID/名称只描述内部组分。GUI 提供波形固定前缀点数（默认 1097）、验证集比例和三类清单文件名；固定节点流程必须启用划分，以便只用训练集拟合温度标准化参数。
- 多材料模式扫描所选上层目录的直接子文件夹，并生成 `material_collection.json`。分别训练的每个 checkpoint 都是标准 `direct_point_field` 完整模型；路由依据 `records[].material_key` 选择模型。项目清理把整个分材料路由组视为一个不可拆分的模型产物。
- 固定节点训练与增量训练目前仅支持 `normal`；旧规则网格模型继续支持 `normal/residual_pinn`。
- 预测保留 `predictions.csv` 原始节点表；每行包含 `sample_id,node_id,x_m,y_m,temperature_k,target_temperature_k,constituent_material_id,interface_side`，展示图不会覆盖该表。
