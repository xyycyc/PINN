# 模型版本与迁移说明

- 旧 checkpoint：没有 `model_kind`，继续由原 `AIReconstructionModel` 读取并输出 `(24,24)`，行为不变。
- 固定节点 checkpoint：当前写出 `checkpoint_version=3`、`model_kind=direct_point_field`，并记录点数、schema、采样指纹、采样版本、训练集温度标准化参数和 CNN/LSTM 分支权重。读取端兼容 v2；缺失的历史分支权重按旧模型等效值 `1.0/1.0` 恢复。
- 固定节点 manifest 与旧 checkpoint 混用时明确报版本错误；点数、采样指纹或标准化参数不一致时拒绝推理，不做静默兼容。
- `build-db` 在所选材料目录检测到嵌套 `worker_*/case_*` 或扁平 `case_*` 布局后使用真实 case 管线，输出到 `database/data_process/<dataset_label>_case_temperature_field`。样本材料路由字段取材料文件夹名；热力 CSV 的材料 ID/名称只描述内部组分。GUI 提供波形固定前缀点数（默认 1097）、验证集比例和三类清单文件名；固定节点流程必须启用划分，以便只用训练集拟合温度标准化参数。
- 多材料模式扫描所选上层目录的直接子文件夹，并生成 `material_collection.json`。分别训练的每个 checkpoint 都是标准 `direct_point_field` 完整模型；路由依据 `records[].material_key` 选择模型。项目清理把整个分材料路由组视为一个不可拆分的模型产物。
- 固定节点训练与增量训练均支持 `normal/residual_pinn`：`normal` 仅温度监督，`residual_pinn` 叠加同材料邻域平滑及离散拉普拉斯损失。旧规则网格模型保留自己的损失定义，不与固定节点损失混用。
- 预测保留 `predictions.csv` 原始节点表；每行包含 `sample_id,node_id,x_m,y_m,temperature_k,target_temperature_k,constituent_material_id,interface_side`，展示图不会覆盖该表。
- 固定节点二维 checkpoint 可选择 `two` 或 `one` 输出。`one` 不改变模型结构，而是在完整二维节点预测后提取归一化 `x=0.5` 的最近采样轴，并额外写出 `axis_predictions.csv/.npz` 和对比图。
- 多材料集合允许材料内部测试比例为 0。post0 外部波形通过 inference-only manifest 挂接到指定材料的 `test` 路由，不进入训练/验证；无真实空间标签时不能把代理场指标解释为真实二维误差。


## 交付后运行兼容与保护（2026-09-12）

- 增量模型的相对 `base_checkpoint` 以该模型所在目录解析，支持整目录迁移。绝对路径继续按原值解析；不会猜测 D/E 盘映射或自动改写外部数据路径。
- 新增量输出不能覆盖基础 checkpoint、已有权重或已有时间戳报告。需要保留同一基础模型的多个增量版本时，应使用新的文件名与时间戳。
- 项目清理扫描完整增量依赖链。选择保留时先核对所有目标目录，保存从基础模型继承的配置，再将各后代作为独立模型保留；不会用已有报告目录覆盖保留结果。原始相对报告布局继续受支持。
- checkpoint 仍使用项目已有的 PyTorch 字典格式，其中可能包含 NumPy 元数据。读取端显式指定 `weights_only=False` 以保留这一契约，未修改模型版本或重写旧权重。PyTorch 2.6 改变了省略该参数时的默认行为，见[官方序列化说明](https://docs.pytorch.org/docs/2.6/notes/serialization.html#torch-load-with-weights-only-true)。仅加载本项目生成或已确认可信来源的 checkpoint。
- R6 开关只改变 GUI 对缺少验证清单的单数据集的放行方式；不会改变模型结构、温度标准化、损失或训练/验证划分，也不会取消已有验证集。
