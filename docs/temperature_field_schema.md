# 固定节点温度场 schema v2

case 唯一键为 `worker/case_ID_温度标识`。输入既可采用 `worker_*/case_*` 嵌套布局，也可在材料目录直接放置 `case_*`；扁平布局的 worker 标识从唯一 `*_ultrasonic_config.json` 所引用的原始热力 CSV 路径恢复。默认选择原始 `receiver_signal.csv`，也可显式选择刚体校正版；`receiver_signal_rigid_motion.csv` 永不作为独立观测。

## 两个不同层级的“材料”

样本级材料由目录决定。单材料模式选择 `.../wumu`，该目录的全部 case 都属于路由字段 `wumu`，正式名称为“钨钼多层材料”。多材料模式选择其上层目录，系统只把直接子文件夹（`wumu`、`steel` 等）识别为不同样本材料，写入 `sample_material_catalog` 和每条记录的 `material_key`。

热力 CSV 的 `thermal_material_id` / `thermal_material_name` 描述一个样本材料内部的组分或层，写入 `constituent_material_catalog` 和 NPZ 的 `material_ids`。对 `wumu` 而言，`layer_1` 是钨层、`layer_2` 是钼层；它们不会成为样本路由字段，也不会各自产生 checkpoint。Window/CLI 的 `--experiment-material` 仅是数据集和输出命名标签，不参与样本材料判断。

## 固定节点数据

温度场 NPZ 包含 `temperature_k float32[P]`、`coordinates_m float32[P,2]`、`node_ids int64[P]`、`material_ids int32[P]`、`interface_side int32[P]`、`sample_weights float32[P]`，并明确记录单位 K、m。这里的 `material_ids` 是组分 ID。采样索引携带源网格 SHA-256 指纹、节点身份、物理坐标、组分、界面侧、损失权重、方法和种子。

训练、验证、测试按 case 划分；温度 z-score 统计量只使用各自训练集拟合。多材料建库生成 `material_collection.json`，每个材料保存独立的 train/validation/test 比例和清单，因此不同材料可以使用不同划分比例。

材料的测试比例允许为 0。若该材料另有纯波形实验测试集，外部 manifest 必须声明 `inference_only=true`，复用该材料训练 manifest 的采样索引、schema、波形长度和标准化参数，并由集合把 `manifests.test` 指向外部清单。原 case 测试清单可作为 `generated_case_test` 留档，但不再作为预测入口。

## 混合训练与分别训练

混合训练从集合中合并各材料的训练记录。因为 direct-point 模型只有一个输出索引空间，只有所有材料的采样网格和波形长度一致时允许混合；否则必须分别训练。

分别训练按 `material_collection.json` 的直接材料条目生成一个完整温度场 checkpoint，例如 `field__wumu.pt`。路由 JSON 的 `routing_field` 为 `records[].material_key`。预测单个 manifest 时读取记录的样本材料字段；预测集合时按每个材料条目调用对应 checkpoint。系统不额外训练波形材料分类器，也不按 `sampling_index.material_ids` 拆分或合并节点预测。

固定节点二维模型始终预测完整节点向量。请求一维输出时，后处理从归一化 `x=0.5` 附近选择最近的采样轴并按 y 排序；因此二维 checkpoint 可输出二维场或一维中心轴，一维旧 checkpoint 不能反向生成二维场。

case 波形按原始时间顺序固定保留连续前 1097 点。该过程不做均匀抽样、插值重采样或分窗，因此一个完整 case 仍只对应一个固定节点温度场。原始波形不足 1097 点时建库直接失败，不做补零。manifest 顶层 `waveform_contract` 记录统一输入契约；record metadata 记录处理策略、固定裁剪长度、原始长度、裁剪后长度和裁剪终止时间。


## 输入校验（2026-09-12）

数据集 manifest 顶层必须是 JSON 对象，`records` 必须是由记录对象组成的列表。训练加载与划分遇到错误结构时给出包含路径的错误；自动发现会跳过结构损坏的候选清单，继续查找可用数据集。

固定节点温度标准化的 `mean_k` 和 `std_k` 必须为有限数值，且 `std_k > 0`；`fit_split` 仍必须为 `train`。这些检查用于拒绝损坏输入，不改变标准化公式、节点数或 case 划分。
