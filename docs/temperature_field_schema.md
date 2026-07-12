# 固定节点温度场 schema v1

case 唯一键为 `worker/case_ID_温度标识`。默认选择原始 `receiver_signal.csv`，也可显式选择刚体校正版；`receiver_signal_rigid_motion.csv` 永不作为独立观测。

温度场 NPZ 包含 `temperature_k float32[P]`、`coordinates_m float32[P,2]`、`node_ids int64[P]`、`material_ids int32[P]`、`interface_side int32[P]`、`sample_weights float32[P]`，并明确记录单位 K、m。`P` 必须在 1 至 10,000 之间。

采样索引携带版本、源网格 SHA-256 指纹、节点身份、物理坐标、材料、界面侧、损失权重、方法和种子。当前采样算法为 `spatial-stratified-v2`：材料总预算按源节点数分配，层内采用空间网格轮询，损失权重按“源分层节点数/采样节点数”计算。相同坐标的接触界面节点以稳定 `interface_side` 区分。旧 manifest 无版本时按 v0 `legacy-grid` 只读兼容，不会将旧规则网格 checkpoint 静默解释成 v1 固定节点模型。

训练、验证、测试按 case 划分；温度 z-score 统计量仅使用训练集拟合，并原样写入三份 manifest 和 checkpoint。
