# Missing / ambiguous / inconsistent items

| 编号 | 内容 | 状态 | 直接证据位置 | 说明 |
|---|---|---|---|---|
| A-007 | 是否直接输入原始超声全波形 | AMBIGUOUS | model/waveform_io.py; manifests | 代码输入 waveform 张量；W30Mo70 manifest 记录 fixed_prefix_native_rate 和 crop_length=1097。是否等同“原始超声全波形”无法直接证明，因为存在裁剪。 |
| A-008 | 是否输入声时/幅值/衰减/振幅谱/中心频率等声学特征 | AMBIGUOUS | data_process/builder.py; model/network.py; model/point_field.py | 旧数据构建记录声学字段，但模型 forward 未显示这些字段作为输入。不能确认正式测试中声学特征作为独立输入。 |
| A-009 | 是否输入材料参数 | NOT_FOUND | model/network.py; model/point_field.py | 未找到材料物性参数进入 forward 的直接证据。 |
| A-010 | 是否输入空间坐标 | NOT_FOUND | model/point_field.py; model/predict.py | 坐标用于输出表和绘图；未找到作为模型输入的直接证据。 |
| A-011 | 是否输入时间坐标 | NOT_FOUND | model/network.py; model/point_field.py | 未找到时间坐标作为独立模型输入的直接证据。 |
| A-012 | 是否输入 dimension/mode/material 条件 | NOT_FOUND | README.md; model/point_field.py | 规则表有 dimension/mode/material，但未作为模型 forward 输入。 |
| B-carbon_silicon | 碳基/硅基复合材料训练集 | NOT_FOUND |  | 未找到可归类为 carbon_silicon 的现有 manifest 记录。 |
| B-TOTAL-SIM | 仿真波形总数是否由直接统计证明达到 1000 组 | NOT_FOUND | all manifest files | 仅统计 source 字段含 sim 的记录；不会把 experiment_case 计入仿真。 |
| B-TEMP-COVERAGE | 仿真记录温度是否覆盖室温至 1500 K | NOT_FOUND | all manifest files | 未找到 source 明确为仿真的记录。 |
| C-022 | 实验数据是否进入增量训练集 | AMBIGUOUS | database/rule/trained_rules.csv | 存在增量 checkpoint 记录，但 rule table 不直接给出新增数据 manifest，不能确认 W30Mo70 实验数据进入增量训练。 |
| D-field-mask | 是否包含 field_mask 或等价字段 | NOT_FOUND | combined manifests or all manifests fallback | field_mask 只在数据加载 batch 中出现，manifest 中未必存在。 |
| E-one_steady | 一维稳态 AI 预测结果 | NOT_FOUND | result/predict | 仅根据现有预测目录、metrics、metadata 和 predictions 表判断。 |
| E-one_transient | 一维瞬态 AI 预测结果 | NOT_FOUND | result/predict | 仅根据现有预测目录、metrics、metadata 和 predictions 表判断。 |
| E-two_transient | 二维瞬态 AI 预测结果 | NOT_FOUND | result/predict | 仅根据现有预测目录、metrics、metadata 和 predictions 表判断。 |
| E-ERR-10PCT | 是否存在相对误差不大于 10% 的直接证据 | NOT_FOUND | metrics.json files | 现有 metrics 提供 K 绝对误差；未发现相对误差 <=10% 的直接字段。 |
| F-001 | 五个测点坐标/编号/误差 | NOT_FOUND |  | 未找到明确五个测点坐标、参考温度、预测温度和五点平均相对误差结果的完整证据。 |
| H-001 | 迁移学习/参数冻结代码 | AMBIGUOUS | cli.py | 找到 fixed_weight/learnable_branch 相关实现和旧规则字段；未找到明确 requires_grad=False 冻结层执行日志。 |
| H-002 | 冻结机制实际执行日志 | NOT_FOUND | result/train/report | 未找到直接证明冻结层生效的执行日志。 |
| I-002 | 提示中的 4 场 0.248766 s / 0.0622 s 记录 | NOT_FOUND |  | 如果为 NOT_FOUND，则不得在报告中采用提示数值。 |
| J-001 | 传统算法耗时及比较口径 | AMBIGUOUS | result/train/report/metal_matrix_waveforms_post0/ai_model_history.csv; result/train/report/metal_matrix_waveforms_post0/ai_model_history.json; result/train/report/one_steady_metal_2026_5_15_2339/ai_model_history.csv; result/train/report/one_steady_metal_2026_5_15_2339/ai_model_history.json; result/train/report/one_steady_metal_matrix_2026_5_16_0155/ai_model_history.csv; result/train/report/one_steady_metal_matrix_2026_5_16_0155/ai_model_history.json | 搜索命中可能来自波形 CSV 数值或代码注释；未找到能证明与 AI 同输入/同工况/同输出规模的正式耗时日志。 |
