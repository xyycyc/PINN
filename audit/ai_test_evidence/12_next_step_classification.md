# Next step classification

| 编号 | 缺失内容 | 当前状态 | 直接证据 | 分类 | 原因 |
|---|---|---|---|---|---|
| A-007 | 是否直接输入原始超声全波形 | AMBIGUOUS | model/waveform_io.py; manifests | C. 只需补充文件或人工确认 | 现有信息不足以唯一确认；可能需要上传外部文件或人工确认口径。 |
| A-008 | 是否输入声时/幅值/衰减/振幅谱/中心频率等声学特征 | AMBIGUOUS | data_process/builder.py; model/network.py; model/point_field.py | C. 只需补充文件或人工确认 | 现有信息不足以唯一确认；可能需要上传外部文件或人工确认口径。 |
| A-009 | 是否输入材料参数 | NOT_FOUND | model/network.py; model/point_field.py | B. 需要修改代码 | 当前代码或输出形式未显示支持该要求。 |
| A-010 | 是否输入空间坐标 | NOT_FOUND | model/point_field.py; model/predict.py | B. 需要修改代码 | 当前代码或输出形式未显示支持该要求。 |
| A-011 | 是否输入时间坐标 | NOT_FOUND | model/network.py; model/point_field.py | B. 需要修改代码 | 当前代码或输出形式未显示支持该要求。 |
| A-012 | 是否输入 dimension/mode/material 条件 | NOT_FOUND | README.md; model/point_field.py | B. 需要修改代码 | 当前代码或输出形式未显示支持该要求。 |
| B-carbon_silicon | 碳基/硅基复合材料训练集 | NOT_FOUND |  | A. 需要补充实验 | 当前仓库没有可直接证明的数据、结果或日志。 |
| B-TOTAL-SIM | 仿真波形总数是否由直接统计证明达到 1000 组 | NOT_FOUND | all manifest files | A. 需要补充实验 | 当前仓库没有可直接证明的数据、结果或日志。 |
| B-TEMP-COVERAGE | 仿真记录温度是否覆盖室温至 1500 K | NOT_FOUND | all manifest files | A. 需要补充实验 | 当前仓库没有可直接证明的数据、结果或日志。 |
| C-022 | 实验数据是否进入增量训练集 | AMBIGUOUS | database/rule/trained_rules.csv | C. 只需补充文件或人工确认 | 现有信息不足以唯一确认；可能需要上传外部文件或人工确认口径。 |
| D-field-mask | 是否包含 field_mask 或等价字段 | NOT_FOUND | combined manifests or all manifests fallback | B. 需要修改代码 | 当前代码或输出形式未显示支持该要求。 |
| E-one_steady | 一维稳态 AI 预测结果 | NOT_FOUND | result/predict | A. 需要补充实验 | 当前仓库没有可直接证明的数据、结果或日志。 |
| E-one_transient | 一维瞬态 AI 预测结果 | NOT_FOUND | result/predict | A. 需要补充实验 | 当前仓库没有可直接证明的数据、结果或日志。 |
| E-two_transient | 二维瞬态 AI 预测结果 | NOT_FOUND | result/predict | A. 需要补充实验 | 当前仓库没有可直接证明的数据、结果或日志。 |
| E-ERR-10PCT | 是否存在相对误差不大于 10% 的直接证据 | NOT_FOUND | metrics.json files | A. 需要补充实验 | 当前仓库没有可直接证明的数据、结果或日志。 |
| F-001 | 五个测点坐标/编号/误差 | NOT_FOUND |  | C. 只需补充文件或人工确认 | 现有信息不足以唯一确认；可能需要上传外部文件或人工确认口径。 |
| H-001 | 迁移学习/参数冻结代码 | AMBIGUOUS | cli.py | C. 只需补充文件或人工确认 | 现有信息不足以唯一确认；可能需要上传外部文件或人工确认口径。 |
| H-002 | 冻结机制实际执行日志 | NOT_FOUND | result/train/report | A. 需要补充实验 | 当前仓库没有可直接证明的数据、结果或日志。 |
| I-002 | 提示中的 4 场 0.248766 s / 0.0622 s 记录 | NOT_FOUND |  | A. 需要补充实验 | 当前仓库没有可直接证明的数据、结果或日志。 |
| J-001 | 传统算法耗时及比较口径 | AMBIGUOUS | result/train/report/metal_matrix_waveforms_post0/ai_model_history.csv; result/train/report/metal_matrix_waveforms_post0/ai_model_history.json; result/train/report/one_steady_metal_2026_5_15_2339/ai_model_history.csv; result/train/report/one_steady_metal_2026_5_15_2339/ai_model_history.json; result/train/report/one_steady_metal_matrix_2026_5_16_0155/ai_model_history.csv; result/train/report/one_steady_metal_matrix_2026_5_16_0155/ai_model_history.json | C. 只需补充文件或人工确认 | 现有信息不足以唯一确认；可能需要上传外部文件或人工确认口径。 |
