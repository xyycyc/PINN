# AI 模型实际输入和输出

| item_id | item_name | status | extracted_value | source_path | source_location | notes |
|---|---|---|---|---|---|---|
| A-001 | 模型定义文件路径 | FOUND | ["model/network.py", "model/point_field.py"] | model/network.py; model/point_field.py | model/network.py:L45; model/point_field.py:L21 |  |
| A-002 | 模型类名称 | FOUND | ["AIReconstructionModel", "DirectPointFieldModel"] | model/network.py; model/point_field.py | model/network.py:L45; model/point_field.py:L21 |  |
| A-003 | 实际训练入口 | FOUND | "python -m ai_model train / ReconstructionTrainer.train" | cli.py; model/trainer.py | cli.py:L715; model/trainer.py:L230 |  |
| A-004 | 实际推理入口 | FOUND | "python -m ai_model predict / predict_and_compare" | cli.py; model/predict.py | cli.py:L802; model/predict.py:L308 |  |
| A-005 | 实际 GUI 调用入口 | FOUND | "python -m ai_model.window; window tabs compose CLI commands" | README.md; window/app.py; window/tabs/predict.py | README.md:L133; window/app.py; window/tabs/predict.py |  |
| A-006 | 模型输入字段 | FOUND | "forward(waveform: torch.Tensor)" | model/network.py; model/point_field.py | model/network.py:L95; model/point_field.py:L46 |  |
| A-007 | 是否直接输入原始超声全波形 | AMBIGUOUS |  | model/waveform_io.py; manifests |  | 代码输入 waveform 张量；W30Mo70 manifest 记录 fixed_prefix_native_rate 和 crop_length=1097。是否等同“原始超声全波形”无法直接证明，因为存在裁剪。 |
| A-008 | 是否输入声时/幅值/衰减/振幅谱/中心频率等声学特征 | AMBIGUOUS | {"legacy_builder_acoustic_fields": ["tof", "amplitude", "center_freq"], "model_forward_input": "waveform only"} | data_process/builder.py; model/network.py; model/point_field.py | data_process/builder.py:L158-L161; model/network.py:L95; model/point_field.py:L46 | 旧数据构建记录声学字段，但模型 forward 未显示这些字段作为输入。不能确认正式测试中声学特征作为独立输入。 |
| A-009 | 是否输入材料参数 | NOT_FOUND |  | model/network.py; model/point_field.py | forward signatures | 未找到材料物性参数进入 forward 的直接证据。 |
| A-010 | 是否输入空间坐标 | NOT_FOUND |  | model/point_field.py; model/predict.py | model/point_field.py:L46 | 坐标用于输出表和绘图；未找到作为模型输入的直接证据。 |
| A-011 | 是否输入时间坐标 | NOT_FOUND |  | model/network.py; model/point_field.py | forward signatures | 未找到时间坐标作为独立模型输入的直接证据。 |
| A-012 | 是否输入 dimension/mode/material 条件 | NOT_FOUND |  | README.md; model/point_field.py | README.md:L346; model/point_field.py:L46 | 规则表有 dimension/mode/material，但未作为模型 forward 输入。 |
| A-013 | 模型输出类型 | FOUND | {"AIReconstructionModel": "field 24x24 + acoustic + temperature", "DirectPointFieldModel": "P fixed-node temperatures"} | model/network.py; model/point_field.py; README.md | model/network.py:L107-L116; model/point_field.py:L46-L58; README.md:L186-L187 |  |
