# 两层 W30Mo70 实验数据

逐条记录索引见 `03_w30mo70_experiment_data.csv`。材料事实由用户确认。

| item_id | item_name | status | extracted_value | source_path | source_location | notes |
|---|---|---|---|---|---|---|
| C-001 | 两层 W30Mo70 实验数据 manifest/记录 | FOUND | {"manifest": "database/data_process/wumu_case_temperature_field/manifest.json", "records": 1000, "temperature_min_k": 300.0, "temperature_max_k": 1500.0} | database/data_process/wumu_case_temperature_field/manifest.json | records[] | 材料身份以用户确认事实为准；manifest 中 material_key 多处为 wumu。 |
| C-020 | 是否具有不少于 20 组实验采集的直接证据 | FOUND | 1000 | database/data_process/wumu_case_temperature_field/manifest.json | records[] | 计数来自 manifest 记录，不仅凭目录文件数量。 |
| C-021 | 实验数据是否进入 AI 测试集 | FOUND | ["database/data_process/wumu_case_temperature_field/test_manifest.json"] | database/data_process/wumu_case_temperature_field/test_manifest.json | file name test_manifest + records |  |
| C-022 | 实验数据是否进入增量训练集 | AMBIGUOUS |  | database/rule/trained_rules.csv | incremental rows | 存在增量 checkpoint 记录，但 rule table 不直接给出新增数据 manifest，不能确认 W30Mo70 实验数据进入增量训练。 |
