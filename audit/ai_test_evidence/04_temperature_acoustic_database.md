# 温度声学参数数据库

若使用本目录生成的索引，均为 `generated_for_audit_only: true`。

| item_id | item_name | status | extracted_value | source_path | source_location | notes |
|---|---|---|---|---|---|---|
| D-001 | 是否存在统一数据库/总 manifest | FOUND | ["database/data_process/metal_matrix/combined_manifest.json", "database/data_process/metal_matrix_waveforms_post0/combined_manifest.json"] | database/data_process/metal_matrix/combined_manifest.json; database/data_process/metal_matrix_waveforms_post0/combined_manifest.json | dataset_name/path contains combined | 发现 combined manifest；是否覆盖全部三类材料需逐项看字段统计。 |
| D-waveform | 是否包含波形路径 | FOUND | "waveform_path" | combined manifests or all manifests fallback | record keys |  |
| D-field | 是否包含温度场路径 | FOUND | "field_path" | combined manifests or all manifests fallback | record keys |  |
| D-temp | 是否包含温度值 | FOUND | "temperature_k" | combined manifests or all manifests fallback | record keys |  |
| D-dim | 是否包含 dimension | FOUND | "dimension" | combined manifests or all manifests fallback | record keys |  |
| D-mode | 是否包含 mode | FOUND | "mode" | combined manifests or all manifests fallback | record keys |  |
| D-material | 是否包含 material | FOUND | "material_key" | combined manifests or all manifests fallback | record keys |  |
| D-source | 是否包含 source | FOUND | "source" | combined manifests or all manifests fallback | record keys |  |
| D-field-mask | 是否包含 field_mask 或等价字段 | NOT_FOUND |  | combined manifests or all manifests fallback | record keys | field_mask 只在数据加载 batch 中出现，manifest 中未必存在。 |
| D-100 | 是否能够提取至少 100 条测试记录 | FOUND | 282 | test_manifest.json files | records[] |  |
