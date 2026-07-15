# AI 温度场测试证据审计总报告

## 1. 审计基本信息

- 仓库地址：https://github.com/xyycyc/PINN.git
- 审计分支：audit/ai-test-evidence
- 基准分支：agent/fixed-node-temperature-field
- 基准 commit：58bedc8461759f9afc96510bdc80dd01c1cfdd11
- 审计 commit：待提交后由 Git 记录；`audit_summary.json` 当前为生成时快照。
- 审计时间：2026-07-15T15:29:48.297400+00:00
- 扫描范围：tracked files、untracked project files、manifest、CSV/JSON/MD/Python/Fortran、已有训练/预测报告、已有 checkpoint 元信息、Git log。
- 未扫描范围及原因：`.git` object internals 未展开；大文件内容未全文哈希；未运行训练、推理、数据生成；未默认执行 checkpoint 参数比较。

## 2. 已确认事实

- 实验测试材料为两层 W30Mo70 钨钼合金多层材料。
  - source_type: user_confirmed
  - status: CONFIRMED

## 3. 现有证据总览

| 检查类别 | CONFIRMED/FOUND | NOT_FOUND | AMBIGUOUS | INCONSISTENT | INACCESSIBLE |
|---|---|---|---|---|---|
| A. AI 模型输入与输出 | 7 | 4 | 2 | 0 | 0 |
| B. 三种材料仿真训练集 | 2 | 3 | 0 | 0 | 0 |
| C. W30Mo70 实验数据 | 3 | 0 | 1 | 0 | 0 |
| D. 温度声学参数数据库 | 9 | 1 | 0 | 0 | 0 |
| E. 四类 AI 预测结果 | 2 | 4 | 0 | 0 | 0 |
| F. 五个测点 | 0 | 1 | 0 | 0 | 0 |
| G. 增量训练 | 4 | 0 | 0 | 0 | 0 |
| H. 迁移学习和冻结 | 0 | 1 | 1 | 0 | 0 |
| I/J. 效率证据 | 1 | 1 | 1 | 0 | 0 |
| K. Windows/Fortran | 2 | 0 | 0 | 0 | 0 |

## 4. AI 模型输入与输出

详见 `01_model_input_output.md`。关键点：模型类、训练入口、推理入口、GUI 入口、forward 输入和输出均按源代码行证据列出；材料参数、空间坐标、时间坐标、dimension/mode/material 作为模型输入未找到直接证据。

## 5. 三种材料仿真训练集

详见 `02_simulation_training_sets.md` 和 `02_simulation_training_sets.csv`。统计来自现有 manifest 记录；配置中的温度范围未作为实际覆盖证据。

## 6. W30Mo70 实验数据

详见 `03_w30mo70_experiment_data.md` 和 `03_w30mo70_experiment_data.csv`。材料身份采用用户确认事实；manifest 和记录只作为数据路径、温度、维度、模式和来源字段证据。

## 7. 温度声学参数数据库

详见 `04_temperature_acoustic_database.md`。本审计生成的 `manifest_summary.csv`、`prediction_runs.csv`、`checkpoint_inventory.csv` 均为审计索引：

```yaml
generated_for_audit_only: true
```

## 8. 四类 AI 预测结果

详见 `05_prediction_results.md` 和 `prediction_runs.csv`。报告区分代码支持、已有预测结果、已有真值、已有误差字段；未将单点或示例结果扩展为完整温度场结论。

## 9. 五个测点

详见 `06_five_measurement_points.md`。未找到明确五个测点坐标、参考温度、预测温度和五点平均相对误差的完整证据。

## 10. 增量训练与迁移学习

详见 `07_incremental_training.md` 和 `08_transfer_learning_and_freezing.md`。已区分增量训练入口、增量 checkpoint/报告存在性、以及参数实际变化证据。主扫描不默认执行 checkpoint 比较。

## 11. 效率证据

详见 `09_timing_evidence.md`。AI metrics 中存在 `eval_total_seconds`/`eval_latency_ms_per_sample` 等字段，但 benchmark 多处为 false；未找到能证明与传统算法同输入、同工况、同输出规模的正式比较口径。未给出正式 speedup。

## 12. Windows、Fortran 与跨语言调用

详见 `10_fortran_and_runtime_evidence.md`。Fortran 源文件、DLL 和 Python bridge 存在；该证据不表示 AI 主体由 Fortran 编写。

## 13. 仍需补充的事项

### 13.1 需要补充实验

| 编号 | 缺失内容 | 当前状态 | 已搜索位置 | 为什么证据不足 |
|---|---|---|---|---|
| B-carbon_silicon | 碳基/硅基复合材料训练集 | NOT_FOUND |  | 未找到可归类为 carbon_silicon 的现有 manifest 记录。 |
| B-TOTAL-SIM | 仿真波形总数是否由直接统计证明达到 1000 组 | NOT_FOUND | all manifest files | 仅统计 source 字段含 sim 的记录；不会把 experiment_case 计入仿真。 |
| B-TEMP-COVERAGE | 仿真记录温度是否覆盖室温至 1500 K | NOT_FOUND | all manifest files | 未找到 source 明确为仿真的记录。 |
| E-one_steady | 一维稳态 AI 预测结果 | NOT_FOUND | result/predict | 仅根据现有预测目录、metrics、metadata 和 predictions 表判断。 |
| E-one_transient | 一维瞬态 AI 预测结果 | NOT_FOUND | result/predict | 仅根据现有预测目录、metrics、metadata 和 predictions 表判断。 |
| E-two_transient | 二维瞬态 AI 预测结果 | NOT_FOUND | result/predict | 仅根据现有预测目录、metrics、metadata 和 predictions 表判断。 |
| E-ERR-10PCT | 是否存在相对误差不大于 10% 的直接证据 | NOT_FOUND | metrics.json files | 现有 metrics 提供 K 绝对误差；未发现相对误差 <=10% 的直接字段。 |
| H-002 | 冻结机制实际执行日志 | NOT_FOUND | result/train/report | 未找到直接证明冻结层生效的执行日志。 |
| I-002 | 提示中的 4 场 0.248766 s / 0.0622 s 记录 | NOT_FOUND |  | 如果为 NOT_FOUND，则不得在报告中采用提示数值。 |

### 13.2 需要修改代码

| 编号 | 缺失内容 | 当前状态 | 已搜索位置 | 为什么证据不足 |
|---|---|---|---|---|
| A-009 | 是否输入材料参数 | NOT_FOUND | model/network.py; model/point_field.py | 未找到材料物性参数进入 forward 的直接证据。 |
| A-010 | 是否输入空间坐标 | NOT_FOUND | model/point_field.py; model/predict.py | 坐标用于输出表和绘图；未找到作为模型输入的直接证据。 |
| A-011 | 是否输入时间坐标 | NOT_FOUND | model/network.py; model/point_field.py | 未找到时间坐标作为独立模型输入的直接证据。 |
| A-012 | 是否输入 dimension/mode/material 条件 | NOT_FOUND | README.md; model/point_field.py | 规则表有 dimension/mode/material，但未作为模型 forward 输入。 |
| D-field-mask | 是否包含 field_mask 或等价字段 | NOT_FOUND | combined manifests or all manifests fallback | field_mask 只在数据加载 batch 中出现，manifest 中未必存在。 |

### 13.3 只需补充文件

| 编号 | 缺失内容 | 当前状态 | 已搜索位置 | 为什么证据不足 |
|---|---|---|---|---|

### 13.4 只需人工确认

| 编号 | 缺失内容 | 当前状态 | 已搜索位置 | 为什么证据不足 |
|---|---|---|---|---|
| A-007 | 是否直接输入原始超声全波形 | AMBIGUOUS | model/waveform_io.py; manifests | 代码输入 waveform 张量；W30Mo70 manifest 记录 fixed_prefix_native_rate 和 crop_length=1097。是否等同“原始超声全波形”无法直接证明，因为存在裁剪。 |
| A-008 | 是否输入声时/幅值/衰减/振幅谱/中心频率等声学特征 | AMBIGUOUS | data_process/builder.py; model/network.py; model/point_field.py | 旧数据构建记录声学字段，但模型 forward 未显示这些字段作为输入。不能确认正式测试中声学特征作为独立输入。 |
| C-022 | 实验数据是否进入增量训练集 | AMBIGUOUS | database/rule/trained_rules.csv | 存在增量 checkpoint 记录，但 rule table 不直接给出新增数据 manifest，不能确认 W30Mo70 实验数据进入增量训练。 |
| F-001 | 五个测点坐标/编号/误差 | NOT_FOUND |  | 未找到明确五个测点坐标、参考温度、预测温度和五点平均相对误差结果的完整证据。 |
| H-001 | 迁移学习/参数冻结代码 | AMBIGUOUS | cli.py | 找到 fixed_weight/learnable_branch 相关实现和旧规则字段；未找到明确 requires_grad=False 冻结层执行日志。 |
| J-001 | 传统算法耗时及比较口径 | AMBIGUOUS | result/train/report/metal_matrix_waveforms_post0/ai_model_history.csv; result/train/report/metal_matrix_waveforms_post0/ai_model_history.json; result/train/report/one_steady_metal_2026_5_15_2339/ai_model_history.csv; result/train/report/one_steady_metal_2026_5_15_2339/ai_model_history.json; result/train/report/one_steady_metal_matrix_2026_5_16_0155/ai_model_history.csv; result/train/report/one_steady_metal_matrix_2026_5_16_0155/ai_model_history.json | 搜索命中可能来自波形 CSV 数值或代码注释；未找到能证明与 AI 同输入/同工况/同输出规模的正式耗时日志。 |

## 14. 可直接用于测试报告的内容

| 编号 | 推荐报告表述 | 原始证据路径 | 证据位置 | 使用限制 |
|---|---|---|---|---|
| FACT-001 | 可写入：实验测试材料 = "两层 W30Mo70 钨钼合金多层材料" | user prompt | 用户明确确认事实 | 仅限按证据原样引用；不得扩展为未证明指标。 |
| A-001 | 可写入：模型定义文件路径 = ["model/network.py", "model/point_field.py"] | model/network.py; model/point_field.py | model/network.py:L45; model/point_field.py:L21 | 仅限按证据原样引用；不得扩展为未证明指标。 |
| A-002 | 可写入：模型类名称 = ["AIReconstructionModel", "DirectPointFieldModel"] | model/network.py; model/point_field.py | model/network.py:L45; model/point_field.py:L21 | 仅限按证据原样引用；不得扩展为未证明指标。 |
| A-003 | 可写入：实际训练入口 = "python -m ai_model train / ReconstructionTrainer.train" | cli.py; model/trainer.py | cli.py:L715; model/trainer.py:L230 | 仅限按证据原样引用；不得扩展为未证明指标。 |
| A-004 | 可写入：实际推理入口 = "python -m ai_model predict / predict_and_compare" | cli.py; model/predict.py | cli.py:L802; model/predict.py:L308 | 仅限按证据原样引用；不得扩展为未证明指标。 |
| A-005 | 可写入：实际 GUI 调用入口 = "python -m ai_model.window; window tabs compose CLI commands" | README.md; window/app.py; window/tabs/predict.py | README.md:L133; window/app.py; window/tabs/predict.py | 仅限按证据原样引用；不得扩展为未证明指标。 |
| A-006 | 可写入：模型输入字段 = "forward(waveform: torch.Tensor)" | model/network.py; model/point_field.py | model/network.py:L95; model/point_field.py:L46 | 仅限按证据原样引用；不得扩展为未证明指标。 |
| A-013 | 可写入：模型输出类型 = {"AIReconstructionModel": "field 24x24 + acoustic + temperature", "DirectPointFieldModel": "P fixed-node temperatures"} | model/network.py; model/point_field.py; README.md | model/network.py:L107-L116; model/point_field.py:L46-L58; README.md:L186-L187 | 仅限按证据原样引用；不得扩展为未证明指标。 |
| C-001 | 可写入：两层 W30Mo70 实验数据 manifest/记录 = {"manifest": "database/data_process/wumu_case_temperature_field/manifest.json", "records": 1000, "temperature_min_k": 300.0, "temperature_max_k": 1500.0} | database/data_process/wumu_case_temperature_field/manifest.json | records[] | 仅限按证据原样引用；不得扩展为未证明指标。 |
| C-020 | 可写入：是否具有不少于 20 组实验采集的直接证据 = 1000 | database/data_process/wumu_case_temperature_field/manifest.json | records[] | 仅限按证据原样引用；不得扩展为未证明指标。 |
| D-100 | 可写入：是否能够提取至少 100 条测试记录 = 282 | test_manifest.json files | records[] | 仅限按证据原样引用；不得扩展为未证明指标。 |
| I-001 | 可写入：AI 推理耗时记录 = [{"run_dir": "result/predict/inference/metal_matrix_waveforms_post0_test", "predictions_csv": "result/predict/inference/metal_matrix_waveforms_post0_test/predictions.csv", "predictions_npz": "result/predict/inference/metal_matrix_waveforms_post0_test/predictions.npz", "samples_metric": 7, "point_count": null, "model_kind": null, "checkpoint": "ai_model\\result\\train\\checkpoint\\metal_matrix_waveforms_post0\\ai_model.pt", "manifest": "ai_model\\database\\data_process\\metal_matrix_waveforms_post0\\test_manifest.json", "eval_total_seconds": 0.18528059998061508, "eval_latency_ms_per_sample": 26.468657140087867, "benchmark_enabled": false, "mae_k": null, "rmse_k": null, "max_abs_k": null, "field_plot_samples": null, "prediction_header": "sample_id;source;material_key;temperature_true_K;temperature_pred_K;abs_error_K;has_field_label"}, {"run_dir": "result/predict/inference/plot_smoke_signed_error", "predictions_csv": "result/predict/inference/plot_smoke_signed_error/predictions.csv", "predictions_npz": "result/predict/inference/plot_smoke_signed_error/predictions.npz", "samples_metric": 200, "point_count": 10000, "model_kind": "direct_point_field", "checkpoint": "D:\\Desktop\\code\\ai_model\\result\\train\\checkpoint\\two_steady_wumu_er_cli_2026_7_14_100\\ai_model.pt", "manifest": "D:\\Desktop\\code\\ai_model\\database\\data_process\\wumu_case_temperature_field\\test_manifest.json", "eval_total_seconds": 2.525187700004608, "eval_latency_ms_per_sample": 12.62593850002304, "benchmark_enabled": false, "mae_k": 4.0308144082946775, "rmse_k": 4.966087075865303, "max_abs_k": 13.40087890625, "field_plot_samples": 1, "prediction_header": "sample_id;node_id;x_m;y_m;temperature_k;target_temperature_k;constituent_material_id;interface_side"}, {"run_dir": "result/predict/inference/point_field_smoke_test", "predictions_csv": "result/predict/inference/point_field_smoke_test/predictions.csv", "predictions_npz": "result/predict/inference/point_field_smoke_test/predictions.npz", "samples_metric": 3, "point_count": 10000, "model_kind": "direct_point_field", "checkpoint": "ai_model\\result\\train\\checkpoint\\point_field_smoke\\smoke.pt", "manifest": "ai_model\\database\\data_process\\metal_matrix_case_temperature_field\\test_manifest.json", "eval_total_seconds": 0.050278899987461045, "eval_latency_ms_per_sample": 16.759633329153683, "benchmark_enabled": null, "mae_k": 252.5427119608561, "rmse_k": 321.6889741393592, "max_abs_k": 557.7281799316406, "field_plot_samples": null, "prediction_header": "sample_id;node_id;x_m;y_m;temperature_k;target_temperature_k;material_id;interface_side"}, {"run_dir": "result/predict/inference/point_field_smoke_v2_test", "predictions_csv": "result/predict/inference/point_field_smoke_v2_test/predictions.csv", "predictions_npz": "result/predict/inference/point_field_smoke_v2_test/predictions.npz", "samples_metric": 4, "point_count": 10000, "model_kind": "direct_point_field", "checkpoint": "ai_model\\result\\train\\checkpoint\\point_field_smoke\\smoke_v2.pt", "manifest": "ai_model\\database\\data_process\\metal_matrix_case_temperature_field\\test_manifest.json", "eval_total_seconds": 0.039972800004761666, "eval_latency_ms_per_sample": 9.993200001190417, "benchmark_enabled": null, "mae_k": 300.60102077941895, "rmse_k": 356.57204956081495, "max_abs_k": 554.9134216308594, "field_plot_samples": null, "prediction_header": "sample_id;node_id;x_m;y_m;temperature_k;target_temperature_k;material_id;interface_side"}, {"run_dir": "result/predict/inference/two_steady_metal_matrix_2026_5_15_2349_2026_5_16_001347", "predictions_csv": "result/predict/inference/two_steady_metal_matrix_2026_5_15_2349_2026_5_16_001347/predictions.csv", "predictions_npz": "result/predict/inference/two_steady_metal_matrix_2026_5_15_2349_2026_5_16_001347/predictions.npz", "samples_metric": 7, "point_count": null, "model_kind": null, "checkpoint": "D:\\Desktop\\code\\ai_model\\result\\train\\checkpoint\\two_steady_metal_matrix_2026_5_15_2349\\ai_model.pt", "manifest": "D:\\Desktop\\code\\ai_model\\database\\data_process\\metal_matrix\\test_manifest.json", "eval_total_seconds": 0.15665039996383712, "eval_latency_ms_per_sample": 22.378628566262446, "benchmark_enabled": 1.0, "mae_k": null, "rmse_k": null, "max_abs_k": null, "field_plot_samples": null, "prediction_header": "sample_id;source;material_key;temperature_true_K;temperature_pred_K;abs_error_K;has_field_label"}, {"run_dir": "result/predict/inference/two_steady_metal_matrix_2026_5_15_2349_2026_5_16_001633", "predictions_csv": "result/predict/inference/two_steady_metal_matrix_2026_5_15_2349_2026_5_16_001633/predictions.csv", "predictions_npz": "result/predict/inference/two_steady_metal_matrix_2026_5_15_2349_2026_5_16_001633/predictions.npz", "samples_metric": 7, "point_count": null, "model_kind": null, "checkpoint": "D:\\Desktop\\code\\ai_model\\result\\train\\checkpoint\\two_steady_metal_matrix_2026_5_15_2349\\ai_model.pt", "manifest": "D:\\Desktop\\code\\ai_model\\database\\data_process\\metal_matrix\\test_manifest.json", "eval_total_seconds": 0.18468710000161082, "eval_latency_ms_per_sample": 26.383871428801545, "benchmark_enabled": false, "mae_k": null, "rmse_k": null, "max_abs_k": null, "field_plot_samples": null, "prediction_header": "sample_id;source;material_key;temperature_true_K;temperature_pred_K;abs_error_K;has_field_label"}, {"run_dir": "result/predict/inference/two_steady_metal_matrix_2026_5_15_2349_2026_5_16_004901", "predictions_csv": "result/predict/inference/two_steady_metal_matrix_2026_5_15_2349_2026_5_16_004901/predictions.csv", "predictions_npz": "result/predict/inference/two_steady_metal_matrix_2026_5_15_2349_2026_5_16_004901/predictions.npz", "samples_metric": 7, "point_count": null, "model_kind": null, "checkpoint": "D:\\Desktop\\code\\ai_model\\result\\train\\checkpoint\\two_steady_metal_matrix_2026_5_15_2349\\2026_5_16_0042.pt", "manifest": "D:\\Desktop\\code\\ai_model\\database\\data_process\\metal_matrix\\test_manifest.json", "eval_total_seconds": 0.2249236000352539, "eval_latency_ms_per_sample": 32.13194286217913, "benchmark_enabled": false, "mae_k": null, "rmse_k": null, "max_abs_k": null, "field_plot_samples": null, "prediction_header": "sample_id;source;material_key;temperature_true_K;temperature_pred_K;abs_error_K;has_field_label"}, {"run_dir": "result/predict/inference/two_steady_metal_matrix_2026_5_15_2349_2026_5_16_004956", "predictions_csv": "result/predict/inference/two_steady_metal_matrix_2026_5_15_2349_2026_5_16_004956/predictions.csv", "predictions_npz": "result/predict/inference/two_steady_metal_matrix_2026_5_15_2349_2026_5_16_004956/predictions.npz", "samples_metric": 7, "point_count": null, "model_kind": null, "checkpoint": "D:\\Desktop\\code\\ai_model\\result\\train\\checkpoint\\two_steady_metal_matrix_2026_5_15_2349\\2026_5_16_0032.pt", "manifest": "D:\\Desktop\\code\\ai_model\\database\\data_process\\metal_matrix\\test_manifest.json", "eval_total_seconds": 0.19692830002168193, "eval_latency_ms_per_sample": 28.132614288811705, "benchmark_enabled": false, "mae_k": null, "rmse_k": null, "max_abs_k": null, "field_plot_samples": null, "prediction_header": "sample_id;source;material_key;temperature_true_K;temperature_pred_K;abs_error_K;has_field_label"}, {"run_dir": "result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_131330", "predictions_csv": "result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_131330/predictions.csv", "predictions_npz": "result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_131330/predictions.npz", "samples_metric": 200, "point_count": 10000, "model_kind": "direct_point_field", "checkpoint": "D:\\Desktop\\code\\ai_model\\result\\train\\checkpoint\\two_steady_wumu_er_cli_2026_7_14_100\\ai_model.pt", "manifest": "D:\\Desktop\\code\\ai_model\\database\\data_process\\wumu_case_temperature_field\\test_manifest.json", "eval_total_seconds": 6.277371700001822, "eval_latency_ms_per_sample": 31.38685850000911, "benchmark_enabled": false, "mae_k": 4.015756273590088, "rmse_k": 4.94884115325271, "max_abs_k": 13.2479248046875, "field_plot_samples": 2, "prediction_header": "sample_id;node_id;x_m;y_m;temperature_k;target_temperature_k;constituent_material_id;interface_side"}, {"run_dir": "result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_135631", "predictions_csv": "result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_135631/predictions.csv", "predictions_npz": "result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_135631/predictions.npz", "samples_metric": 200, "point_count": 10000, "model_kind": "direct_point_field", "checkpoint": "D:\\Desktop\\code\\ai_model\\result\\train\\checkpoint\\two_steady_wumu_er_cli_2026_7_14_100\\ai_model.pt", "manifest": "D:\\Desktop\\code\\ai_model\\database\\data_process\\wumu_case_temperature_field\\test_manifest.json", "eval_total_seconds": 1.1408969000040088, "eval_latency_ms_per_sample": 5.704484500020044, "benchmark_enabled": false, "mae_k": 4.015756273590088, "rmse_k": 4.94884115325271, "max_abs_k": 13.2479248046875, "field_plot_samples": 4, "prediction_header": "sample_id;node_id;x_m;y_m;temperature_k;target_temperature_k;constituent_material_id;interface_side"}, {"run_dir": "result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_142211", "predictions_csv": "result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_142211/predictions.csv", "predictions_npz": "result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_142211/predictions.npz", "samples_metric": 200, "point_count": 10000, "model_kind": "direct_point_field", "checkpoint": "D:\\Desktop\\code\\ai_model\\result\\train\\checkpoint\\two_steady_wumu_er_cli_2026_7_14_100\\ai_model.pt", "manifest": "D:\\Desktop\\code\\ai_model\\database\\data_process\\wumu_case_temperature_field\\test_manifest.json", "eval_total_seconds": 1.1759702000053949, "eval_latency_ms_per_sample": 5.879851000026974, "benchmark_enabled": false, "mae_k": 4.015756273590088, "rmse_k": 4.94884115325271, "max_abs_k": 13.2479248046875, "field_plot_samples": 4, "prediction_header": "sample_id;node_id;x_m;y_m;temperature_k;target_temperature_k;constituent_material_id;interface_side"}, {"run_dir": "result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_143411", "predictions_csv": "result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_143411/predictions.csv", "predictions_npz": "result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_143411/predictions.npz", "samples_metric": 200, "point_count": 10000, "model_kind": "direct_point_field", "checkpoint": "D:\\Desktop\\code\\ai_model\\result\\train\\checkpoint\\two_steady_wumu_er_cli_2026_7_14_100\\ai_model.pt", "manifest": "D:\\Desktop\\code\\ai_model\\database\\data_process\\wumu_case_temperature_field\\test_manifest.json", "eval_total_seconds": 1.1720786000005319, "eval_latency_ms_per_sample": 5.860393000002659, "benchmark_enabled": false, "mae_k": 4.015756273590088, "rmse_k": 4.94884115325271, "max_abs_k": 13.2479248046875, "field_plot_samples": 4, "prediction_header": "sample_id;node_id;x_m;y_m;temperature_k;target_temperature_k;constituent_material_id;interface_side"}, {"run_dir": "result/predict/inference/two_steady_wumu_matrix_2026_7_14_1157_2026_7_14_151954", "predictions_csv": "result/predict/inference/two_steady_wumu_matrix_2026_7_14_1157_2026_7_14_151954/predictions.csv", "predictions_npz": "result/predict/inference/two_steady_wumu_matrix_2026_7_14_1157_2026_7_14_151954/predictions.npz", "samples_metric": 200, "point_count": 10000, "model_kind": "direct_point_field", "checkpoint": "D:\\Desktop\\code\\ai_model\\result\\train\\checkpoint\\two_steady_wumu_matrix_2026_7_14_1157\\ai_model.pt", "manifest": "D:\\Desktop\\code\\ai_model\\database\\data_process\\wumu_case_temperature_field\\test_manifest.json", "eval_total_seconds": 1.3133252000043285, "eval_latency_ms_per_sample": 6.5666260000216425, "benchmark_enabled": false, "mae_k": 1.3934736314239502, "rmse_k": 1.7506939776618713, "max_abs_k": 4.919677734375, "field_plot_samples": 6, "prediction_header": "sample_id;node_id;x_m;y_m;temperature_k;target_temperature_k;constituent_material_id;interface_side"}] | result/predict/**/metrics.json | eval_total_seconds/eval_latency_ms_per_sample | 仅限按证据原样引用；不得扩展为未证明指标。 |
| K-001 | 可写入：Windows/Fortran 源文件和编译产物 = ["fortran/__init__.py", "fortran/__pycache__", "fortran/ai_numeric.dll", "fortran/ai_numeric.f90", "fortran/ai_numeric.mod", "fortran/build_fortran.ps1", "fortran/native_bridge.py", "fortran/README.md"] | fortran/ | file list | 仅限按证据原样引用；不得扩展为未证明指标。 |
| K-002 | 可写入：Python/Fortran 调用接口 = [{"path": "fortran/ai_numeric.f90", "line": 6, "text": "subroutine compute_prediction_metrics(y_true, y_pred, n, mae, rmse, max_error) &"}, {"path": "fortran/ai_numeric.f90", "line": 7, "text": "bind(C, name=\"compute_prediction_metrics\")"}, {"path": "fortran/ai_numeric.f90", "line": 38, "text": "end subroutine compute_prediction_metrics"}, {"path": "fortran/ai_numeric.f90", "line": 41, "text": "subroutine average_waveform_pairs(time_values, voltage_values, n_runs, n_points, &"}, {"path": "fortran/ai_numeric.f90", "line": 42, "text": "time_average, voltage_average) bind(C, name=\"average_waveform_pairs\")"}, {"path": "fortran/ai_numeric.f90", "line": 70, "text": "end subroutine average_waveform_pairs"}, {"path": "fortran/build_fortran.ps1", "line": 2, "text": "[string]$Compiler = \"gfortran\""}, {"path": "fortran/build_fortran.ps1", "line": 8, "text": "$OutputFile = Join-Path $SourceDir \"ai_numeric.dll\""}, {"path": "fortran/build_fortran.ps1", "line": 11, "text": "throw \"Fortran compiler '$Compiler' was not found in PATH.\""}, {"path": "fortran/build_fortran.ps1", "line": 17, "text": "-static-libgfortran `"}, {"path": "fortran/build_fortran.ps1", "line": 24, "text": "throw \"Fortran compilation failed with exit code $LASTEXITCODE.\""}, {"path": "fortran/build_fortran.ps1", "line": 27, "text": "Write-Host \"Built Fortran library: $OutputFile\""}, {"path": "fortran/native_bridge.py", "line": 3, "text": "import ctypes"}, {"path": "fortran/native_bridge.py", "line": 13, "text": "return \"ai_numeric.dll\""}, {"path": "fortran/native_bridge.py", "line": 20, "text": "def _load_library() -> ctypes.CDLL:"}, {"path": "fortran/native_bridge.py", "line": 24, "text": "f\"Fortran library not found: {library_path}. \""}, {"path": "fortran/native_bridge.py", "line": 26, "text": "\"-File fortran/build_fortran.ps1\""}, {"path": "fortran/native_bridge.py", "line": 29, "text": "library = ctypes.CDLL(str(library_path))"}, {"path": "fortran/native_bridge.py", "line": 30, "text": "double_pointer = ctypes.POINTER(ctypes.c_double)"}, {"path": "fortran/native_bridge.py", "line": 32, "text": "library.compute_prediction_metrics.argtypes = ["}] | fortran/ai_numeric.f90; fortran/build_fortran.ps1 | grep hits | 仅限按证据原样引用；不得扩展为未证明指标。 |

## 15. 文件索引

| 文件 | 用途 |
|---|---|
| audit/ai_test_evidence/00_repository_inventory.md | audit output |
| audit/ai_test_evidence/01_model_input_output.md | audit output |
| audit/ai_test_evidence/02_simulation_training_sets.csv | audit output |
| audit/ai_test_evidence/02_simulation_training_sets.md | audit output |
| audit/ai_test_evidence/03_w30mo70_experiment_data.csv | audit output |
| audit/ai_test_evidence/03_w30mo70_experiment_data.md | audit output |
| audit/ai_test_evidence/04_temperature_acoustic_database.md | audit output |
| audit/ai_test_evidence/05_prediction_results.md | audit output |
| audit/ai_test_evidence/06_five_measurement_points.md | audit output |
| audit/ai_test_evidence/07_incremental_training.md | audit output |
| audit/ai_test_evidence/08_transfer_learning_and_freezing.md | audit output |
| audit/ai_test_evidence/09_timing_evidence.md | audit output |
| audit/ai_test_evidence/10_fortran_and_runtime_evidence.md | audit output |
| audit/ai_test_evidence/11_missing_ambiguous_inconsistent_items.md | audit output |
| audit/ai_test_evidence/12_next_step_classification.md | audit output |
| audit/ai_test_evidence/audit_summary.json | audit output |
| audit/ai_test_evidence/checkpoint_comparison_two_steady_metal_matrix_2026_5_16_0042.csv | audit output |
| audit/ai_test_evidence/checkpoint_comparison_two_steady_metal_matrix_2026_5_16_0042.json | audit output |
| audit/ai_test_evidence/checkpoint_inventory.csv | audit output |
| audit/ai_test_evidence/evidence_index.csv | audit output |
| audit/ai_test_evidence/file_hashes.csv | audit output |
| audit/ai_test_evidence/FINAL_AUDIT_REPORT.md | audit output |
| audit/ai_test_evidence/manifest_summary.csv | audit output |
| audit/ai_test_evidence/prediction_runs.csv | audit output |
| audit/ai_test_evidence/README.md | audit output |
| audit/ai_test_evidence/scripts/compare_checkpoints.py | audit script |
| audit/ai_test_evidence/scripts/scan_repository.py | audit script |
| audit/ai_test_evidence/scripts/summarize_manifests.py | audit script |
