## ai_model 数据目录规范（2026-05）

当前版本将输入与输出统一锚定为两个可配置根目录：

- 输入根目录：`data`（可通过 `--data-root` 修改）
- 输出根目录：`result`（可通过 `--result-root` 修改）

相对路径默认都相对于仓库根目录解析。

### 输入目录（data）

- 原始数据：放在 `data` 下任意子目录（推荐 `data/raw/...`）
- 数据库缓存：`data/cache/database/`
  - `fields/*.npy`
  - `waveforms/*.npy`
  - `simulation_manifest.json`
  - `experiment_manifest.json`
  - `combined_manifest.json`

### 输出目录（result）

- 训练 checkpoint：
  - `result/train/checkpoint/<训练名称>/<checkpoint文件名>`
- 训练报告与曲线：
  - `result/train/report/<训练名称>/`
- 推理结果（单次）：
  - `result/predict/inference/<推理名称>/`
- 推理结果（批量）：
  - `result/predict/batch/<推理名称>/`

### 命名规则

- 训练支持 `--train-name` 与 `--checkpoint-name`
  - 未传 `--train-name` 时，默认用 checkpoint 文件名去后缀作为训练名称
- 推理支持 `--predict-name` 与 `--predict-kind`
  - `--predict-kind` 取值：`inference` / `batch`
  - 未传 `--predict-name` 时，默认沿用历史命名：`<checkpoint_stem>_predictions`

### GUI（window）同步

`build-db / train / validate / predict / online-update / demo` 标签页已支持：

- 输入根目录（`--data-root`）
- 输出根目录（`--result-root`）

其中 `train`、`predict` 支持填写名称参数；**增量训练**（`online-update`）默认将权重写入基础 checkpoint 同级目录（`<训练时间>.pt`），报告写入 `report/<基础任务名>/Incremental/<训练时间>/`，并在完成后**自动登记**到 `database/rule/trained_rules.csv`（与全量训练相同，可按三元组在下拉框中匹配）。GUI 刷新规则映射时会补登记磁盘上尚未写入 CSV 的检查点。
