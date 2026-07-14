# ai_model：单波形到固定物理节点温度场

本目录提供温度场建库、训练、预测、校验、增量训练和图形窗口。当前主流程将一个实验 case 表示为：

```text
一条接收波形 + 固定物理节点上的温度标签
```

真实 case 使用版本化、确定性的空间分层采样，将约 23 万个有限元节点降采样为 10,000 个代表节点；温度始终以 K 落盘，坐标始终以 m 落盘，并保留节点、材料及接触界面两侧身份。

相关设计文档：

- [GUI/CLI 兼容基线](docs/compatibility_baseline.md)
- [固定节点温度场 schema v2](docs/temperature_field_schema.md)
- [模型版本与迁移说明](docs/model_migration.md)
- [图形窗口说明](window/README.md)
- [任务状态与实验路线](task%20log.md)

## 1. 环境准备

推荐使用 Python 3.10 或更高版本。在 `ai_model` 的上一级目录打开 PowerShell：

```powershell
cd D:\Desktop\code
python --version
python -m pip install --upgrade pip
python -m pip install torch numpy pandas tqdm matplotlib scipy Pillow
```

如需 CUDA，请安装与本机 CUDA/显卡驱动匹配的 PyTorch。`tkinter` 通常随 Windows Python 一起安装，可这样检查：

```powershell
python -c "import tkinter; print(tkinter.TkVersion)"
```

确认主程序可用：

```powershell
python -m ai_model --help
```

所有本文中的 `python -m ai_model ...` 命令都应在 `ai_model` 的上一级目录执行；也可以传绝对路径。

## 2. 目录约定

默认输入根目录是 `ai_model/database`，默认输出根目录是 `ai_model/result`：

```text
ai_model/
├── database/
│   ├── raw/                         # 原始实验/仿真文件
│   ├── data_process/                # 建库后的 manifest、波形和温度场
│   └── rule/                        # 材料和 checkpoint 规则登记
├── result/
│   ├── train/checkpoint/            # 模型 checkpoint
│   ├── train/report/                # 训练曲线、历史和统计
│   └── predict/
│       ├── inference/               # 单次推理
│       └── batch/                   # 批量推理
├── window/                          # 中文图形窗口
└── cli.py                           # CLI 入口
```

可用 `--data-root` 和 `--result-root` 修改两个根目录。相对路径按 `ai_model` 包目录解析；manifest/checkpoint 也可直接传绝对路径。

## 3. 真实 case 数据格式

推荐原始数据布局：

```text
database/raw/calibration_sweep/
└── worker_01/
    └── case_0000_T0300p000K/
        ├── configs/
        │   └── *_ultrasonic_config.json
        ├── heat/thermomechanical_steady/csv/
        │   └── thermomechanical_steady_nodes.csv
        ├── ultrasonic/
        │   ├── receiver_signal.csv
        │   ├── receiver_signal_rigid_corrected.csv
        │   └── receiver_signal_rigid_motion.csv
        └── logs/
```

也支持将 case 直接放在实验根目录下的扁平布局，例如：

```text
E:/pinn_data/wumu/
├── case_0000_T0300p000K/
├── case_0001_T0301p202K/
└── case_0999_T1500p000K/
```

扁平布局下，建库程序会从每个 case 的唯一 `*_ultrasonic_config.json` 中读取
`config.io.temperature_csv_path`，恢复原始 `worker_XX/case_...` 唯一键；配置缺失、
指向其他 case 或存在歧义时会明确报错，不会伪造 worker 编号。

约束：

- 超声 JSON 的 `config.io.temperature_csv_path` 必须指向同一 case 的热力 CSV。
- 默认使用 `receiver_signal.csv`；可由后端显式选择刚体校正版。
- `receiver_signal_rigid_motion.csv` 只是派生校正量，永远不会被当作独立实验。
- 热力 CSV 至少需要 `node_id,x,y,T`；建议同时包含 `thermal_material_id` 和 `dup_target_surface`。
- 相同 `(x,y)` 的接触界面两侧节点按 `node_id/material/interface_side` 区分，不按坐标合并。

当实验目录包含上述嵌套或扁平 case 布局时，默认 GUI/CLI 建库流程会自动识别真实 case，并输出到：

```text
database/data_process/<dataset_label>_case_temperature_field/
```

其中主要文件为：

```text
manifest.json
train_manifest.json
validation_manifest.json
test_manifest.json
split_config.json
mesh_audit.json
sampling_index.npz
waveforms/*.npy
temperature_fields/*.npz
```

固定节点默认划分为训练 70%、验证 10%、测试 20%。温度均值和标准差只由训练集拟合，再原样用于验证和测试，避免信息泄漏。

## 4. 最推荐操作：图形窗口

启动窗口：

```powershell
python -m ai_model.window
```

主流程 Tab 顺序保持为：

1. 路径设置
2. 构建数据库
3. 训练模型
4. 预测对比
5. 校验数据
6. 增量训练
7. 一键演示
8. 项目清理

### 4.1 路径设置

- 输入根目录：默认 `database`。
- 输出根目录：默认 `result`。
- 子进程工作目录：建议留空，默认使用 `ai_model` 的上一级目录。
- Python 解释器：使用启动窗口的 Python。
- 如需持久化当前值，使用菜单“配置 → 把当前表单保存为默认值”。

### 4.2 构建数据库

单材料建库时直接选择材料文件夹，例如 `E:/pinn_data/wumu`：

- 跳过内置仿真：开启。
- 实验目录：默认 `raw/wumu`，也可浏览到外部的 `E:/pinn_data/wumu`。
- 数据集命名标签：默认 `wumu`，只影响目录/任务命名；样本材料路由字段取所选文件夹名。
- 多材料输入：平时关闭；开启后选择 `wumu` 等材料文件夹的上层目录，并在动态表中逐材料填写划分比例。
- case 波形固定前缀长度：默认 `1097` 点，只裁掉尾部，不重采样。
- 自动划分：开启。
- 测试集比例：默认 `0.2`。
- 验证集比例：固定节点默认 `0.1`，训练集为剩余 `0.7`。
- 随机种子：默认 `42`。

点击“执行”后，窗口会完成配对审计、网格一致性审计、10,000 点采样、真实温度标签提取以及 case 级划分。

### 4.3 训练模型

- 训练页直接选择包含 `split_config.json`、train manifest 和 validation manifest 的数据集文件夹；默认自动选择最新文件夹。
- train manifest 参与反向传播，validation manifest 每轮只做前向评估；最终 checkpoint 恢复为 validation loss 最低的 epoch。
- 规则维度：一维选 `one`，二维固定节点 case 建议选 `two`。
- 规则模式：稳态选 `steady`，瞬态选 `transient`。
- 材料名称应与本次训练任务一致。
- 设备可选 `cuda / cpu / auto / gpu`。
- 填写训练轮数并点击“执行”。
- 选择 `material_collection.json` 后可勾选“按材料分别训练 checkpoint”，为每个材料文件夹生成一个完整模型和路由 JSON。
- CNN/LSTM 数值在固定模式下是固定权重，在可学习模式下是两个权重的初始值。
- 固定节点模型目前只支持 `normal`；`residual_pinn` 仅用于旧规则网格模型，界面会在启动任务前拦截错误组合。

训练器根据 manifest 自动选择模型：

- schema v2 固定节点 manifest → `direct_point_field`，输出 10,000 点。
- 旧 manifest → 原 `AIReconstructionModel`，输出 `(24,24)`。

训练完成后 checkpoint 会按规则三元组登记，供预测和增量训练页选择。

### 4.4 预测对比

- 测试 manifest 默认自动选择最新 `test_manifest.json`，也可关闭自动选择后手工指定。
- 只显示与当前 manifest 模型类型兼容的 checkpoint。
- 可选择单 checkpoint 或材料路由；自动发现路由时只选择与当前数据集目录匹配的文件。
- 可选择是否绘图和测速；固定节点与材料路由同样会实际执行样本数和纯前向测速设置。
- 默认完整继承 checkpoint 的预处理；仅勾选“覆盖 checkpoint 预处理”时使用本页参数。
- 点击“执行”后，固定节点模型会输出原始节点预测表、NPZ、metrics 和 metadata。

### 4.5 校验数据

自动选择最新 combined manifest，检查记录数、实验/仿真数量、材料和温度范围。固定节点真实 case 不包含内置仿真时，旧的“3.3 仿真数量”条件可能显示 false；这不等于 schema、采样或训练链路失败。

### 4.6 增量训练

- 自动选择最新 combined manifest。
- 只列出与 manifest 类型兼容的基础 checkpoint。
- 默认继承基础 checkpoint 的设备、预处理、训练模式和权重；预处理需要显式勾选才会覆盖。
- 点击执行时会保留用户在表单中显式修改的增量轮数、设备和权重，不会再次静默恢复默认值。
- 新权重写入基础任务对应目录，报告写入 `Incremental/<时间戳>/`。
- 完成后自动写入 `database/rule/trained_rules.csv`。

### 4.7 一键演示

点击一次完成建库、划分、训练和校验。正常模式选择单个材料目录（例如 `wumu`）；开启“多材料输入”后选择这些材料文件夹的上层目录，并为每种材料独立设置 train/validation/test 比例。只有多材料上层目录可以启用分别训练；单个 `wumu` 始终训练为一个完整 checkpoint。

### 4.8 项目清理

按规则三元组扫描 checkpoint、报告、预测结果和规则登记，并实时沿用“路径设置”的根目录。根目录改变时旧选择会立即失效，后端也拒绝清理当前 checkpoint 根目录以外的文件。分别训练产生的材料 checkpoint 属于一个完整路由组，清理其中任意一个时会在二次确认后整组清理，避免留下失效路由。清理是不可逆操作，执行前请核对窗口列出的路径。

窗口日志支持停止任务、复制、清空和保存；Windows 中文/空格路径及 UTF-8 输出已通过回归测试。

## 5. CLI 快速完整流程

以下示例均从 `D:\Desktop\code` 执行。

### 5.1 建库

```powershell
python -m ai_model build-db `
  --data-root database `
  --result-root result `
  --skip-simulation `
  --experiment-dir E:/pinn_data/wumu `
  --experiment-material wumu `
  --experiment-limit 1000 `
  --waveform-crop-length 1097 `
  --split-dataset `
  --split-test-ratio 0.2 `
  --split-validation-ratio 0.1 `
  --split-seed 42
```

实验目录中存在嵌套 `worker_*/case_*` 或扁平 `case_*` 布局时，上述命令自动进入真实固定节点管线。样本级材料路由字段取所选文件夹名，因此这里是 `wumu`，正式名称为“钨钼多层材料”。热力 CSV 内部的 `layer_1`（钨）和 `layer_2`（钼）只是该材料的组分层，不参与 checkpoint 数量和样本路由。`--experiment-material` 只作为数据集/输出命名标签。

多材料建库时选择 `wumu` 等材料文件夹的上层目录；每个直接子文件夹是一种独立样本材料：

```powershell
python -m ai_model build-db `
  --data-root database `
  --result-root result `
  --skip-simulation `
  --experiment-dir E:/pinn_data `
  --experiment-material multi_material_v1 `
  --multi-material-input `
  --material-split wumu=0.7,0.1,0.2 `
  --material-split steel=0.6,0.1,0.3 `
  --split-dataset
```

输出的 `material_collection.json` 保存每种材料各自的清单和划分比例。未显式传入 `--material-split` 的材料使用 0.7/0.1/0.2。

如果目录不包含可识别的 case，命令才会尝试旧版数值温度 CSV 导入。旧版导入若没有得到任何有效记录会以非零状态退出，并在输出目录写入 `*_import_report.json`，列出跳过或解析失败原因；不会再生成“0 条记录但 exit_code=0”的空数据库。

旧 CSV/仿真建库示例：

```powershell
python -m ai_model build-db `
  --data-root database `
  --result-root result `
  --no-skip-simulation `
  --sim-per-material 400 `
  --experiment-dir raw/legacy_experiment `
  --experiment-material metal_matrix `
  --experiment-limit 2000
```

`build-db` 参数：

| 参数 | 含义/默认值 |
| --- | --- |
| `--data-root` | 输入根目录，默认 `database` |
| `--result-root` | 输出根目录，默认 `result` |
| `--sim-per-material` | 每种内置材料仿真数量，默认 400 |
| `--experiment-dir` | 实验目录，默认 `raw/10times` |
| `--experiment-material` | 数据集命名标签；不决定固定节点样本的材料类型 |
| `--multi-material-input` | 将实验目录的直接子文件夹识别为多种样本材料 |
| `--material-split` | `材料文件夹=train,validation,test`，可重复传入 |
| `--experiment-limit` | 实验 case/CSV 上限；0 表示不导入 |
| `--waveform-crop-length` | 固定节点 case 原始波形连续前缀点数，默认 1097 |
| `--skip-simulation / --no-skip-simulation` | 跳过/启用内置仿真，默认跳过 |
| `--external-sim-dir` | 外部仿真 CSV 目录，默认关闭 |
| `--external-sim-material` | 外部仿真材料；空值时复用实验材料 |
| `--external-sim-limit` | 外部仿真上限；负数不限，0 不导入 |
| `--split-dataset / --no-split-dataset` | 是否自动划分，默认开启 |
| `--split-test-ratio` | 测试集比例，默认 0.2 |
| `--split-validation-ratio` | 固定节点验证集比例，默认 0.1 |
| `--split-seed` | 划分/采样随机种子，默认 42 |
| `--split-experiment-policy` | 旧数据策略：`uniform / all_experiment_train / all_experiment_test` |
| `--train-manifest-name` | 训练 manifest 文件名 |
| `--validation-manifest-name` | 固定节点验证 manifest 文件名 |
| `--test-manifest-name` | 测试 manifest 文件名 |

### 5.2 训练

固定节点训练示例：

```powershell
python -m ai_model train `
  --data-root database `
  --result-root result `
  --manifest ai_model/database/data_process/metal_matrix_case_temperature_field `
  --train-name metal_matrix_point_v1 `
  --checkpoint-name ai_model.pt `
  --rule-dimension two `
  --rule-mode steady `
  --rule-material metal_matrix `
  --device cuda `
  --early-stopping-patience 10 `
  --epochs 500
```

训练公共参数：

| 参数 | 含义 |
| --- | --- |
| `--manifest` | 训练数据集文件夹；自动读取 `split_config.json` 指向的 train/validation 清单，仍兼容直接传 train manifest |
| `--validation-manifest` | 可选，显式覆盖自动解析的 validation manifest |
| `--train-name` | checkpoint/report 子目录名 |
| `--checkpoint-name` | checkpoint 文件名，默认 `ai_model.pt` |
| `--separate-materials / --no-separate-materials` | 对 `material_collection.json` 中每个样本材料训练完整 checkpoint，默认关闭 |
| `--rule-dimension` | `one / two` |
| `--rule-mode` | `steady / transient` |
| `--rule-material` | 材料规则名称；三项规则参数必须同时传或同时不传 |
| `--device` | `cuda / cpu / auto / gpu` |
| `--epochs` | 训练轮数，CLI 默认 20 |
| `--early-stopping-patience` | validation loss 连续未改善多少代后停止，默认 10；必须大于 0 |
| `--training-mode` | `normal / residual_pinn`；固定节点仅支持 `normal` |
| `--physics-residual-weight` | residual PINN 残差权重，默认 0.1 |
| `--learnable-branch-weights` | 启用可学习分支权重 |
| `--fixed-weight-cnn` | CNN 固定权重，或可学习模式的初始值 |
| `--fixed-weight-lstm` | LSTM 固定权重，或可学习模式的初始值 |

训练会在每代结束后计算 validation loss。连续达到“早停耐心代数”仍未刷新最佳值时，训练立即停止，并恢复 validation loss 最低那一代的模型权重；没有 validation manifest 时不会触发早停。

旧版曾暴露 `--fixed-weight-material/--fixed-weight-dimension/--fixed-weight-mode`，但当前网络没有这三个前向分支。参数名仅为旧命令识别而保留，显式传入会报清晰错误，避免产生“权重已生效”的假象；规则 CSV 中的同名列仅用于读取历史记录。

多材料固定节点集合可选择分别训练：

```powershell
python -m ai_model train `
  --manifest ai_model/database/data_process/multi_material_v1_multi_material_temperature_field `
  --train-name material_checkpoints_v1 `
  --checkpoint-name field.pt `
  --rule-dimension two `
  --rule-mode steady `
  --rule-material multi_material_v1 `
  --separate-materials
```

若集合有三种材料，会生成三个 `field__<文件夹名>.pt` 和一个 `field__material_router.json`。每个 checkpoint 都预测该材料的完整固定节点温度场，使用该材料自己的训练划分更新权重，并用对应验证划分选择最佳 epoch。关闭分别训练时，CLI 会分别合并各材料的 train/validation 划分；只有采样网格和波形长度完全一致时允许混合训练，否则要求启用分别训练。

### 5.3 预测

显式指定 checkpoint：

```powershell
python -m ai_model predict `
  --data-root database `
  --result-root result `
  --manifest ai_model/database/data_process/metal_matrix_case_temperature_field/test_manifest.json `
  --checkpoint ai_model/result/train/checkpoint/metal_matrix_point_v1/ai_model.pt `
  --output-dir ai_model/result/predict/inference/metal_matrix_point_v1_test `
  --device cuda `
  --plots `
  --num-field-samples 4 `
  --benchmark `
  --benchmark-warmup-samples 64 `
  --benchmark-runs 3
```

也可通过已登记规则自动查找 checkpoint：

```powershell
python -m ai_model predict `
  --manifest ai_model/database/data_process/metal_matrix_case_temperature_field/test_manifest.json `
  --rule-dimension two `
  --rule-mode steady `
  --rule-material metal_matrix
```

样本材料自动路由：

```powershell
python -m ai_model predict `
  --manifest ai_model/database/data_process/multi_material_v1_multi_material_temperature_field/material_collection.json `
  --material-router ai_model/result/train/checkpoint/material_checkpoints_v1/field__material_router.json `
  --output-dir ai_model/result/predict/inference/material_checkpoints_v1_test
```

预测参数：

| 参数 | 含义 |
| --- | --- |
| `--manifest` | 必填，待预测 manifest |
| `--checkpoint` | checkpoint；或使用完整规则三元组自动解析 |
| `--material-router` | 样本材料路由 JSON；依据记录/材料文件夹的 `material_key` 调用对应完整 checkpoint |
| `--output-dir` | 明确的输出目录 |
| `--predict-name` | 未传输出目录时使用的任务名 |
| `--plots / --no-plots` | 开关预测图，默认关闭 |
| `--num-field-samples` | 绘图样本数，默认 6 |
| `--benchmark / --no-benchmark` | 开关纯前向测速，默认关闭 |
| `--benchmark-warmup-samples` | 预热样本数，默认 64 |
| `--benchmark-runs` | 计时 batch 轮数，默认 3 |
| `--predict-kind` | 内部兼容参数：`inference / batch` |

固定节点 `predictions.csv` 每行对应一个物理节点：

```text
sample_id,node_id,x_m,y_m,temperature_k,target_temperature_k,constituent_material_id,interface_side
```

同时生成：

- `predictions.npz`：完整数组。
- `metrics.json`：全场、边界、界面、各材料、高温区的 MAE/RMSE/最大误差，以及参数量、内存和推理时间。
- `metadata.json`：checkpoint、schema、采样版本、网格指纹、单位和训练集标准化参数。
- `point_field_compare.png`：仅在启用绘图时生成，不覆盖原始节点表。

这里不训练额外的波形材料分类器。单个 manifest 直接读取 `records[].material_key`；集合预测读取材料文件夹对应的集合条目，然后调用该材料的完整 checkpoint。`sampling_index.material_ids` 仅描述材料内部的组分层。

### 5.4 校验

```powershell
python -m ai_model validate `
  --data-root database `
  --result-root result `
  --manifest ai_model/database/data_process/metal_matrix_case_temperature_field/manifest.json
```

报告写入：

```text
result/train/report/validation/requirement_3_3_report.json
```

### 5.5 增量训练

```powershell
python -m ai_model online-update `
  --data-root database `
  --result-root result `
  --manifest ai_model/database/data_process/metal_matrix_case_temperature_field/train_manifest.json `
  --checkpoint ai_model/result/train/checkpoint/metal_matrix_point_v1/ai_model.pt `
  --output-name point_update_001.pt `
  --rule-dimension two `
  --rule-mode steady `
  --rule-material metal_matrix `
  --device cuda `
  --epochs 20
```

如不传 `--output-name`，使用训练时间作为文件名。固定节点增量训练会严格检查 checkpoint 版本、点数和温度标准化参数。

### 5.6 一键演示

```powershell
python -m ai_model demo `
  --data-root database `
  --result-root result `
  --train-name point_demo `
  --device cpu `
  --epochs 1 `
  --experiment-limit 20 `
  --experiment-material wumu `
  --waveform-crop-length 1097 `
  --split-dataset `
  --split-test-ratio 0.2 `
  --split-validation-ratio 0.1 `
  --split-seed 42 `
  --rule-dimension two `
  --rule-mode steady `
  --rule-material wumu
```

正式训练时应增大 `--epochs`；1 epoch 只适合验证链路，不代表工程精度。固定节点多材料演示可额外传 `--separate-materials`。

### 5.7 查看任意命令的实时帮助

```powershell
python -m ai_model build-db --help
python -m ai_model train --help
python -m ai_model predict --help
python -m ai_model validate --help
python -m ai_model online-update --help
python -m ai_model demo --help
```

## 6. 波形预处理参数

`train / predict / online-update / demo` 共用以下参数：

```text
--preprocess clip,smooth,detrend,robust_norm
--clip-quantile 1.0
--smooth-window 11
```

可选步骤：

- `clip`：按分位数裁峰。
- `smooth`：滑动窗口平滑，窗口自动转成奇数。
- `detrend`：去基线趋势。
- `robust_norm`：稳健归一化。

模型输入前始终执行 z-score，并记录波形统计。不要在 `--preprocess` 中重复写 `zscore`。

## 7. 模型与 checkpoint 兼容规则

### 固定节点模型

```text
checkpoint_version = 3
model_kind = direct_point_field
schema_version = 1
sampling_version = spatial-stratified-v2
```

checkpoint 记录 `point_count`、采样指纹、训练集温度统计和 CNN/LSTM 分支权重。当前写出版本为 3；读取端仍兼容版本 2，并把旧模型缺失的两个权重恢复为其历史等效值 `1.0/1.0`。预测或增量训练时，点数、采样或标准化任一项不一致都会明确报错。

### 旧模型

没有 `model_kind` 的历史 checkpoint 继续按旧规则网格模型加载，输出 `(24,24)`。不会把旧 checkpoint 静默解释为固定节点模型，也不会把固定节点 checkpoint 用到旧 manifest。

## 8. 实验与批处理脚本

以下脚本不属于六个稳定主入口，但仍可独立执行。先查看帮助：

```powershell
python -m ai_model.batch.batch_test_modes --help
python -m ai_model.batch.batch_preprocess_test --help
python -m ai_model.batch.search_fixed_weights --help
python -m ai_model.batch.plot_batch_test_loss_curves --help
python -m ai_model.batch.rerun_predict --help
```

常用示例：

```powershell
# 四种模式对比
python -m ai_model.batch.batch_test_modes --data-root database/raw --result-root result --epochs 5000 --device cuda --seed 42

# 多种预处理批量对比
python -m ai_model.batch.batch_preprocess_test --data-root database/raw --result-root result --epochs 5000 --device cuda

# 当前模型有效的 CNN × LSTM 固定权重笛卡尔积搜索
python -m ai_model.batch.search_fixed_weights --data-root database/raw --result-root result --epochs 1000 --device cuda
```

批处理脚本主要服务旧批量实验；固定节点主流程优先使用六个 CLI 入口或 Window。旧 `physical_weights` 搜索维度现会明确拒绝非 1.0 值，因为当前网络没有对应物性分支，继续搜索只会重复等价训练。

## 9. 自动化测试

从 `ai_model` 上一级目录执行：

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest discover -s ai_model\tests -v
```

测试覆盖：

- CLI 子命令和默认值。
- Window Tab 顺序与命令拼装。
- 固定节点 schema 校验。
- 采样确定性、材料预算和界面节点保留。
- case 级无泄漏划分。
- 训练集温度标准化。
- Dataset 到 10,000 点模型的基本数据契约。
- Window 对旧/新 split_config 的发现。

## 10. 常见问题

### 10.1 `No module named ai_model`

终端位置错误。切换到 `ai_model` 上一级目录：

```powershell
cd D:\Desktop\code
python -m ai_model --help
```

### 10.2 CUDA 不可用

先用 CPU 验证：

```powershell
python -m ai_model train ... --device cpu --epochs 1
```

然后确认：

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### 10.3 checkpoint 与 manifest 不兼容

固定节点 manifest 只能使用 `model_kind=direct_point_field`、`checkpoint_version=2/3` 的 checkpoint（当前训练写出 v3）。请确认训练和预测使用同一 `sampling_index.npz`、相同 point count 和相同训练集标准化参数。

### 10.4 建库提示固定节点不能与旧仿真合并

发现 calibration_sweep 时保持：

```text
--skip-simulation
不传 --external-sim-dir
```

### 10.5 校验报告 `meets_3_3=false`

该字段沿用旧项目对仿真数量、材料数量和温度跨度的要求。仅含真实 calibration case 的固定节点数据可能不满足这些旧数量条件；应同时查看 mesh audit、schema、划分泄漏检查和实际预测指标。

### 10.6 Windows 中文输出乱码

Window 子进程会自动设置 UTF-8。直接运行 CLI 时可执行：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## 11. 当前验收边界

数据管线、固定采样、Window/CLI 适配、版本检查和预测节点导出已经可用。正式模型是否达到工程要求，仍取决于项目方确认的 MAE、RMSE、最大误差、界面误差、推理时间及内存阈值。不要用 1 epoch smoke checkpoint 声称模型精度达标。
