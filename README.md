# ai_model：基于超声全波形与物理约束的人工智能温度场重构系统

## 快速开始与版本

当前目录为 `E:\code\ai_model`。已交付版本保存在 [`submitted` 分支](https://github.com/xyycyc/PINN/tree/submitted)（`626dc24`）；交付后重构使用 `codex/refactor-portability`。修改范围与待批准事项见[重构记录](docs/refactoring.md)，本轮全局检查见[审核记录](docs/audit_2026-09-12.md)，文档版本见[文档索引](docs/README.md)。

在项目目录直接执行：

```powershell
cd E:\code\ai_model
python -m pip install -r requirements.txt
python run.py --help
python run.py --gui
```

上面的 `python` 必须是已安装项目依赖的解释器。可用 `python -c "import sys; print(sys.executable)"` 确认；本轮验证使用 `D:\anaconda\python.exe`（Python 3.12.4），Python 安装位置与 E 盘项目位置可以不同。若系统默认 Python 缺少依赖，可先激活原环境，或在 PowerShell 中用 `& 'D:\anaconda\python.exe' run.py --gui` 启动。

预测指标和重复波形平均还需要本地 Fortran 库。源码受版本控制，编译出的 DLL 被 Git 忽略；首次检出或迁移后缺少 DLL 时，从项目根执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File fortran/build_fortran.ps1
python tools/check_environment.py
```

构建需要已有 GNU Fortran 编译器；已有可加载的库时可直接运行环境检查。详细说明见 [Fortran 构建与验证](fortran/README.md)。环境检查使用小数组验证数值接口，不读取业务数据。

`python run.py train ...` 等价于在上一级目录运行 `python -m ai_model train ...`。下文保留包入口示例，兼容原有脚本。

第一次使用 GUI：先设置输入/输出根目录，再依次建库、选择数据集训练、选择同一数据集对应的模型预测。默认手动选择数据清单；如需自动选择最新清单，可在各页面开启相应开关。程序不会自动下载原始数据或模型；原始数据、权重和运行结果被 `.gitignore` 排除，需要另外保留。

| 入口或模块 | 当前职责 |
| --- | --- |
| `build-db` / `data_process/` | 识别 case、波形与节点温度配对、采样、划分、清单与多材料集合 |
| `train` / `model/trainer.py` | 固定节点或旧网格模型训练、验证、早停与 checkpoint 登记 |
| `predict` / `model/predict.py` | 单模型或材料路由预测、可选绘图和测速 |
| `validate` | 数据需求核查；不重新运行预测或重算历史验收误差 |
| `online-update` | 基于兼容 checkpoint 对全部可训练参数继续优化，写出新权重 |
| `demo` | 连续建库、划分、训练和校验；不包含自动预测 |
| `window/` | Tkinter 操作界面与后台命令执行 |
| `batch/` | 保留的批量实验脚本；默认主窗口不显示这些页面 |
| `tests/` | 数据、模型、GUI/CLI 兼容性回归测试 |
| `cli.py` / `cli_arguments.py` / `commands.py` | 轻量入口 / 参数声明 / 六个独立命令处理函数 |
| `paths.py` | 配置、CLI 与 GUI 共用的项目路径解析 |

本文的数据规模、误差和耗时属于历史交付记录，不表示任意数据集或环境下都能取得同样结果。本轮代码检查和测试没有重新测量这些指标。

## 1. 系统概述

`ai_model` 是面向多层材料、金属基复合材料和碳基/硅基复合材料的人工智能温度场重构系统。系统以超声全波形张量为直接输入，通过 CNN–LSTM–BP 融合网络学习波形传播特征与温度场之间的非线性映射，并将温度场数据监督与基于固定物理节点空间邻接关系的 PINN 物理约束设计相结合，实现一维、二维、稳态和瞬态温度场的智能重构。

系统当前实现温度—声学参数数据库构建、超声波形导入与预处理、模型训练与保存、温度场预测、精度与效率评估、在线增量训练、多材料模型管理和结果导出。增量训练会继续优化全部可训练参数；参数冻结与部分层更新属于后续扩展设计。Windows 图形化界面提供八个主流程入口，Python 模块与 Fortran 数值库共同完成计算和结果记录。

系统面向以下三类材料：

1. 两层 W30Mo70 钨钼合金多层材料；
2. W 基体/SiC 颗粒金属基复合材料；
3. SiC 颗粒/CVI-SiC 基体硅基复合材料。

## 2. 历史交付数据与性能指标

### 2.1 数据资源

| 数据类别 | 材料体系 | 数据量 |
| --- | --- | ---: |
| 仿真数据 | 两层 W30Mo70 钨钼合金多层材料 | 1000 组 |
| 仿真数据 | W 基体/SiC 颗粒金属基复合材料 | 200 组 |
| 仿真数据 | SiC 颗粒/CVI-SiC 基体硅基复合材料 | 200 组 |
| 仿真数据合计 | 三类材料波形—温度场配对数据 | 1400 组 |
| 实验数据 | 两层 W30Mo70 钨钼合金多层材料 | 不少于 20 组 |

仿真数据覆盖 300～1500 K，记录超声全波形、物理节点坐标、节点温度、材料组分、边界身份和接触界面身份。实验数据用于实际波形输入、温度场预测和代表性测点对比。

项目文档与交付目标统一表述为 300～1500 K。程序执行需求核查时，以训练集仿真记录数作为样本数量口径，并采用 350～1450 K 的端点容差判断温度覆盖，避免有限采样或浮点误差导致覆盖范围误判。

### 2.2 温度重构精度

四类温度场反演工况采用五个代表性测点平均温度的相对误差进行评价：

| 工况 | 五点平均温度相对误差 | 评价结果 |
| --- | ---: | --- |
| 一维稳态 | 0.7221% | 小于 10% |
| 一维瞬态 | 3.3637% | 小于 10% |
| 二维稳态 | 0.7095% | 小于 10% |
| 二维瞬态 | 3.3221% | 小于 10% |

详细测点数据、指标定义和计算过程见[温度场重构误差指标与效率评估说明](docs/error_analysis.md)。

### 2.3 推理效率

4 个完整温度场的总推理耗时约为 0.248766 s，单个完整温度场平均推理耗时约为 0.0622 s。与传统反演方法相比：

| 方法 | 单场平均耗时/s | 相对 AI 加速比 |
| --- | ---: | ---: |
| AI 温度场重构 | 0.0622 | 1.0 |
| 共轭梯度法 | 18.67 | 300.2 |
| 灵敏度法 | 11.10 | 178.5 |
| 最速下降法 | 12.44 | 200.0 |

## 3. 技术原理

### 3.1 CNN–LSTM–BP 融合模型

系统模型以超声全波形张量为直接输入，采用 CNN 与 LSTM 双分支结构：

- CNN 分支提取局部波形形态、幅值变化、频域相关特征和温度敏感模式；
- LSTM 分支提取回波到达顺序、传播演化和长短期时序依赖；
- 两路特征经加权拼接后，由 BP 全连接网络完成非线性特征融合和温度场回归；
- 输出端生成固定物理节点上的完整二维温度场；
- 一维温度分布由二维温度场的固定中心线提取。

程序中的模型标识为 `DirectPointFieldModel`。其数据流可概括为：

```text
超声全波形张量
├── CNN 分支：局部形态与温度敏感特征
└── LSTM 分支：传播时序与长短期依赖
          ↓
      分支加权拼接
          ↓
      BP 全连接融合
          ↓
10,000 个固定物理节点二维温度场
          ↓
二维完整场 / 一维固定中心线结果
```

### 3.2 PINN 物理约束

固定节点温度场的物理约束依据节点真实物理坐标和材料信息建立空间邻接关系，在同一材料内部计算相邻节点温度差形成的空间平滑先验，并通过离散空间拉普拉斯残差约束预测温度场的局部空间一致性。接触界面两侧节点保持独立材料和界面身份，避免将不同材料或界面两侧节点作为普通连续节点直接平滑。

系统默认采用温度场监督训练；物理约束训练作为固定节点模型的扩展训练形式，通过空间平滑先验与离散拉普拉斯残差实现。

固定节点 `normal` 模式只计算温度监督损失；`residual_pinn` 模式的训练目标为：

$$
L=
L_{\mathrm{field}}
+0.05L_{\mathrm{smooth}}
+\lambda_pL_{\mathrm{laplacian}}
$$

其中：

- \(L_{\mathrm{field}}\) 为固定物理节点温度场监督损失；
- \(L_{\mathrm{smooth}}\) 为物理相邻节点之间温度差形成的空间平滑先验；
- \(L_{\mathrm{laplacian}}\) 为基于固定节点空间邻接关系构造的离散拉普拉斯残差；
- \(\lambda_p\) 为物理残差权重。

### 3.3 固定物理节点温度场

二维有限元模型包含约 23 万个原始节点。系统采用确定性的空间分层采样方法形成 10,000 个代表物理节点，并保留：

- 原始节点编号；
- 物理坐标；
- 温度标签；
- 材料组分；
- 边界身份；
- 接触界面两侧身份。

模型输出为 10,000 个固定物理节点上的完整二维温度场。固定采样索引保证训练、验证、测试和工程预测采用一致的物理节点定义，并支持结果回填、空间绘图和误差分区统计。

一维温度分布由二维场归一化 `x=0.5` 的固定中心线提取，按纵向坐标排序后输出温度曲线、参考曲线和误差曲线。

## 4. 系统功能

系统提供以下功能：

- 三材料温度—声学参数数据库构建；
- 超声全波形自动识别、导入、裁剪和平滑处理；
- 仿真波形与有限元温度场自动配对；
- 训练集、验证集和测试集按材料独立配置；
- CNN–LSTM–BP 模型训练、保存和加载；
- 数据监督与 PINN 物理约束融合；
- 一维、二维、稳态和瞬态温度场预测；
- 多材料共享模型，以及分材料完整模型与材料路由管理；
- 五测点精度、全场精度和分区精度分析；
- GPU 推理计时与传统反演方法效率比较；
- 在线增量训练，以及基于参数冻结与部分权重更新的迁移学习扩展设计；
- checkpoint、数据格式与采样索引一致性检查；
- Windows 图形化操作和命令行批处理；
- CSV、NPZ、JSON 和图像结果导出；
- 数据、模型、配置、日志和结果全过程追溯。

## 5. 环境准备

推荐使用 Python 3.10 或更高版本。在 `ai_model` 的上一级目录打开 PowerShell：

```powershell
cd E:\code
python --version
python -m pip install -r ai_model/requirements.txt
```

GPU 运行环境应安装与本机 CUDA 和显卡驱动相匹配的 PyTorch。Windows 图形界面使用 `tkinter`，可通过以下命令检查：

```powershell
python -c "import tkinter; print(tkinter.TkVersion)"
```

检查系统入口：

```powershell
python -m ai_model --help
```

本文中的 `python -m ai_model ...` 命令均在 `ai_model` 的上一级目录执行。

## 6. 目录与数据组织

默认输入根目录为 `ai_model/database`，默认输出根目录为 `ai_model/result`：

```text
ai_model/
├── database/
│   ├── raw/                         # 原始实验与仿真数据
│   ├── data_process/                # 建库后的清单、波形和温度场
│   └── rule/                        # 材料与模型规则登记
├── result/
│   ├── train/checkpoint/            # 模型文件
│   ├── train/report/                # 训练报告和统计
│   └── predict/
│       ├── inference/               # 单次预测
│       └── batch/                   # 批量预测
├── window/                          # Windows 图形化界面
└── cli.py                           # 命令行入口
```

### 6.1 标准 case 数据

系统支持分层和扁平两种 case 目录：

```text
worker_01/
└── case_0000_T0300p000K/
    ├── configs/
    │   └── *_ultrasonic_config.json
    ├── heat/thermomechanical_steady/csv/
    │   └── thermomechanical_steady_nodes.csv
    └── ultrasonic/
        └── receiver_signal.csv
```

```text
E:/pinn_data/wumu/
├── case_0000_T0300p000K/
├── case_0001_T0301p202K/
└── case_0999_T1500p000K/
```

温度场 CSV 的核心字段为：

```text
node_id,x,y,T,thermal_material_id,dup_target_surface
```

其中温度单位为 K，坐标单位为 m。相同坐标处的接触界面两侧节点通过节点编号、材料编号和界面身份分别保存。

### 6.2 建库输出

完成数据配对、固定节点采样和数据划分后，数据集目录包含：

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

多材料数据库通过 `material_collection.json` 记录每种材料的数据清单、划分比例和模型路由信息。逐材料配置使三类材料可根据任务需要形成训练、验证和测试组合；独立实验波形可作为指定材料的测试来源。

## 7. Windows 图形化操作

启动图形界面：

```powershell
python -m ai_model.window
```

界面依次提供：

1. 路径设置；
2. 构建数据库；
3. 训练模型；
4. 预测对比；
5. 校验数据；
6. 增量训练；
7. 一键演示；
8. 项目清理。

准备参数时，日志默认收起以留出更多表单空间；开始任务会自动展开。顶部显示运行、停止和完成状态及已用时间，右下角按钮按功能命名；运行期间其它执行按钮和项目删除按钮禁用。

命令页左下角的「预览命令」可以检查并复制当前命令，不启动任务。长说明会自动换行，数字格式错误会标记输入框并滚动到对应位置。顶部「保存表单」保存全部页面，空数字必须补全后才能保存；旧配置中的无效字段会恢复内置默认值，并展开日志列出字段，原文件需手动保存才会更新；「打开输出目录」打开当前路径配置对应的结果文件夹。重载配置保留当前页面、已有日志和日志显示状态。

### 7.1 路径设置

设置输入根目录、输出根目录和任务工作目录。Python 解释器固定为启动 GUI 的解释器。项目相对路径以 `ai_model` 包目录为基准，原始输入相对路径（如 `raw/wumu`）以输入根目录为基准；外部绝对路径需要在迁移后核对。

### 7.2 构建数据库

单材料任务选择材料 case 文件夹；多材料任务选择三类材料文件夹的共同上级目录，并在材料表中分别设置训练、验证和测试比例。

系统自动完成：

- case 发现与波形—温度场配对；
- 波形列和时间列读取；
- 温度单位与坐标单位统一；
- 约 23 万节点到 10,000 固定节点的确定性采样；
- 边界和接触界面节点保留；
- 数据划分及训练集统计量生成；
- 数据格式、网格和配对完整性检查。

对于独立实验波形，在“外部测试导入”中选择 post0 CSV 目录，设置挂接材料、数据集名称、时间列和波形列。系统将实验波形登记到指定材料的测试清单，用于模型预测和测点对比。

### 7.3 训练模型

在训练页选择数据集目录或 `material_collection.json`，设置任务维度、稳瞬态模式、材料规则名称、设备、训练轮数和早停参数。

多材料任务提供两种管理方式：

- 共享模型：将三类材料的训练清单合并后训练一个模型，作为默认方式；
- 分材料模型：为每种材料训练完整 checkpoint，并生成材料路由文件。

训练页默认要求验证清单。经 R6 批准新增的「允许无验证集训练」开关默认关闭，仅放行单数据集缺少验证清单的情况；这时不执行验证早停或最佳验证模型选择。已有验证集继续使用，多材料集合仍按各材料的 train/validation 划分训练。CLI 保留单清单无验证集训练的兼容行为。

分材料训练中，训练轮数对每种材料分别生效；共享训练中，训练轮数对应整个混合数据集。验证清单用于每轮评估和最佳模型选择，训练完成后模型按规则登记，供预测与增量训练调用。

### 7.4 预测对比

预测页可选择单材料测试清单、多材料集合或独立实验测试清单。

- 共享模型根据集合清单依次预测各材料数据；
- 分材料模型根据材料路由自动选择对应 checkpoint；
- 二维输出生成完整 10,000 节点温度场；
- 一维输出从二维场提取归一化 `x=0.5` 中心线；
- 绘图功能生成真实值、预测值和误差对比图；
- 性能评估功能记录单场耗时、吞吐率和显存信息。

### 7.5 校验数据

校验入口对选定清单或材料集合执行数据需求核查并写出 `requirement_3_3_report.json`。节点、采样和数据划分的一致性检查分布在建库、数据加载和训练/预测入口；预测误差由 `predict` 生成，`validate` 不重新预测。

### 7.6 增量训练与迁移学习

增量训练页加载基础 checkpoint 和新增数据清单，可配置训练轮数、设备和分支权重。当前优化全部可训练参数，没有冻结编码器或仅训练部分层的选项；新增冻结策略需另行设计。新模型与基础任务建立关联，训练报告写入独立时间目录，并登记到模型规则库。增量输出不得覆盖基础权重、已有增量文件或已有报告目录；出现冲突时需使用新的文件名或在新的时间戳下重试。

### 7.7 一键演示

一键演示按照界面配置连续执行建库、划分、训练和校验，用于展示系统完整业务链路。多材料模式按逐材料配置完成数据组织与模型训练。

### 7.8 项目清理

项目清理页按任务规则展示模型、训练报告和预测结果。先扫描关联产物并检查删除预览，再点击固定在底部的「执行删除…」；保留原有确认流程。任务运行期间暂时禁止删除，结束后会重新构建删除计划。依赖扫描覆盖全部增量后代；选择保留时先检查所有目标是否冲突，并把继承配置保存到独立模型。清理路径必须位于当前配置对应的 checkpoint、训练报告或预测目录内；元数据含越界路径或目标重名时会拒绝操作。

## 8. 命令行操作

图形界面覆盖主要业务流程；命令行适合批量任务、服务器运行和自动化执行。CLI 所有小数参数均拒绝 `NaN` 和无穷大，避免非有限值进入训练或数据处理。

### 8.1 单材料建库

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

### 8.2 多材料建库

```powershell
python -m ai_model build-db `
  --data-root database `
  --result-root result `
  --skip-simulation `
  --experiment-dir E:/pinn_data `
  --experiment-material multi_material_v1 `
  --multi-material-input `
  --material-split wumu=0.7,0.1,0.2 `
  --material-split metal=0.7,0.1,0.2 `
  --material-split silicon=0.7,0.1,0.2 `
  --split-dataset
```

### 8.3 导入独立实验测试波形

```powershell
python -m ai_model build-db `
  --data-root database `
  --result-root result `
  --skip-simulation `
  --experiment-dir E:/pinn_data `
  --experiment-material multi_material_v1 `
  --multi-material-input `
  --material-split wumu=0.8,0.2,0 `
  --material-split metal=0.7,0.2,0.1 `
  --material-split silicon=0.7,0.2,0.1 `
  --external-test-dir E:/raw/wumu_exp_post0 `
  --external-test-material wumu `
  --external-test-dataset-name wumu_exp `
  --external-test-signal-column amplitude_filtered_residual `
  --external-test-time-column time_s `
  --external-test-time-min-s 0 `
  --split-dataset
```

### 8.4 模型训练

```powershell
python -m ai_model train `
  --data-root database `
  --result-root result `
  --manifest ai_model/database/data_process/multi_material_v1_multi_material_temperature_field `
  --train-name multi_material_temperature_v1 `
  --checkpoint-name ai_model.pt `
  --rule-dimension two `
  --rule-mode steady `
  --rule-material multi_material `
  --device cuda `
  --early-stopping-patience 10 `
  --epochs 500
```

该命令默认训练一个多材料共享模型。如需为每种材料生成独立 checkpoint 和材料路由文件，在相同命令中增加：

```text
--separate-materials
```

### 8.5 温度场预测

共享多材料模型预测：

```powershell
python -m ai_model predict `
  --data-root database `
  --result-root result `
  --manifest ai_model/database/data_process/multi_material_v1_multi_material_temperature_field/material_collection.json `
  --checkpoint ai_model/result/train/checkpoint/multi_material_temperature_v1/ai_model.pt `
  --predict-name multi_material_prediction `
  --plots `
  --prediction-dimension two `
  --benchmark
```

分材料模型与材料路由预测：

```powershell
python -m ai_model predict `
  --data-root database `
  --result-root result `
  --manifest ai_model/database/data_process/multi_material_v1_multi_material_temperature_field/material_collection.json `
  --material-router ai_model/result/train/checkpoint/material_checkpoints_v1/ai_model__material_router.json `
  --predict-name routed_material_prediction `
  --plots `
  --prediction-dimension one `
  --benchmark
```

`--prediction-dimension two` 输出完整二维节点场；`--prediction-dimension one` 输出二维场的固定中心线结果。

### 8.6 数据校验

```powershell
python -m ai_model validate `
  --data-root database `
  --result-root result `
  --manifest ai_model/database/data_process/multi_material_v1_multi_material_temperature_field/material_collection.json
```

`validate` 入口会识别 `material_collection.json` 的集合类型，逐材料解析对应的 combined manifest，并汇总记录数、材料数、温度范围和数据校验结果。

### 8.7 在线增量训练

```powershell
python -m ai_model online-update `
  --data-root database `
  --result-root result `
  --manifest ai_model/database/data_process/wumu_case_temperature_field/train_manifest.json `
  --checkpoint ai_model/result/train/checkpoint/wumu_temperature_v1/ai_model.pt `
  --output-name wumu_incremental_v2.pt `
  --rule-dimension two `
  --rule-mode steady `
  --rule-material wumu `
  --device cuda `
  --epochs 20
```

### 8.8 一键演示

```powershell
python -m ai_model demo `
  --data-root database `
  --result-root result `
  --train-name point_demo `
  --device cpu `
  --epochs 20 `
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

查看命令帮助：

```powershell
python -m ai_model build-db --help
python -m ai_model train --help
python -m ai_model predict --help
python -m ai_model validate --help
python -m ai_model online-update --help
python -m ai_model demo --help
```

## 9. 波形预处理

训练、预测、增量训练和一键演示共用以下预处理参数：

```text
--preprocess clip,smooth,detrend,robust_norm
--clip-quantile 1.0
--smooth-window 11
```

处理步骤包括：

- `clip`：按分位数处理脉冲峰值；
- `smooth`：采用滑动窗口平滑波形；
- `detrend`：消除基线趋势；
- `robust_norm`：采用稳健统计量完成幅值归一化。

可选预处理配置随 checkpoint 保存，预测与增量训练默认继承；显式覆盖才改变这些配置。波形 z-score 在模型输入阶段按样本计算。固定节点温度标准化使用训练集统计量，并随 checkpoint 保存。

## 10. 模型与结果管理

模型文件记录模型类型、固定节点数量、采样指纹、温度统计量、波形预处理参数、CNN/LSTM 分支权重和任务规则。系统采用版本化 checkpoint 管理机制，根据模型元数据自动匹配对应的数据格式和推理流程，并在训练、预测和增量训练入口执行一致性检查。

预测目录主要包含：

- `predictions.csv`：节点编号、坐标、预测温度、参考温度、材料和界面身份；
- `predictions.npz`：完整温度场数组；
- `metrics.json`：全场、边界、界面、材料分区、高温区的 MAE/RMSE/最大绝对误差，以及样本数、节点数、模型与 checkpoint 信息、评估耗时、参数量和内存信息；选择一维结果时增加中心轴指标，启用独立测速时增加 benchmark 字段；
- `metadata.json`：模型、数据、采样、单位及运行环境信息；
- `point_field_compare.png`：启用绘图并选择二维结果时生成的温度场对比图；
- `axis_predictions.csv/.npz`：选择一维结果时生成的中心线预测结果；
- `axis_compare.png`：选择一维结果并启用绘图时生成的真实值、预测值和误差对比图。

五个代表性测点的验收指标依据固定测点映射和节点预测结果形成独立评价记录，不作为 `metrics.json` 的固定字段。

训练目录保存 checkpoint、损失曲线、训练过程记录、最佳轮次和任务统计。数据清单、模型元数据、预测日志及结果文件共同构成全过程追溯链路。

## 11. 误差评价方法

合同验收主指标为五个代表性测点平均温度的相对误差：

$$
T_{\mathrm{ref,avg}}=\frac{1}{5}\sum_{i=1}^{5}T_{\mathrm{ref},i}
$$

$$
T_{\mathrm{AI,avg}}=\frac{1}{5}\sum_{i=1}^{5}T_{\mathrm{AI},i}
$$

$$
\delta_T=
\frac{|T_{\mathrm{AI,avg}}-T_{\mathrm{ref,avg}}|}
{\max(|T_{\mathrm{ref,avg}}|,\varepsilon)}\times100\%
$$

其中，\(\varepsilon\) 为防止参考平均温度接近零时出现零分母的正数。合同验收只使用上述“先计算五点平均温度，再计算相对误差”的公式。

MAE、RMSE、最大绝对误差、中心化 RMSE、空间相关系数、边界误差、界面误差、材料分区误差和高温区误差用于分析温度场空间分布质量和定位局部误差。

效率评价采用：

$$
\mathrm{Speedup}=\frac{t_{\mathrm{traditional}}}{t_{\mathrm{AI}}}
$$

AI 方法与传统方法统一以完成一个完整温度场反演任务所需的平均时间作为统计口径。在满足温度反演精度要求的基础上，计算传统方法平均耗时与 AI 单场平均推理耗时之比。

## 12. 常见操作提示

### 12.1 Python 包入口

在 `ai_model` 的上一级目录执行：

```powershell
cd E:\code
python -m ai_model --help
```

### 12.2 GPU 环境检查

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### 12.3 Windows 中文输出

图形界面自动采用 UTF-8。命令行任务可设置：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

### 12.4 模型与数据一致性

预测和增量训练时，系统根据 checkpoint 元数据检查模型类型、节点数量、采样索引、波形长度和温度统计量。选择同一建库任务生成的数据清单和模型，即可保持完整的数据处理与物理节点对应关系。

## 13. 系统能力与质量保证

本系统形成三材料温度—声学数据库、CNN–LSTM–BP 温度场重构模型、PINN 物理约束融合方法及配套的软件操作和结果分析能力，支持：

- 三类材料、1400 组仿真配对数据和不少于 20 组实验数据；
- 300～1500 K 温度范围；
- 一维、二维、稳态和瞬态温度场重构；
- 10,000 个固定物理节点完整二维温度场；
- 二维温度场固定中心线的一维结果；
- 在线增量训练，以及基于参数冻结与部分权重更新的迁移学习扩展设计；
- GPU 快速推理和效率统计；
- Windows 图形化界面；
- Python 人工智能模块与 Fortran 数值计算模块协同；
- 多材料模型路由和版本化 checkpoint 管理；
- 数据、模型、配置、日志和结果全过程追溯；
- 自动化测试、数据质量检查和版本一致性验证。

系统通过标准化数据格式、确定性物理节点采样、模型元数据管理和完整结果导出，为温度场重构任务提供统一、稳定、可追溯的技术支撑。

## 14. 开发与验证

在 `E:\code` 执行：

```powershell
python -m pip install pytest
python -m pytest ai_model/tests -q
```

先运行 `python tools/check_environment.py` 确认解释器和 Fortran 库。也可以在 `E:\code\ai_model` 直接执行 `python -m pytest -q`；`pyproject.toml` 已配置测试目录和包路径。模型格式见[模型迁移说明](docs/model_migration.md)，GUI 操作细节见[窗口说明](window/README.md)，历史交互约束见[兼容基线](docs/compatibility_baseline.md)。

GUI 使用体验回归可在项目根运行 `python tools/gui_smoke.py`；Windows 上附加 `--screenshots` 可同时生成实际窗口截图（需要 Pillow 和 pywin32）。脚本使用临时数据和配置，报告默认写入 `audit/gui_usability/smoke/`；详情见[窗口说明](window/README.md#交互回归)。
