# ai_model：基于超声全波形与物理约束的人工智能温度场重构系统

## 1. 系统概述

`ai_model` 是面向多层材料、金属基复合材料和碳基/硅基复合材料的人工智能温度场重构系统。系统以超声全波形张量为直接输入，通过 CNN–LSTM–BP 融合网络学习波形传播特征与温度场之间的非线性映射，并将温度场数据监督与 PINN 物理约束相结合，实现一维、二维、稳态和瞬态温度场的智能重构。

系统覆盖温度—声学参数数据库构建、超声波形导入与预处理、模型训练与保存、温度场预测、精度与效率评估、在线增量训练、参数冻结、迁移学习、多材料模型管理和结果导出。Windows 图形化界面提供完整操作入口，Python 人工智能模块与 Fortran 数值计算模块协同完成数据生成、模型训练、预测分析及全过程追溯。

系统面向以下三类材料：

1. 两层 W30Mo70 钨钼合金多层材料；
2. W 基体/SiC 颗粒金属基复合材料；
3. SiC 颗粒/CVI-SiC 基体硅基复合材料。

## 2. 交付数据与性能指标

### 2.1 数据资源

| 数据类别 | 材料体系 | 数据量 |
| --- | --- | ---: |
| 仿真数据 | 两层 W30Mo70 钨钼合金多层材料 | 1000 组 |
| 仿真数据 | W 基体/SiC 颗粒金属基复合材料 | 200 组 |
| 仿真数据 | SiC 颗粒/CVI-SiC 基体硅基复合材料 | 200 组 |
| 仿真数据合计 | 三类材料波形—温度场配对数据 | 1400 组 |
| 实验数据 | 两层 W30Mo70 钨钼合金多层材料 | 不少于 20 组 |

仿真数据覆盖 300～1500 K，记录超声全波形、物理节点坐标、节点温度、材料组分、边界身份和接触界面身份。实验数据用于实际波形输入、温度场预测和代表性测点对比。

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

模型训练将温度场数据监督损失与 PINN 物理约束相结合，在控制预测温度与参考温度数值偏差的同时，引入温度场空间连续性、热传导规律、边界条件及合理温度范围等约束，提高重构结果的物理一致性和稳定性。

训练目标可概括为：

$$
L=L_{\mathrm{MSE}}+\lambda_p L_{\mathrm{phys}}+\lambda_s L_{\mathrm{smooth}}
$$

其中：

- \(L_{\mathrm{MSE}}\) 为温度场数据监督损失；
- \(L_{\mathrm{phys}}\) 为 PINN 物理约束损失；
- \(L_{\mathrm{smooth}}\) 为空间连续性和平滑约束；
- \(\lambda_p\) 和 \(\lambda_s\) 为对应权重。

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
- 多材料共享模型和分材料模型管理；
- 五测点精度、全场精度和分区精度分析；
- GPU 推理计时与传统反演方法效率比较；
- 在线增量训练、参数冻结和迁移学习；
- checkpoint、数据格式与采样索引一致性检查；
- Windows 图形化操作和命令行批处理；
- CSV、NPZ、JSON 和图像结果导出；
- 数据、模型、配置、日志和结果全过程追溯。

## 5. 环境准备

推荐使用 Python 3.10 或更高版本。在 `ai_model` 的上一级目录打开 PowerShell：

```powershell
cd D:\Desktop\code
python --version
python -m pip install --upgrade pip
python -m pip install torch numpy pandas tqdm matplotlib scipy Pillow
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

### 7.1 路径设置

设置输入根目录、输出根目录、Python 解释器和任务工作目录。表单参数可保存为用户配置，供后续任务复用。

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

- 共享模型：将三类材料的训练清单合并后训练一个模型；
- 分材料模型：为每种材料训练完整 checkpoint，并生成材料路由文件。

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

校验模块综合检查数据规模、数据格式、材料信息、温度范围、网格一致性、采样索引、数据划分和预测指标，并输出完整核查报告。

### 7.6 增量训练与迁移学习

增量训练页加载基础 checkpoint 和新增数据清单，可配置训练轮数、设备、分支权重及参数冻结策略。新模型与基础任务建立关联，训练报告写入独立时间目录，并登记到模型规则库。

### 7.7 一键演示

一键演示按照界面配置连续执行建库、划分、训练和校验，用于展示系统完整业务链路。多材料模式按逐材料配置完成数据组织与模型训练。

### 7.8 项目清理

项目清理页按任务规则展示模型、训练报告和预测结果，执行前提供路径核对和二次确认，便于维护规范的交付目录。

## 8. 命令行操作

图形界面覆盖主要业务流程；命令行适合批量任务、服务器运行和自动化执行。

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

如需为每种材料生成独立模型，在相同命令中增加：

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

分材料路由预测：

```powershell
python -m ai_model predict `
  --data-root database `
  --result-root result `
  --manifest ai_model/database/data_process/multi_material_v1_multi_material_temperature_field/material_collection.json `
  --material-router ai_model/result/train/checkpoint/material_checkpoints/ai_model__material_router.json `
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

波形统计参数随模型保存，并在预测和增量训练阶段保持一致。

## 10. 模型与结果管理

模型文件记录模型类型、固定节点数量、采样指纹、温度统计量、波形预处理参数、CNN/LSTM 分支权重和任务规则。系统采用版本化 checkpoint 管理机制，根据模型元数据自动匹配对应的数据格式和推理流程，并在训练、预测和增量训练入口执行一致性检查。

预测目录主要包含：

- `predictions.csv`：节点编号、坐标、预测温度、参考温度、材料和界面身份；
- `predictions.npz`：完整温度场数组；
- `metrics.json`：全场、边界、界面、材料分区和高温区误差；
- `metadata.json`：模型、数据、采样、单位及运行环境信息；
- `point_field_compare.png`：二维温度场对比图；
- `axis_predictions.csv/.npz`：一维中心线预测结果；
- `axis_compare.png`：一维真实值、预测值和误差对比图。

训练目录保存 checkpoint、损失曲线、训练过程记录、最佳轮次和任务统计。数据清单、模型元数据、预测日志及结果文件共同构成全过程追溯链路。

## 11. 误差评价方法

合同验收主指标为五个代表性测点平均温度的相对误差：

$$
T_{\mathrm{ref,avg}}=\frac{1}{5}\sum_{i=1}^{5}T_{\mathrm{ref},i}
$$

$$
T_{\mathrm{pred,avg}}=\frac{1}{5}\sum_{i=1}^{5}T_{\mathrm{pred},i}
$$

$$
E_{\mathrm{5point}}=
\frac{|T_{\mathrm{pred,avg}}-T_{\mathrm{ref,avg}}|}
{|T_{\mathrm{ref,avg}}|}\times100\%
$$

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
cd D:\Desktop\code
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
- 在线增量训练、参数冻结和迁移学习；
- GPU 快速推理和效率统计；
- Windows 图形化界面；
- Python 人工智能模块与 Fortran 数值计算模块协同；
- 多材料模型路由和版本化 checkpoint 管理；
- 数据、模型、配置、日志和结果全过程追溯；
- 自动化测试、数据质量检查和版本一致性验证。

系统通过标准化数据格式、确定性物理节点采样、模型元数据管理和完整结果导出，为温度场重构任务提供统一、稳定、可追溯的技术支撑。
