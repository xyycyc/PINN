# ai_model 图形化操作面板

`ai_model.window` 是 `ai_model` 子项目的中文 GUI 操作面板，目标是把
主文档 [`../README.md`](../README.md) 中的命令行用法用图形界面包装起来，方便选择模式、修改超参数、
查看产物，而不必每次都手敲 `python -m ai_model ...`。

## 启动方式

在项目根目录 `E:\code\ai_model` 执行 `python run.py --gui`，或保留下面的包入口。

在已将 `ai_model` 加入 Python 路径的环境中执行（常见：在 **`ai_model` 的上一级目录** 打开终端）：

```bash
python -m ai_model.window
```

依赖：
- Python 3.10+
- `tkinter`（Python 标准库自带）
- `Pillow`（结果浏览 Tab 的图片预览，缺失时仅退化为提示文本）
- 训练/推理子进程依赖见项目根 [`requirements.txt`](../requirements.txt)

## 界面结构

主窗口提供八个功能页，顺序与交付版一致。浅色背景、分组表单、统一标题和强调色按钮区分主要操作。

- 顶部提供「保存表单」「打开输出目录」，以及持续可见的任务状态和已用时间。
- 中间为功能页。表单可滚动，底部操作按钮始终可见；输入框下方显示可自动换行的说明，键盘切换焦点时会自动滚动到对应字段。
- 日志默认收起，开始任务自动展开；可以手动收起或拖动分隔条调整高度。完整日志支持复制、保存和跟随最新输出。
- 状态栏显示保存、复制、任务结束等反馈；文件、运行、配置、帮助菜单继续保留。

初始窗口会根据屏幕大小缩小，通常为 `1280x820`，最小为 `1000x640`。

## 各 Tab 对照表

| 分组 | Tab | 对应命令 |
| --- | --- | --- |
| 主流程 | 路径设置 | （共用输入/输出根目录和子进程工作目录；Python 固定为启动窗口的解释器；写入 `settings.json` 的 `common`） |
| 主流程 | 构建数据库 | `python -m ai_model build-db` |
| 主流程 | 训练模型 | `python -m ai_model train` |
| 主流程 | 预测对比 | `python -m ai_model predict` |
| 主流程 | 校验数据 | `python -m ai_model validate` |
| 主流程 | 增量训练 | `python -m ai_model online-update` |
| 主流程 | 一键演示 | `python -m ai_model demo` |
| 主流程 | 项目清理 | （内置：按规则组合扫描并清理关联权重/报告/预测结果与 CSV 登记） |

每个命令页先填写参数，再使用底部操作栏：

- **预览命令**：检查并复制当前表单生成的命令，不启动任务；同时显示工作目录。
- **开始训练 / 开始预测 / 开始校验等**：校验输入后启动后台任务。数字错误直接标记对应输入框，避免关闭弹窗后还要寻找错误字段；`NaN` 和无穷大也视为无效数字。
- **停止任务**：仅在有任务时可用，确认后在后台终止子进程，等待期间界面可继续操作。任务结束后恢复执行按钮。

「路径设置」只显示「保存共用路径」，不会启动任务；它只保存 `common` 段。顶部「保存表单」保存全部页面，有无效数字时会切换到出错页，且不部分改写配置。重新加载配置会保留当前页面、日志、折叠状态和自动滚动偏好。

输入/输出根目录和子进程工作目录集中在路径页，其它页面使用当前表单的值；Python 始终使用启动窗口的解释器。项目路径相对包目录，实验及外部原始数据目录相对输入根目录，历史 `database/...` 写法仍相对包目录。文件选择器使用同样的路径规则，路径尚不存在时从最近的现有父目录打开。

项目清理页的扫描和删除按钮固定在底部，删除预览可以滚动。已有任务执行时禁止删除，任务结束后重新生成清理计划；原来的确认和依赖处理规则保持不变。

## 超参数与模式

下面这些参数在多个 Tab 中复用，含义与主项目 [`README.md`](../README.md) 中 CLI 说明一致：

- **设备 `--device`**：`cuda / cpu / auto / gpu`
- **训练模式 `--training-mode`**：`normal / residual_pinn`
- **物理残差权重 `--physics-residual-weight`**：仅 `residual_pinn` 生效
- **可学习分支权重 `--learnable-branch-weights`**：勾选启用，
  CNN/LSTM 输入值作为初值；不勾选时则固定为表单值（默认均为 `0.75`）
- **预处理流水线 `--preprocess`**：逗号分隔，可选
  `clip,smooth,detrend,robust_norm`；模型输入前会自动执行 z-score，不要在流水线中重复填写，
  也可在 “预设” 下拉里直接选常用组合

各 Tab 还会暴露各自专属的参数（如 `--sim-per-material`、
`--experiment-limit`、`--num-field-samples`、`--pipelines` 等），
名字与 CLI 完全对齐，便于查阅 [`README.md`](../README.md)。

近期兼容行为：

- 构建数据库和一键演示可扫描多材料上层目录，并为每个直接子材料文件夹单独设置 train/validation/test 比例；多材料模式隐藏旧全局比例，只采用逐材料表；
- 构建数据库页可把 post0 原始波形作为 `inference_only` 外部测试集挂接到指定材料，外部样本不参与训练和验证；
- 训练和演示可对 `material_collection.json` 中每种样本材料分别训练完整 checkpoint；预测页按记录的 `material_key` 选择对应模型；
- 训练运行参数只展示当前网络实际存在的 CNN、LSTM 两个固定分支权重；
- 训练、预测、校验、增量训练默认手动选择清单；需要自动发现时可开启“自动使用最新清单”。已有配置显式开启自动发现时继续保留；
- 预测/增量训练默认继承 checkpoint 预处理，只有勾选覆盖开关才发送本页预处理参数。
- 二维固定节点模型可选择输出完整二维场，或在预测后提取归一化 `x=0.5` 中心轴生成一维对比图；一维旧模型只能选择一维输出。
- 固定节点训练与增量训练支持 `normal` 和 `residual_pinn`；后者增加空间平滑与离散拉普拉斯损失。
- `wumu` 是一种完整的“钨钼多层材料”；`layer_1/layer_2` 只是内部组分层，不会拆成两个 checkpoint。
- 一键演示会登记生成的混合 checkpoint 或分材料完整 checkpoint，预测页可直接发现。

## 结果浏览 Tab（当前界面暂时隐藏，代码保留）

- 默认根目录为 `result`（相对 **ai_model 包根目录** 解析），可点 “浏览…” 切换；
- 左侧目录树支持懒加载（点击展开时才扫描子目录），不会卡住大目录；
- 右侧预览：
  - `*.png/.jpg/.gif` 等图片：用 Pillow 缩放后展示；
  - `*.json`：自动美化（`json.dumps(indent=2, ensure_ascii=False)`）；
  - `*.csv/.txt/.md/.log`：直接以等宽字体展示，最大 256 KB，多余截断；
  - 其他类型：仅显示元信息，可点击 “在系统中打开”。

## 子进程行为说明

- `runner.py` 强制设置 `PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`，
  避免 Windows 控制台默认 GBK 把中文 tqdm/print 打成乱码；
- 子进程的 **cwd** 默认为 **ai_model 包目录的上一级**（与在上一级目录执行 `python -m ai_model <subcommand>` 一致），
  可在「路径设置」中填写「子进程工作目录」覆盖；**Python 解释器固定为启动 GUI 的 Python**；
- 一次只允许一个任务运行；停止任务先 `terminate` 5 秒等待，
  超时再 `kill`，等待在后台完成；
- 关闭窗口前若发现仍有任务在跑，会弹窗确认是否一起终止，并在后台结束任务后关闭窗口。
- 修改子进程工作目录后对下一次执行立即生效；保存设置用于下次启动。
- 日志按批次刷新，任务状态由主线程处理；重载配置保留已有日志。
- 日志中的 Windows 命令按 PowerShell 语法引用，支持路径含空格、单引号和特殊字符。

## 文件结构

```
ai_model/window/
├── README.md            # 本文档
├── __init__.py          # 暴露 launch 入口
├── __main__.py          # python -m ai_model.window 入口
├── app.py               # 主窗口、Notebook、日志面板、状态栏
├── runner.py            # 子进程运行器（流式输出 + 终止）
├── widgets.py           # 响应式表单、路径选择与错误定位
├── theme.py             # 统一配色、字体与 ttk 样式
├── task_status.py       # 任务状态和经过时间
├── dialogs.py           # 只读命令预览
└── tabs/
    ├── __init__.py
    ├── base.py              # BaseCommandTab 公共基类
    ├── build_db.py          # build-db
    ├── train.py             # train
    ├── predict.py           # predict
    ├── validate.py          # validate
    ├── online_update.py     # online-update
    ├── demo.py              # demo
    ├── batch_modes.py       # 批量实验（当前界面暂不展示）
    ├── batch_preprocess.py  # 批量实验（当前界面暂不展示）
    ├── search_weights.py    # 批量实验（当前界面暂不展示）
    ├── plot_loss.py         # 结果分析（当前界面暂不展示）
    ├── rerun_predict.py     # 结果分析（当前界面暂不展示）
    ├── result_browser.py    # 结果浏览（当前界面暂不展示）
    └── manage_artifacts.py  # 项目清理（主流程）
```

如果以后 `cli.py` / `ai_model.batch` 下脚本增加了新参数，只需要在对应 Tab 里
再加几行 `Labeled*` 控件并把它拼到 `compose_command()` 里即可。

## 交互回归

在项目根运行：

```powershell
python -m pytest -q
python tools/gui_smoke.py
python tools/gui_smoke.py --screenshots
```

`gui_smoke.py` 用 Tk 自身的事件循环模拟选页、按钮、输入和重载。三个窗口尺寸覆盖八个页面；检查路径输入宽度、控件边界和底部按钮可见性。还验证输入报错与定位、失败保存不改配置、预览不启动进程、路径选择、真实 `validate` 子进程、停止、失败状态、运行中禁止清理和重载日志保留。校验使用空记录的临时清单，验证的是命令执行与界面反馈，不能用来判断业务数据是否达到验收要求。

默认输出为 `audit/gui_usability/smoke/report.json` 和 `session.log`；`--screenshots` 额外保存实际窗口截图，使用 Windows 的 PrintWindow，只捕获脚本创建的窗口，需要 Pillow 和 pywin32。测试配置、数据和命令产物均位于临时目录，不使用生产训练数据和模型。
