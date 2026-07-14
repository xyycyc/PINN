# ai_model 图形化操作面板

`ai_model.window` 是 `ai_model` 子项目的中文 GUI 操作面板，目标是把
主文档 [`../README.md`](../README.md) 中的命令行用法用图形界面包装起来，方便选择模式、修改超参数、
查看产物，而不必每次都手敲 `python -m ai_model ...`。

## 启动方式

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

应用主窗口由三块组成：

1. 顶部 **Notebook**：当前仅展示 “主流程” 分组（批量实验与结果分析已暂时隐藏），
   分组下挂多个功能 Tab。
2. 中下部 **运行日志面板**：实时显示子进程 stdout/stderr，支持
   清空 / 复制 / 保存日志、自动滚动到底部。
3. 底部 **状态栏**：显示当前任务状态与最近一条命令摘要。

工具栏（菜单）提供 “停止当前任务 / 退出 / 关于” 等。

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

每个命令型 Tab 都遵循统一布局：

- 上方表单：各 Tab 专属参数；主流程将「输入/输出根目录」和 **子进程工作目录** 集中在 **路径设置**，
  其它主流程 Tab 执行命令时会自动带上输入/输出根目录；Python 始终使用启动窗口的解释器，工作目录可由路径页覆盖；
  设备、训练轮数、权重、预处理等仍在各自功能页配置；
- 右下两个按钮：
  - **执行**：根据表单内容拼出 `<启动窗口的 python> -m ai_model <subcommand> --opt val ...`，
    在后台启动子进程并实时把输出写到日志面板；子进程 **cwd** 默认为 **ai_model 包目录的上一级**（便于 `python -m ai_model`），可在路径设置中覆盖；
  - **停止**：先 `terminate`，超时则 `kill`，安全终止当前任务。

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

- 构建数据库和一键演示可扫描多材料上层目录，并为每个直接子材料文件夹单独设置 train/validation/test 比例；
- 训练和演示可对 `material_collection.json` 中每种样本材料分别训练完整 checkpoint；预测页按记录的 `material_key` 选择对应模型；
- 训练运行参数只展示当前网络实际存在的 CNN、LSTM 两个固定分支权重；
- 训练、预测、校验、增量训练默认自动选择最新清单，关闭开关后严格使用手填清单；
- 预测/增量训练默认继承 checkpoint 预处理，只有勾选覆盖开关才发送本页预处理参数。
- 固定节点训练与增量训练目前只支持 `normal`；界面会在启动子进程前拦截 `residual_pinn`。
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
  超时再 `kill`；
- 关闭窗口前若发现仍有任务在跑，会弹窗确认是否一起终止。

## 文件结构

```
ai_model/window/
├── README.md            # 本文档
├── __init__.py          # 暴露 launch 入口
├── __main__.py          # python -m ai_model.window 入口
├── app.py               # 主窗口、Notebook、日志面板、状态栏
├── runner.py            # 子进程运行器（流式输出 + 终止）
├── widgets.py           # 复用控件 + 中文字体/样式
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
