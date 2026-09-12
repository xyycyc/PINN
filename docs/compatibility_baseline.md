# GUI/CLI 兼容基线（2026-07-13）

窗口标题为“ai_model 图形化操作面板”，尺寸 `1280x820`，最小尺寸 `1080x720`。主流程 Tab 顺序固定为：路径设置、构建数据库、训练模型、预测对比、校验数据、增量训练、一键演示、项目清理。公共区域包括运行日志、清空/复制/保存日志、自动滚动和状态栏；菜单包括文件、运行、配置、帮助。所有命令型 Tab 保留“执行/停止”。

字段默认值快照为 `window/settings.json`。命令由各 Tab 的 `compose_command()` 生成，入口保持 `python -m ai_model <subcommand>`。必须保留的子命令为 `build-db`、`train`、`validate`、`online-update`、`predict`、`demo`。关键默认值：`data_root=database`、`result_root=result`、`device=cuda`、`training_mode=normal`、`physics_residual_weight=0.1`、`epochs=20`、划分种子 42、固定节点验证/测试比例 0.1/0.2、case 波形固定前缀长度 1097。

最小流程产物：旧网格建库生成 manifest、`waveforms/*.npy`、`fields/*.npy`；固定节点 case 建库写入 `database/data_process/<标签>_case_temperature_field/`，包含 `temperature_fields/*.npz`、采样定义和 train/validation/test 清单。多材料建库另生成 `material_collection.json`，每个直接子材料文件夹具有独立划分。训练生成 `result/train/checkpoint/<任务名>/*.pt` 与 report；材料分开训练还生成 `*__material_router.json`。预测生成 `predictions.{npz,csv}`、`metadata.json` 和 `metrics.json`；校验生成 `requirement_3_3_report.json`。

必须维持的交互契约：关闭“自动使用最新清单”后不得覆盖手选 manifest；自定义 data/result 根目录须同时作用于规则发现与项目清理；切换清理根目录必须使旧 checkpoint 选择失效，且后端不得删除当前 checkpoint 根以外的文件；推理和增量训练默认继承 checkpoint 预处理，执行增量训练时不得覆盖用户刚修改的 runtime；固定节点绘图样本数与 benchmark 必须真实生效；分材料 checkpoint 清理按完整路由组处理，不能留下损坏的路由 JSON；Demo 生成的 checkpoint 必须登记后供预测页选择。


## 2026-09-12 交付后变更

交付快照固定在 `submitted`（`626dc24`）。本轮保留六个 CLI 命令的参数与默认值；旧 `ai_model.cli` 导入接口通过兼容转发保留。

- R1 已获用户批准：新 GUI 配置默认手动选择数据清单。自动发现仍可开启，已有配置显式保存的 `auto_manifest=true` 不被改写；自动发现测试需主动开启开关。
- 可在包目录执行 `python run.py --help`、`python run.py --gui`，原 `python -m ai_model` 入口不变。
- 任务停止和退出等待改在后台完成；Tk 控件只由主线程更新。
- 子进程工作目录从当前表单读取，保存操作只决定下次启动的默认值。
- 配置提交失败保留原文件和内存值，重新加载使用当前配置文件并保留日志。
