# GUI/CLI 兼容基线（2026-07-12）

窗口标题为“ai_model 图形化操作面板”，尺寸 `1280x820`，最小尺寸 `1080x720`。主流程 Tab 顺序固定为：路径设置、构建数据库、训练模型、预测对比、校验数据、增量训练、一键演示、项目清理。公共区域包括运行日志、清空/复制/保存日志、自动滚动和状态栏；菜单包括文件、运行、配置、帮助。所有命令型 Tab 保留“执行/停止”。

字段默认值快照为 `window/settings.json`。命令由各 Tab 的 `compose_command()` 生成，入口保持 `python -m ai_model <subcommand>`。必须保留的子命令为 `build-db`、`train`、`validate`、`online-update`、`predict`、`demo`。关键默认值：`data_root=database`、`result_root=result`、`device=cuda`、`training_mode=normal`、`physics_residual_weight=0.1`、`epochs=20`、划分种子 42。

最小流程产物：建库生成 manifest、`waveforms/*.npy`、`fields/*.npy`；训练生成 `result/train/checkpoint/<任务名>/*.pt` 与 report；预测生成 `predictions.{npz,csv}` 和 `metrics.json`；校验生成 `requirement_3_3_report.json`。新版 case 温度场独立写入 `database/case_temperature_field`，不静默改变旧产物含义。
