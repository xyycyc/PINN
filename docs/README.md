# 文档索引与适用版本

本索引针对交付后维护分支 `codex/refactor-portability`。当前功能以源码、主 README 和窗口说明为准；已交付快照为 [`submitted`](https://github.com/xyycyc/PINN/tree/submitted)（`626dc24`）。

| 当前使用与维护资料 | 内容 |
| --- | --- |
| [项目 README](../README.md) | 入口、安装与迁移、主要流程、命令示例 |
| [窗口操作说明](../window/README.md) | 八个功能页、保存与恢复、无验证集开关、任务与清理 |
| [Fortran 构建说明](../fortran/README.md) | 本地数值库依赖、编译、自检 |
| [模型迁移说明](model_migration.md) | checkpoint 契约、基础模型引用和增量保护 |
| [固定节点数据格式](temperature_field_schema.md) | 记录、采样、材料与标准化契约 |
| [兼容基线与变更](compatibility_baseline.md) | 交付时约束及已批准的后续变更 |
| [重构与审批记录](refactoring.md) | 分支备份、历轮验证、待确认功能 |
| [本轮全局审核](audit_2026-09-12.md) | 发现、修复、测试及剩余限制 |

## 历史交付与研究资料

[历史误差说明](error_analysis.md)、[研究审核](../research_audit/README.md)和[交付归档](../output/README.md)保留其原有数据来源与时间。它们不能证明当前代码在任意新数据集上达到历史精度或耗时。

`output/` 中既有 Word/PDF 手册、培训记录、`before_delivery` 和 QA 文件，以及根目录和 `文本/` 下的原始文档，均作为交付或编写过程资料保留。当前 GUI 已有改动，使用维护分支时按上表操作。历史文档不批量替换截图、签署内容、数据或日期；这样可保留提交给甲方版本的事实记录。需要再次对外交付 Word/PDF 时，应基于现行说明生成带新版本号的交付包。
