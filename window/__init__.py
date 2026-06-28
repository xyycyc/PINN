"""ai_model 的中文图形化操作窗口（基于 tkinter）。

提供模型构建、训练、预测、校验、增量训练、项目清理等主流程可视化界面，
所有窗口相关代码集中在 ``ai_model/window`` 子包下，便于统一维护。

启动方式（需能从 Python 路径导入 ``ai_model`` 包；子进程默认在 **包目录的上一级** 作为 cwd 执行 ``python -m ai_model``，与命令行常见用法一致）::

    python -m ai_model.window

路径设置页可指定 **子进程工作目录**（写入 ``settings.json`` 的 ``common``）；解释器固定为启动 GUI 的 Python。
"""

from .app import AiModelApp, launch

__all__ = ["AiModelApp", "launch"]
