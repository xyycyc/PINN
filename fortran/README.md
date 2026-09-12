# Fortran 数值库

`ai_numeric.f90` 提供预测误差计算和重复实验波形平均，`native_bridge.py` 通过 `ctypes` 调用。两类数组均使用 `real(c_double)`，长度使用 `integer(c_int64_t)`，接口通过 `ISO_C_BINDING` 声明。

- `compute_prediction_metrics`：MAE、RMSE 和最大绝对误差，由预测模块使用。
- `average_waveform_pairs`：重复实验的时间和电压数组平均。

## Windows 构建

在项目根执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File fortran/build_fortran.ps1
python tools/check_environment.py
```

需要 GNU Fortran（`gfortran`）在 PATH 中，并与 Python 使用相同位数。也可通过 `-Compiler '编译器绝对路径'` 指定已有编译器。脚本从本目录源码生成 `ai_numeric.dll`，静态链接 Fortran/GCC 运行库；修改源码后需要重新构建。

DLL 和 `.mod` 文件属于本机构建产物，被 Git 忽略。首次检出仓库或迁移后仅安装 `requirements.txt` 不会生成该库。已有适配本机的可信 DLL 时可以直接验证；不要用不同架构的库替换。

`check_environment.py` 显示所用 Python 路径，检查运行依赖，用已知小数组校验两种数值接口。GUI 交互另用 `python tools/gui_smoke.py` 检查。脚本不会读取生产数据或训练模型。

Python 包装器在库缺失或无法加载时明确失败，没有 NumPy 静默替代实现。Linux/macOS 分别需要 `libai_numeric.so` / `libai_numeric.dylib`；仓库当前构建脚本面向 Windows，其他平台应按相同接口自行构建。

本轮本机验证：Python 3.12.4、64 位；从仓库源码构建 DLL 后，两种接口的小数组检查均通过。更多环境记录见[审核记录](../docs/audit_2026-09-12.md)。
