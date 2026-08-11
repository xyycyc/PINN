# Current PINN physics assessment

## What `residual_pinn` actually computes

- Legacy grid: unscaled finite-difference `d2x + d2y`, squared and averaged；另有相邻像素 smoothness。
- Fixed nodes: same-material k-nearest-neighbor graph；`|T_i-T_j|` smoothness + normalized graph Laplacian squared。
- Interface nodes are excluded from the graph residual；boundary nodes are excluded from the Laplacian residual。

## Missing physical terms

训练 loss 未使用 density、heat capacity、thermal conductivity、transient derivative、heat source、heat flux、initial condition、boundary condition、interface temperature/flux continuity 或 material-dependent coefficients。没有量纲、空间尺度或时间尺度一致的 heat-equation residual。

原始 solver CSV/config **包含**部分 `rho/cp/k` 和边界信息，但模型 loss 没有消费这些信息。名称中的 `PINN` 因而不能作为科学物理模型证据。

## Decision

**Can the current PINN formulation be scientifically reused as-is? No.**

It is useful as an engineering spatial regularizer, but should not automatically become the paper's physical model. 未来若使用 physics constraint，必须从可识别的实验 measurement operator、热方程、材料/边界/界面条件重新定义并做量纲验证。
