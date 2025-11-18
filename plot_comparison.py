# 文件名: plot_comparison.py
# 描述: 绘制PINN的预测结果。
# (已更新) 现在可以根据问题是1D还是2D，自动选择绘制线图或热力图。

import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_results(model, device, problem):
    """
    绘制并对比PINN预测和真实解。

    自动检测问题是1D (e.g., HarmonicOscillator) 还是 2D (e.g., HeatEquation)
    并选择合适的绘图方式。
    """
    print("Verifying result...")

    # 获取问题的类名 (e.g., "HarmonicOscillator" 或 "HeatEquation")
    problem_name = type(problem).__name__

    model.eval()  # 将模型设置为评估模式

    # ==========================================================
    # 路径 1: 1D 问题 (例如: HarmonicOscillator)
    # ==========================================================
    if problem_name == "HarmonicOscillator":

        # 准备测试点 (t)
        t_test = torch.linspace(problem.domain_start, problem.domain_end, 200).view(-1, 1).to(device)
        with torch.no_grad():
            u_pred = model(t_test)

        # 真实解
        u_true = torch.sin(t_test)

        # 绘图对比 (1D 线图)
        plt.figure(figsize=(10, 6))
        plt.plot(t_test.cpu().numpy(), u_true.cpu().numpy(), 'r-', label='True Solution (u = sin(t))')
        plt.plot(t_test.cpu().numpy(), u_pred.cpu().numpy(), 'b--', label='PINN Prediction')
        plt.title(f'PINN Solution for {problem_name}')
        plt.xlabel('Time (t)')
        plt.ylabel('Position (u)')
        plt.legend()
        plt.grid(True)
        plt.show()

    # ==========================================================
    # 路径 2: 2D 问题 (例如: HeatEquation, WaveEquation, etc.)
    # ==========================================================
    else:
        print(f"Detected 2D problem: {problem_name}. Generating heatmap plot...")

        # 检查问题是否具有 2D 域属性
        if not all(hasattr(problem, attr) for attr in
                   ['domain_t_start', 'domain_t_end', 'domain_x_start', 'domain_x_end']):
            print(f"Plotter Error: {problem_name} object is missing required domain attributes "
                  "(domain_t_start, domain_x_start, etc.)")
            return

        # 1. 创建一个 (t, x) 网格用于绘图
        n_t = 100  # t 轴的分辨率
        n_x = 100  # x 轴的分辨率

        t_linspace = torch.linspace(problem.domain_t_start, problem.domain_t_end, n_t).to(device)
        x_linspace = torch.linspace(problem.domain_x_start, problem.domain_x_end, n_x).to(device)

        # 创建网格 (t_grid, x_grid)
        t_grid, x_grid = torch.meshgrid(t_linspace, x_linspace, indexing='ij')

        # 2. 准备模型输入
        # 将 (t, x) 网格展平为 [N, 2] 的张量
        tx_test_flat = torch.stack([t_grid.flatten(), x_grid.flatten()], dim=1)

        # 3. 获取模型预测
        with torch.no_grad():
            u_pred_flat = model(tx_test_flat)

        # 4. 将预测结果重塑为 [n_t, n_x] 的网格
        u_pred_grid = u_pred_flat.view(n_t, n_x).cpu().numpy()

        # 5. 绘制热力图 (Heatmap)
        plt.figure(figsize=(10, 6))
        # 我们使用 pcolormesh 来绘制热力图，x轴是t，y轴是x
        c = plt.pcolormesh(
            t_grid.cpu().numpy(),
            x_grid.cpu().numpy(),
            u_pred_grid,
            shading='gouraud',  # 'gouraud' 或 'auto'
            cmap='viridis'  # (e.g., 'viridis', 'jet', 'coolwarm')
        )
        plt.colorbar(c, label="Predicted u(t, x)")

        plt.title(f'PINN Solution for {problem_name}')
        plt.xlabel('Time (t)')
        plt.ylabel('Space (x)')
        plt.show()