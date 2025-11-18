# 文件名: problem.py
# 描述: 封装了物理问题(简谐振子)的
#       - 数据生成 (data_generate)
#       - 损失计算 (loss_cal)

import torch
import torch.nn as nn
import numpy as np


class HarmonicOscillator:
    """
    封装了简谐振子问题 (u_tt + u = 0) 的所有逻辑。

    边界条件 (BCs):
        - u(0) = 0
        - u'(0) = 1
    内部锚点:
        - u(pi) = 0
    """

    def __init__(self, device, N_physics=1000):
        self.device = device
        self.N_physics = N_physics
        self.domain_start = 0.0
        self.domain_end = 2.0 * np.pi

        self.loss_fn = nn.MSELoss()

        # --- 准备固定的“真实”数据 (BCs 和锚点) ---
        # 1. 初始条件 @ t=0
        self.t_bc = torch.tensor([[self.domain_start]], requires_grad=True).to(self.device)
        self.u_bc_target = torch.tensor([[0.0]]).to(self.device)  # u(0) = 0
        self.v_bc_target = torch.tensor([[1.0]]).to(self.device)  # u'(0) = 1 (速度)

        # 2. 内部锚点 @ t=pi
        self.t_internal = torch.tensor([[np.pi]], requires_grad=True).to(self.device)
        self.u_internal_target = torch.tensor([[0.0]]).to(self.device)  # u(pi) = 0

        print("Problem (Harmonic Oscillator) initialized.")

    def data_generate(self):
        """
        (用户请求) 生成动态的物理配置点 (t_physics)。
        """
        # 每次迭代动态生成物理点 (配置点)
        t_physics = (torch.rand(self.N_physics, 1) * (self.domain_end - self.domain_start) + self.domain_start).to(
            self.device)
        t_physics.requires_grad = True
        return t_physics

    def loss_cal(self, model, t_physics):
        """
        (用户请求) 计算所有损失 (数据损失 + 物理损失)。
        """

        # --- 损失1：数据损失 (Data Loss) ---
        # 1a. 初始位置 u(0) = 0
        u_at_bc = model(self.t_bc)
        loss_bc_u = self.loss_fn(u_at_bc, self.u_bc_target)

        # 1b. 初始速度 u'(0) = 1
        u_t_at_bc = torch.autograd.grad(
            outputs=u_at_bc,
            inputs=self.t_bc,
            grad_outputs=torch.ones_like(u_at_bc).to(self.device),
            create_graph=True,
            retain_graph=True
        )[0]
        loss_bc_ut = self.loss_fn(u_t_at_bc, self.v_bc_target)

        # 1c. 内部锚点 u(pi) = 0
        u_at_internal = model(self.t_internal)
        loss_data_internal = self.loss_fn(u_at_internal, self.u_internal_target)

        loss_data = loss_bc_u + loss_bc_ut + loss_data_internal

        # --- 损失2：物理损失 (Physics Loss) ---
        # 方程为: u_tt + u = 0

        # 1. 计算 u(t)
        u = model(t_physics)

        # 2. 计算一阶导 u_t = du/dt (速度)
        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t_physics,
            grad_outputs=torch.ones_like(u).to(self.device),
            create_graph=True,
            retain_graph=True
        )[0]

        # 3. 计算二阶导 u_tt = d^2u/dt^2 (加速度)
        u_tt = torch.autograd.grad(
            outputs=u_t,
            inputs=t_physics,
            grad_outputs=torch.ones_like(u_t).to(self.device),
            create_graph=True,
            retain_graph=True
        )[0]

        # 4. 计算物理方程残差: residual = u_tt + u
        residual = u_tt + u

        # 5. 计算物理损失 (目标: residual = 0)
        loss_physics = self.loss_fn(residual, torch.zeros_like(residual).to(self.device))

        return loss_data, loss_physics


# ==========================================================
# 1. 热力学: 热传导方程 (Heat Equation)
# ==========================================================
class HeatEquation:
    """
    封装了 1D 热传导方程 (u_t = alpha * u_xx) 的所有逻辑。

    问题设定:
    - 求解域: t in [0, 1], x in [0, 1]
    - 初始条件 (IC) @ t=0: u(0, x) = sin(pi * x)
    - 边界条件 (BC) @ x=0, x=1: u(t, 0) = 0, u(t, 1) = 0
    """

    def __init__(self, device, N_physics=1000):
        self.device = device
        self.N_physics = N_physics

        # 物理和求解域参数
        self.alpha = 0.1  # 热扩散系数
        self.domain_t_start = 0.0
        self.domain_t_end = 1.0
        self.domain_x_start = 0.0
        self.domain_x_end = 1.0

        self.loss_fn = nn.MSELoss()

        # --- 准备固定的“真实”数据 (IC 和 BCs) ---

        # 1. 初始条件 (IC) @ t=0
        N_ic = 100
        x_ic = torch.linspace(self.domain_x_start, self.domain_x_end, N_ic).view(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        self.tx_ic = torch.cat([t_ic, x_ic], dim=1).to(self.device).requires_grad_(True)
        # 目标: u(0, x) = sin(pi * x)
        self.u_ic_target = torch.sin(np.pi * x_ic).to(self.device)

        # 2. 边界条件 (BC) @ x=0 和 x=1
        N_bc = 100
        t_bc = torch.linspace(self.domain_t_start, self.domain_t_end, N_bc).view(-1, 1)
        # @ x=0
        x_bc_start = torch.zeros_like(t_bc)
        self.tx_bc_start = torch.cat([t_bc, x_bc_start], dim=1).to(self.device).requires_grad_(True)
        # @ x=1
        x_bc_end = torch.ones_like(t_bc)
        self.tx_bc_end = torch.cat([t_bc, x_bc_end], dim=1).to(self.device).requires_grad_(True)
        # 目标: u(t, 0) = 0, u(t, 1) = 0
        self.u_bc_target = torch.zeros_like(t_bc).to(self.device)

        print("Problem (Heat Equation) initialized.")

    def data_generate(self):
        """
        生成动态的物理配置点 (t, x)。
        """
        # 每次迭代动态生成物理点 (配置点)
        t_physics = (torch.rand(self.N_physics, 1) * (self.domain_t_end - self.domain_t_start) + self.domain_t_start)
        x_physics = (torch.rand(self.N_physics, 1) * (self.domain_x_end - self.domain_x_start) + self.domain_x_start)

        tx_physics = torch.cat([t_physics, x_physics], dim=1).to(self.device)
        tx_physics.requires_grad = True

        # (变量名适配) 返回的张量现在是 (t, x) 对
        return tx_physics

    def loss_cal(self, model, tx_physics):
        """
        计算所有损失 (数据损失 + 物理损失)。
        """

        # --- 损失1：数据损失 (Data Loss = IC + BC) ---
        # 1a. 初始条件 u(0, x)
        u_at_ic = model(self.tx_ic)
        loss_ic = self.loss_fn(u_at_ic, self.u_ic_target)

        # 1b. 边界条件 u(t, 0)
        u_at_bc_start = model(self.tx_bc_start)
        loss_bc_1 = self.loss_fn(u_at_bc_start, self.u_bc_target)

        # 1c. 边界条件 u(t, 1)
        u_at_bc_end = model(self.tx_bc_end)
        loss_bc_2 = self.loss_fn(u_at_bc_end, self.u_bc_target)

        loss_data = loss_ic + loss_bc_1 + loss_bc_2

        # --- 损失2：物理损失 (Physics Loss) ---
        # 方程为: u_t - alpha * u_xx = 0

        # 1. 计算 u(t, x)
        u = model(tx_physics)

        # 2. 计算一阶和二阶导数
        # (t, x) 是输入, u 是输出
        # u_grads 会包含 [du/dt, du/dx]
        u_grads = torch.autograd.grad(
            outputs=u,
            inputs=tx_physics,
            grad_outputs=torch.ones_like(u).to(self.device),
            create_graph=True, retain_graph=True
        )[0]

        # 提取 du/dt
        u_t = u_grads[:, [0]]  # 切片 [:, [0]] 保持维度
        # 提取 du/dx
        u_x = u_grads[:, [1]]

        # 3. 计算二阶导 u_xx
        # 我们对 u_x (一阶空间导) 再次求导
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=tx_physics,
            grad_outputs=torch.ones_like(u_x).to(self.device),
            create_graph=True, retain_graph=True
        )[0][:, [1]]  # 同样，我们只关心对 x 的导数

        # 4. 计算物理方程残差
        residual = u_t - self.alpha * u_xx

        # 5. 计算物理损失 (目标: residual = 0)
        loss_physics = self.loss_fn(residual, torch.zeros_like(residual).to(self.device))

        return loss_data, loss_physics


# ==========================================================
# 2. 声学/力学: 波动方程 (Wave Equation)
# ==========================================================
class WaveEquation:
    """
    封装了 1D 波动方程 (u_tt = c^2 * u_xx) 的所有逻辑。

    问题设定:
    - 求解域: t in [0, 1], x in [0, 1]
    - 初始条件 (IC) @ t=0: u(0, x) = sin(pi * x), u_t(0, x) = 0
    - 边界条件 (BC) @ x=0, x=1: u(t, 0) = 0, u(t, 1) = 0
    """

    def __init__(self, device, N_physics=1000):
        self.device = device
        self.N_physics = N_physics

        # 物理和求解域参数
        self.c = 1.0  # 波速
        self.domain_t_start = 0.0
        self.domain_t_end = 1.0
        self.domain_x_start = 0.0
        self.domain_x_end = 1.0

        self.loss_fn = nn.MSELoss()

        # --- 准备固定的“真实”数据 (IC 和 BCs) ---

        # 1. 初始条件 (IC) @ t=0
        N_ic = 100
        x_ic = torch.linspace(self.domain_x_start, self.domain_x_end, N_ic).view(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        self.tx_ic = torch.cat([t_ic, x_ic], dim=1).to(self.device).requires_grad_(True)
        # 目标1: u(0, x) = sin(pi * x)
        self.u_ic_target = torch.sin(np.pi * x_ic).to(self.device)
        # 目标2: u_t(0, x) = 0 (初始速度)
        self.v_ic_target = torch.zeros_like(x_ic).to(self.device)

        # 2. 边界条件 (BC) @ x=0 和 x=1 (与HeatEquation相同)
        N_bc = 100
        t_bc = torch.linspace(self.domain_t_start, self.domain_t_end, N_bc).view(-1, 1)
        x_bc_start = torch.zeros_like(t_bc)
        self.tx_bc_start = torch.cat([t_bc, x_bc_start], dim=1).to(self.device).requires_grad_(True)
        x_bc_end = torch.ones_like(t_bc)
        self.tx_bc_end = torch.cat([t_bc, x_bc_end], dim=1).to(self.device).requires_grad_(True)
        self.u_bc_target = torch.zeros_like(t_bc).to(self.device)

        print("Problem (Wave Equation) initialized.")

    def data_generate(self):
        """
        生成动态的物理配置点 (t, x)。
        """
        t_physics = (torch.rand(self.N_physics, 1) * (self.domain_t_end - self.domain_t_start) + self.domain_t_start)
        x_physics = (torch.rand(self.N_physics, 1) * (self.domain_x_end - self.domain_x_start) + self.domain_x_start)
        tx_physics = torch.cat([t_physics, x_physics], dim=1).to(self.device)
        tx_physics.requires_grad = True
        return tx_physics

    def loss_cal(self, model, tx_physics):
        """
        计算所有损失 (数据损失 + 物理损失)。
        """

        # --- 损失1：数据损失 (Data Loss = ICs + BCs) ---
        # 1a. 初始位置 u(0, x)
        u_at_ic = model(self.tx_ic)
        loss_ic_u = self.loss_fn(u_at_ic, self.u_ic_target)

        # 1b. 初始速度 u_t(0, x)
        u_t_at_ic = torch.autograd.grad(
            outputs=u_at_ic,
            inputs=self.tx_ic,
            grad_outputs=torch.ones_like(u_at_ic).to(self.device),
            create_graph=True, retain_graph=True
        )[0][:, [0]]  # 提取 du/dt
        loss_ic_v = self.loss_fn(u_t_at_ic, self.v_ic_target)

        # 1c. 边界条件 (同 HeatEquation)
        u_at_bc_start = model(self.tx_bc_start)
        loss_bc_1 = self.loss_fn(u_at_bc_start, self.u_bc_target)
        u_at_bc_end = model(self.tx_bc_end)
        loss_bc_2 = self.loss_fn(u_at_bc_end, self.u_bc_target)

        loss_data = loss_ic_u + loss_ic_v + loss_bc_1 + loss_bc_2

        # --- 损失2：物理损失 (Physics Loss) ---
        # 方程为: u_tt - c^2 * u_xx = 0

        # 1. 计算 u(t, x)
        u = model(tx_physics)

        # 2. 计算一阶导数 [du/dt, du/dx]
        u_grads = torch.autograd.grad(
            outputs=u,
            inputs=tx_physics,
            grad_outputs=torch.ones_like(u).to(self.device),
            create_graph=True, retain_graph=True
        )[0]
        u_t = u_grads[:, [0]]
        u_x = u_grads[:, [1]]

        # 3. 计算二阶导 u_tt 和 u_xx
        u_tt = torch.autograd.grad(
            outputs=u_t,
            inputs=tx_physics,
            grad_outputs=torch.ones_like(u_t).to(self.device),
            create_graph=True, retain_graph=True
        )[0][:, [0]]  # 提取 d(u_t)/dt

        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=tx_physics,
            grad_outputs=torch.ones_like(u_x).to(self.device),
            create_graph=True, retain_graph=True
        )[0][:, [1]]  # 提取 d(u_x)/dx

        # 4. 计算物理方程残差
        residual = u_tt - (self.c ** 2) * u_xx

        # 5. 计算物理损失 (目标: residual = 0)
        loss_physics = self.loss_fn(residual, torch.zeros_like(residual).to(self.device))

        return loss_data, loss_physics


# ==========================================================
# 3. 力学: 伯格斯方程 (Burgers' Equation)
# ==========================================================
class BurgersEquation:
    """
    封装了 1D 粘性伯格斯方程 (u_t + u * u_x = nu * u_xx) 的所有逻辑。

    问题设定:
    - 求解域: t in [0, 1], x in [-1, 1]
    - 粘性系数: nu = 0.01 / pi
    - 初始条件 (IC) @ t=0: u(0, x) = -sin(pi * x)
    - 边界条件 (BC) @ x=-1, x=1: u(t, -1) = 0, u(t, 1) = 0
    """

    def __init__(self, device, N_physics=1000):
        self.device = device
        self.N_physics = N_physics

        # 物理和求解域参数
        self.nu = (0.01 / np.pi)  # 粘性系数
        self.domain_t_start = 0.0
        self.domain_t_end = 1.0
        self.domain_x_start = -1.0
        self.domain_x_end = 1.0

        self.loss_fn = nn.MSELoss()

        # --- 准备固定的“真实”数据 (IC 和 BCs) ---

        # 1. 初始条件 (IC) @ t=0
        N_ic = 100
        x_ic = torch.linspace(self.domain_x_start, self.domain_x_end, N_ic).view(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        self.tx_ic = torch.cat([t_ic, x_ic], dim=1).to(self.device).requires_grad_(True)
        # G: u(0, x) = -sin(pi * x)
        self.u_ic_target = -torch.sin(np.pi * x_ic).to(self.device)

        # 2. 边界条件 (BC) @ x=-1 和 x=1
        N_bc = 100
        t_bc = torch.linspace(self.domain_t_start, self.domain_t_end, N_bc).view(-1, 1)
        # @ x=-1
        x_bc_start = -torch.ones_like(t_bc)
        self.tx_bc_start = torch.cat([t_bc, x_bc_start], dim=1).to(self.device).requires_grad_(True)
        # @ x=1
        x_bc_end = torch.ones_like(t_bc)
        self.tx_bc_end = torch.cat([t_bc, x_bc_end], dim=1).to(self.device).requires_grad_(True)
        # 目标: u(t, -1) = 0, u(t, 1) = 0
        self.u_bc_target = torch.zeros_like(t_bc).to(self.device)

        print("Problem (Burgers' Equation) initialized.")

    def data_generate(self):
        """
        生成动态的物理配置点 (t, x)。
        """
        t_physics = (torch.rand(self.N_physics, 1) * (self.domain_t_end - self.domain_t_start) + self.domain_t_start)
        x_physics = (torch.rand(self.N_physics, 1) * (self.domain_x_end - self.domain_x_start) + self.domain_x_start)
        tx_physics = torch.cat([t_physics, x_physics], dim=1).to(self.device)
        tx_physics.requires_grad = True
        return tx_physics

    def loss_cal(self, model, tx_physics):
        """
        计算所有损失 (数据损失 + 物理损失)。
        """

        # --- 损失1：数据损失 (Data Loss = IC + BC) ---
        # 1a. 初始条件 u(0, x)
        u_at_ic = model(self.tx_ic)
        loss_ic = self.loss_fn(u_at_ic, self.u_ic_target)

        # 1b. 边界条件 (同 HeatEquation)
        u_at_bc_start = model(self.tx_bc_start)
        loss_bc_1 = self.loss_fn(u_at_bc_start, self.u_bc_target)
        u_at_bc_end = model(self.tx_bc_end)
        loss_bc_2 = self.loss_fn(u_at_bc_end, self.u_bc_target)

        loss_data = loss_ic + loss_bc_1 + loss_bc_2

        # --- 损失2：物理损失 (Physics Loss) ---
        # 方程为: u_t + u * u_x - nu * u_xx = 0

        # 1. 计算 u(t, x)
        u = model(tx_physics)

        # 2. 计算一阶导数 [du/dt, du/dx]
        u_grads = torch.autograd.grad(
            outputs=u,
            inputs=tx_physics,
            grad_outputs=torch.ones_like(u).to(self.device),
            create_graph=True, retain_graph=True
        )[0]
        u_t = u_grads[:, [0]]
        u_x = u_grads[:, [1]]

        # 3. 计算二阶导 u_xx
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=tx_physics,
            grad_outputs=torch.ones_like(u_x).to(self.device),
            create_graph=True, retain_graph=True
        )[0][:, [1]]  # 提取 d(u_x)/dx

        # 4. 计算物理方程残差 (包含非线性项 u * u_x)
        residual = u_t + u * u_x - self.nu * u_xx

        # 5. 计算物理损失 (目标: residual = 0)
        loss_physics = self.loss_fn(residual, torch.zeros_like(residual).to(self.device))

        return loss_data, loss_physics

    # ==========================================================
    # 2. 热声耦合: 温度相关的超声波传输
    # ==========================================================


class CoupledWaveEquation:
    """
    封装了 1D 波动方程 (u_tt = c(T(x))^2 * u_xx) 的所有逻辑。
    这模拟了超声波在具有 *非均匀温度场* 的介质(钨钼合金)中传输。

    问题设定:
    - 求解域: t in [0, 1], x in [0, 1]
    - (耦合) 已知温度场: T(x) = 300 + 700 * x (从300K线性升温到1000K)
    - (耦合) 波速-温度关系 (示意): c(T) = 5000 - 0.5 * (T - 300)
    - 初始条件 (IC) @ t=0: u(0, x) = sin(pi * x), u_t(0, x) = 0
    - 边界条件 (BC) @ x=0, x=1: u(t, 0) = 0, u(t, 1) = 0
    """

    def __init__(self, device, N_physics=1000):
        self.device = device
        self.N_physics = N_physics

        # 求解域参数
        self.domain_t_start = 0.0
        self.domain_t_end = 1.0
        self.domain_x_start = 0.0
        self.domain_x_end = 1.0

        self.loss_fn = nn.MSELoss()

        # --- (耦合关键) W-Mo 介质的物理属性 ---
        self.T_start = 300.0  # 300K
        self.T_end = 1000.0  # 1000K
        self.c_base = 5000.0  # m/s (在T_start时的声速)
        self.c_beta = 0.5  # m/s/K (声速随温度变化的系数)

        # --- 准备固定的“真实”数据 (IC 和 BCs) ---

        # 1. 初始条件 (IC) @ t=0
        N_ic = 100
        x_ic = torch.linspace(self.domain_x_start, self.domain_x_end, N_ic).view(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        self.tx_ic = torch.cat([t_ic, x_ic], dim=1).to(self.device).requires_grad_(True)
        # 目标1: u(0, x) = sin(pi * x) (初始形状)
        self.u_ic_target = torch.sin(np.pi * x_ic).to(self.device)
        # 目标2: u_t(0, x) = 0 (初始速度)
        self.v_ic_target = torch.zeros_like(x_ic).to(self.device)

        # 2. 边界条件 (BC) @ x=0 和 x=1 (两端固定)
        N_bc = 100
        t_bc = torch.linspace(self.domain_t_start, self.domain_t_end, N_bc).view(-1, 1)
        x_bc_start = torch.zeros_like(t_bc)
        self.tx_bc_start = torch.cat([t_bc, x_bc_start], dim=1).to(self.device).requires_grad_(True)
        x_bc_end = torch.ones_like(t_bc)
        self.tx_bc_end = torch.cat([t_bc, x_bc_end], dim=1).to(self.device).requires_grad_(True)
        self.u_bc_target = torch.zeros_like(t_bc).to(self.device)  # u(t, 0)=0, u(t, 1)=0

        print("Problem (Coupled Wave Equation) initialized.")

    def get_learnable_parameters(self):
        """(新函数) 此问题没有可学习参数，返回空列表"""
        return []

    def get_wave_speed(self, x):
        """(新函数) 核心: 计算空间 x 处的温度 T(x) 和波速 c(T(x))"""
        # 1. 计算 T(x) = T_start + (T_end - T_start) * x
        T_x = self.T_start + (self.T_end - self.T_start) * x
        # 2. 计算 c(T) = c_base - beta * (T - T_ref)
        c_at_x = self.c_base - self.c_beta * (T_x - self.T_start)
        return c_at_x

    def data_generate(self):
        """
        生成动态的物理配置点 (t, x)。
        """
        t_physics = (
                torch.rand(self.N_physics, 1) * (self.domain_t_end - self.domain_t_start) + self.domain_t_start)
        x_physics = (
                torch.rand(self.N_physics, 1) * (self.domain_x_end - self.domain_x_start) + self.domain_x_start)

        tx_physics = torch.cat([t_physics, x_physics], dim=1).to(self.device)
        tx_physics.requires_grad = True
        return tx_physics  # (变量名适配)

    def loss_cal(self, model, tx_physics):
        """
        计算所有损失 (数据损失 + 物理损失)。
        """

        # --- 损失1：数据损失 (Data Loss = ICs + BCs) ---
        # 1a. 初始位置 u(0, x)
        u_at_ic = model(self.tx_ic)
        loss_ic_u = self.loss_fn(u_at_ic, self.u_ic_target)

        # 1b. 初始速度 u_t(0, x)
        u_t_at_ic = torch.autograd.grad(
            outputs=u_at_ic, inputs=self.tx_ic,
            grad_outputs=torch.ones_like(u_at_ic).to(self.device),
            create_graph=True, retain_graph=True
        )[0][:, [0]]
        loss_ic_v = self.loss_fn(u_t_at_ic, self.v_ic_target)

        # 1c. 边界条件
        u_at_bc_start = model(self.tx_bc_start)
        u_at_bc_end = model(self.tx_bc_end)
        loss_bc = self.loss_fn(u_at_bc_start, self.u_bc_target) + self.loss_fn(u_at_bc_end, self.u_bc_target)

        loss_data = loss_ic_u + loss_ic_v + loss_bc

        # --- 损失2：物理损失 (Physics Loss) ---
        # 方程为: u_tt - c(T(x))^2 * u_xx = 0

        u = model(tx_physics)

        u_grads = torch.autograd.grad(
            outputs=u, inputs=tx_physics,
            grad_outputs=torch.ones_like(u).to(self.device),
            create_graph=True, retain_graph=True
        )[0]
        u_t = u_grads[:, [0]]
        u_x = u_grads[:, [1]]

        u_tt = torch.autograd.grad(
            outputs=u_t, inputs=tx_physics,
            grad_outputs=torch.ones_like(u_t).to(self.device),
            create_graph=True, retain_graph=True
        )[0][:, [0]]

        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=tx_physics,
            grad_outputs=torch.ones_like(u_x).to(self.device),
            create_graph=True, retain_graph=True
        )[0][:, [0]]

        # (耦合关键) 获取随x变化的波速 c(T(x))
        x_physics = tx_physics[:, [1]]  # 提取x坐标
        c_at_x = self.get_wave_speed(x_physics)

        # (耦合关键) 波速 c(T(x))^2 乘以 u_xx
        residual = u_tt - (c_at_x ** 2) * u_xx

        loss_physics = self.loss_fn(residual, torch.zeros_like(residual).to(self.device))

        return loss_data, loss_physics
