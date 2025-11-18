# 文件名: main.py
# 描述: 主训练脚本 (已模块化)
#       - 使用 argparse 选择模型和问题
#       - 自动匹配网络输入/输出维度
#       - 执行训练循环并绘图

import torch
import torch.optim as optim
import argparse
from plot_comparison import plot_results

# (假设网络架构在 network.py 中)
from network import MLP, FourierFeatureMLP, ResNetMLP, SIREN

# (假设物理问题在 constrain.py 中)
from constrain import HarmonicOscillator, HeatEquation, BurgersEquation, WaveEquation, CoupledWaveEquation

# --- 1. 配置中心 ---

# 定义每个问题的元数据，特别是输入/输出维度
PROBLEM_CONFIG = {
    # 您的 1D 问题
    "HarmonicOscillator": {"class": HarmonicOscillator, "input_dim": 1, "output_dim": 1},

    # 您的 2D 问题 (t, x)
    "HeatEquation": {"class": HeatEquation, "input_dim": 2, "output_dim": 1},
    "BurgersEquation": {"class": BurgersEquation, "input_dim": 2, "output_dim": 1},
    "WaveEquation": {"class": WaveEquation, "input_dim": 2, "output_dim": 1},
    "CoupledWaveEquation": {"class": CoupledWaveEquation, "input_dim": 2, "output_dim": 1},
    # (您可以在此处添加 InverseHeatEquation 等)
}

# 为不同架构定义"默认"的超参数配置
# (您可以在命令行中覆盖 lr, epochs, n_physics)
NETWORK_CONFIG = {
    # 通用
    "hidden_dim": 256,
    "num_layers": 5,

    # FourierFeatureMLP 专用
    "num_freqs": 8,
    "sigma": 5.0,

    # ResNetMLP 专用
    "num_blocks": 4,

    # SIREN 专用
    "omega": 30.0
}


def get_network(name, input_dim, output_dim, config):
    """
    根据名称和配置实例化网络架构。
    """
    if name == "MLP":
        return MLP(input_dim=input_dim, output_dim=output_dim,
                   hidden_dim=config['hidden_dim'], num_layers=config['num_layers'])

    elif name == "FourierFeatureMLP":
        return FourierFeatureMLP(input_dim=input_dim, output_dim=output_dim,
                                 hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
                                 num_freqs=config['num_freqs'], sigma=config['sigma'])

    elif name == "ResNetMLP":
        return ResNetMLP(input_dim=input_dim, output_dim=output_dim,
                         hidden_dim=config['hidden_dim'], num_blocks=config['num_blocks'])

    elif name == "SIREN":
        return SIREN(input_dim=input_dim, output_dim=output_dim,
                     hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
                     omega=config['omega'])
    else:
        raise ValueError(f"未知的网络架构: {name}")


def main(args):
    """
    主执行函数。
    """
    # --- 1. 设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 实验设置 ---")
    print(f"使用设备: {device}")
    print(f"选择问题: {args.problem_name}")
    print(f"选择架构: {args.model_name}")

    # --- 2. 初始化物理问题 (封装了数据和Loss) ---
    if args.problem_name not in PROBLEM_CONFIG:
        raise ValueError(f"未知问题: {args.problem_name}。请在 PROBLEM_CONFIG 中注册。")

    problem_info = PROBLEM_CONFIG[args.problem_name]
    ProblemClass = problem_info["class"]
    problem = ProblemClass(device=device, N_physics=args.n_physics)

    # --- 3. 初始化模型 ---
    # (自动从问题中获取 input_dim 和 output_dim)
    input_dim = problem_info["input_dim"]
    output_dim = problem_info["output_dim"]

    pinn = get_network(args.model_name, input_dim, output_dim, NETWORK_CONFIG).to(device)

    # --- 4. 初始化优化器 ---
    # (自动包含 "反问题" 的可学习参数)
    learnable_params = list(pinn.parameters())
    if hasattr(problem, 'get_learnable_parameters'):
        learnable_params += problem.get_learnable_parameters()
        print("注意: 优化器已包含问题中的可学习参数 (用于反问题)。")

    optimizer = optim.Adam(learnable_params, lr=args.lr)

    # --- 5. 训练循环 ---
    print(f"--- 开始训练 (Epochs: {args.epochs}, LR: {args.lr}) ---")

    for epoch in range(args.epochs):

        # (从问题中生成动态数据点)
        # (注意: 'physics_points' 是一个通用名称)
        # (对于1D问题, 它是 t_physics; 对于2D问题, 它是 tx_physics)
        physics_points = problem.data_generate()

        optimizer.zero_grad()

        # (从问题中计算损失)
        # (我们假设所有 'loss_cal' 都接受 (model, points) 两个参数)
        loss_data, loss_physics = problem.loss_cal(pinn, physics_points)

        # (应用命令行中的权重)
        loss = args.a * loss_data + args.b * loss_physics

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item():.4f}, '
                  f'Data Loss: {loss_data.item():.4f}, '
                  f'Physics Loss: {loss_physics.item():.4f}')

    print("--- 训练完成 ---")

    # --- 6. 验证和绘图 ---
    plot_results(pinn, device, problem)



# --- Python标准入口点 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINN 训练器")

    # --- 核心选择 ---
    parser.add_argument('--model_name', type=str, default='MLP',
                        choices=['MLP', 'FourierFeatureMLP', 'ResNetMLP', 'SIREN'],
                        help='要使用的网络架构')
    parser.add_argument('--problem_name', type=str, default='CoupledWaveEquation',
                        choices=list(PROBLEM_CONFIG.keys()),
                        help='要解决的物理约束问题')

    # --- 训练超参数 ---
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='学习率')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='训练的总轮数')
    parser.add_argument('--n_physics', type=int, default=1000,
                        help='每轮生成的物理配置点数量')

    # --- 损失权重超参数 ---
    parser.add_argument('--a', type=float, default=10.0,
                        help='数据损失 (loss_data) 的权重')
    parser.add_argument('--b', type=float, default=1.0,
                        help='物理损失 (loss_physics) 的权重')

    args = parser.parse_args()
    main(args)