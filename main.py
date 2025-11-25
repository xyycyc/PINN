# 文件名: main.py
# 描述: 主训练脚本 (已模块化)
#       - 使用 argparse 选择模型和问题
#       - 自动匹配网络输入/输出维度
#       - 执行训练循环并绘图
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.optim as optim
import argparse
from plot_comparison import plot_results

# (假设网络架构在 network.py 中)
from network import MLP, FourierFeatureMLP, ResNetMLP, SIREN

# (假设物理问题在 constrain.py 中)
from constrain import HarmonicOscillator, HeatEquation, BurgersEquation, WaveEquation, CoupledWaveEquation,compute_label_mse

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
    if name == "MLP":
        return MLP(input_dim, output_dim, config['hidden_dim'], config['num_layers'])
    elif name == "FourierFeatureMLP":
        return FourierFeatureMLP(input_dim, output_dim, config['hidden_dim'], config['num_layers'], config['num_freqs'],
                                 config['sigma'])
    elif name == "ResNetMLP":
        return ResNetMLP(input_dim, output_dim, config['hidden_dim'], config['num_blocks'])
    elif name == "SIREN":
        return SIREN(input_dim, output_dim, config['hidden_dim'], config['num_layers'], config['omega'])
    raise ValueError(f"Unknown network: {name}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 实验设置: {args.problem_name} | {args.model_name} ---")
    print(f"--- 权重: a(Data)={args.a}, b(Physics)={args.b}, c(Label MSE)={args.c} ---")

    # 1. 初始化问题
    if args.problem_name not in PROBLEM_CONFIG:
        raise ValueError(f"Unknown problem: {args.problem_name}")

    p_info = PROBLEM_CONFIG[args.problem_name]
    problem = p_info["class"](device=device, N_physics=args.n_physics)

    # 2. 初始化网络
    pinn = get_network(args.model_name, p_info["input_dim"], p_info["output_dim"], NETWORK_CONFIG).to(device)

    # 3. 初始化优化器
    learnable_params = list(pinn.parameters())
    if hasattr(problem, 'get_learnable_parameters'):
        learnable_params += problem.get_learnable_parameters()
    optimizer = optim.Adam(learnable_params, lr=args.lr)

    # [简化] 直接读取统一接口 (得益于第一步的修改)
    if problem.x_label is not None:
        print(f"检测到标签数据: Shape {problem.x_label.shape}")
    else:
        print("无标签数据 (MSE Loss 将为 0)")

    print("Starting training...")
    for epoch in range(args.epochs):

        physics_points = problem.data_generate()

        optimizer.zero_grad()

        # A. 原始两部分 Loss (Data + Physics)
        loss_data, loss_physics = problem.loss_cal(pinn, physics_points)

        # B. 新增 Label MSE Loss
        # 直接调用导入的函数，传入 problem 中的统一属性
        loss_mse = compute_label_mse(pinn, problem.x_label, problem.u_label)

        # C. 组合总 Loss
        loss = args.a * loss_data + args.b * loss_physics + args.c * loss_mse

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/{args.epochs}] Total: {loss.item():.5f} | '
                  f'Data: {loss_data.item():.5f} | '
                  f'Phy: {loss_physics.item():.5f} | '
                  f'Label: {loss_mse.item():.5f}')

    print("Training finished.")
    try:
        plot_results(pinn, device, problem)
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    # ... (argparse 部分保持不变) ...
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MLP',
                        choices=['MLP', 'FourierFeatureMLP', 'ResNetMLP', 'SIREN'])
    parser.add_argument('--problem_name', type=str, default='HarmonicOscillator', choices=list(PROBLEM_CONFIG.keys()))
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--n_physics', type=int, default=1000)
    parser.add_argument('--a', type=float, default=1.0, help='Weight for Data Loss')
    parser.add_argument('--b', type=float, default=1.0, help='Weight for Physics Loss')
    parser.add_argument('--c', type=float, default=10.0, help='Weight for Label MSE Loss')

    args = parser.parse_args()
    main(args)