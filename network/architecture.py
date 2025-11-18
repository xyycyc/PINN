import numpy as np
import torch.nn as nn
import torch
class MLP(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim, num_layers):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.Tanh(),
            nn.Linear(40, hidden_dim),  # 加宽的网络
            nn.Tanh(),
            nn.Linear(hidden_dim, 40),
            nn.Tanh(),
            nn.Linear(40, output_dim)
        )

    def forward(self, t):
        return self.net(t)


# 文件名: architecture.py
# 描述: 包含用于PINN的可选网络架构。
#       - FourierFeatureMLP (用于高频)
#       - ResNetMLP (用于深度网络)
#       - SIREN (用于高精度导数)

# ==========================================================
# 架构 1: 傅里叶特征网络 (Fourier Feature Network)
# ==========================================================
class FourierFeatureMLP(nn.Module):
    """
    傅里叶特征MLP (源自 NeRF)。

    此架构在将输入 (t, x) 送入标准MLP之前，
    首先将其映射到高维傅里叶空间，使其极擅长学习高频函数。
    """

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers,
                 num_freqs=10, sigma=10.0):
        """
        参数:
            input_dim (int): 2 (t, x)
            output_dim (int): 1 (u)
            hidden_dim (int): 隐藏层的宽度 (e.g., 256)
            num_layers (int): 隐藏层的数量 (e.g., 8)
            num_freqs (int): 'L' in NeRF, 傅里叶特征的数量
            sigma (float): 傅里叶特征的尺度 (高斯分布的标准差)
        """
        super(FourierFeatureMLP, self).__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs

        # 1. 傅里叶特征映射
        # 随机采样一个固定的 (B) 矩阵
        # B 矩阵形状为 [mapping_size, input_dim]
        # mapping_size 就是 num_freqs
        B_matrix = torch.randn(num_freqs, input_dim) * sigma

        # 注册为 buffer，这样它就是模型状态的一部分，但不是可训练参数
        self.register_buffer('B_matrix', B_matrix)

        # 2. 实际的MLP
        # 映射后的输入维度为: input_dim * 2 * num_freqs
        # (因为每个频率有 sin 和 cos 两个分量)
        mlp_input_dim = 2 * num_freqs

        layers = []
        # 输入层
        layers.append(nn.Linear(mlp_input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, tx):
        # tx 的形状: [batch_size, input_dim]

        # 1. 傅里叶特征编码
        # (tx @ self.B_matrix.T) -> [batch_size, num_freqs]
        # (2.0 * np.pi * ...) 是标准 NeRF 做法
        tx_proj = 2.0 * np.pi * (tx @ self.B_matrix.T)

        # [batch_size, 2 * num_freqs]
        tx_mapped = torch.cat([torch.sin(tx_proj), torch.cos(tx_proj)], dim=-1)

        # 2. 通过MLP
        return self.net(tx_mapped)


# ==========================================================
# 架构 2: 残差网络 (ResNet-MLP)
# ==========================================================
class ResBlock(nn.Module):
    """
    一个简单的残差块 (ResNet Block)。
    """

    def __init__(self, hidden_dim):
        super(ResBlock, self).__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        # x -> layer1 -> act -> layer2 -> (+) -> act
        #                                  |
        #                                  x
        identity = x
        out = self.activation(self.layer1(x))
        out = self.layer2(out)
        out = self.activation(out + identity)  # 跳跃连接
        return out


class ResNetMLP(nn.Module):
    """
    使用残差块构建的深度MLP。

    允许网络非常深，同时通过跳跃连接保持梯度流。
    注意：这要求隐藏维度 (hidden_dim) 保持一致。
    """

    def __init__(self, input_dim, output_dim, hidden_dim, num_blocks):
        """
        参数:
            input_dim (int): 2 (t, x)
            output_dim (int): 1 (u)
            hidden_dim (int): 隐藏层的宽度 (e.g., 256)
            num_blocks (int): ResBlock 的数量 (e.g., 4)
        """
        super(ResNetMLP, self).__init__()

        # 1. 输入层 (将 2D 映射到 hidden_dim)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        # 2. 堆叠的残差块
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim) for _ in range(num_blocks)]
        )

        # 3. 输出层 (将 hidden_dim 映射到 1D)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, tx):
        x = self.input_layer(tx)
        x = self.res_blocks(x)
        return self.output_layer(x)


# ==========================================================
# 架构 3: SIREN (Sinusoidal Representation Network)
# ==========================================================
class SirenLayer(nn.Module):
    """
    SIREN 模型的单层，使用 sin(omega * (Wx + b))
    """

    def __init__(self, in_features, out_features, omega=30.0, is_first_layer=False):
        super(SirenLayer, self).__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)

        # SIREN 的成功在极大程度上依赖于这种特殊的权重初始化
        self.init_weights(is_first_layer)

    def init_weights(self, is_first_layer):
        in_dim = self.linear.in_features
        if is_first_layer:
            # 第一层: Uniform(-1/in_dim, 1/in_dim)
            bound = 1.0 / in_dim
        else:
            # 隐藏层: Uniform(-sqrt(6/in_dim)/omega, sqrt(6/in_dim)/omega)
            bound = np.sqrt(6.0 / in_dim) / self.omega

        nn.init.uniform_(self.linear.weight, -bound, bound)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class SIREN(nn.Module):
    """
    SIREN 架构，完全由 SirenLayer 组成。

    这种网络天生适合表示导数，非常适合PINN。
    """

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, omega=30.0):
        """
        参数:
            input_dim (int): 2 (t, x)
            output_dim (int): 1 (u)
            hidden_dim (int): 隐藏层的宽度 (e.g., 256)
            num_layers (int): 隐藏层的数量 (e.g., 4)
            omega (float): 控制 sin 频率的超参数
        """
        super(SIREN, self).__init__()

        layers = []

        # 第一层 (特殊初始化)
        layers.append(SirenLayer(input_dim, hidden_dim, omega=omega, is_first_layer=True))

        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega=omega))

        self.net = nn.Sequential(*layers)

        # SIREN 的最后一层是一个标准的线性层，*不* 带 sin 激活
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # (输出层使用标准Xavier/Glorot初始化，这里nn.Linear默认的Kaiming即可)

    def forward(self, tx):
        x = self.net(tx)
        return self.output_layer(x)