% 一维超声固体测温：恒定热流反演及温度重建
clear; clc; close all;

%% 1. 物理参数设置 (参考不锈钢材料)
L = 0.1;                % 棒长 (m)
rho = 7930;             % 密度 (kg/m^3)
cp = 500;               % 比热容 (J/(kg*K))
k = 16.3;               % 导热系数 (W/(m*K))
alpha = k / (rho * cp); % 热扩散率

% 声速-温度模型: V(T) = a*T + b (T单位: K)
va = -0.6487; 
vb = 5934.9;

% 离散化设置
Nx = 50;                % 空间网格
dx = L / (Nx-1);
x = linspace(0, L, Nx);
Nt = 40;                % 时间步数
dt = 2.0;               % 时间步长 (s) - 适当增加时长利于观察
T_init = 300;           % 初始温度 (K)

%% 2. 模拟“真实”环境 (生成实验观测数据)
% 设定一个恒定的真实热流值
q_const_true = 4e5;     % 400 kW/m^2
q_true = ones(1, Nt) * q_const_true; 

% 求解正向问题：得到真实温度场演变
T_true_history = solve_heat_1d_core(q_true, L, Nx, Nt, dt, rho, cp, k, T_init);

% 计算观测到的 TOF (声束穿过全长再返回)
TOF_measured = zeros(Nt, 1);
for i = 1:Nt
    V_dist = va * T_true_history(i, :) + vb;
    TOF_measured(i) = 2 * sum(dx ./ V_dist); % 往返时间
end

% 注意：此处暂时不加噪声，以验证算法闭环的准确性

%% 3. 逆问题重建 (Optimization)
% 目标：反演常数值 q
% 初始猜想 (假设完全不知道热流)
q_guess_init = 1e5; 

% 优化配置：增加残差缩放因子 (1e9 转换到纳秒量级)
% 这能显著提高 lsqnonlin 对微小变化的识别能力
scale_factor = 1e9; 
options = optimoptions('lsqnonlin', 'Display', 'iter', 'FunctionTolerance', 1e-10, 'StepTolerance', 1e-10);

% 方案 A: 假设已知热流是常数 (反演一个标量)
fprintf('\n--- 正在进行单参数(常数)反演 ---\n');
q_inverted_scalar = lsqnonlin(@(q) (objective_scalar(q, TOF_measured, L, Nx, Nt, dt, rho, cp, k, T_init, va, vb) * scale_factor), ...
                              q_guess_init, 0, 1e7, options);

% 方案 B: 假设热流随时间变 (反演一个序列，如文献所示)
fprintf('\n--- 正在进行多参数(序列)反演 ---\n');
q_start_vec = ones(1, Nt) * q_guess_init;
q_inverted_vec = lsqnonlin(@(q) (objective_vec(q, TOF_measured, L, Nx, Nt, dt, rho, cp, k, T_init, va, vb) * scale_factor), ...
                           q_start_vec, zeros(1, Nt), 1e7 * ones(1, Nt), options);

%% 4. 结果验证与绘图
T_reconstructed = solve_heat_1d_core(q_inverted_vec, L, Nx, Nt, dt, rho, cp, k, T_init);

figure('Color', 'w', 'Name', 'Reconstruction Results');
subplot(1,2,1);
plot(0:dt:(Nt-1)*dt, q_true/1e3, 'k-', 'LineWidth', 2); hold on;
plot(0:dt:(Nt-1)*dt, q_inverted_vec/1e3, 'r--', 'LineWidth', 1.5);
xlabel('时间 (s)'); ylabel('热流 (kW/m^2)');
title('热流反演对比'); legend('真实热流', '反演热流'); grid on;

subplot(1,2,2);
plot(x, T_true_history(end, :), 'k-', 'LineWidth', 2); hold on;
plot(x, T_reconstructed(end, :), 'r--', 'LineWidth', 1.5);
xlabel('位置 (m)'); ylabel('温度 (K)');
title(['t = ', num2str((Nt-1)*dt), 's 时刻温度分布']);
legend('真实温度', '重建温度'); grid on;

fprintf('\n反演完成！设定热流: %.2f, 反演常数热流: %.2f\n', q_const_true, q_inverted_scalar);

%% --- 核心函数：一维热传导隐式求解器 ---
function T_hist = solve_heat_1d_core(q_flux, L, Nx, Nt, dt, rho, cp, k, T_init)
    dx = L / (Nx-1);
    alpha = k / (rho * cp);
    r = alpha * dt / (dx^2);
    T = ones(Nx, 1) * T_init;
    T_hist = zeros(Nt, Nx);
    T_hist(1,:) = T;
    
    % 构造隐式算子矩阵 A (满足 (1+2r)T_i - rT_{i-1} - rT_{i+1} = T_old)
    A = diag((1 + 2*r) * ones(Nx, 1)) + diag(-r * ones(Nx-1, 1), 1) + diag(-r * ones(Nx-1, 1), -1);
    
    % 边界修正 (中心差分格式)
    % x=0 (注入热流 q): A(1,2) = -2r
    A(1, 2) = -2*r; 
    % x=L (绝热边界): A(N, N-1) = -2r
    A(Nx, Nx-1) = -2*r;
    
    invA = inv(A); % 预计算逆矩阵提高效率
    
    for i = 2:Nt
        B = T;
        % 热流项注入 (系数推导自边界能量平衡)
        B(1) = B(1) + (2 * r * dx / k) * q_flux(i);
        T = invA * B;
        T_hist(i, :) = T';
    end
end

% 标量优化目标函数
function res = objective_scalar(q_val, TOF_m, L, Nx, Nt, dt, rho, cp, k, T_init, va, vb)
    q_vec = ones(1, Nt) * q_val;
    res = objective_vec(q_vec, TOF_m, L, Nx, Nt, dt, rho, cp, k, T_init, va, vb);
end

% 序列优化目标函数
function res = objective_vec(q_vec, TOF_m, L, Nx, Nt, dt, rho, cp, k, T_init, va, vb)
    T_h = solve_heat_1d_core(q_vec, L, Nx, Nt, dt, rho, cp, k, T_init);
    dx = L / (Nx-1);
    TOF_c = zeros(length(TOF_m), 1);
    for i = 1:Nt
        V_d = va * T_h(i, :) + vb;
        TOF_c(i) = 2 * sum(dx ./ V_d);
    end
    res = TOF_c - TOF_m;
end
