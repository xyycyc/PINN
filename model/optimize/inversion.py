"""
一维超声固体测温：恒定热流反演及温度重建
从 inversion.m 转换的 Python 实现

在 cmd 中跑不同优化方法的指令（从项目根: python -m ai_model.model.optimize.inversion，或在 model/optimize 目录下: python inversion.py）：
  python inversion.py                    # 默认：least_squares（与 MATLAB lsqnonlin 一致）
  python inversion.py least_squares      # 非线性最小二乘
  python inversion.py L-BFGS-B           # 拟牛顿 + 边界
  python inversion.py SLSQP              # 序列二次规划
  python inversion.py trust-constr       # 信赖域
  python inversion.py differential_evolution   # 差分进化（全局，较慢）
"""
import os
import sys
import time
from datetime import datetime
import json

import numpy as np
from scipy.optimize import least_squares, minimize, differential_evolution
import matplotlib.pyplot as plt

# 可选优化器: 'least_squares' | 'L-BFGS-B' | 'SLSQP' | 'trust-constr' | 'differential_evolution'
# 也可通过命令行参数覆盖，如: python inversion.py L-BFGS-B
OPTIMIZER = "least_squares"

# 使用支持中文的字体（若系统无则回退默认）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def solve_heat_1d_core(q_flux, L, Nx, Nt, dt, rho, cp, k, T_init):
    """
    一维热传导隐式求解器。
    边界：x=0 恒定热流 q，x=L 绝热。
    """
    dx = L / (Nx - 1)
    alpha = k / (rho * cp)
    r = alpha * dt / (dx ** 2)
    T = np.ones(Nx) * T_init
    T_hist = np.zeros((Nt, Nx))
    T_hist[0, :] = T

    # 隐式差分矩阵 A: (1+2r)T_i - r*T_{i-1} - r*T_{i+1} = T_old
    A = (
        np.diag((1 + 2 * r) * np.ones(Nx))
        + np.diag(-r * np.ones(Nx - 1), 1)
        + np.diag(-r * np.ones(Nx - 1), -1)
    )
    A[0, 1] = -2 * r   # x=0 热流边界
    A[Nx - 1, Nx - 2] = -2 * r  # x=L 绝热边界

    invA = np.linalg.inv(A)

    for i in range(1, Nt):
        B = T.copy()
        B[0] = B[0] + (2 * r * dx / k) * q_flux[i]
        T = invA @ B
        T_hist[i, :] = T

    return T_hist


def objective_vec(q_vec, TOF_m, L, Nx, Nt, dt, rho, cp, k, T_init, va, vb, scale_factor):
    """序列热流下的 TOF 残差（用于优化）。"""
    q_vec = np.atleast_1d(q_vec).ravel()
    if q_vec.size == 1:
        q_vec = np.ones(Nt) * q_vec[0]
    T_h = solve_heat_1d_core(q_vec, L, Nx, Nt, dt, rho, cp, k, T_init)
    dx = L / (Nx - 1)
    TOF_c = np.zeros(Nt)
    for i in range(Nt):
        V_d = va * T_h[i, :] + vb
        TOF_c[i] = 2 * np.sum(dx / V_d)
    return (TOF_c - TOF_m) * scale_factor


def objective_scalar(q_val, TOF_m, L, Nx, Nt, dt, rho, cp, k, T_init, va, vb, scale_factor):
    """标量常数热流下的 TOF 残差。"""
    q_vec = np.ones(Nt) * q_val
    return objective_vec(q_vec, TOF_m, L, Nx, Nt, dt, rho, cp, k, T_init, va, vb, scale_factor)


def run_optimization(
    obj_fun,
    x0,
    bounds,
    optimizer_name=None,
    residual_scale: float = 1.0,
):
    """
    统一调用不同优化器。obj_fun(x) 返回残差向量（会被最小化 sum(res^2)）。
    返回 (x_opt, nfev, meta)。
    """
    name = (optimizer_name or OPTIMIZER).strip()

    if name == "least_squares":
        res = least_squares(
            obj_fun,
            x0,
            bounds=bounds,
            ftol=1e-10,
            xtol=1e-10,
            verbose=2,
        )
        meta = {
            "optimizer": name,
            "options": {"ftol": 1e-10, "xtol": 1e-10, "verbose": 2},
            "dim": int(np.atleast_1d(x0).ravel().size),
            "bounds_lb0": float(np.atleast_1d(bounds[0]).ravel()[0]),
            "bounds_ub0": float(np.atleast_1d(bounds[1]).ravel()[0]),
        }
        return res.x, res.nfev, meta

    # 以下优化器需要标量目标: 0.5 * sum(residual^2)
    def scalar_obj(x):
        r = obj_fun(x)
        return 0.5 * np.sum(r.astype(float) ** 2)

    lb = np.atleast_1d(bounds[0])
    ub = np.atleast_1d(bounds[1])
    n = len(x0)

    if name == "L-BFGS-B":
        res = minimize(
            scalar_obj,
            x0,
            method="L-BFGS-B",
            bounds=list(zip(lb, ub)),
            options={
                # 目标函数由残差平方构成：适当收紧 ftol，有助于得到更稳定的局部最优
                "ftol": 1e-12,
                # 有限差分步长：与 q 的量级更匹配，避免梯度估计近似为 0
                "eps": 1.0,
                "maxfun": 100000,
                "maxiter": 50000,
            },
        )
        meta = {
            "optimizer": name,
            "options": {"ftol": 1e-12, "maxfun": 100000, "maxiter": 50000},
            "dim": int(np.atleast_1d(x0).ravel().size),
            "bounds_lb0": float(lb.ravel()[0]),
            "bounds_ub0": float(ub.ravel()[0]),
        }
        return res.x, res.nfev, meta
    if name == "SLSQP":
        res = minimize(
            scalar_obj,
            x0,
            method="SLSQP",
            bounds=list(zip(lb, ub)),
            options={
                # SLSQP 对数值尺度较敏感；收紧 ftol，同时用更合适的步长 eps
                "ftol": 1e-14,
                "maxiter": 50000,
                # 有限差分步长：避免数值噪声下梯度估计过小
                "eps": 1.0,
            },
        )
        meta = {
            "optimizer": name,
            "options": {"ftol": 1e-14, "maxiter": 50000, "eps": 1.0},
            "dim": int(np.atleast_1d(x0).ravel().size),
            "bounds_lb0": float(lb.ravel()[0]),
            "bounds_ub0": float(ub.ravel()[0]),
        }
        return res.x, res.nfev, meta
    if name == "trust-constr":
        from scipy.optimize import Bounds

        res = minimize(
            scalar_obj,
            x0,
            method="trust-constr",
            bounds=Bounds(lb, ub),
            options={
                # 信赖域法：同时收紧梯度与步长停止准则
                "gtol": 1e-12,
                "xtol": 1e-12,
                "barrier_tol": 1e-12,
                "maxiter": 20000,
            },
        )
        meta = {
            "optimizer": name,
            "options": {"gtol": 1e-12, "xtol": 1e-12, "barrier_tol": 1e-12, "maxiter": 20000},
            "dim": int(np.atleast_1d(x0).ravel().size),
            "bounds_lb0": float(lb.ravel()[0]),
            "bounds_ub0": float(ub.ravel()[0]),
        }
        return res.x, res.nfunc, meta
    if name == "differential_evolution":
        # 全局优化，无梯度，耗时长，适合初值差或多峰
        res = differential_evolution(
            scalar_obj,
            bounds=list(zip(lb, ub)),
            seed=42,
            maxiter=500,
            popsize=min(20, max(6, n * 2)),
            atol=1e-7,
            tol=1e-7,
            polish=True,
            workers=1,
        )
        meta = {
            "optimizer": name,
            "options": {
                "seed": 42,
                "maxiter": 500,
                "popsize": int(min(20, max(6, n * 2))),
                "atol": 1e-7,
                "tol": 1e-7,
                "polish": True,
                "workers": 1,
            },
            "dim": int(np.atleast_1d(x0).ravel().size),
            "bounds_lb0": float(lb.ravel()[0]),
            "bounds_ub0": float(ub.ravel()[0]),
        }
        return res.x, res.nfev, meta

    raise ValueError(f"未知优化器: {name}, 可选: least_squares, L-BFGS-B, SLSQP, trust-constr, differential_evolution")


def main():
    run_optimizer = sys.argv[1].strip() if len(sys.argv) > 1 else OPTIMIZER
    t_total_start = time.perf_counter()

    # ========== 1. 物理参数设置 (参考不锈钢) ==========
    L = 0.1
    rho = 7930
    cp = 500
    k = 16.3
    alpha = k / (rho * cp)

    # 声速-温度模型: V(T) = va*T + vb (T 单位: K)
    va = -0.6487
    vb = 5934.9

    Nx = 50
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx)
    Nt = 40
    dt = 2.0
    T_init = 300

    # ========== 2. 模拟“真实”环境（生成观测数据） ==========
    t0 = time.perf_counter()
    q_const_true = 4e5  # 400 kW/m^2
    q_true = np.ones(Nt) * q_const_true

    T_true_history = solve_heat_1d_core(q_true, L, Nx, Nt, dt, rho, cp, k, T_init)

    TOF_measured = np.zeros(Nt)
    for i in range(Nt):
        V_dist = va * T_true_history[i, :] + vb
        TOF_measured[i] = 2 * np.sum(dx / V_dist)
    t_forward = time.perf_counter() - t0

    # ========== 3. 逆问题反演 ==========
    q_guess_init = 1e5
    scale_factor = 1e9

    # 方案 A：反演单一常数热流
    print(f"\n--- 正在进行单参数(常数)反演 (优化器: {run_optimizer}) ---")
    def obj_scalar(q):
        return objective_scalar(
            q[0], TOF_measured, L, Nx, Nt, dt, rho, cp, k, T_init, va, vb, scale_factor
        )
    t0 = time.perf_counter()
    q_inverted_scalar, nfev_scalar, meta_scalar = run_optimization(
        obj_scalar,
        [q_guess_init],
        (0, 1e7),
        optimizer_name=run_optimizer,
        residual_scale=scale_factor,
    )
    q_inverted_scalar = float(q_inverted_scalar[0])
    t_scalar = time.perf_counter() - t0

    # 方案 B：反演时间序列热流
    print(f"\n--- 正在进行多参数(序列)反演 (优化器: {run_optimizer}) ---")
    q_start_vec = np.ones(Nt) * q_guess_init
    def obj_vec(q):
        return objective_vec(
            q, TOF_measured, L, Nx, Nt, dt, rho, cp, k, T_init, va, vb, scale_factor
        )
    t0 = time.perf_counter()
    q_inverted_vec, nfev_vec, meta_vec = run_optimization(
        obj_vec,
        q_start_vec,
        (np.zeros(Nt), 1e7 * np.ones(Nt)),
        optimizer_name=run_optimizer,
        residual_scale=scale_factor,
    )
    t_vec = time.perf_counter() - t0

    # ========== 4. 结果验证与绘图 ==========
    T_reconstructed = solve_heat_1d_core(
        q_inverted_vec, L, Nx, Nt, dt, rho, cp, k, T_init
    )
    t_axis = np.arange(Nt) * dt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('white')
    fig.suptitle('Reconstruction Results', fontsize=12)

    ax1.plot(t_axis, q_true / 1e3, 'k-', lw=2, label='真实热流')
    ax1.plot(t_axis, q_inverted_vec / 1e3, 'r--', lw=1.5, label='反演热流')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('热流 (kW/m^2)')
    ax1.set_title('热流反演对比')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x, T_true_history[-1, :], 'k-', lw=2, label='真实温度')
    ax2.plot(x, T_reconstructed[-1, :], 'r--', lw=1.5, label='重建温度')
    ax2.set_xlabel('位置 (m)')
    ax2.set_ylabel('温度 (K)')
    ax2.set_title(f't = {(Nt-1)*dt} s 时刻温度分布')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # 图片输出：保存到 result 目录，文件名带时间戳
    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(result_dir, f"inversion_{timestamp}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n图片已保存: {fig_path}")
    plt.show()

    # 温度重建误差（末时刻）
    T_true_final = T_true_history[-1, :]
    T_recon_final = T_reconstructed[-1, :]
    temp_rmse = np.sqrt(np.mean((T_recon_final - T_true_final) ** 2))
    temp_max_err = np.max(np.abs(T_recon_final - T_true_final))

    t_total = time.perf_counter() - t_total_start

    # ========== 结果总结 ==========
    print("\n" + "=" * 60)
    print("                        结果总结")
    print("=" * 60)
    print(f"  真实热流 (恒定):     {q_const_true:>12.2f}  W/m^2  ({q_const_true/1e3:.2f} kW/m^2)")
    print(f"  反演常数热流:        {q_inverted_scalar:>12.2f}  W/m^2  ({q_inverted_scalar/1e3:.2f} kW/m^2)")
    print(f"  反演序列热流 均值:   {np.mean(q_inverted_vec):>12.2f}  W/m^2")
    print(f"  反演序列热流 标准差: {np.std(q_inverted_vec):>12.2f}  W/m^2")
    print("-" * 60)
    print(f"  末时刻温度 RMSE:     {temp_rmse:>12.4f}  K")
    print(f"  末时刻温度 最大误差: {temp_max_err:>12.4f}  K")
    print("-" * 60)
    print("  耗时统计:")
    print(f"    正问题(生成观测):  {t_forward:>8.3f}  s")
    print(f"    单参数反演:        {t_scalar:>8.3f}  s  (函数调用 {nfev_scalar} 次)")
    print(f"    多参数反演:        {t_vec:>8.3f}  s  (函数调用 {nfev_vec} 次)")
    print(f"    总耗时:            {t_total:>8.3f}  s")
    print("-" * 60)
    print(f"  输出图片: {fig_path}")
    print("=" * 60)

    # 保存“终端风格结果总结文本”与“参数信息（优化器设置）”
    # 方便你后续在 result 目录里直接对比不同方法的效果。
    temp_rmse_f = float(temp_rmse)
    temp_max_err_f = float(temp_max_err)
    temp_text_lines = []
    temp_text_lines.append("=" * 60)
    temp_text_lines.append("                        结果总结")
    temp_text_lines.append("=" * 60)
    temp_text_lines.append(
        f"  真实热流 (恒定):     {q_const_true:>12.2f}  W/m^2  ({q_const_true/1e3:.2f} kW/m^2)"
    )
    temp_text_lines.append(
        f"  反演常数热流:        {q_inverted_scalar:>12.2f}  W/m^2  ({q_inverted_scalar/1e3:.2f} kW/m^2)"
    )
    temp_text_lines.append(
        f"  反演序列热流 均值:   {np.mean(q_inverted_vec):>12.2f}  W/m^2"
    )
    temp_text_lines.append(
        f"  反演序列热流 标准差: {np.std(q_inverted_vec):>12.2f}  W/m^2"
    )
    temp_text_lines.append("-" * 60)
    temp_text_lines.append(f"  末时刻温度 RMSE:     {temp_rmse_f:>12.4f}  K")
    temp_text_lines.append(f"  末时刻温度 最大误差: {temp_max_err_f:>12.4f}  K")
    temp_text_lines.append("-" * 60)
    temp_text_lines.append("  耗时统计:")
    temp_text_lines.append(f"    正问题(生成观测):  {float(t_forward):>8.3f}  s")
    temp_text_lines.append(
        f"    单参数反演:        {float(t_scalar):>8.3f}  s  (函数调用 {nfev_scalar} 次)"
    )
    temp_text_lines.append(
        f"    多参数反演:        {float(t_vec):>8.3f}  s  (函数调用 {nfev_vec} 次)"
    )
    temp_text_lines.append(f"    总耗时:            {float(t_total):>8.3f}  s")
    temp_text_lines.append("-" * 60)
    temp_text_lines.append(f"  输出图片: {fig_path}")
    temp_text_lines.append("=" * 60)
    temp_report_text = "\n".join(temp_text_lines)

    report_txt_path = os.path.join(
        result_dir, f"inversion_{timestamp}_{run_optimizer}_report.txt"
    )
    payload = {
        "algorithm": "inversion",
        "optimizer_name": run_optimizer,
        "seed": 42,
        "physical_params": {
            "L": L,
            "rho": rho,
            "cp": cp,
            "k": k,
            "va": va,
            "vb": vb,
            "Nx": Nx,
            "Nt": Nt,
            "dt": dt,
            "T_init": T_init,
        },
        "initial_guess": {"q_guess_init": q_guess_init, "scale_factor": scale_factor},
        "optimizer_scalar_meta": meta_scalar,
        "optimizer_sequence_meta": meta_vec,
        "results": {
            "q_inverted_scalar": float(q_inverted_scalar),
            "q_inverted_vec_mean": float(np.mean(q_inverted_vec)),
            "q_inverted_vec_std": float(np.std(q_inverted_vec)),
            "temp_rmse_K": temp_rmse_f,
            "temp_max_err_K": temp_max_err_f,
            "t_forward_s": float(t_forward),
            "t_scalar_s": float(t_scalar),
            "t_vec_s": float(t_vec),
            "t_total_s": float(t_total),
            "nfev_scalar": int(nfev_scalar),
            "nfev_vec": int(nfev_vec),
        },
        "figure_path": fig_path,
    }
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(temp_report_text + "\n")
    with open(
        os.path.join(result_dir, f"inversion_{timestamp}_{run_optimizer}_params.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
