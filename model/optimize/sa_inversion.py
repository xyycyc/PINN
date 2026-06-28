"""
退火算法(Simulated Annealing) 求解 inversion.py 对应的逆问题
目标函数与物理参数保持与 model/optimize/inversion.py 一致。

运行方式：
  python -m ai_model.model.optimize.sa_inversion
或在 model/optimize 目录下：
  python sa_inversion.py
"""

import os
import sys
import time
from datetime import datetime

import json
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 物理参数（必须与 optimize/inversion.py 保持一致）
# -----------------------------
L = 0.1
rho = 7930
cp = 500
k = 16.3

va = -0.6487
vb = 5934.9

Nx = 50
Nt = 40
dt = 2.0
T_init = 300

q_const_true = 4e5
q_guess_init = 1e5

scale_factor = 1e9
bounds_scalar = (0.0, 1e7)
bounds_vec = (0.0, 1e7)


def build_inverse_problem():
    dx = L / (Nx - 1)
    alpha = k / (rho * cp)
    r = alpha * dt / (dx**2)

    A = (
        np.diag((1.0 + 2.0 * r) * np.ones(Nx))
        + np.diag(-r * np.ones(Nx - 1), 1)
        + np.diag(-r * np.ones(Nx - 1), -1)
    )
    A[0, 1] = -2.0 * r
    A[Nx - 1, Nx - 2] = -2.0 * r
    invA = np.linalg.inv(A)

    bf = (2.0 * r * dx / k)

    def solve_heat_1d_core_cached(q_flux):
        q_flux = np.atleast_1d(q_flux).astype(float).ravel()
        if q_flux.size == 1:
            q_flux = np.ones(Nt) * q_flux[0]
        if q_flux.size != Nt:
            raise ValueError(f"q_flux length must be {Nt}, got {q_flux.size}")

        T = np.ones(Nx) * T_init
        T_hist = np.zeros((Nt, Nx))
        T_hist[0, :] = T

        for i in range(1, Nt):
            B = T.copy()
            B[0] = B[0] + bf * q_flux[i]
            T = invA @ B
            T_hist[i, :] = T
        return T_hist

    q_true = np.ones(Nt) * q_const_true
    T_true_history = solve_heat_1d_core_cached(q_true)

    TOF_measured = np.zeros(Nt)
    for i in range(Nt):
        V_dist = va * T_true_history[i, :] + vb
        TOF_measured[i] = 2.0 * np.sum(dx / V_dist)

    def objective_vec_cached(q_vec):
        q_vec = np.atleast_1d(q_vec).astype(float).ravel()
        if q_vec.size == 1:
            q_vec = np.ones(Nt) * q_vec[0]
        T_h = solve_heat_1d_core_cached(q_vec)
        TOF_c = np.zeros(Nt)
        for i in range(Nt):
            V_d = va * T_h[i, :] + vb
            TOF_c[i] = 2.0 * np.sum(dx / V_d)
        return (TOF_c - TOF_measured) * scale_factor

    def target_scalar(q_val):
        r = objective_vec_cached(np.ones(Nt) * float(q_val))
        return 0.5 * float(np.sum(r.astype(float) ** 2))

    def target_vector(q_vec):
        r = objective_vec_cached(q_vec)
        return 0.5 * float(np.sum(r.astype(float) ** 2))

    return {
        "solve_heat": solve_heat_1d_core_cached,
        "T_true_history": T_true_history,
        "q_true": q_true,
        "target_scalar": target_scalar,
        "target_vector": target_vector,
    }


def anneal_minimize(
    rng,
    target_fn,
    dim,
    lb,
    ub,
    x0,
    max_steps=20000,
    T_init_factor=1.0,
    cooling=0.995,
    step_sigma_frac=0.02,
):
    lb = float(lb)
    ub = float(ub)
    span = ub - lb

    x = np.asarray(x0, dtype=float).ravel().copy()
    if x.size != dim:
        raise ValueError("x0 dim mismatch")
    x = np.clip(x, lb, ub)

    cur_val = float(target_fn(x))
    T = max(1e-12, cur_val * T_init_factor)
    sigma0 = step_sigma_frac * span

    evals = 1
    best_x = x.copy()
    best_val = cur_val

    for step in range(max_steps):
        # annealing step size decays with temperature
        sigma = sigma0 * np.sqrt(max(T, 1e-12) / max(T_init_factor * cur_val, 1e-12))
        sigma = max(1e-12, sigma)

        proposal = x + rng.normal(0.0, sigma, size=dim)
        proposal = np.clip(proposal, lb, ub)

        prop_val = float(target_fn(proposal))
        evals += 1
        delta = prop_val - cur_val

        accept = False
        if delta < 0:
            accept = True
        else:
            p = np.exp(-delta / max(T, 1e-12))
            if rng.random() < p:
                accept = True

        if accept:
            x = proposal
            cur_val = prop_val
            if cur_val < best_val:
                best_val = cur_val
                best_x = x.copy()

        T *= cooling
        if T < 1e-15:
            break

    return best_x, best_val, evals


def main():
    rng = np.random.default_rng(42)
    prob = build_inverse_problem()
    x_axis = np.linspace(0, L, Nx)
    t_axis = np.arange(Nt) * dt

    sa_scalar_params = {
        "max_steps": 11000,
        "T_init_factor": 1.0,
        "cooling": 0.9965,
        "step_sigma_frac": 0.008,
    }
    sa_seq_params = {
        # 40维序列反演：需要更长退火过程
        "max_steps": 15000,
        "T_init_factor": 1.0,
        "cooling": 0.996,
        "step_sigma_frac": 0.003,
    }

    # ========== 1) 常数热流反演（dim=1） ==========
    t_start = time.perf_counter()
    print("\n--- SA：单参数(常数)反演 ---")
    best_x_scalar, best_f_scalar, evals_scalar = anneal_minimize(
        rng,
        target_fn=lambda q: prob["target_scalar"](float(np.atleast_1d(q).ravel()[0])),
        dim=1,
        lb=bounds_scalar[0],
        ub=bounds_scalar[1],
        x0=[q_guess_init],
        **sa_scalar_params,
    )
    q_inverted_scalar = float(best_x_scalar[0])
    t_scalar = time.perf_counter() - t_start

    # ========== 2) 时间序列热流反演（dim=Nt） ==========
    t_start = time.perf_counter()
    print("\n--- SA：多参数(序列)反演 ---")
    best_x_vec, best_f_vec, evals_vec = anneal_minimize(
        rng,
        target_fn=lambda q: prob["target_vector"](q),
        dim=Nt,
        lb=bounds_vec[0],
        ub=bounds_vec[1],
        x0=np.ones(Nt) * q_guess_init,
        **sa_seq_params,
    )
    q_inverted_vec = best_x_vec.astype(float)
    t_vec = time.perf_counter() - t_start

    # ========== 3) 重建温度与误差 ==========
    T_reconstructed = prob["solve_heat"](q_inverted_vec)
    T_true_history = prob["T_true_history"]

    T_true_final = T_true_history[-1, :]
    T_recon_final = T_reconstructed[-1, :]
    temp_rmse = float(np.sqrt(np.mean((T_recon_final - T_true_final) ** 2)))
    temp_max_err = float(np.max(np.abs(T_recon_final - T_true_final)))

    # ========== 4) 绘图与输出 ==========
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    q_true = prob["q_true"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("white")
    fig.suptitle("Reconstruction Results (SA)", fontsize=12)

    ax1.plot(t_axis, q_true / 1e3, "k-", lw=2, label="真实热流")
    ax1.plot(t_axis, q_inverted_vec / 1e3, "r--", lw=1.5, label="反演热流")
    ax1.set_xlabel("时间 (s)")
    ax1.set_ylabel("热流 (kW/m^2)")
    ax1.set_title("热流反演对比")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x_axis, T_true_final, "k-", lw=2, label="真实温度")
    ax2.plot(x_axis, T_recon_final, "r--", lw=1.5, label="重建温度")
    ax2.set_xlabel("位置 (m)")
    ax2.set_ylabel("温度 (K)")
    ax2.set_title(f"t = {(Nt - 1) * dt} s 时刻温度分布")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(result_dir, f"sa_inversion_{timestamp}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n图片已保存: {fig_path}")

    report_lines = []
    report_lines.append(f"图片已保存: {fig_path}")
    report_lines.append("=" * 60)
    report_lines.append("                        结果总结 (SA)")
    report_lines.append("=" * 60)
    report_lines.append(
        f"  真实热流 (恒定):     {q_const_true:>12.2f}  W/m^2  ({q_const_true/1e3:.2f} kW/m^2)"
    )
    report_lines.append(
        f"  反演常数热流:        {q_inverted_scalar:>12.2f}  W/m^2  ({q_inverted_scalar/1e3:.2f} kW/m^2)"
    )
    report_lines.append(
        f"  反演序列热流 均值:   {np.mean(q_inverted_vec):>12.2f}  W/m^2"
    )
    report_lines.append(
        f"  反演序列热流 标准差: {np.std(q_inverted_vec):>12.2f}  W/m^2"
    )
    report_lines.append("-" * 60)
    report_lines.append(f"  末时刻温度 RMSE:     {temp_rmse:>12.4f}  K")
    report_lines.append(f"  末时刻温度 最大误差: {temp_max_err:>12.4f}  K")
    report_lines.append("-" * 60)
    report_lines.append("  耗时统计:")
    report_lines.append(
        f"    单参数反演:        {t_scalar:>8.3f}  s  (函数评估 {evals_scalar} 次)"
    )
    report_lines.append(
        f"    多参数反演:        {t_vec:>8.3f}  s  (函数评估 {evals_vec} 次)"
    )
    report_lines.append("=" * 60)
    report_text = "\n".join(report_lines)

    print("\n" + "=" * 60)
    print("                        结果总结 (SA)")
    print("=" * 60)
    print(
        f"  真实热流 (恒定):     {q_const_true:>12.2f}  W/m^2  ({q_const_true/1e3:.2f} kW/m^2)"
    )
    print(
        f"  反演常数热流:        {q_inverted_scalar:>12.2f}  W/m^2  ({q_inverted_scalar/1e3:.2f} kW/m^2)"
    )
    print(f"  反演序列热流 均值:   {np.mean(q_inverted_vec):>12.2f}  W/m^2")
    print(
        f"  反演序列热流 标准差: {np.std(q_inverted_vec):>12.2f}  W/m^2"
    )
    print("-" * 60)
    print(f"  末时刻温度 RMSE:     {temp_rmse:>12.4f}  K")
    print(f"  末时刻温度 最大误差: {temp_max_err:>12.4f}  K")
    print("-" * 60)
    print("  耗时统计:")
    print(f"    单参数反演:        {t_scalar:>8.3f}  s  (函数评估 {evals_scalar} 次)")
    print(f"    多参数反演:        {t_vec:>8.3f}  s  (函数评估 {evals_vec} 次)")
    print("=" * 60)

    report_txt_path = os.path.join(result_dir, f"sa_inversion_{timestamp}_report.txt")
    payload = {
        "algorithm": "SA",
        "seed": 42,
        "sa_scalar_params": sa_scalar_params,
        "sa_seq_params": sa_seq_params,
        "q_const_true": q_const_true,
        "q_guess_init": q_guess_init,
        "results": {
            "q_inverted_scalar": q_inverted_scalar,
            "q_inverted_vec_mean": float(np.mean(q_inverted_vec)),
            "q_inverted_vec_std": float(np.std(q_inverted_vec)),
            "temp_rmse": temp_rmse,
            "temp_max_err": temp_max_err,
            "t_scalar_s": t_scalar,
            "t_vec_s": t_vec,
            "evals_scalar": int(evals_scalar),
            "evals_vec": int(evals_vec),
        },
        "figure_path": fig_path,
    }
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    with open(
        os.path.join(result_dir, f"sa_inversion_{timestamp}_params.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    plt.show()


if __name__ == "__main__":
    main()

