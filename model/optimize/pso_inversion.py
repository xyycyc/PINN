"""
粒子群算法(PSO) 求解 inversion.py 对应的逆问题
目标函数与物理参数保持与 model/optimize/inversion.py 一致。

运行方式：
  python -m ai_model.model.optimize.pso_inversion
或在 model/optimize 目录下：
  python pso_inversion.py
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

q_const_true = 4e5  # 400 kW/m^2
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
        "dx": dx,
        "solve_heat": solve_heat_1d_core_cached,
        "T_true_history": T_true_history,
        "TOF_measured": TOF_measured,
        "q_true": q_true,
        "target_scalar": target_scalar,
        "target_vector": target_vector,
    }


def pso_minimize(
    rng,
    target_fn,
    dim,
    lb,
    ub,
    x0=None,
    swarm_size=50,
    max_iter=120,
    w_start=0.9,
    w_end=0.4,
    c1=1.6,
    c2=1.6,
    init_spread_frac=0.25,
):
    lb = float(lb)
    ub = float(ub)
    if x0 is None:
        x0 = np.full(dim, (lb + ub) / 2.0, dtype=float)
    x0 = np.asarray(x0, dtype=float).ravel()
    if x0.size != dim:
        raise ValueError("x0 dim mismatch")

    span = ub - lb
    init_spread = init_spread_frac * span

    # init swarm
    pos = np.empty((swarm_size, dim), dtype=float)
    vel = np.zeros((swarm_size, dim), dtype=float)
    for i in range(swarm_size):
        if rng.random() < 0.7:
            pos[i] = np.clip(x0 + rng.normal(0.0, init_spread, size=dim), lb, ub)
        else:
            pos[i] = rng.uniform(lb, ub, size=dim)
        vel[i] = rng.normal(0.0, 0.05 * span, size=dim)

    pbest_pos = pos.copy()
    pbest_val = np.empty(swarm_size, dtype=float)
    for i in range(swarm_size):
        pbest_val[i] = target_fn(pos[i])
    gbest_idx = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = float(pbest_val[gbest_idx])

    evals = swarm_size

    for it in range(max_iter):
        w = w_start + (w_end - w_start) * (it / max(1, max_iter - 1))
        for i in range(swarm_size):
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            vel[i] = (
                w * vel[i]
                + c1 * r1 * (pbest_pos[i] - pos[i])
                + c2 * r2 * (gbest_pos - pos[i])
            )
            pos[i] = pos[i] + vel[i]

            # project to bounds
            clipped = np.clip(pos[i], lb, ub)
            pos[i] = clipped

            val = target_fn(pos[i])
            evals += 1
            if val < pbest_val[i]:
                pbest_val[i] = float(val)
                pbest_pos[i] = pos[i].copy()
                if val < gbest_val:
                    gbest_val = float(val)
                    gbest_pos = pos[i].copy()

    return gbest_pos, gbest_val, evals


def main():
    rng = np.random.default_rng(42)
    prob = build_inverse_problem()
    x_axis = np.linspace(0, L, Nx)
    t_axis = np.arange(Nt) * dt

    pso_scalar_params = {
        "swarm_size": 25,
        "max_iter": 90,
        "w_start": 0.9,
        "w_end": 0.34,
        "c1": 1.7,
        "c2": 1.7,
        "init_spread_frac": 0.09,
    }
    pso_seq_params = {
        # 40维序列反演：提高粒子群规模与迭代次数
        "swarm_size": 60,
        "max_iter": 110,
        "w_start": 0.9,
        "w_end": 0.3,
        "c1": 1.7,
        "c2": 1.7,
        "init_spread_frac": 0.08,
    }

    # ========== 1) 常数热流反演（dim=1） ==========
    t_start = time.perf_counter()
    print("\n--- PSO：单参数(常数)反演 ---")
    best_x_scalar, best_f_scalar, evals_scalar = pso_minimize(
        rng,
        target_fn=lambda q: prob["target_scalar"](float(np.atleast_1d(q).ravel()[0])),
        dim=1,
        lb=bounds_scalar[0],
        ub=bounds_scalar[1],
        x0=[q_guess_init],
        **pso_scalar_params,
    )
    q_inverted_scalar = float(best_x_scalar[0])
    t_scalar = time.perf_counter() - t_start

    # ========== 2) 时间序列热流反演（dim=Nt） ==========
    t_start = time.perf_counter()
    print("\n--- PSO：多参数(序列)反演 ---")
    best_x_vec, best_f_vec, evals_vec = pso_minimize(
        rng,
        target_fn=lambda q: prob["target_vector"](q),
        dim=Nt,
        lb=bounds_vec[0],
        ub=bounds_vec[1],
        x0=np.ones(Nt) * q_guess_init,
        **pso_seq_params,
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
    fig.suptitle("Reconstruction Results (PSO)", fontsize=12)

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
    fig_path = os.path.join(result_dir, f"pso_inversion_{timestamp}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n图片已保存: {fig_path}")

    report_lines = []
    report_lines.append(f"图片已保存: {fig_path}")
    report_lines.append("=" * 60)
    report_lines.append("                        结果总结 (PSO)")
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
    print("                        结果总结 (PSO)")
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

    report_txt_path = os.path.join(result_dir, f"pso_inversion_{timestamp}_report.txt")
    payload = {
        "algorithm": "PSO",
        "seed": 42,
        "pso_scalar_params": pso_scalar_params,
        "pso_seq_params": pso_seq_params,
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
        os.path.join(result_dir, f"pso_inversion_{timestamp}_params.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    plt.show()


if __name__ == "__main__":
    main()

