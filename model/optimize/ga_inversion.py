"""
遗传算法(Genetic Algorithm) 求解 inversion.py 对应的逆问题
目标函数与物理参数保持与 model/optimize/inversion.py 一致。

运行方式：
  python -m ai_model.model.optimize.ga_inversion
或在 model/optimize 目录下：
  python ga_inversion.py
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

# 观测生成：恒定热流（必须与 optimize/inversion.py 保持一致）
q_const_true = 4e5  # 400 kW/m^2
q_guess_init = 1e5

scale_factor = 1e9
bounds_scalar = (0.0, 1e7)
bounds_vec = (0.0, 1e7)  # 每个时间点一个热流变量


def build_inverse_problem():
    """
    构建逆问题求解器所需的缓存量：
    - 预计算隐式格式的 invA
    - 预计算 TOF_measured 与真值温度
    """
    dx = L / (Nx - 1)
    alpha = k / (rho * cp)
    r = alpha * dt / (dx**2)

    # implicit matrix A
    A = (
        np.diag((1.0 + 2.0 * r) * np.ones(Nx))
        + np.diag(-r * np.ones(Nx - 1), 1)
        + np.diag(-r * np.ones(Nx - 1), -1)
    )
    A[0, 1] = -2.0 * r
    A[Nx - 1, Nx - 2] = -2.0 * r
    invA = np.linalg.inv(A)

    # boundary contribution factor for B[0]
    bf = (2.0 * r * dx / k)

    def solve_heat_1d_core_cached(q_flux):
        """
        一维热传导隐式求解器（与 optimize/inversion.py 数学形式一致），但复用 invA 加速。
        边界：x=0 恒定热流 q，x=L 绝热。
        """
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

    # forward true
    q_true = np.ones(Nt) * q_const_true
    T_true_history = solve_heat_1d_core_cached(q_true)

    # TOF_measured
    TOF_measured = np.zeros(Nt)
    for i in range(Nt):
        V_dist = va * T_true_history[i, :] + vb
        TOF_measured[i] = 2.0 * np.sum(dx / V_dist)

    def objective_vec_cached(q_vec):
        """
        与 optimize/inversion.py objective_vec 完全一致：
        返回残差向量（未平方），尺度因子 scale_factor 在此处乘入。
        """
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
        r = objective_vec_cached(np.ones(Nt) * q_val)
        return 0.5 * float(np.sum(r.astype(float) ** 2))

    def target_vector(q_vec):
        r = objective_vec_cached(q_vec)
        return 0.5 * float(np.sum(r.astype(float) ** 2))

    return {
        "dx": dx,
        "solve_heat": solve_heat_1d_core_cached,
        "T_true_history": T_true_history,
        "TOF_measured": TOF_measured,
        "objective_vec": objective_vec_cached,
        "target_scalar": target_scalar,
        "target_vector": target_vector,
        "q_true": q_true,
    }


def tournament_select(rng, fitness, k=3):
    """在适应度更小的个体中做锦标赛选择（minimize）。"""
    idxs = rng.integers(0, len(fitness), size=k)
    best = idxs[np.argmin(fitness[idxs])]
    return best


def blend_crossover(rng, p1, p2, alpha=0.5):
    """
    BLX-alpha 混合交叉：
    新个体每个基因在 [min- alpha*range, max+ alpha*range] 上均匀采样。
    """
    c = np.empty_like(p1)
    for j in range(p1.size):
        x1, x2 = p1[j], p2[j]
        x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
        spread = x_max - x_min
        low = x_min - alpha * spread
        high = x_max + alpha * spread
        c[j] = rng.uniform(low, high)
    return c


def mutate_gaussian(rng, child, lb, ub, sigma, mutation_rate=0.2):
    """高斯变异 + 裁剪到边界。"""
    out = child.copy()
    dim = out.size
    for j in range(dim):
        if rng.random() < mutation_rate:
            out[j] += rng.normal(0.0, sigma)
    return np.clip(out, lb, ub)


def ga_minimize(
    rng,
    target_fn,
    dim,
    lb,
    ub,
    x0,
    pop_size=50,
    generations=40,
    elite_frac=0.1,
    crossover_alpha=0.5,
    mutation_rate=0.2,
    sigma0_frac=0.05,
):
    """
    简单实数编码 GA（minimize）。
    返回 (best_x, best_f, eval_count)
    """
    lb = np.full(dim, lb, dtype=float)
    ub = np.full(dim, ub, dtype=float)
    x0 = np.asarray(x0, dtype=float).ravel()
    if x0.size != dim:
        raise ValueError("x0 dim mismatch")

    # init population: around x0 + uniform fallback
    pop = np.empty((pop_size, dim), dtype=float)
    spread = 0.2 * (ub - lb)
    for i in range(pop_size):
        if rng.random() < 0.7:
            pop[i] = np.clip(x0 + rng.normal(0.0, spread), lb, ub)
        else:
            pop[i] = rng.uniform(lb, ub)

    eval_count = 0
    sigma0 = sigma0_frac * (ub[0] - lb[0])

    elite_count = max(1, int(np.ceil(pop_size * elite_frac)))

    best_x = None
    best_f = float("inf")

    for g in range(generations):
        fitness = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            fitness[i] = target_fn(pop[i])
        eval_count += pop_size

        order = np.argsort(fitness)
        if float(fitness[order[0]]) < best_f:
            best_f = float(fitness[order[0]])
            best_x = pop[order[0]].copy()

        # elitism
        new_pop = [pop[idx].copy() for idx in order[:elite_count]]

        # linearly decreasing mutation sigma
        t = g / max(1, generations - 1)
        sigma = sigma0 * (1.0 - t)
        while len(new_pop) < pop_size:
            i1 = tournament_select(rng, fitness, k=3)
            i2 = tournament_select(rng, fitness, k=3)
            p1 = pop[i1]
            p2 = pop[i2]
            child = blend_crossover(rng, p1, p2, alpha=crossover_alpha)
            child = mutate_gaussian(
                rng, child, lb, ub, sigma=sigma, mutation_rate=mutation_rate
            )
            new_pop.append(child)

        pop = np.stack(new_pop, axis=0)

    return best_x, best_f, eval_count


def main():
    rng = np.random.default_rng(42)
    prob = build_inverse_problem()
    x_axis = np.linspace(0, L, Nx)
    t_axis = np.arange(Nt) * dt

    ga_scalar_params = {
        "pop_size": 32,
        "generations": 45,
        "elite_frac": 0.18,
        "crossover_alpha": 0.55,
        "mutation_rate": 0.25,
        "sigma0_frac": 0.05,
    }
    ga_seq_params = {
        # 40维序列反演搜索预算需更大，避免过早停在较差局部点
        "pop_size": 50,
        "generations": 80,
        "elite_frac": 0.18,
        "crossover_alpha": 0.55,
        "mutation_rate": 0.25,
        "sigma0_frac": 0.06,
    }

    # ========== 1) 常数热流反演（dim=1） ==========
    t_start = time.perf_counter()
    print("\n--- GA：单参数(常数)反演 ---")
    best_x_scalar, best_f_scalar, evals_scalar = ga_minimize(
        rng,
        target_fn=lambda q: prob["target_scalar"](float(np.atleast_1d(q).ravel()[0])),
        dim=1,
        lb=bounds_scalar[0],
        ub=bounds_scalar[1],
        x0=[q_guess_init],
        **ga_scalar_params,
    )
    q_inverted_scalar = float(best_x_scalar[0])
    t_scalar = time.perf_counter() - t_start

    # ========== 2) 时间序列热流反演（dim=Nt） ==========
    t_start = time.perf_counter()
    print("\n--- GA：多参数(序列)反演 ---")
    best_x_vec, best_f_vec, evals_vec = ga_minimize(
        rng,
        target_fn=lambda q: prob["target_vector"](q),
        dim=Nt,
        lb=bounds_vec[0],
        ub=bounds_vec[1],
        x0=np.ones(Nt) * q_guess_init,
        **ga_seq_params,
    )
    q_inverted_vec = best_x_vec.astype(float)
    t_vec = time.perf_counter() - t_start

    # ========== 3) 重建温度与误差 ==========
    T_reconstructed = prob["solve_heat"](q_inverted_vec)

    q_true = prob["q_true"]
    T_true_history = prob["T_true_history"]

    T_true_final = T_true_history[-1, :]
    T_recon_final = T_reconstructed[-1, :]
    temp_rmse = float(np.sqrt(np.mean((T_recon_final - T_true_final) ** 2)))
    temp_max_err = float(np.max(np.abs(T_recon_final - T_true_final)))

    # ========== 4) 绘图与输出 ==========
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("white")
    fig.suptitle("Reconstruction Results (GA)", fontsize=12)

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
    fig_path = os.path.join(result_dir, f"ga_inversion_{timestamp}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n图片已保存: {fig_path}")

    report_lines = []
    report_lines.append(f"图片已保存: {fig_path}")
    report_lines.append("=" * 60)
    report_lines.append("                        结果总结 (GA)")
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
    print("                        结果总结 (GA)")
    print("=" * 60)
    print(
        f"  真实热流 (恒定):     {q_const_true:>12.2f}  W/m^2  ({q_const_true/1e3:.2f} kW/m^2)"
    )
    print(
        f"  反演常数热流:        {q_inverted_scalar:>12.2f}  W/m^2  ({q_inverted_scalar/1e3:.2f} kW/m^2)"
    )
    print(f"  反演序列热流 均值:   {np.mean(q_inverted_vec):>12.2f}  W/m^2")
    print(f"  反演序列热流 标准差: {np.std(q_inverted_vec):>12.2f}  W/m^2")
    print("-" * 60)
    print(f"  末时刻温度 RMSE:     {temp_rmse:>12.4f}  K")
    print(f"  末时刻温度 最大误差: {temp_max_err:>12.4f}  K")
    print("-" * 60)
    print("  耗时统计:")
    print(f"    单参数反演:        {t_scalar:>8.3f}  s  (函数评估 {evals_scalar} 次)")
    print(f"    多参数反演:        {t_vec:>8.3f}  s  (函数评估 {evals_vec} 次)")
    print("=" * 60)

    # 保存“终端风格”的结果文本 + 参数信息
    report_txt_path = os.path.join(result_dir, f"ga_inversion_{timestamp}_report.txt")
    payload = {
        "algorithm": "GA",
        "seed": 42,
        "ga_scalar_params": ga_scalar_params,
        "ga_seq_params": ga_seq_params,
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
        os.path.join(result_dir, f"ga_inversion_{timestamp}_params.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    plt.show()


if __name__ == "__main__":
    main()

