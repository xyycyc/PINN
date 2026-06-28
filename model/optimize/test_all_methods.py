"""
测试脚本：对 `optimize.py` 支持的所有反演方法做统一对比统计与绘图。

比较对象（与 optimize.py 的分发一致，去掉 alias 的重名）：
  ga
  pso
  sa
  least_squares
  L-BFGS-B
  SLSQP
  trust-constr
  differential_evolution

输出：
  写入新建目录：model/optimize/result/test_all_methods_<timestamp>/
    - summary.csv
    - comparison.png      （q(t) 与最终温度 T(x) 对比）
"""

from __future__ import annotations

import os
import time
from datetime import datetime
import argparse

import numpy as np

# 尽量避免在无 GUI 环境里阻塞
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import Bounds, differential_evolution, least_squares, minimize

try:
    from .ga_inversion import ga_minimize
    from .pso_inversion import pso_minimize
    from .sa_inversion import anneal_minimize
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from ga_inversion import ga_minimize
    from pso_inversion import pso_minimize
    from sa_inversion import anneal_minimize
#
# 注意：这里不要复用 inversion.py 的 solve_heat_1d_core/objective_vec
# （其内部每次都会反复求 inv(A)，在 GA/PSO/SA/DE 里会非常慢）。
# 本脚本会自己做一份与 inversion.py 数学一致的缓存求解器。


# -----------------------------
# 物理参数与 inversion.py 保持一致
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

x = np.linspace(0, L, Nx)
dx = L / (Nx - 1)
t_axis = np.arange(Nt) * dt


def build_forward_and_objective():
    alpha = k / (rho * cp)
    r = alpha * dt / (dx**2)

    # implicit matrix A（与 inversion.py 数学形式一致）
    A = (
        np.diag((1.0 + 2.0 * r) * np.ones(Nx))
        + np.diag(-r * np.ones(Nx - 1), 1)
        + np.diag(-r * np.ones(Nx - 1), -1)
    )
    A[0, 1] = -2.0 * r
    A[Nx - 1, Nx - 2] = -2.0 * r
    invA = np.linalg.inv(A)

    # 边界项系数（与 inversion.py solve_heat_1d_core 一致）
    bf = (2.0 * r * dx / k)

    def solve_heat_cached(q_flux: np.ndarray) -> np.ndarray:
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
    T_true_history = solve_heat_cached(q_true)

    # TOF_measured 生成方式与 model/optimize/inversion.py 一致
    TOF_measured = np.zeros(Nt)
    for i in range(Nt):
        V_dist = va * T_true_history[i, :] + vb
        TOF_measured[i] = 2 * np.sum(dx / V_dist)

    def obj_residual_vec(q_vec: np.ndarray) -> np.ndarray:
        q_vec = np.atleast_1d(q_vec).astype(float).ravel()
        if q_vec.size == 1:
            q_vec = np.ones(Nt) * q_vec[0]
        if q_vec.size != Nt:
            raise ValueError(f"q_vec length must be {Nt}, got {q_vec.size}")

        T_h = solve_heat_cached(q_vec)
        TOF_c = np.zeros(Nt)
        for i in range(Nt):
            V_d = va * T_h[i, :] + vb
            TOF_c[i] = 2 * np.sum(dx / V_d)
        return (TOF_c - TOF_measured) * scale_factor

    return q_true, T_true_history, solve_heat_cached, obj_residual_vec


def scalar_from_residual(residual_vec: np.ndarray) -> float:
    # 与 model/optimize/inversion.py run_optimization 中 scalar_obj = 0.5*sum(res^2) 一致
    r = residual_vec.astype(float)
    return 0.5 * float(np.sum(r**2))


def compute_temp_error(q_seq, T_true_history, solve_heat_cached):
    T_reconstructed = solve_heat_cached(q_seq)
    T_true_final = T_true_history[-1, :]
    T_recon_final = T_reconstructed[-1, :]
    temp_rmse = float(np.sqrt(np.mean((T_recon_final - T_true_final) ** 2)))
    temp_max_err = float(np.max(np.abs(T_recon_final - T_true_final)))
    return temp_rmse, temp_max_err, T_reconstructed


def solve_all_methods(de_maxiter: int, de_popsize: int, seed: int = 42):
    q_true, T_true_history, solve_heat_cached, obj_residual_vec = (
        build_forward_and_objective()
    )

    bounds_vec = (np.zeros(Nt), 1e7 * np.ones(Nt))
    x0_vec = np.ones(Nt) * q_guess_init

    results = {}

    def run(method_name, solver_fn, params: dict):
        print(f"\n=== Running {method_name} ===")
        t0 = time.perf_counter()
        q_seq, meta = solver_fn()
        runtime = time.perf_counter() - t0
        temp_rmse, temp_max_err, T_reconstructed = compute_temp_error(
            q_seq, T_true_history, solve_heat_cached
        )
        results[method_name] = {
            "q_seq": q_seq,
            "runtime_s": runtime,
            "temp_rmse": temp_rmse,
            "temp_max_err": temp_max_err,
            "meta": meta,
            "params": params,
            "T_reconstructed": T_reconstructed,
        }
        print(
            f"{method_name}: temp_rmse={temp_rmse:.6f}, temp_max_err={temp_max_err:.6f}, runtime={runtime:.2f}s"
        )

    # GA / PSO / SA
    # 这里直接使用与脚本 main() 同步的“默认超参数”，确保比较一致。
    run(
        "ga",
        lambda: (
            ga_minimize(
                rng=np.random.default_rng(seed),
                target_fn=lambda q: scalar_from_residual(obj_residual_vec(q)),
                dim=Nt,
                lb=0.0,
                ub=1e7,
                x0=np.ones(Nt) * q_guess_init,
                pop_size=50,
                generations=80,
                elite_frac=0.18,
                crossover_alpha=0.55,
                mutation_rate=0.25,
                sigma0_frac=0.06,
            )[0],
            {"method": "ga"},
        )
        ,
        {
            "ga_seq_params": {
                "pop_size": 50,
                "generations": 80,
                "elite_frac": 0.18,
                "crossover_alpha": 0.55,
                "mutation_rate": 0.25,
                "sigma0_frac": 0.06,
            },
            "seed": seed,
        },
    )

    run(
        "pso",
        lambda: (
            pso_minimize(
                rng=np.random.default_rng(seed),
                target_fn=lambda q: scalar_from_residual(obj_residual_vec(q)),
                dim=Nt,
                lb=0.0,
                ub=1e7,
                x0=np.ones(Nt) * q_guess_init,
                swarm_size=60,
                max_iter=110,
                w_start=0.9,
                w_end=0.3,
                c1=1.7,
                c2=1.7,
                init_spread_frac=0.08,
            )[0],
            {"method": "pso"},
        )
        ,
        {
            "pso_seq_params": {
                "swarm_size": 60,
                "max_iter": 110,
                "w_start": 0.9,
                "w_end": 0.3,
                "c1": 1.7,
                "c2": 1.7,
                "init_spread_frac": 0.08,
            },
            "seed": seed,
        },
    )

    run(
        "sa",
        lambda: (
            anneal_minimize(
                rng=np.random.default_rng(seed),
                target_fn=lambda q: scalar_from_residual(obj_residual_vec(q)),
                dim=Nt,
                lb=0.0,
                ub=1e7,
                x0=np.ones(Nt) * q_guess_init,
                max_steps=15000,
                T_init_factor=1.0,
                cooling=0.996,
                step_sigma_frac=0.003,
            )[0],
            {"method": "sa"},
        )
        ,
        {
            "sa_seq_params": {
                "max_steps": 15000,
                "T_init_factor": 1.0,
                "cooling": 0.996,
                "step_sigma_frac": 0.003,
            },
            "seed": seed,
        },
    )

    # least_squares / L-BFGS-B / SLSQP / trust-constr / differential_evolution
    def solver_least_squares():
        res = least_squares(
            obj_residual_vec,
            x0_vec,
            bounds=bounds_vec,
            ftol=1e-10,
            xtol=1e-10,
            verbose=0,
        )
        return res.x, {"nfev": int(res.nfev)}

    run(
        "least_squares",
        solver_least_squares,
        {
            "options": {"ftol": 1e-10, "xtol": 1e-10, "verbose": 0},
            "bounds": {"lb": 0.0, "ub": 1e7},
        },
    )
    # inversion 别名：与 model/optimize/inversion.py 默认 OPTIMIZER=least_squares 一致
    results["inversion"] = dict(results["least_squares"])

    def solver_lbfgsb():
        def scalar_obj(q):
            return scalar_from_residual(obj_residual_vec(q))

        lb, ub = bounds_vec
        res = minimize(
            scalar_obj,
            x0_vec,
            method="L-BFGS-B",
            bounds=list(zip(lb, ub)),
            options={"ftol": 1e-12, "maxfun": 100000, "maxiter": 50000, "eps": 1.0},
        )
        return res.x, {"nfev": int(res.nfev)}

    run(
        "L-BFGS-B",
        solver_lbfgsb,
        {
            "options": {"ftol": 1e-12, "maxfun": 100000, "maxiter": 50000, "eps": 1.0},
            "bounds": {"lb": 0.0, "ub": 1e7},
        },
    )

    def solver_slsqp():
        def scalar_obj(q):
            return scalar_from_residual(obj_residual_vec(q))

        lb, ub = bounds_vec
        res = minimize(
            scalar_obj,
            x0_vec,
            method="SLSQP",
            bounds=list(zip(lb, ub)),
            options={"ftol": 1e-14, "maxiter": 50000, "eps": 1.0},
        )
        return res.x, {"nfev": int(res.nfev)}

    run(
        "SLSQP",
        solver_slsqp,
        {
            "options": {"ftol": 1e-14, "maxiter": 50000, "eps": 1.0},
            "bounds": {"lb": 0.0, "ub": 1e7},
        },
    )

    def solver_trust_constr():
        def scalar_obj(q):
            return scalar_from_residual(obj_residual_vec(q))

        lb, ub = bounds_vec
        res = minimize(
            scalar_obj,
            x0_vec,
            method="trust-constr",
            bounds=Bounds(lb, ub),
            options={
                "gtol": 1e-12,
                "xtol": 1e-12,
                "barrier_tol": 1e-12,
                "maxiter": 20000,
            },
        )
        # trust-constr 的调用次数字段在 scipy 内部略有差异
        nfev = getattr(res, "nfev", None)
        return res.x, {"nfev": None if nfev is None else int(nfev)}

    run(
        "trust-constr",
        solver_trust_constr,
        {
            "options": {
                "gtol": 1e-12,
                "xtol": 1e-12,
                "barrier_tol": 1e-12,
                "maxiter": 20000,
            },
            "bounds": {"lb": 0.0, "ub": 1e7},
        },
    )

    def solver_de():
        def scalar_obj(q):
            return scalar_from_residual(obj_residual_vec(q))

        lb, ub = bounds_vec
        res = differential_evolution(
            scalar_obj,
            bounds=list(zip(lb, ub)),
            seed=seed,
            maxiter=de_maxiter,
            popsize=de_popsize,
            atol=1e-8,
            tol=1e-8,
            polish=True,
            workers=1,
        )
        return res.x, {"nfev": int(res.nfev)}

    # DE 默认最耗时；这里给一个较稳妥的预算，避免你跑一次等待太久。
    run(
        "differential_evolution",
        solver_de,
        {
            "options": {
                "seed": seed,
                "maxiter": de_maxiter,
                "popsize": de_popsize,
                "atol": 1e-8,
                "tol": 1e-8,
                "polish": True,
                "workers": 1,
            },
            "bounds": {"lb": 0.0, "ub": 1e7},
        },
    )

    return q_true, results


def save_outputs(q_true: np.ndarray, T_true_history: np.ndarray, results: dict[str, dict]):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "result", f"test_all_methods_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # summary.csv
    rows = []
    for name, info in results.items():
        rows.append(
            [
                name,
                info["temp_rmse"],
                info["temp_max_err"],
                info["runtime_s"],
                float(np.sqrt(np.mean((info["q_seq"] - q_true) ** 2))),
            ]
        )
    header = ["method", "temp_rmse_K", "temp_max_err_K", "runtime_s", "q_rmse_W_m2"]
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(
                f"{r[0]},{r[1]:.8f},{r[2]:.8f},{r[3]:.4f},{r[4]:.4f}\n"
            )

    # comparison.png: q(t) + T(x)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # q(t)
    ax1.plot(t_axis, q_true / 1e3, "k-", lw=2.0, label="true")
    for name, info in results.items():
        ax1.plot(t_axis, info["q_seq"] / 1e3, lw=1.3, label=name)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("q (kW/m^2)")
    ax1.set_title("Reconstructed heat flux q(t)")
    ax1.grid(True)
    ax1.legend(fontsize=7)

    # T(x) at final time
    # 取每个方法重建温度的末时刻曲线
    for name, info in results.items():
        T_last = info["T_reconstructed"][-1, :]
        ax2.plot(x, T_last, lw=1.3, label=name)
    # true curve
    T_true_final = T_true_history[-1, :]
    ax2.plot(x, T_true_final, "k-", lw=2.0, label="true")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("T (K)")
    ax2.set_title("Final temperature T(x)")
    ax2.grid(True)
    ax2.legend(fontsize=7)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, "comparison.png")
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")

    # params.json: 记录每个方法实际使用的算法超参数
    import json

    params_out = {}
    for name, info in results.items():
        params_out[name] = {
            "params": info.get("params", {}),
            "temp_rmse": info["temp_rmse"],
            "temp_max_err": info["temp_max_err"],
            "runtime_s": info["runtime_s"],
            "meta": info.get("meta", {}),
        }
    params_json_path = os.path.join(out_dir, "params.json")
    with open(params_json_path, "w", encoding="utf-8") as f:
        json.dump(params_out, f, ensure_ascii=False, indent=2)

    # report.txt：简短的“终端风格”总结（方便你直接看）
    report_txt_path = os.path.join(out_dir, "report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("搜索方法结果总结（按 temp_rmse 升序）\n")
        f.write("=" * 60 + "\n")
        for name in sorted(results.keys(), key=lambda n: results[n]["temp_rmse"]):
            info = results[name]
            f.write(
                f"{name:>22s}: temp_rmse={info['temp_rmse']:.6f}K, temp_max_err={info['temp_max_err']:.6f}K, runtime={info['runtime_s']:.2f}s\n"
            )
        f.write("=" * 60 + "\n")

    print(f"\n对比结果已保存到：{out_dir}")
    print(f" - summary: {csv_path}")
    print(f" - figure : {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--de_maxiter", type=int, default=150, help="differential_evolution 的 maxiter")
    parser.add_argument("--de_popsize", type=int, default=40, help="differential_evolution 的 popsize")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    q_true, results = solve_all_methods(
        de_maxiter=args.de_maxiter,
        de_popsize=args.de_popsize,
        seed=args.seed,
    )
    # solve_all_methods 还没有返回 T_true_history；这里重算一份真解用于绘图
    # （省去修改函数签名带来的连锁改动）
    # 如果你更偏好严格一致，可继续把 T_true_history 一起返回。
    _, T_true_history, _, _ = build_forward_and_objective()
    save_outputs(q_true, T_true_history, results)


if __name__ == "__main__":
    main()

