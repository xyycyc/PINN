#!/usr/bin/env python3
"""对 10times 十次实验中共有温度点的波形取均值，输出到 metal_avg。"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ...fortran import average_waveform_pairs
except ImportError:
    # Support direct execution: python database/raw/compute_metal_avg.py
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from fortran import average_waveform_pairs

HEADER_LINES = 15
SKIP_ROWS = 15
RUN_DIR_PATTERN = re.compile(r"^\d{2}$")
TEMP_STEM_PATTERN = re.compile(r"^[-+]?\d+$")


def _discover_runs(source_dir: Path) -> list[Path]:
    runs = sorted(
        (p for p in source_dir.iterdir() if p.is_dir() and RUN_DIR_PATTERN.fullmatch(p.name)),
        key=lambda p: p.name,
    )
    if len(runs) != 10:
        raise ValueError(f"期望 10 次实验子目录 (01–10)，实际找到 {len(runs)} 个: {source_dir}")
    return runs


def _temperature_stems(run_dir: Path) -> set[str]:
    stems: set[str] = set()
    for csv_path in run_dir.glob("*.csv"):
        if TEMP_STEM_PATTERN.fullmatch(csv_path.stem):
            stems.add(csv_path.stem)
    return stems


def common_temperatures(runs: list[Path]) -> list[str]:
    per_run = [_temperature_stems(run_dir) for run_dir in runs]
    shared = set.intersection(*per_run)
    return sorted(shared, key=int)


def read_waveform(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(csv_path, header=None, skiprows=SKIP_ROWS, usecols=[0, 1])
    time_s = frame.iloc[:, 0].to_numpy(dtype=np.float64)
    voltage = frame.iloc[:, 1].to_numpy(dtype=np.float64)
    if time_s.size == 0:
        raise ValueError(f"波形为空: {csv_path}")
    return time_s, voltage


def read_header_lines(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
        return [handle.readline().rstrip("\n\r") for _ in range(HEADER_LINES)]


def patch_hoffset(header_lines: list[str], hoffset: float) -> list[str]:
    patched: list[str] = []
    for line in header_lines:
        if "HOffset" in line:
            if '"' in line:
                patched.append(f'"HOffset"               ,{hoffset:.6E}   ')
            else:
                patched.append(f"HOffset,{hoffset:.6E}")
        else:
            patched.append(line)
    return patched


def format_wave_row(time_s: float, voltage: float) -> str:
    return f"  {time_s:.7E},       {voltage:.2E},"


def write_averaged_csv(
    output_path: Path,
    header_lines: list[str],
    time_s: np.ndarray,
    voltage: np.ndarray,
) -> None:
    header_lines = patch_hoffset(header_lines, float(time_s[0]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in header_lines:
            handle.write(f"{line}\n")
        handle.write("\n")
        for t, v in zip(time_s, voltage, strict=True):
            handle.write(f"{format_wave_row(t, v)}\n")


def average_temperature(runs: list[Path], temperature: str) -> tuple[np.ndarray, np.ndarray]:
    time_stack: list[np.ndarray] = []
    voltage_stack: list[np.ndarray] = []
    expected_len: int | None = None

    for run_dir in runs:
        csv_path = run_dir / f"{temperature}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"缺少温度文件: {csv_path}")
        time_s, voltage = read_waveform(csv_path)
        if expected_len is None:
            expected_len = time_s.size
        elif time_s.size != expected_len:
            raise ValueError(
                f"温度 {temperature} 在 {run_dir.name} 中长度为 {time_s.size}，"
                f"与参考长度 {expected_len} 不一致"
            )
        time_stack.append(time_s)
        voltage_stack.append(voltage)

    return average_waveform_pairs(
        np.stack(time_stack, axis=0),
        np.stack(voltage_stack, axis=0),
    )


def compute_metal_avg(
    source_dir: Path,
    output_dir: Path,
    *,
    verbose: bool = True,
) -> dict[str, object]:
    runs = _discover_runs(source_dir)
    temperatures = common_temperatures(runs)
    if not temperatures:
        raise ValueError(f"未找到十次实验共有的温度 CSV: {source_dir}")

    template_dir = runs[0]
    skipped = sorted(
        set().union(*[_temperature_stems(run_dir) for run_dir in runs]) - set(temperatures),
        key=int,
    )

    for temperature in temperatures:
        time_avg, voltage_avg = average_temperature(runs, temperature)
        header_lines = read_header_lines(template_dir / f"{temperature}.csv")
        write_averaged_csv(output_dir / f"{temperature}.csv", header_lines, time_avg, voltage_avg)
        if verbose:
            print(f"  {temperature}.csv  ({voltage_avg.size} 点)")

    summary = {
        "numeric_backend": "fortran",
        "run_count": len(runs),
        "temperature_count": len(temperatures),
        "temperatures": temperatures,
        "skipped_temperatures": skipped,
        "output_dir": str(output_dir),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="计算 10times 十次实验中共有温度波形的均值，写入 metal_avg。"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=script_dir / "10times",
        help="十次实验根目录（含 01–10 子目录，默认 database/raw/10times）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "metal_avg",
        help="输出目录（默认 database/raw/metal_avg）",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="不打印逐文件进度")
    args = parser.parse_args(argv)

    source_dir = args.source.resolve()
    output_dir = args.output.resolve()
    if not source_dir.is_dir():
        print(f"错误: 源目录不存在: {source_dir}", file=sys.stderr)
        return 1

    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    try:
        summary = compute_metal_avg(source_dir, output_dir, verbose=not args.quiet)
    except (ValueError, FileNotFoundError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    print(
        f"完成: {summary['temperature_count']} 个共有温度点 "
        f"（十次实验各 {summary['run_count']} 组）"
    )
    if summary["skipped_temperatures"]:
        print(
            "已跳过非十次共有温度:",
            ", ".join(summary["skipped_temperatures"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
