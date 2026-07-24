"""子进程运行器：把 ``python -m ai_model ...`` 的输出实时回灌到 GUI。

使用方式（典型）::

    runner = CommandRunner(log_callback=lambda line: print(line, end=""))
    runner.run(["python", "-m", "ai_model", "validate"], on_finish=...)
    # ...
    runner.stop()

设计要点
--------
1. 真正执行命令的是后台线程，避免阻塞 tkinter 主循环；
2. ``log_callback`` 在工作线程被调用，调用方需自行做线程安全转发
   （本项目里统一通过 ``queue.Queue`` + ``Tk.after`` 转发到 UI）；
3. 默认子进程 ``cwd`` 为 ``ai_model`` 包目录的上一级，便于 ``python -m ai_model``；
   调用 ``run(..., cwd=...)`` 可覆盖（例如从 ``settings.json`` 读取的路径）；
4. 强制设置 ``PYTHONIOENCODING=utf-8`` / ``PYTHONUTF8=1``，
   否则 Windows 控制台默认 GBK 容易把中文日志打成乱码；
5. 通过 ``stop()`` 主动终止仍在运行的进程，先尝试 ``terminate``，
   超时再 ``kill``。
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


LogCallback = Callable[[str], None]
FinishCallback = Callable[[int | None], None]


def ai_model_package_dir() -> Path:
    """``ai_model`` 包根目录（含 ``window/``、``cli.py`` 等），与 ``AIModelConfig.repo_root`` 一致。"""

    return Path(__file__).resolve().parents[1]


def default_subprocess_cwd() -> Path:
    """运行 ``python -m ai_model ...`` 的推荐工作目录：包目录的上一级，便于解析顶层包名。"""

    return ai_model_package_dir().parent


def format_command(cmd: Sequence[str]) -> str:
    """把命令格式化为可在 shell 复制运行的字符串，方便日志显示。"""

    parts: list[str] = []
    for item in cmd:
        text = str(item)
        if not text:
            parts.append('""')
            continue
        if any(ch.isspace() for ch in text) or any(ch in text for ch in '"\''):
            quoted = text.replace('"', '\\"')
            parts.append(f'"{quoted}"')
        else:
            parts.append(text)
    return " ".join(parts)


class CommandRunner:
    """单实例命令执行器，同一时刻只允许跑一条命令。"""

    def __init__(
        self,
        log_callback: LogCallback | None = None,
        default_cwd: Path | None = None,
    ) -> None:
        self._log = log_callback or (lambda line: None)
        # 默认 CWD = ai_model 的上一级（便于 ``python -m ai_model``）；调用 ``run(..., cwd=...)`` 可覆盖。
        self._default_cwd = Path(default_cwd) if default_cwd else default_subprocess_cwd()
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def default_cwd(self) -> Path:
        return self._default_cwd

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def run(
        self,
        cmd: Sequence[str],
        on_finish: FinishCallback | None = None,
        cwd: Path | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> bool:
        """启动一条命令，返回是否成功启动。"""

        if self.is_running:
            self._log("[runner] 已有任务正在执行，请先停止当前任务。\n")
            return False

        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})
        # ``launch_root`` is intentionally configurable in the Window.  Keep the
        # local ai_model package importable even when that working directory is
        # outside the repository (the default cwd happens to work without this).
        package_parent = str(ai_model_package_dir().parent)
        python_path = [item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item]
        if not any(Path(item).resolve() == Path(package_parent).resolve() for item in python_path):
            python_path.insert(0, package_parent)
        env["PYTHONPATH"] = os.pathsep.join(python_path)

        cwd_path = Path(cwd) if cwd else self._default_cwd
        cmd_text = format_command(cmd)
        self._log(f"[runner] CWD = {cwd_path}\n")
        self._log(f"[runner] $ {cmd_text}\n")

        try:
            popen_kwargs: dict[str, Any] = dict(
                cwd=str(cwd_path),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            self._proc = subprocess.Popen(list(cmd), **popen_kwargs)
        except FileNotFoundError as exc:
            self._log(f"[runner] 启动失败: {exc}\n")
            self._proc = None
            if on_finish is not None:
                on_finish(-1)
            return False

        self._thread = threading.Thread(
            target=self._stream_output,
            args=(on_finish,),
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is None:
            return
        if proc.poll() is not None:
            return
        self._log("[runner] 正在请求终止子进程...\n")
        try:
            proc.terminate()
        except Exception as exc:  # pragma: no cover - 依赖系统行为
            self._log(f"[runner] terminate 失败: {exc}\n")
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._log("[runner] terminate 超时，强制 kill。\n")
            try:
                proc.kill()
            except Exception as exc:  # pragma: no cover
                self._log(f"[runner] kill 失败: {exc}\n")

    def _stream_output(self, on_finish: FinishCallback | None) -> None:
        proc = self._proc
        assert proc is not None
        try:
            assert proc.stdout is not None
            with proc.stdout:
                for line in proc.stdout:
                    self._log(line)
        except Exception as exc:  # pragma: no cover
            self._log(f"[runner] 读取输出异常: {exc}\n")
        finally:
            try:
                exit_code = proc.wait()
            except Exception:
                exit_code = -1
            self._log(f"[runner] 进程结束，exit_code = {exit_code}\n")
            with self._lock:
                self._proc = None
            if on_finish is not None:
                try:
                    on_finish(exit_code)
                except Exception as exc:  # pragma: no cover
                    self._log(f"[runner] on_finish 回调异常: {exc}\n")


def python_module_command(
    module: str,
    *args: str,
    executable: str | None = None,
) -> list[str]:
    """构造 ``<python> -m <module> args...`` 命令。"""

    exe = (executable or "").strip() or sys.executable
    return [exe, "-m", module, *[str(a) for a in args]]


def shlex_parse(text: str) -> list[str]:
    """供 GUI 显示命令时反向解析使用。"""

    return shlex.split(text, posix=os.name != "nt")
