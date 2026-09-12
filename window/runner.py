"""Run one command at a time and stream output without calling Tk from workers."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..paths import PACKAGE_ROOT

LogCallback = Callable[[str], None]
FinishCallback = Callable[[int | None], None]


def ai_model_package_dir() -> Path:
    return PACKAGE_ROOT


def default_subprocess_cwd() -> Path:
    return ai_model_package_dir().parent


def format_command(cmd: Sequence[str]) -> str:
    """Return a copyable PowerShell command on Windows, or a POSIX command."""
    parts = [str(item) for item in cmd]
    if os.name == "nt":
        return "& " + " ".join("'" + item.replace("'", "''") + "'" for item in parts)
    return shlex.join(parts)


def _subprocess_environment(extra_env: dict[str, str] | None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    package_parent = str(ai_model_package_dir().parent)
    python_path = [item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item]
    # Always prefer this checkout over any stale path inherited after migration.
    python_path = [
        item
        for item in python_path
        if Path(item).resolve() != Path(package_parent).resolve()
    ]
    env["PYTHONPATH"] = os.pathsep.join([package_parent, *python_path])
    return env


class CommandRunner:
    """Keep a task occupied until its output is drained; capture its own process."""

    def __init__(
        self, log_callback: LogCallback | None = None, default_cwd: Path | None = None
    ) -> None:
        self._log = log_callback or (lambda line: None)
        self._default_cwd = (
            Path(default_cwd) if default_cwd else default_subprocess_cwd()
        )
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop_thread: threading.Thread | None = None
        self._stop_target: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    @property
    def default_cwd(self) -> Path:
        return self._default_cwd

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None

    def run(
        self,
        cmd: Sequence[str],
        on_finish: FinishCallback | None = None,
        cwd: Path | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> bool:
        """Start one process atomically; report OS launch errors through callbacks."""
        cwd_path = Path(cwd) if cwd else self._default_cwd
        rejected = False
        error: OSError | ValueError | None = None
        with self._lock:
            if self._proc is not None:
                rejected = True
            else:
                try:
                    if not cmd:
                        raise ValueError("命令不能为空")
                    kwargs: dict[str, Any] = dict(
                        cwd=str(cwd_path),
                        env=_subprocess_environment(extra_env),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        bufsize=1,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    if os.name == "nt":
                        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                    proc = subprocess.Popen([str(item) for item in cmd], **kwargs)
                    self._proc = proc
                except (OSError, ValueError) as exc:
                    error = exc
        if rejected:
            self._log("[runner] 已有任务正在执行，请先停止当前任务。\n")
            return False
        self._log(f"[runner] CWD = {cwd_path}\n[runner] $ {format_command(cmd)}\n")
        if error is not None:
            self._log(f"[runner] 启动失败: {error}\n")
            self._notify_finish(on_finish, -1)
            return False
        self._thread = threading.Thread(
            target=self._stream_output, args=(proc, on_finish), daemon=True
        )
        self._thread.start()
        return True

    def request_stop(self) -> None:
        """Schedule termination without blocking the GUI event loop."""
        with self._lock:
            proc = self._proc
            if proc is None:
                return
            if (
                self._stop_target is proc
                and self._stop_thread is not None
                and self._stop_thread.is_alive()
            ):
                return
            self._stop_target = proc
            self._stop_thread = threading.Thread(
                target=self._stop_process, args=(proc,), daemon=True
            )
            self._stop_thread.start()

    def stop(self) -> None:
        """Synchronous stop for non-GUI callers; the app uses request_stop()."""
        with self._lock:
            proc = self._proc
        if proc is not None:
            self._stop_process(proc)

    def _stop_process(self, proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        self._log("[runner] 正在请求终止子进程...\n")
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._log("[runner] terminate 超时，强制 kill。\n")
                proc.kill()
                proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._log(f"[runner] 终止失败: {exc}\n")

    def _stream_output(
        self, proc: subprocess.Popen[str], on_finish: FinishCallback | None
    ) -> None:
        try:
            assert proc.stdout is not None
            with proc.stdout:
                for line in proc.stdout:
                    self._log(line)
        except Exception as exc:  # pragma: no cover - broken output pipe
            self._log(f"[runner] 读取输出异常: {exc}\n")
        finally:
            exit_code = proc.wait()
            self._log(f"[runner] 进程结束，exit_code = {exit_code}\n")
            with self._lock:
                if self._proc is proc:
                    self._proc = None
            self._notify_finish(on_finish, exit_code)

    def _notify_finish(self, callback: FinishCallback | None, exit_code: int) -> None:
        if callback is not None:
            try:
                callback(exit_code)
            except Exception as exc:  # pragma: no cover - external callback
                self._log(f"[runner] on_finish 回调异常: {exc}\n")


def python_module_command(
    module: str, *args: str, executable: str | None = None
) -> list[str]:
    exe = (executable or "").strip() or sys.executable
    return [exe, "-m", module, *[str(arg) for arg in args]]


def shlex_parse(text: str) -> list[str]:
    """Legacy tokenization helper; command execution always uses argument lists."""
    return shlex.split(text, posix=os.name != "nt")
