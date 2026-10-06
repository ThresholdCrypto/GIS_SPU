# -*- coding: utf-8 -*-
"""SPU 通信量剖析：从 SPU 自己的 profile 日志里取出 `send/recv bytes`。

依据（0.9.5 实测，不是推断）
--------------------------
`libspu.RuntimeConfig.enable_pphlo_profile = True` 后，执行期会输出两类行：

    pphlo.multiply, executed 1 times, duration 0.000183387s, \
        send bytes 2048 recv bytes 2048, send actions 1, recv actions 1
    Link details: total send bytes 4681, recv bytes 4605, send actions 321, \
        recv actions 316

这是当前唯一的通信量来源：`spu.libspu.pyi` 里 `Context` **没有**任何字节计数
接口，Python 侧也拿不到执行统计对象。所以只能读日志。

两个必须知道的坑
----------------
1. 这些行由 **C 层 spdlog** 写出，`contextlib.redirect_stdout` 抓不到——
   必须做 **fd 级**重定向（`os.dup2`），见 `capture_native_logs`；
2. 通信量**不是每一条都解析确定**。按 `--repeat 5` 实测（`docs/mpc_comm_baseline.json`）：
   `DistanceLE` / `WeightedSum` 上 ABY3 / SEMI2K / SECURENN **五次完全一致**
   （0.000% 抖动），REF2K 恒为 0 B；但
   `TemporalOverlap` 的 ABY3 与 CHEETAH 的 `DistanceLE` / `WeightedSum` 会抖，
   幅度实测 0.03%–10.1%（ABY3 × `TemporalOverlap` 随环宽变宽而变大：FM32 3.9%、
   FM64 5.0%、FM128 10.1%）。所以报数一律给**中位 + 区间**，
   不要拿单次值当结论——`summarize_repeats` 已经这么做。

第三个坑：原生日志是**进程级全局开关**
------------------------------------
`libspu.logging.setup_logging` 是 C++ 侧的进程级单例配置（不是 per-simulator）。
本仓库的 PSI 路径在每次求交前会把它**关掉**
（`backends/psi_backend/runtime.py` 的 `_set_native_log(quiet=True)`：
`enable_console_logger=False` + `log_level=ERROR`），好让编译器输出干净。
一旦它被关过，**同一进程里后续所有 SPU 运行的 pphlo profile 行都不会再出现**——
实测：先跑一次 PSI，再开 profiling 取通信量，fd 重定向拿到的文本长度是 0。

所以 `capture_comm=True` 必须先把 console logger 重新打开，这就是
`enable_native_console_log()` 存在的唯一理由。两个附带后果要记住：

* 打开后原生日志会**一直响**到有人再关它（PSI 每次自己会关），
  所以带 profiling 的运行会刷屏——benchmark 运行器本就建议重定向日志；
* `system_log_path` 一律改指 `/dev/null`。它的默认值是相对路径 `'spu.log'`，
  不改就会在**当前工作目录**落一个日志文件（编译器不该往用户项目里丢东西）。
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import tempfile
from typing import Any, Iterator

#: 链路级汇总行
_LINK_RE = re.compile(
    r"Link details: total send bytes (?P<send>\d+),\s*"
    r"recv bytes (?P<recv>\d+),\s*"
    r"send actions (?P<send_actions>\d+),\s*"
    r"recv actions (?P<recv_actions>\d+)"
)

#: 逐原语行：`pphlo.<op>, executed N times, duration Xs, send bytes A recv bytes B, ...`
_OP_RE = re.compile(
    r"pphlo\.(?P<name>[\w\.]+),\s*"
    r"executed (?P<executions>\d+) times,\s*"
    r"duration (?P<duration>[0-9.eE+-]+)s,\s*"
    r"send bytes (?P<send>\d+)\s*recv bytes (?P<recv>\d+),\s*"
    r"send actions (?P<send_actions>\d+),\s*"
    r"recv actions (?P<recv_actions>\d+)"
)


def parse_comm_profile(text: str) -> dict[str, Any]:
    """从 profile 日志文本里取出通信量。

    取**最后一条** `Link details`（一次运行可能有多段），逐原语表同理按名字合并。
    解析不到就返回空字典的对应项——**不填数字**。
    """

    total: dict[str, int] = {}
    for match in _LINK_RE.finditer(text or ""):
        total = {
            "comm_send_bytes": int(match.group("send")),
            "comm_recv_bytes": int(match.group("recv")),
            "comm_send_actions": int(match.group("send_actions")),
            "comm_recv_actions": int(match.group("recv_actions")),
        }

    per_op: dict[str, dict[str, int | float]] = {}
    for match in _OP_RE.finditer(text or ""):
        name = match.group("name")
        entry = per_op.setdefault(
            name,
            {"send_bytes": 0, "recv_bytes": 0, "executions": 0, "duration_s": 0.0},
        )
        entry["send_bytes"] = int(entry["send_bytes"]) + int(match.group("send"))
        entry["recv_bytes"] = int(entry["recv_bytes"]) + int(match.group("recv"))
        entry["executions"] = int(entry["executions"]) + int(match.group("executions"))
        entry["duration_s"] = float(entry["duration_s"]) + float(match.group("duration"))

    if total:
        total["comm_total_bytes"] = total["comm_send_bytes"] + total["comm_recv_bytes"]
    if per_op:
        total["comm_by_primitive"] = per_op
    return total


class CapturedLogs:
    """fd 级捕获到的文本（读一次即可；上下文退出后失效）。"""

    def __init__(self, files: tuple[Any, ...]) -> None:
        self._files = files
        self._text: str | None = None

    def read(self) -> str:
        if self._text is None:
            chunks: list[str] = []
            for handle in self._files:
                handle.flush()
                handle.seek(0)
                chunks.append(handle.read().decode("utf-8", errors="replace"))
            self._text = "".join(chunks)
        return self._text


def enable_native_console_log(level: str = "INFO") -> bool:
    """把 SPU 原生 console logger 打开（通信量只能从它拿到）。

    为什么需要它：原生日志开关是**进程级**的，而本仓库的 PSI 路径会把它关掉
    （见模块 docstring "第三个坑"）。关过之后，同一进程里 pphlo profile 行
    就再也不出现了——通信量会静默变成"取不到"。

    与 `backends/psi_backend/runtime.py::_set_native_log` 同构：C++ 侧接口不可
    内省，所以只按实测可用的字段设置，**任何失败都返回 False 而不是抛**。
    调用方据此决定"这栏留空"，而不是猜一个数填进去。

    Args:
        level: 日志级别名（默认 `INFO`；profile 行就是这个级别）。

    Returns:
        是否成功设置。`False` 表示无法保证 console logger 打开。
    """

    try:
        import spu.libspu as libspu

        logging = getattr(libspu, "logging", None)
        if logging is None or not hasattr(logging, "setup_logging"):
            return False
        options = logging.LogOptions()
        options.enable_console_logger = True
        log_level = getattr(logging, "LogLevel", None)
        if log_level is not None and hasattr(log_level, level):
            options.log_level = getattr(log_level, level)
        # 默认值是相对路径 'spu.log'，会在 CWD 落盘；一律改指 /dev/null。
        options.system_log_path = os.devnull
        logging.setup_logging(options)
    except Exception:
        return False
    return True


@contextlib.contextmanager
def capture_native_logs() -> Iterator[CapturedLogs]:
    """把 fd 1 / 2 重定向到临时文件，yield 一个可读取捕获文本的对象。

    SPU 的 profile 行走 C 层 spdlog，只有 fd 级重定向能拿到。
    退出时无条件恢复原 fd（异常路径也要恢复，否则会把进程的 stdout 弄坏）。
    """

    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    files = (tempfile.TemporaryFile(), tempfile.TemporaryFile())
    captured = CapturedLogs(files)
    try:
        os.dup2(files[0].fileno(), 1)
        os.dup2(files[1].fileno(), 2)
        yield captured
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        for handle in files:
            handle.close()
