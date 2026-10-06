# -*- coding: utf-8 -*-
"""SPU 通信量剖析：日志解析与 fd 级捕获（P2-2）。

通信量是 SPU 自己 pphlo profiling 的产物，Python 侧拿不到结构化对象——
只能读它写出的日志行。因此这一层的正确性完全落在**解析器**上：
解析器读错一位，基线就会报出一个假的通信量。测试因此分三块：

1. 解析器对**真实格式**的日志行给出预期数值（用 0.9.5 实测的原文样式）；
2. 解析器对**缺失 / 畸形**输入返回空——不猜、不补零（"空就是空"）；
3. fd 级捕获真的拿得到 C 层 spdlog 的输出（`redirect_stdout` 拿不到，
   这正是它必须存在的原因）。

真实执行路径（`capture_comm=True`）的断言在 `test_benchmark_mpc.py` 的
`TestRealMpcBenchmark` 里，那里才有 SPU。
"""

from __future__ import annotations

import os

import pytest

from backends.spu_backend.profile import (
    capture_native_logs,
    enable_native_console_log,
    parse_comm_profile,
)
from tests._helpers import has_spu

# 0.9.5 实测原文样式（见 backends/spu_backend/profile.py 模块说明）
_REAL_LINK = (
    "Link details: total send bytes 4681, recv bytes 4605, "
    "send actions 321, recv actions 316"
)
_REAL_OP = (
    "pphlo.multiply, executed 1 times, duration 0.000183387s, "
    "send bytes 2048 recv bytes 2048, send actions 1, recv actions 1"
)


class TestParseCommProfile:
    def test_link_line_gives_the_totals(self):
        parsed = parse_comm_profile(_REAL_LINK)
        assert parsed["comm_send_bytes"] == 4681
        assert parsed["comm_recv_bytes"] == 4605
        assert parsed["comm_total_bytes"] == 9286  # send + recv
        assert parsed["comm_send_actions"] == 321
        assert parsed["comm_recv_actions"] == 316

    def test_primitive_line_is_summed_by_name(self):
        text = "\n".join([_REAL_OP, _REAL_OP.replace("2048", "16")])
        parsed = parse_comm_profile(text)["comm_by_primitive"]
        assert set(parsed) == {"multiply"}
        assert parsed["multiply"]["executions"] == 2
        assert parsed["multiply"]["send_bytes"] == 2048 + 16
        assert parsed["multiply"]["recv_bytes"] == 2048 + 16

    def test_last_link_line_wins(self):
        """一段日志可能有多次运行的分段汇总；取最后一次才算"这次跑的"。"""

        text = "Link details: total send bytes 1, recv bytes 1, send actions 1, recv actions 1\n"
        text += _REAL_LINK
        parsed = parse_comm_profile(text)
        assert parsed["comm_send_bytes"] == 4681

    def test_missing_lines_return_nothing_not_zero(self):
        """解析不到就什么都不给——补 0 会变成"零通信量"这种假结论。"""

        assert parse_comm_profile("") == {}
        assert parse_comm_profile("libspu: starting simulator\n") == {}

    def test_malformed_link_line_is_ignored(self):
        text = "Link details: total send bytes abc, recv bytes 1, send actions 1, recv actions 1"
        assert parse_comm_profile(text) == {}

    def test_primitive_without_link_line_still_reports_ops(self):
        parsed = parse_comm_profile(_REAL_OP)
        # 没有链路汇总行 → 不给总量（不许用原语之和冒充）
        assert "comm_total_bytes" not in parsed
        assert parsed["comm_by_primitive"]["multiply"]["send_bytes"] == 2048

    def test_realistic_log_shape(self):
        """贴近真实输出：混着框架日志、多原语、一次汇总。"""

        text = "\n".join(
            [
                "[2026-10-06 12:00:00.000] [info] [simulator] world size = 3",
                _REAL_OP,
                _REAL_OP.replace("pphlo.multiply", "pphlo.convert").replace("2048", "8"),
                "[2026-10-06 12:00:00.100] [info] [profiler] profiling report:",
                _REAL_LINK,
            ]
        )
        parsed = parse_comm_profile(text)
        assert parsed["comm_total_bytes"] == 9286
        assert sorted(parsed["comm_by_primitive"]) == ["convert", "multiply"]


class TestCaptureNativeLogs:
    def test_captures_c_layer_style_output(self):
        """fd 级重定向抓得到直接写 fd 1 的内容（`print` 抓得到不代表够用）。"""

        with capture_native_logs() as logs:
            os.write(1, b"pphlo.multiply, executed 1 times\n")
            os.write(2, b"Link details: total send bytes 1, recv bytes 2, send actions 0, recv actions 0\n")
            text = logs.read()
        assert "pphlo.multiply" in text
        assert "Link details" in text

    def test_stdout_is_restored_after_the_block(self):
        """退出后 fd 1 仍然可用：再做一次捕获还能正常读到内容。"""

        with capture_native_logs():
            pass
        with capture_native_logs() as logs:
            os.write(1, b"after-restore\n")
            text = logs.read()
        assert "after-restore" in text

    def test_stdout_is_restored_even_when_the_body_raises(self):
        """异常路径也必须恢复 fd——否则整个进程的 stdout 会静默指向一个已关闭的临时文件。"""

        try:
            with capture_native_logs():
                raise RuntimeError("boom")
        except RuntimeError as exc:
            assert str(exc) == "boom"

        with capture_native_logs() as logs:
            os.write(1, b"survived\n")
            text = logs.read()
        assert "survived" in text


class TestEnableNativeConsoleLog:
    """进程级开关的打开动作：**拿不到就说拿不到**，不许假装成功。"""

    def test_never_raises_and_reports_honestly(self):
        result = enable_native_console_log()
        assert result is (True if has_spu() else False)

    def test_unknown_level_name_does_not_crash(self):
        """级别名写错时退化成"只开 console、不改级别"，而不是抛异常。"""

        assert enable_native_console_log(level="NO_SUCH_LEVEL") is has_spu()

    def test_does_not_drop_a_log_file_into_the_cwd(self, tmp_path, monkeypatch):
        """`system_log_path` 默认是相对路径 `'spu.log'`——绝不能在 CWD 落盘。"""

        if not has_spu():
            pytest.skip("需要真实 SPU（该路径只在设置成功时才可能落盘）")

        monkeypatch.chdir(tmp_path)
        assert enable_native_console_log() is True
        assert not (tmp_path / "spu.log").exists()
