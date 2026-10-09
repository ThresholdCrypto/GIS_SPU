# -*- coding: utf-8 -*-
"""private-join-and-compute 测试桩：不装上游二进制也能验证 PI-Sum 接入层。

与 PSI-CA 桩（``_psi_ca_stub.py``，替换的是模块导入点）不同，PI-Sum 的上游
产物是**两个可执行文件**，因此桩替换的是 runtime 里写明"唯一进程启动点"的
``spawn_pjc``，以及 capability 里做 flag 核查的 ``probe_binary_flags``。

桩的行为：把真实参数里的两份输入 CSV 读回来做**明文复算**，再按上游原文格式
打印结果行 ``Client: The intersection size is <N> and the intersection-sum is <S>``。
这样桩同时验证三件事：CSV 落盘形态、角色映射（哪一侧带关联值）、
结果行解析与折算口径。

边界说明：桩只保证"接口形态正确 + 结果按明文集合计算"，**不能替代真机验证**
（真实协议的密码学语义不在桩的职责内）。
"""

from __future__ import annotations

import os
from pathlib import Path

#: 上游 server 就绪行（``server.cc`` 原文形态）
SERVER_READY_LINE = "Server: listening on 127.0.0.1:10501\n"
#: 上游结果行模板（``client_impl.cc`` PrintOutput 原文形态）
RESULT_LINE_TEMPLATE = (
    "Client: The intersection size is {count} and the intersection-sum is {total}\n"
)


class FakePjcProcess:
    """假的子进程：stdout 是行迭代器，wait / poll / terminate 按 Popen 最小面实现。"""

    def __init__(self, lines, returncode: int = 0):
        self.stdout = iter(list(lines))
        self.returncode: int | None = None
        self._final = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self._final
        return self._final

    def terminate(self):
        if self.returncode is None:
            self.returncode = -15


def read_server_csv(path: str) -> list[int]:
    """按上游约定读 server CSV：每行恰好 1 列（无表头）。"""

    codes: list[int] = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                codes.append(int(line))
    return codes


def read_client_csv(path: str) -> dict[int, int]:
    """按上游约定读 client CSV：每行恰好 2 列（标识符, 非负整数）。"""

    pairs: dict[int, int] = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            code, _, value = line.partition(",")
            pairs[int(code)] = int(value)
    return pairs


def reference_intersection_sum(client_csv: str, server_csv: str) -> tuple[int, int]:
    """按明文复算 (交集基数, 交集内关联值之和)——用的是两份真实 CSV。"""

    client = read_client_csv(client_csv)
    server = {int(code) for code in read_server_csv(server_csv)}
    common = sorted(set(client) & server)
    return len(common), sum(client[code] for code in common)


class FakePjc:
    """假的上游构建产物：server 只报就绪，client 打印明文复算的结果行。"""

    def __init__(
        self,
        *,
        client_returncode: int = 0,
        with_result_line: bool = True,
        count_offset: int = 0,
    ):
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.server_csv: str | None = None
        self.results: list[tuple[int, int]] = []
        self.client_returncode = client_returncode
        self.with_result_line = with_result_line
        #: 故意报错的偏移（0 = 如实报）——用于验证「不一致即失败」这条纪律
        self.count_offset = count_offset

    def spawn(self, binary, args):
        name = Path(str(binary)).name
        argmap = dict(item.lstrip("-").split("=", 1) for item in args)
        self.calls.append((name, argmap))
        if name.startswith("server"):
            self.server_csv = argmap["server_data_file"]
            return FakePjcProcess(["Server: loading data... \n", SERVER_READY_LINE])
        assert self.server_csv is not None, "client 在没有 server 的情况下被启动"
        count, total = reference_intersection_sum(
            argmap["client_data_file"], self.server_csv
        )
        count += self.count_offset
        self.results.append((count, total))
        lines = ["Client: Loading data...\n", "Client: Generating keys...\n"]
        if self.with_result_line:
            lines.append(RESULT_LINE_TEMPLATE.format(count=count, total=total))
        return FakePjcProcess(lines, returncode=self.client_returncode)


def make_bin_dir(tmp_path) -> str:
    """造一个"构建产物目录"：两个占位文件即可（``--help`` 探测由桩接管）。"""

    for name in ("client", "server"):
        path = Path(tmp_path) / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass
    return str(tmp_path)


def install(monkeypatch, fake: FakePjc | None = None) -> FakePjc:
    """把桩装进 PI-Sum 后端的两个探测点，返回假产物（供断言调用序列）。"""

    from backends.psi_sum_backend import capability, runtime

    fake = fake or FakePjc()
    monkeypatch.setattr(runtime, "spawn_pjc", fake.spawn)
    monkeypatch.setattr(capability, "probe_binary_flags", lambda *a, **k: ())
    return fake
