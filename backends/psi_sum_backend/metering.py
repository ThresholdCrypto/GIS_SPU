# -*- coding: utf-8 -*-
"""PI-Sum 资源计量件（Phase 10）：通信量（回环中继）与峰值内存（procfs 探针）。

两条计量都以**如实**为第一纪律：

- ``CountingRelay``：在 client 与 server 之间插入一个回环 TCP 中继，逐字节
  统计两个方向的**应用层字节**（含 gRPC / HTTP2 封装；不含 TCP/IP 头，
  也不推测协议开销）。它只数它转发的字节，不做任何换算或估计。
- ``PeakRssProbe``：在子进程存活期间采样 ``/proc/<pid>/status`` 的 ``VmHWM``
  （内核维护的峰值水位——读到的数字是**峰值**，采样间隔只决定"最后一次
  观测发生在何时"）。进程退出后 `/proc` 项不再提供 VmHWM（僵尸态只剩
  State 行，已实测），所以退出前最后窗口 ≤ 采样间隔的增长无法观测到；
  该口径在结果里按实际采样间隔标注，不假装是精确退出值。
  探针需要真实 pid：假进程（测试桩）没有 pid —— 探针如实报"未取到读数"。

两个中继方向的口径（写进记录时就按这个口径命名）：
``client_to_server_bytes`` = client 发出、server 收到的字节；
``server_to_client_bytes`` = server 发出、client 收到的字节。
"""

from __future__ import annotations

import os
import re
import socket
import threading
import time
from typing import Any

#: VmHWM 采样间隔（秒）。20 ms：协议执行以秒计，窗口误差可忽略；
#: 每次采样只读一个 1–2 KiB 的 procfs 文件，对被测进程的性能扰动可忽略。
VMHWM_SAMPLE_INTERVAL_S = 0.02

#: /proc/<pid>/status 里 VmHWM 行（单位 kB）
_VMHWM_PATTERN = re.compile(r"^VmHWM:\s+(\d+)\s+kB$", re.MULTILINE)

#: 中继单次搬运的块大小（64 KiB：回环上单次 recv 的常见上限附近）
_PUMP_CHUNK = 65536


def vmhwm_mb_from_status(text: str) -> float | None:
    """从 ``/proc/<pid>/status`` 文本里取 VmHWM（MiB）；解析不到返回 None。

    单位换算与 ``backends.psi_backend.benchmark`` 的 ru_maxrss 口径一致
    （kB / 1024 = MiB）。
    """

    match = _VMHWM_PATTERN.search(text or "")
    if match is None:
        return None
    return round(int(match.group(1)) / 1024.0, 3)


class PeakRssProbe:
    """采样某个 pid 的 VmHWM 峰值水位（线程内定期读，stop() 后取最大值）。

    只接受**真实 pid**；pid 为 None（测试桩进程 / 非本地进程）时
    ``available`` 为 False，start/stop 都是无副作用的空操作，读数记 None。
    """

    def __init__(
        self, pid: int | None, interval: float = VMHWM_SAMPLE_INTERVAL_S
    ) -> None:
        self._pid = pid if isinstance(pid, int) and pid > 0 else None
        self._interval = float(interval)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._peak_mb: float | None = None

    @property
    def available(self) -> bool:
        """探针是否能真正读到读数（需要真实 pid）。"""

        return self._pid is not None

    @property
    def interval_ms(self) -> float:
        return round(self._interval * 1000.0, 3)

    def start(self) -> None:
        if not self.available:
            return
        thread = threading.Thread(target=self._run, name="psi-sum-rss", daemon=True)
        thread.start()
        self._thread = thread

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self._interval)
        self._sample_once()

    def _sample_once(self) -> None:
        assert self._pid is not None
        try:
            with open(f"/proc/{self._pid}/status", encoding="utf-8") as handle:
                value = vmhwm_mb_from_status(handle.read())
        except OSError:
            return
        if value is None:
            return
        if self._peak_mb is None or value > self._peak_mb:
            self._peak_mb = value

    def stop(self) -> float | None:
        """停止采样并返回观测到的峰值（MiB）；从未读到过则 None。"""

        if not self.available:
            return None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 5 + 0.5)
        self._sample_once()
        return self._peak_mb


class CountingRelay:
    """回环 TCP 中继：转发连接并双向逐字节计数（只为计量，不做任何改写）。

    用法::

        relay = CountingRelay("127.0.0.1", 10501)
        relay.start()
        # client 的 --port 指向 relay.port；server 仍监听 10501
        ...
        relay.stop()
        totals = relay.totals()   # client_to_server / server_to_client / total

    说明：半关闭按 HTTP/2 的实际用法处理（一端 EOF → 对端 shutdown(WR)），
    连接断开即收尾；计数在线程锁内累加，无丢字节路径。
    """

    def __init__(self, upstream_host: str, upstream_port: int) -> None:
        self._upstream = (str(upstream_host), int(upstream_port))
        self._listener: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self._live: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._counts = {"client_to_server_bytes": 0, "server_to_client_bytes": 0}
        self._stopping = threading.Event()
        self._port: int | None = None

    @property
    def port(self) -> int:
        """中继监听端口（start() 之后才有值；由内核分配，避免端口撞车）。"""

        if self._port is None:
            raise RuntimeError("中继尚未启动：先调用 start()")
        return self._port

    def start(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(16)
        listener.settimeout(0.2)
        self._listener = listener
        self._port = int(listener.getsockname()[1])
        thread = threading.Thread(
            target=self._accept_loop, name="psi-sum-relay", daemon=True
        )
        thread.start()
        self._threads.append(thread)

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stopping.is_set():
            try:
                downstream, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                upstream = socket.create_connection(self._upstream, timeout=5.0)
            except OSError:
                # 上游不可达：立刻断开下游，让调用方按连接失败暴露问题
                downstream.close()
                continue
            for sock in (downstream, upstream):
                sock.settimeout(None)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                self._live.update((downstream, upstream))
            handler = threading.Thread(
                target=self._handle_connection,
                args=(downstream, upstream),
                name="psi-sum-relay-conn",
                daemon=True,
            )
            handler.start()
            self._threads.append(handler)

    def _handle_connection(
        self, downstream: socket.socket, upstream: socket.socket
    ) -> None:
        pumps = [
            threading.Thread(
                target=self._pump,
                args=(downstream, upstream, "client_to_server_bytes"),
                daemon=True,
            ),
            threading.Thread(
                target=self._pump,
                args=(upstream, downstream, "server_to_client_bytes"),
                daemon=True,
            ),
        ]
        for pump in pumps:
            pump.start()
        for pump in pumps:
            pump.join()
        self._close_pair(downstream, upstream)

    def _pump(self, source: socket.socket, sink: socket.socket, counter: str) -> None:
        try:
            while True:
                chunk = source.recv(_PUMP_CHUNK)
                if not chunk:
                    # 半关闭：告诉对端这一方向已经结束（HTTP/2 连接的正常收尾）
                    try:
                        sink.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
                    return
                with self._lock:
                    self._counts[counter] += len(chunk)
                sink.sendall(chunk)
        except OSError:
            return

    def _close_pair(self, *sockets: socket.socket) -> None:
        with self._lock:
            for sock in sockets:
                self._live.discard(sock)
        for sock in sockets:
            try:
                sock.close()
            except OSError:
                pass

    def stop(self) -> None:
        """停止监听并收掉所有连接；幂等（重复调用无副作用）。"""

        self._stopping.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        with self._lock:
            live = list(self._live)
        for sock in live:
            try:
                sock.close()
            except OSError:
                pass
        for thread in self._threads:
            thread.join(timeout=5.0)

    def totals(self) -> dict[str, Any]:
        """两个方向的字节数 + 合计（不含 TCP/IP 头；中继不推测协议开销）。"""

        with self._lock:
            counts = dict(self._counts)
        counts["total_bytes"] = (
            counts["client_to_server_bytes"] + counts["server_to_client_bytes"]
        )
        return counts
