# -*- coding: utf-8 -*-
"""Private Intersection-Sum 执行路径（Google private-join-and-compute）。

与兄弟 PSI 路径的关系
---------------------
三条**独立路径、三种泄漏承诺**，刻意不合并：

- ``backends/psi_backend``（libpsi 求交）：接收方在协议内部获得**交集本体**；
- ``backends/psi_ca_backend``（OpenMined PSI-CA）：只交给 client 一个**基数**；
- 本模块（PI-Sum）：交给 client **基数 + 交集内关联值之和**两个数，
  交集本体同样不交给任一方——多出来的那个"和"是本档的用途，也是本档的
  新风险面（上游 README 自陈：唯一值或过小交集可由和反推成员）。

因此本模块只承接 ``CellSetIntersect`` 的 ``REVEAL_INTERSECTION_SUM`` 档；
其余算子 / 策略**显式拒绝**（不做布尔判定、不产出交集本体）。

角色映射（必须知情）
--------------------
上游协议是**不对称**的：server 持标识符，client 持标识符 + 关联值，
**两者都只由 client 得知**。本后端固定：

    client = 左侧输入（带关联值，对应 libpsi 的 receiver_rank=0 口径）
    server = 右侧输入（只持标识符）

输入与输出形态（均对上游源码逐字核对，不是推测）
------------------------------------------------
- server CSV：每行**恰好 1 列**（标识符）；client CSV：每行**恰好 2 列**
  （标识符, 非负 int64 关联值）；**无表头**（上游按列数校验，表头即报错）；
- 结果行（``client_impl.cc`` PrintOutput 原文）：
  ``Client: The intersection size is <N> and the intersection-sum is <S>``；
- 就差机器：两侧 gRPC 都是 ``LocalCredentials(LOCAL_TCP)``，
  所以 server/client 必须在**同一台机**上跑，端口由本模块统一分配。

实现细节
--------
- 关联值由调用方显式给出（``left_values``：码 → 值）；**缺值直接报错**，
  不静默按 0 补齐——0 补齐会把"数据没接上"伪装成"和为 0"；
- 和值须落在 int64 内（上游 ``ToIntValue``），超界由上游报错，本模块原样转述；
- 临时工作目录自建（0700）+ 进程退出兜底清理，与 libpsi 路径同一纪律。
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence

from ..psi_backend.geosot_optimizer import GEOSOT_OPTIMIZER_VERSION, prepare_grid_codes
from ..psi_backend.input_adapter import check_layout_agreement
from ..psi_backend.protocol_registry import RESULT_SEMANTICS_EXACT
from ..psi_backend.runtime import PSI_RUNTIME_WORLD_SIZE, PsiRunResult
from . import capability
from .capability import (
    PSI_SUM_CLIENT_BINARY,
    PSI_SUM_SERVER_BINARY,
    PSI_SUM_SUPPORTED_OPS,
    PSI_SUM_SUPPORTED_POLICIES,
    PSI_SUM_UPSTREAM,
    PSI_SUM_UPSTREAM_COMMIT,
    PsiSumCapabilityReport,
    binary_path,
    check_psi_sum_capabilities,
    resolve_bin_dir,
)
from .metering import CountingRelay, PeakRssProbe, VMHWM_SAMPLE_INTERVAL_S

#: 结果里的协议名（展示用；不是 SPU/libpsi 枚举成员，不放进协议注册表）
PSI_SUM_PROTOCOL = "PJC-PI-SUM"

#: 协议内部泄漏面登记码（与 libpsi 的 intersection-body、PSI-CA 的 count-only 并列）
PSI_SUM_PROTOCOL_LEAK = "count+sum"

#: 由 client（=左侧输入）得知结果（对应 libpsi 的 receiver_rank=0 口径）
PSI_SUM_LEARNING_RANK = 0

#: 本后端唯一承接的结果策略（业务层同时获得基数与和）
PSI_SUM_RESULT_POLICY = "REVEAL_INTERSECTION_SUM"

#: 逐算子泄漏面登记（本后端只有这一档）
PSI_SUM_OP_LEAKS: Mapping[str, str] = {
    "CellSetIntersect": (
        "只出「基数 + 交集内和」：协议只计算这两个数（client=左侧），"
        "交集本体不交给任一方；两个数本身对 client 可见"
    ),
}

#: server 就绪行的匹配串（``server.cc`` 原文：``Server: listening on <port>``）
PSI_SUM_READY_MARKER = "Server: listening on"

#: 结果行的唯一解析锚点（``client_impl.cc`` PrintOutput 原文）
PSI_SUM_RESULT_PATTERN = re.compile(
    r"The intersection size is (\d+) and the intersection-sum is (-?\d+)"
)

#: 上游 ``client.cc`` 的 paillier_modulus_size 默认值（本项目不新增旋钮）
PSI_SUM_PAILLIER_MODULUS_SIZE = 1536

#: 默认监听地址：上游默认是 ``0.0.0.0:10501``（监听全网卡）；本项目收紧到回环
PSI_SUM_DEFAULT_PORT = "127.0.0.1:10501"
#: server 启动到报告就绪的等待上限（秒）
PSI_SUM_STARTUP_TIMEOUT_S = 60.0
#: client 走完协议的等待上限（秒）
PSI_SUM_PROTOCOL_TIMEOUT_S = 600.0
#: 上游 int64 上界（关联值与和值都必须落在 [0, 2**63-1]）
PSI_SUM_INT64_MAX = 2**63 - 1

#: 进程存活期间自建的工作目录（atexit 兜底清理；正常路径走 finally）
_ACTIVE_WORKDIRS: set[str] = set()


def _chmod(path: str, mode: int) -> None:
    """尽力收紧权限；非 POSIX / 权限不足时静默跳过（不影响正确性）。"""

    try:
        os.chmod(path, mode)
    except OSError:
        return


def _make_workdir() -> str:
    """自建临时工作目录：随机名 + 0700 + 进程退出兜底清理。"""

    path = tempfile.mkdtemp(prefix="geo_psi_sum_")
    _chmod(path, 0o700)
    _ACTIVE_WORKDIRS.add(path)
    return path


def _release_workdir(path: str) -> None:
    _ACTIVE_WORKDIRS.discard(path)


def _cleanup_active_workdirs() -> None:
    """进程退出兜底：清掉 finally 未及清理的工作目录（幂等）。"""

    for path in list(_ACTIVE_WORKDIRS):
        shutil.rmtree(path, ignore_errors=True)
        _ACTIVE_WORKDIRS.discard(path)


atexit.register(_cleanup_active_workdirs)


def _jsonable(value: Any) -> Any:
    """把结果值整理成可 JSON 化的形态（tuple → list，保序）。"""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


# --------------------------------------------------------------------------
# 子进程：唯一的启动点（测试可替换）
# --------------------------------------------------------------------------


def spawn_pjc(binary: str, args: Sequence[str]) -> Any:
    """启动一个上游可执行文件（**唯一进程启动点**，测试可替换）。

    用文本模式把 stderr 并入 stdout：上游把进度与结果都打在 stdout 上，
    结果行是唯一解析锚点；错误信息在 stderr，合流后仍能按行读到。
    """

    return subprocess.Popen(
        [binary, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _terminate(proc: Any) -> None:
    """尽力收掉子进程；进程已退出 / 假进程不支持时静默跳过。"""

    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
    except Exception:
        return


def _read_until(proc: Any, marker: str, timeout: float) -> str | None:
    """逐行读到含 ``marker`` 的行；超时/EOF 返回 None（并收掉进程）。

    超时用 timer 兜底：读 ``proc.stdout`` 是阻塞的，只靠 deadline 判断
    会一直卡在 readline 上；timer 到点终止进程让读立即返回 EOF。
    """

    deadline_timer = threading.Timer(timeout, _terminate, args=(proc,))
    deadline_timer.daemon = True
    deadline_timer.start()
    try:
        for line in proc.stdout:
            if marker in line:
                return line
        return None
    except Exception:
        return None
    finally:
        deadline_timer.cancel()


def _read_to_end(proc: Any, timeout: float) -> tuple[str, int | None]:
    """把进程输出读干净；返回 (全部文本, 退出码)。超时收掉进程。"""

    deadline_timer = threading.Timer(timeout, _terminate, args=(proc,))
    deadline_timer.daemon = True
    deadline_timer.start()
    chunks: list[str] = []
    try:
        for line in proc.stdout:
            chunks.append(line)
        try:
            code = proc.wait(timeout=timeout)
        except Exception:
            code = None
    except Exception:
        code = None
    finally:
        deadline_timer.cancel()
    return "".join(chunks), code


# --------------------------------------------------------------------------
# 输入落盘：上游只吃 CSV
# --------------------------------------------------------------------------


def _write_server_csv(path: str, codes: Sequence[int]) -> None:
    """server CSV：每行恰好 1 列（上游按列数校验，多一列即报错）。"""

    text = "".join(f"{int(code)}\n" for code in codes)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    _chmod(path, 0o600)


def _write_client_csv(path: str, codes: Sequence[int], values: Mapping[int, int]) -> None:
    """client CSV：每行恰好 2 列（标识符, 非负 int64 关联值）。

    只写裸字段、不加引号：格网码与关联值都是纯数字，不含分隔符或引号，
    不需要转义（上游 ``SplitCsvLine`` 对裸字段与带引号字段都接受）。
    """

    lines = [f"{int(code)},{int(values[int(code)])}\n" for code in codes]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(lines))
    _chmod(path, 0o600)


# --------------------------------------------------------------------------
# 明文对拍与结果折叠
# --------------------------------------------------------------------------


def _policy_dict() -> dict[str, Any]:
    """PI-Sum 档的结果策略登记（业务暴露 + 协议泄漏，两栏并排）。"""

    return {
        "policy": PSI_SUM_RESULT_POLICY,
        "business_value": "count+sum",
        "protocol_leak": PSI_SUM_PROTOCOL_LEAK,
        "executable": True,
        "leak_disclosure": (
            "协议内部泄漏面：交集基数 + 交集内关联值之和——PI-Sum 只计算这两个数，"
            "交集本体不离开任一方；但和值本身可反推成员（唯一值 / 过小交集），"
            "上游 README 明言其缓解措施未实现，见 docs/PSI_SUM_CAPABILITY.md"
        ),
    }


def _fold_reference(
    reference: Any, count: int, total: int, values: Mapping[int, int]
) -> bool | None:
    """把明文参考折算到「基数 + 和」口径再比对；折算不出返回 None（不硬比）。"""

    if reference is None or isinstance(reference, bool):
        return None
    try:
        codes = sorted({int(item) for item in reference})
        expected_total = sum(int(values[code]) for code in codes)
    except (TypeError, ValueError, KeyError):
        return None
    return len(codes) == count and expected_total == total


def _validate_values(
    codes: Sequence[int], values: Mapping[int, int] | None
) -> str | None:
    """校验关联值覆盖面与取值范围；返回错误说明（None = 通过）。

    覆盖面必须完整：缺一个码就不跑协议——按 0 补齐会把"数据没接上"
    伪装成"和为 0"，那是假结果，不是容差。
    """

    if values is None:
        return (
            "PI-Sum 必须显式提供关联值（left_values：码 → 非负整数）："
            "上游 client 侧 CSV 每行必须是「标识符, 非负 int64 值」两列，"
            "没有值的集合请改用 --psi-count psi-ca（只出基数）"
        )
    missing = [code for code in codes if int(code) not in values]
    if missing:
        return (
            f"关联值覆盖不全：{len(missing)} 个码没有对应值（前几个：{missing[:5]}）；"
            "本后端不按 0 补齐——请补齐 left_values 或改走只出计数的 PSI-CA 档"
        )
    bad = [
        code
        for code in codes
        if not isinstance(values[int(code)], int)
        or isinstance(values[int(code)], bool)
        or values[int(code)] < 0
        or values[int(code)] > PSI_SUM_INT64_MAX
    ]
    if bad:
        return (
            f"关联值越界：{len(bad)} 个码的值不是 [0, 2**63-1] 内的整数"
            f"（前几个：{bad[:5]}）；上游按非负 int64 解析，负数/超界直接报错"
        )
    return None


def run_psi_intersection_sum(
    left: Sequence[int],
    right: Sequence[int],
    *,
    left_values: Mapping[int, int] | None = None,
    op: str = "CellSetIntersect",
    result_policy: str | None = None,
    left_layout: Mapping[str, Any] | None = None,
    right_layout: Mapping[str, Any] | None = None,
    party_binding: Sequence[Mapping[str, Any]] | None = None,
    reference_fn: Any = None,
    report: PsiSumCapabilityReport | None = None,
    bin_dir: str | os.PathLike[str] | None = None,
    port: str | None = None,
    measure_communication: bool = False,
    measure_memory: bool = False,
    quiet: bool = True,
) -> PsiRunResult:
    """两方格网集合的**交集内求和**计算（PI-Sum，private-join-and-compute）。

    Args:
        left / right: 两方 64 位格网码集合。角色：client=left（**带关联值**，
            并得到结果），server=right（只持标识符）。
        left_values: 码 → 非负整数关联值；必须覆盖去重后的全部左侧码。
        op: 只接受 ``CellSetIntersect``（见 ``PSI_SUM_SUPPORTED_OPS``）。
        result_policy: 只接受 ``REVEAL_INTERSECTION_SUM``（None 等同）。
        left_layout / right_layout: 两方格网布局清单；双方都给出且不一致时
            在进入协议前以 LAYOUT_MISMATCH 拒绝（与 libpsi / PSI-CA 同源）。
        party_binding: 逐输入的参与方绑定披露，由编译期装配；原样登记。
        reference_fn: 明文参考实现（返回交集码序列）；给出后折算到
            「基数 + 和」口径做一致性比对。
        report: 能力核查结论（测试与编译期可注入；缺省现场探测）。
        bin_dir: 上游构建产物目录（缺省读环境变量 ``GIS_SPU_PJC_BIN_DIR``）。
        port: gRPC 监听地址（缺省 ``127.0.0.1:10501``，故意不用上游默认的
            ``0.0.0.0``——本路径只在本机回环内跑）。
        measure_communication: 在 client 与 server 之间插入回环 TCP 中继，
            逐字节采集两个方向的通信量（``result.communication``）。
            中继多一跳，本次 ``timings_ms`` 包含该开销；不测时保持原路径。
        measure_memory: 对两个子进程挂 procfs VmHWM 采样探针，采集峰值
            RSS（``result.memory``；探针拿不到真实 pid 时如实留空）。
        quiet: 预留与兄弟后端一致的签名；上游子进程的 stdout 只进 notes。
    """

    result = PsiRunResult(
        status="unavailable",
        op=op,
        protocol=PSI_SUM_PROTOCOL,
        curve=None,
        protocol_params={
            "impl": PSI_SUM_UPSTREAM,
            "commit": PSI_SUM_UPSTREAM_COMMIT[:7],
            "paillier_modulus_size": PSI_SUM_PAILLIER_MODULUS_SIZE,
        },
        broadcast_result=False,
        # 本路径没有 libpsi 配置对象（配置闭环只对 libpsi 路径成立），
        # 不伪造 runtime_config；实际参数见 protocol_params 与 notes。
        runtime_config={},
        world_size=PSI_RUNTIME_WORLD_SIZE,
        receiver_rank=PSI_SUM_LEARNING_RANK,
        reveals=PSI_SUM_OP_LEAKS.get(op, "（未登记泄漏面，请补充）"),
        semantics="交集内求和：(|A∩B|, Σ 关联值)——两个数都由 client 得知",
        result_semantics=RESULT_SEMANTICS_EXACT,
        result_policy=_policy_dict(),
    )
    result.party_binding = tuple(dict(item) for item in (party_binding or ()))
    result.notes = result.notes + (
        "PI-Sum 角色映射：client=左侧输入（带关联值、获得结果，对应 "
        "receiver_rank=0），server=右侧输入（只持标识符）",
        "本路径落盘：上游只吃 CSV（server 1 列/行、client 2 列/行，无表头），"
        "在自建 0700 临时目录内生成，退出即清",
        "传输面：server/client 都走 gRPC LocalCredentials(LOCAL_TCP)，"
        "只能同机、无 TLS、无身份认证",
    )
    if measure_communication:
        result.notes = result.notes + (
            "通信量计量：client 接到回环 TCP 中继（内核分配端口）再转发给 "
            "server；中继逐字节统计两个方向（应用层字节，含 gRPC/HTTP2 封装，"
            "不含 TCP/IP 头）",
        )
    if measure_memory:
        result.notes = result.notes + (
            "内存计量：procfs VmHWM 采样探针（间隔 "
            f"{VMHWM_SAMPLE_INTERVAL_S * 1000:.0f} ms）挂在两个子进程上——"
            "数值是内核峰值水位；最后一次采样后到退出的窗口 ≤ 采样间隔，"
            "口径随结果登记，不假装是精确退出值",
        )

    if op not in PSI_SUM_SUPPORTED_OPS:
        result.status = "error"
        result.error = (
            f"pjc 后端只承接 {tuple(PSI_SUM_SUPPORTED_OPS)} 的交集内求和档"
            f"（{PSI_SUM_RESULT_POLICY}）；收到算子 {op!r}。其余算子请走 libpsi "
            "求交路径（backends.psi_backend）或 PSI-CA 计数档"
            "（backends.psi_ca_backend）"
        )
        return result

    requested = (
        PSI_SUM_RESULT_POLICY
        if result_policy is None
        else str(result_policy).upper().strip()
    )
    if requested != PSI_SUM_RESULT_POLICY:
        result.status = "error"
        result.error = (
            f"pjc 后端只能执行 {PSI_SUM_RESULT_POLICY}（同时出基数与和）；"
            f"收到策略 {requested!r}。PI-Sum 不产出交集本体、也不做布尔判定——"
            "需要交集本体请改用 libpsi 求交路径"
        )
        return result

    # 布局握手前置检查：与 libpsi 路径同源、同文案口径（不依赖环境，离线也拦得住）
    if left_layout is not None or right_layout is not None:
        agreement = check_layout_agreement(left_layout, right_layout)
        result.layout_agreement = agreement.to_dict()
        if agreement.agreement is False:
            result.status = "error"
            result.error = (
                "LAYOUT_MISMATCH: 两方格网布局不一致，已禁止进入 PI-Sum；"
                f"mismatches={list(agreement.mismatches)}；"
                "请两方对齐 ir.grid_code_layout_manifest()"
                "（layout_id / version / 位分配）"
            )
            return result

    report = report or check_psi_sum_capabilities(bin_dir)
    if not report.runnable:
        result.blockers = report.blockers
        result.notes = result.notes + (
            "当前环境无法真实执行 PI-Sum；结果栏位保持空缺而不以推测值填充"
            "（见 blockers 与 docs/PSI_SUM_CAPABILITY.md）",
        )
        return result

    _t_enter = time.perf_counter()

    # 输入校验 + 排序去重：与 PSI 家族同一口径（去重是正确性前提）
    left_prep = prepare_grid_codes(left)
    right_prep = prepare_grid_codes(right)
    result.optimizer = {
        "applied": True,
        "version": GEOSOT_OPTIMIZER_VERSION,
        "sorted": True,
        "left": left_prep.to_dict(),
        "right": right_prep.to_dict(),
    }
    left_codes = [int(code) for code in left_prep.codes]
    right_codes = [int(code) for code in right_prep.codes]
    removed = left_prep.duplicate_count + right_prep.duplicate_count
    if removed:
        result.notes = result.notes + (
            "Geo-RR22 预处理：已排序并去重（left 移除 "
            f"{left_prep.duplicate_count} 个、right 移除 {right_prep.duplicate_count} 个"
            "重复码）。去重是正确性前提，不是可选优化",
        )
    result.original_count = len(left_codes)

    values_problem = _validate_values(left_codes, left_values)
    if values_problem is not None:
        result.status = "error"
        result.error = values_problem
        result.timings_ms["total_ms"] = round(
            (time.perf_counter() - _t_enter) * 1000.0, 3
        )
        return result
    assert left_values is not None  # _validate_values 已保证非 None

    # 明文参考（先算；失败不阻塞协议，与其它 PSI 路径一致）
    if reference_fn is not None:
        try:
            result.reference = reference_fn(left_codes, right_codes)
        except Exception:
            result.reference = None

    # 空输入：按集合论直接给值，不启动两个子进程（与兄弟路径同一口径）
    if not left_codes or not right_codes:
        result.status = "empty-input"
        result.value = (0, 0)
        result.intersection_count = 0
        result.intersection_unique_count = 0
        result.optimizer = {
            "applied": False,
            "reason": "空输入：未启动 PI-Sum 协议，未执行预处理",
        }
        result.agreement = _fold_reference(result.reference, 0, 0, left_values)
        result.timings_ms["total_ms"] = round(
            (time.perf_counter() - _t_enter) * 1000.0, 3
        )
        return result

    # ---------------- 协议执行（上游两个可执行文件，本机回环 gRPC） ----------------
    workdir = _make_workdir()
    server_proc = None
    client_proc = None
    server_probe: PeakRssProbe | None = None
    client_probe: PeakRssProbe | None = None
    relay: CountingRelay | None = None
    client_peak_mb: float | None = None
    server_peak_mb: float | None = None
    listen_port = port or PSI_SUM_DEFAULT_PORT
    client_port = listen_port
    if measure_communication:
        parsed = _split_host_port(listen_port)
        if parsed is None:
            result.status = "error"
            result.error = (
                f"通信量计量需要 host:port 形式的监听地址（收到 {listen_port!r}）；"
                "中继起不来时不退回未计量执行——否则“没测到”会被读成“测得”"
            )
            shutil.rmtree(workdir, ignore_errors=True)
            _release_workdir(workdir)
            return result
        relay = CountingRelay(parsed[0], parsed[1])
        relay.start()
        client_port = f"127.0.0.1:{relay.port}"
    try:
        server_csv = os.path.join(workdir, "server_data.csv")
        client_csv = os.path.join(workdir, "client_data.csv")
        # 输入 I/O 计时（口径见 docs/PSI_SUM_CAPABILITY.md §7.2）：只覆盖
        # 「把两份输入 CSV 落到本机临时目录」这一段。上游结果从 stdout 读回，
        # 本档没有输出文件读取段，故不产生 io_read_ms。
        _t_io = time.perf_counter()
        _write_server_csv(server_csv, right_codes)
        _write_client_csv(client_csv, left_codes, left_values)
        result.timings_ms["io_write_ms"] = round(
            (time.perf_counter() - _t_io) * 1000.0, 3
        )

        server_bin = report.binaries.get(
            PSI_SUM_SERVER_BINARY
        ) or binary_path(resolve_bin_dir(bin_dir) or "", PSI_SUM_SERVER_BINARY)
        client_bin = report.binaries.get(
            PSI_SUM_CLIENT_BINARY
        ) or binary_path(resolve_bin_dir(bin_dir) or "", PSI_SUM_CLIENT_BINARY)

        _t_proto = time.perf_counter()
        server_proc = spawn_pjc(
            server_bin,
            [f"--server_data_file={server_csv}", f"--port={listen_port}"],
        )
        if measure_memory:
            server_probe = PeakRssProbe(_proc_pid(server_proc))
            server_probe.start()
        if _read_until(server_proc, PSI_SUM_READY_MARKER, PSI_SUM_STARTUP_TIMEOUT_S) is None:
            result.status = "error"
            result.error = (
                f"上游 server 未在 {PSI_SUM_STARTUP_TIMEOUT_S:.0f}s 内报告就绪"
                f"（期望输出含 {PSI_SUM_READY_MARKER!r}）："
                "端口被占用、二进制不可执行或数据文件被拒都会走到这里；"
                "原始输出见 notes"
            )
            result.notes = result.notes + (
                f"server 原始输出：{_drain_text(server_proc)!r}",
            )
            result.timings_ms["pi_sum_execute_ms"] = round(
                (time.perf_counter() - _t_proto) * 1000.0, 3
            )
            return result

        client_proc = spawn_pjc(
            client_bin,
            [
                f"--client_data_file={client_csv}",
                f"--port={client_port}",
                f"--paillier_modulus_size={PSI_SUM_PAILLIER_MODULUS_SIZE}",
            ],
        )
        if measure_memory:
            client_probe = PeakRssProbe(_proc_pid(client_proc))
            client_probe.start()
        output, returncode = _read_to_end(client_proc, PSI_SUM_PROTOCOL_TIMEOUT_S)
        if client_probe is not None:
            client_peak_mb = client_probe.stop()
    except Exception as exc:
        result.status = "error"
        result.error = f"PI-Sum 执行失败：{type(exc).__name__}: {exc}"
        result.timings_ms["total_ms"] = round(
            (time.perf_counter() - _t_enter) * 1000.0, 3
        )
        return result
    finally:
        _terminate(client_proc)
        _terminate(server_proc)
        if server_probe is not None:
            server_peak_mb = server_probe.stop()
        if relay is not None:
            relay.stop()
        # 工作目录里只剩两份输入 CSV：退出即删（atexit 兜底）
        shutil.rmtree(workdir, ignore_errors=True)
        _release_workdir(workdir)

    if relay is not None:
        totals = relay.totals()
        result.communication = {
            "send_bytes": totals["client_to_server_bytes"],
            "recv_bytes": totals["server_to_client_bytes"],
            "total_bytes": totals["total_bytes"],
            "client_to_server_bytes": totals["client_to_server_bytes"],
            "server_to_client_bytes": totals["server_to_client_bytes"],
            "meter": "loopback-relay",
            "direction": "send=client→server；recv=server→client",
            "note": (
                "回环 TCP 中继逐字节计数（应用层字节，含 gRPC/HTTP2 封装；"
                "不含 TCP/IP 头）；中继多一跳，timings_ms 含该开销"
            ),
        }
    if measure_memory:
        peaks = [
            value for value in (client_peak_mb, server_peak_mb) if value is not None
        ]
        result.memory = {
            "client_peak_rss_mb": client_peak_mb,
            "server_peak_rss_mb": server_peak_mb,
            "peak_rss_mb": max(peaks) if peaks else None,
            "probe": "procfs-VmHWM",
            "sampling_interval_ms": round(VMHWM_SAMPLE_INTERVAL_S * 1000.0, 3),
            "note": (
                "两个子进程各记一条；peak_rss_mb 取较大者（并发两进程，不是合计）。"
                "数值是内核峰值水位；最后一次采样后到退出的窗口 ≤ 采样间隔"
            ),
        }

    result.timings_ms["pi_sum_execute_ms"] = round(
        (time.perf_counter() - _t_proto) * 1000.0, 3
    )

    match = PSI_SUM_RESULT_PATTERN.search(output)
    if match is None:
        tail = "\n".join(line for line in output.strip().splitlines()[-6:])
        result.status = "error"
        result.error = (
            "PI-Sum 未产出可解析的结果行（期望匹配 "
            f"{PSI_SUM_RESULT_PATTERN.pattern!r}）；client 退出码={returncode}；"
            f"输出尾部：{tail!r}"
        )
        result.timings_ms["total_ms"] = round(
            (time.perf_counter() - _t_enter) * 1000.0, 3
        )
        return result

    count = int(match.group(1))
    total = int(match.group(2))
    if returncode not in (0, None) and count == 0 and total == 0:
        # 退出码非 0 却报出 0/0：不当作成功（可能是半途失败后仍打印了默认值）
        result.status = "error"
        result.error = (
            f"client 退出码={returncode} 而结果行为 0/0，按失败处理；"
            f"输出尾部：{_tail(output)!r}"
        )
        result.timings_ms["total_ms"] = round(
            (time.perf_counter() - _t_enter) * 1000.0, 3
        )
        return result

    result.status = "ok"
    result.value = (count, total)
    result.intersection_count = count
    result.intersection_unique_count = count
    result.agreement = _fold_reference(result.reference, count, total, left_values)
    result.notes = result.notes + (
        f"PI-Sum 结果：交集基数={count}，交集内关联值之和={total}；"
        f"client 退出码={returncode}",
    )
    if report.version:
        result.protocol_params = dict(result.protocol_params)
        result.protocol_params["impl_commit"] = report.version
    # 本档声明 exact 语义：与明文不一致就是执行结论失败，而不是"带警示的 ok"
    # （与 libpsi 精确协议、PSI-CA 同一纪律：不接受"跑了但不对"）。
    if result.agreement is False:
        result.status = "error"
        result.error = (
            f"PI-Sum 结果与明文参考不一致：PI-Sum=（{count}, {total}），"
            f"明文参考={_jsonable(result.reference)!r}"
        )
    result.timings_ms["total_ms"] = round((time.perf_counter() - _t_enter) * 1000.0, 3)
    return result


def _split_host_port(listen: str) -> tuple[str, int] | None:
    """把 ``host:port`` 拆成 (host, port)；不合形返回 None（不猜）。"""

    text = str(listen).strip()
    host, _, port = text.rpartition(":")
    if not host or not port.isdigit():
        return None
    return host, int(port)


def _proc_pid(proc: Any) -> int | None:
    """真实子进程的 pid；测试桩（无 pid）返回 None（探针如实留空）。"""

    pid = getattr(proc, "pid", None)
    return pid if isinstance(pid, int) and pid > 0 else None


def _tail(text: str, lines: int = 6) -> str:
    """取输出尾部若干行（错误信息里的可读上下文）。"""

    return "\n".join(line for line in text.strip().splitlines()[-lines:])


def _drain_text(proc: Any) -> str:
    """把已经死掉的进程的输出读干净（用于错误上下文；不阻塞太久）。"""

    if proc is None:
        return ""
    try:
        output, _ = _read_to_end(proc, 5.0)
        return _tail(output)
    except Exception:
        return ""
