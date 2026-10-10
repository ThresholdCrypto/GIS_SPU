# -*- coding: utf-8 -*-
"""PSI-Cardinality 执行路径：只出交集基数的两方计算（openmined-psi）。

与 ``backends/psi_backend.runtime.run_psi_intersection`` 的关系
---------------------------------------------------------------
两条**独立路径、两种泄漏承诺**，刻意不合并：

- libpsi 求交：接收方在协议内部获得**交集本体**；``REVEAL_COUNT``
  只收缩业务层暴露，不改变协议泄漏面；
- PSI-CA：协议只计算并返回**计数**，交集本体不交给任一方。

因此本模块只承接 ``CellSetIntersect`` 的计数档（``REVEAL_COUNT``）；
其余算子 / 策略**显式拒绝**（不做交集、不做布尔判定）。

角色映射（必须知情）
--------------------
OpenMined PSI 是**不对称**协议：client 查询、server 持库，计数由 client
侧得到。本后端固定 client=左侧输入（对应 libpsi 的 ``receiver_rank=0``
口径），server=右侧输入。两侧集合都只在半诚实模型下受保护
（见 docs/PSI_CA_CAPABILITY.md §威胁模型）。

实现细节（均与上游行为对齐，有出处）
------------------------------------
- client / server 都以 ``reveal_intersection=False`` 创建：上游
  ``psi_server.cpp`` 的 ProcessRequest 拒绝两侧标志不一致的请求；
- ``CreateSetupMessage`` 的 fpr 参数对 RAW 档被忽略（上游源码注释原文），
  本项目固定传 0.0、不提供假阳率旋钮——不制造"已配置精度"的错觉；
- 输入先做 Geo-RR22 预处理（排序 + 去重），与 PSI 家族口径一致；
- 全程内存 protobuf，不落盘、不建临时文件（libpsi 路径需要 CSV）。
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from ..psi_backend.geosot_optimizer import GEOSOT_OPTIMIZER_VERSION, prepare_grid_codes
from ..psi_backend.input_adapter import check_layout_agreement
from ..psi_backend.protocol_registry import RESULT_SEMANTICS_EXACT
from ..psi_backend.result_policy import REVEAL_COUNT
from ..psi_backend.runtime import PsiRunResult, PSI_RUNTIME_WORLD_SIZE
from . import capability
from .capability import (
    PSI_CA_STRUCTURE,
    PSI_CA_SUPPORTED_OPS,
    PsiCaCapabilityReport,
    check_psi_ca_capabilities,
)

#: 结果里的协议名（展示用；不是 SPU/libpsi 枚举成员，不放进协议注册表）
PSI_CA_PROTOCOL = "PSI-CA"

#: 逐算子泄漏面登记（本后端只有这一档）
PSI_CA_OP_LEAKS: Mapping[str, str] = {
    "CellSetIntersect": (
        "只出交集基数：协议只计算并返回计数（client=左侧），"
        "交集本体不交给任一方；计数本身对 client 可见"
    ),
}

#: 协议内部泄漏面登记码（与 libpsi 的 intersection-body 并列、互不相同）
PSI_CA_PROTOCOL_LEAK = "count-only"

#: client 侧学习计数（对应 libpsi 的 receiver_rank=0 口径）
PSI_CA_LEARNING_RANK = 0


def _ca_policy_dict() -> dict[str, Any]:
    """CA 档的结果策略登记（业务暴露 + 协议泄漏，两栏并排）。

    ``leak_disclosure`` 是给 CLI 的整句泄漏披露：libpsi 档那句
    "接收方仍获得交集本体"在 CA 档是**错的**，必须换成本档的原文。
    """

    return {
        "policy": REVEAL_COUNT,
        "business_value": "count",
        "protocol_leak": PSI_CA_PROTOCOL_LEAK,
        "executable": True,
        "leak_disclosure": (
            "协议内部泄漏面：仅交集基数——PSI-CA 只计算计数，"
            "交集本体不离开任一方（与 libpsi 求交不同：后者把交集本体交给接收方）"
        ),
    }


def _agree_count(reference: Any, count: int) -> bool | None:
    """把明文参考折算到计数口径再比对；折算不出返回 None（不硬比）。"""

    if reference is None or isinstance(reference, bool):
        return None
    try:
        expected = len(set(int(value) for value in reference))
    except (TypeError, ValueError):
        return None
    return expected == count


#: 通信量计量口径名：协议消息的 protobuf 载荷（不是网络观测；口径见
#: docs/PSI_CA_CAPABILITY.md §7.1）。
PSI_CA_COMMUNICATION_METER = "protobuf-payload"


def _message_payload_sizes(
    setup: Any, request: Any, response: Any
) -> dict[str, int] | None:
    """三条协议消息的序列化载荷（字节）；任一取不到就返回 None（不填 0）。

    之所以量「消息载荷」而不是「网络字节」：本档是**进程内链路**
    （client / server 对象同进程、无 socket），没有可观测的网络；而在上线的
    真实部署里，这三条消息就是要过网的全部内容——方向出自上游 proto 注释
    （`private_set_intersection/proto/psi.proto`：Request 发往 server、
    Response 发回 client）与 API 形态（ServerSetup 只由 server 产出、
    只由 client 消费，故必须传到 client）。
    """

    sizes: dict[str, int] = {}
    messages = (("Request", request), ("ServerSetup", setup), ("Response", response))
    for name, message in messages:
        serializer = getattr(message, "SerializeToString", None)
        if not callable(serializer):
            return None
        try:
            sizes[name] = len(serializer())
        except Exception:
            return None
    return sizes


def run_psi_cardinality(
    left: Sequence[int],
    right: Sequence[int],
    *,
    op: str = "CellSetIntersect",
    result_policy: str | None = None,
    left_layout: Mapping[str, Any] | None = None,
    right_layout: Mapping[str, Any] | None = None,
    party_binding: Sequence[Mapping[str, Any]] | None = None,
    reference_fn: Any = None,
    report: PsiCaCapabilityReport | None = None,
    quiet: bool = True,
) -> PsiRunResult:
    """两方格网集合的**只出基数**计算（PSI-Cardinality，openmined-psi）。

    Args:
        left / right: 两方的 64 位格网码集合。角色：client=left（获得计数），
            server=right。
        op: 只接受 ``CellSetIntersect``（见 ``PSI_CA_SUPPORTED_OPS``）。
        result_policy: 只接受 ``REVEAL_COUNT``（None 等同——本后端只有这一档）；
            其它策略显式拒绝，不静默降级也不伪造交集。
        left_layout / right_layout: 两方格网布局清单；双方都给出且不一致时
            在进入协议前以 LAYOUT_MISMATCH 拒绝（与 libpsi 路径同源）。
        party_binding: 逐输入的参与方绑定披露，由编译期装配；原样登记。
        reference_fn: 明文参考实现（返回交集码序列）；给出后折算到计数口径
            做一致性比对。
        report: 能力核查结论（测试与编译期可注入；缺省现场探测）。
        quiet: 预留与兄弟后端一致的签名；本路径不产生 stdout 输出。
    """

    result = PsiRunResult(
        status="unavailable",
        op=op,
        protocol=PSI_CA_PROTOCOL,
        curve=None,
        protocol_params={"structure": PSI_CA_STRUCTURE, "impl": capability.PSI_CA_DISTRIBUTION},
        broadcast_result=False,
        # 本路径没有 libpsi 配置对象（配置闭环只对 libpsi 路径成立），
        # 不伪造 runtime_config；实际参数见 protocol_params 与 notes。
        runtime_config={},
        world_size=PSI_RUNTIME_WORLD_SIZE,
        receiver_rank=PSI_CA_LEARNING_RANK,
        reveals=PSI_CA_OP_LEAKS.get(op, "（未登记泄漏面，请补充）"),
        semantics="只出计数：|A∩B|（业务层获得计数；本档不产出交集本体）",
        result_semantics=RESULT_SEMANTICS_EXACT,
        result_policy=_ca_policy_dict(),
    )
    result.party_binding = tuple(dict(item) for item in (party_binding or ()))
    result.notes = result.notes + (
        "PSI-CA 角色映射：client=左侧输入（获得计数的一方，对应 receiver_rank=0），"
        "server=右侧输入；计数由 client 侧计算",
        "本路径不落盘：全程内存 protobuf 消息（无临时 CSV）",
    )

    if op not in PSI_CA_SUPPORTED_OPS:
        result.status = "error"
        result.error = (
            f"psi-ca 后端只承接 {tuple(PSI_CA_SUPPORTED_OPS)} 的计数档"
            f"（{REVEAL_COUNT}）；收到算子 {op!r}。其余算子请走 libpsi 求交路径"
            "（backends.psi_backend）"
        )
        return result

    requested = REVEAL_COUNT if result_policy is None else str(result_policy).upper().strip()
    if requested != REVEAL_COUNT:
        result.status = "error"
        result.error = (
            f"psi-ca 只能执行 {REVEAL_COUNT}（只出基数）；收到策略 {requested!r}。"
            "PSI-CA 不产出交集本体、也不做布尔判定——需要交集本体请改用 libpsi 求交路径"
        )
        return result

    # 布局握手前置检查：与 libpsi 路径同源、同文案口径（不依赖环境，离线也拦得住）
    if left_layout is not None or right_layout is not None:
        agreement = check_layout_agreement(left_layout, right_layout)
        result.layout_agreement = agreement.to_dict()
        if agreement.agreement is False:
            result.status = "error"
            result.error = (
                "LAYOUT_MISMATCH: 两方格网布局不一致，已禁止进入 PSI-CA；"
                f"mismatches={list(agreement.mismatches)}；"
                "请两方对齐 ir.grid_code_layout_manifest()"
                "（layout_id / version / 位分配）"
            )
            return result

    report = report or check_psi_ca_capabilities()
    if not report.runnable:
        result.blockers = report.blockers
        result.notes = result.notes + (
            "当前环境无法真实执行 PSI-CA；计数结果栏位保持空缺而不以推测值填充"
            "（见 blockers 与 docs/PSI_CA_CAPABILITY.md）",
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
    removed = left_prep.duplicate_count + right_prep.duplicate_count
    if removed:
        result.notes = result.notes + (
            "Geo-RR22 预处理：已排序并去重（left 移除 "
            f"{left_prep.duplicate_count} 个、right 移除 {right_prep.duplicate_count} 个"
            "重复码）。去重是正确性前提，不是可选优化",
        )
    left_codes = [int(code) for code in left_prep.codes]
    right_codes = [int(code) for code in right_prep.codes]
    result.original_count = len(left_codes)

    # 明文参考（先算；失败不阻塞 PSI-CA，与 libpsi 路径一致）
    if reference_fn is not None:
        try:
            result.reference = reference_fn(left_codes, right_codes)
        except Exception:
            result.reference = None

    # 空输入：按集合论直接给值，不启动协议（与 libpsi 路径同一口径）
    if not left_codes or not right_codes:
        result.status = "empty-input"
        result.value = 0
        result.intersection_count = 0
        result.intersection_unique_count = 0
        result.optimizer = {
            "applied": False,
            "reason": "空输入：未启动 PSI-CA 协议，未执行预处理",
        }
        result.agreement = _agree_count(result.reference, 0)
        result.timings_ms["total_ms"] = (time.perf_counter() - _t_enter) * 1000.0
        return result

    # ---------------- 协议执行（调用上游实现；全程内存消息） ----------------
    payload: dict[str, int] | None = None
    try:
        psi_mod = capability.import_openmined_psi()
        client_items = [str(code) for code in left_codes]
        server_items = [str(code) for code in right_codes]

        client = psi_mod.client.CreateWithNewKey(False)
        server = psi_mod.server.CreateWithNewKey(False)
        setup = server.CreateSetupMessage(
            0.0,  # RAW 档忽略 fpr（上游 psi_server.cpp 源码注释）
            len(client_items),
            server_items,
            psi_mod.DataStructure.RAW,
        )
        request = client.CreateRequest(client_items)
        response = server.ProcessRequest(request)
        count = int(client.GetIntersectionSize(setup, response))
        payload = _message_payload_sizes(setup, request, response)
    except Exception as exc:
        result.status = "error"
        result.error = f"PSI-CA 执行失败：{type(exc).__name__}: {exc}"
        result.timings_ms["total_ms"] = (time.perf_counter() - _t_enter) * 1000.0
        return result

    result.timings_ms["psi_execute_ms"] = (time.perf_counter() - _t_enter) * 1000.0
    result.status = "ok"
    if payload is None:
        result.notes = result.notes + (
            "未能取到协议消息载荷（上游 API 漂移：消息对象缺少 SerializeToString）"
            "——通信量留空，不以 0 或推测值填充",
        )
    else:
        send_bytes = payload["Request"]
        recv_bytes = payload["ServerSetup"] + payload["Response"]
        result.communication = {
            "send_bytes": send_bytes,
            "recv_bytes": recv_bytes,
            "total_bytes": send_bytes + recv_bytes,
            "client_to_server_bytes": send_bytes,
            "server_to_client_bytes": recv_bytes,
            "by_message": dict(payload),
            "meter": PSI_CA_COMMUNICATION_METER,
            "direction": (
                "send=client→server（Request）；"
                "recv=server→client（ServerSetup + Response）"
            ),
            "note": (
                "协议三条消息的 protobuf 载荷（SerializeToString 实测）；"
                "本档是进程内链路、无 socket，故不含任何传输封装"
                "（gRPC / HTTP2 / TLS）与 TCP/IP 头——真实部署的网络字节 ≥ 此值"
            ),
        }
    result.intersection_count = count
    result.intersection_unique_count = count
    result.value = count
    result.agreement = _agree_count(result.reference, count)
    # 本档声明 exact 语义：计数与明文不一致就是执行结论失败，而不是
    # "带警示的 ok"（与 libpsi 精确协议同一纪律：不接受"跑了但不对"）。
    if result.agreement is False:
        result.status = "error"
        result.error = (
            f"PSI-CA 计数与明文参考不一致：PSI-CA={count}，"
            f"明文={result.reference!r}"
        )
    if report.version:
        result.protocol_params = dict(result.protocol_params)
        result.protocol_params["impl_version"] = report.version
    result.timings_ms["total_ms"] = (time.perf_counter() - _t_enter) * 1000.0
    return result
