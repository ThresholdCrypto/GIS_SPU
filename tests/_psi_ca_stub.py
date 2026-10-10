# -*- coding: utf-8 -*-
"""openmined-psi 测试桩：不装 openmined-psi 也能验证 PSI-CA 接入层。

桩按上游 2.0.6 的 API 形态复刻（``client`` / ``server`` 子入口 + ``DataStructure``
枚举 + ``ServerSetup`` / ``Request`` / ``Response`` 类型），只实现本项目用到的
调用序列，并把每次调用记入 ``stub.calls`` 供测试断言。

边界说明：桩只保证"接口形态正确 + 计数按明文集合计算"，**不能替代真机验证**
（真实协议的密码学语义不在桩的职责内）。
"""

from __future__ import annotations

import types

#: 上游模块级属性清单（missing_api 核对用；桩必须齐全才能代表 2.0.6 形态）
MODULE_ATTRS = ("client", "server", "DataStructure", "ServerSetup", "Request", "Response")

#: 桩里的「消息载荷」基数 / 每条目增量（字节）。
#: 只用来验证计量接线与方向拆分，**不代表真实编码长度**（真实长度由 protobuf
#: 与上游实现决定，真机读数见 docs/PSI_CA_CAPABILITY.md §7.1）。
PAYLOAD_BASE_BYTES = 16
PAYLOAD_PER_ITEM_BYTES = 8


class _DataStructure:
    RAW = "RAW"
    GCS = "GCS"
    BLOOM_FILTER = "BLOOM_FILTER"


class _Setup:
    def __init__(self, items):
        self.items = list(items)

    def SerializeToString(self) -> bytes:
        return b"S" * (
            PAYLOAD_BASE_BYTES + PAYLOAD_PER_ITEM_BYTES * len(self.items)
        )


class _Request:
    def __init__(self, items):
        self.items = list(items)

    def SerializeToString(self) -> bytes:
        return b"Q" * (
            PAYLOAD_BASE_BYTES + PAYLOAD_PER_ITEM_BYTES * len(self.items)
        )


class _Response:
    def __init__(self, request):
        self.request = request

    def SerializeToString(self) -> bytes:
        return b"P" * (
            PAYLOAD_BASE_BYTES
            + PAYLOAD_PER_ITEM_BYTES * len(self.request.items)
        )


class _Client:
    def __init__(self, owner):
        self._owner = owner

    def CreateWithNewKey(self, reveal_intersection: bool):
        self._owner.calls.append(("client.CreateWithNewKey", reveal_intersection))
        return self

    def CreateRequest(self, data):
        items = [str(item) for item in data]
        self._owner.calls.append(("client.CreateRequest", tuple(items)))
        return _Request(items)

    def GetIntersectionSize(self, setup, response):
        self._owner.calls.append(("client.GetIntersectionSize",))
        return len(set(setup.items) & set(response.request.items))


class _Server:
    def __init__(self, owner):
        self._owner = owner

    def CreateWithNewKey(self, reveal_intersection: bool):
        self._owner.calls.append(("server.CreateWithNewKey", reveal_intersection))
        return self

    def CreateSetupMessage(self, fpr, num_client_inputs, inputs, ds):
        self._owner.calls.append(
            ("server.CreateSetupMessage", fpr, num_client_inputs, ds)
        )
        return _Setup(inputs)

    def ProcessRequest(self, request):
        self._owner.calls.append(("server.ProcessRequest",))
        return _Response(request)


class PsiCaStub:
    """上游模块形态的测试桩（模块级属性齐全，可过 missing_api 核对）。"""

    def __init__(self, *, version: str = "2.0.6"):
        self.__version__ = version
        self.calls: list[tuple] = []
        self.client = _Client(self)
        self.server = _Server(self)
        self.DataStructure = _DataStructure
        self.ServerSetup = _Setup
        self.Request = _Request
        self.Response = _Response


class _DropProxy:
    """把指定方法藏起来的代理：验证 missing_api 能核出 API 漂移。"""

    def __init__(self, target, dropped):
        self._target = target
        self._dropped = frozenset(dropped)

    def __getattr__(self, name):
        if name in self._dropped:
            raise AttributeError(name)
        return getattr(self._target, name)


def make_stub(*, version: str = "2.0.6") -> PsiCaStub:
    return PsiCaStub(version=version)


def without_api(stub, *, module_attrs=(), methods=()):
    """返回"缺了某些 API"的桩；module_attrs 删模块属性，methods 藏子入口方法。"""

    ns = types.SimpleNamespace()
    for name in MODULE_ATTRS:
        if name in module_attrs:
            continue
        obj = getattr(stub, name)
        dropped = [method for owner, method in methods if owner == name]
        if dropped:
            obj = _DropProxy(obj, dropped)
        setattr(ns, name, obj)
    ns.__version__ = stub.__version__
    return ns


def install_stub_psi(monkeypatch, *, version: str = "2.0.6") -> PsiCaStub:
    """把桩装进 PSI-CA 后端的唯一导入点，返回桩（供断言调用序列）。"""

    from backends.psi_ca_backend import capability

    stub = make_stub(version=version)
    monkeypatch.setattr(capability, "import_openmined_psi", lambda: stub)
    return stub
