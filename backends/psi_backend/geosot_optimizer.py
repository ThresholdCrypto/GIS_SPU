# -*- coding: utf-8 -*-
"""Geo-RR22 预处理：GridCode 的排序 / 去重 / 前缀压缩 / 分桶（任务书 §8 / §9）。

定位
====
本模块只做**确定性、无损、可审计**的集合整形，不做密码学优化：

    原始 CellSet（可能乱序、含重复）
        → sort + dedup（sorted unique —— 进入 PSI 的键集合，执行路径默认启用）
        → compress（前缀分组表示；分析与后续候选剪枝用，不进协议）
        → partition（按高位前缀分桶；后续候选剪枝用，不进协议）

PSI 交换的是完整的 64 位码本身，所以 compress / partition 的产物**不是**
PSI 输入；它们为 §10 的"候选集剪枝"预留表示。候选剪枝尚未实现——
本模块不会因为"看起来能删"而丢弃任何码（剪枝只能删"确定不相交"部分，
且必须另行设计验证，见 docs/GEO_RR22_DESIGN.md）。

去重是正确性前提（不是可选优化）
================================
spu 0.9.5 实测（2026-09）：

- ``PROTOCOL_RR22``：重复键直接失败
      Paxos error, Duplicate keys were detected ...
- ``PROTOCOL_KKRT``：重复键直接失败
      Cannot find empty bin in stash ...
- ``PROTOCOL_ECDH``：能跑完，但 intersection_count 按重复乘数膨胀
      （N=2^12、dup=0.25 实测 3072 vs 唯一 2048），不能按唯一集合解读。

因此 ``runtime.run_psi_intersection`` 默认先调用 ``prepare_grid_codes()``；
只有基准 / 诊断需要复现原始协议行为时才显式 ``optimize_input=False``
（见 docs/BENCHMARK_PROTOCOL.md 与 docs/GEO_RR22_DESIGN.md）。

版本与指纹（跨方一致性）
========================
预处理是双方一致性敏感步骤：两侧必须用同一版规则。本模块提供：

- ``GEOSOT_OPTIMIZER_VERSION``：规则版本，随预处理语义变化递增；
- ``fingerprint_grid_codes()``：对 sorted unique 码集合取 SHA-256 的前 16 个
  十六进制字符。两侧各持一个指纹，供审计 / 对拍（"预处理握手"）。
  指纹只覆盖码集合本身，不引入新的信息暴露。
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Iterable

from .input_adapter import validate_grid_code

#: 预处理规则版本：排序口径 / 去重口径 / 表示格式变化时递增。
GEOSOT_OPTIMIZER_VERSION = "0.1.0"

#: 指纹长度（SHA-256 十六进制字符数）。16 字符 = 64 bit，够做相等性对拍。
FINGERPRINT_HEX_CHARS = 16


def _validated(codes: Iterable[int]) -> list[int]:
    out: list[int] = []
    for index, code in enumerate(codes):
        out.append(validate_grid_code(code, where=f"codes[{index}]"))
    return out


def sort_grid_codes(codes: Iterable[int]) -> tuple[int, ...]:
    """升序排序（保留重复；去重请用 ``deduplicate_grid_codes``）。"""

    return tuple(sorted(_validated(codes)))


def deduplicate_grid_codes(codes: Iterable[int]) -> tuple[int, ...]:
    """去重，保留**首次出现**的顺序（确定性；不隐式排序）。"""

    seen: dict[int, None] = {}
    for code in _validated(codes):
        seen.setdefault(code, None)
    return tuple(seen)


@dataclass(frozen=True)
class GridCodePreparation:
    """一次预处理的结果：sorted unique 码 + 规模统计（进入 PSI 的形态）。"""

    codes: tuple[int, ...]
    input_count: int
    duplicate_count: int

    @property
    def unique_count(self) -> int:
        return len(self.codes)

    @property
    def fingerprint(self) -> str:
        return fingerprint_grid_codes(self.codes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_count": self.input_count,
            "unique_count": self.unique_count,
            "duplicate_count": self.duplicate_count,
            "fingerprint": self.fingerprint,
            "version": GEOSOT_OPTIMIZER_VERSION,
        }


def prepare_grid_codes(codes: Iterable[int]) -> GridCodePreparation:
    """执行路径使用的组合：一次校验 → 去重 → 排序。"""

    raw = _validated(codes)
    unique = sorted(set(raw))
    return GridCodePreparation(
        codes=tuple(unique),
        input_count=len(raw),
        duplicate_count=len(raw) - len(unique),
    )


def fingerprint_grid_codes(codes: Iterable[int]) -> str:
    """sorted unique 码集合的审计指纹（SHA-256 前 16 个十六进制字符）。"""

    ordered = sorted(set(_validated(codes)))
    packed = b"" if not ordered else struct.pack(f">{len(ordered)}Q", *ordered)
    return hashlib.sha256(packed).hexdigest()[:FINGERPRINT_HEX_CHARS]


@dataclass(frozen=True)
class GridCodeGroup:
    """前缀分组：高 ``shared_bytes`` 个字节相同，低字节在 ``suffixes`` 列出。"""

    shared_bytes: int
    prefix: int
    suffixes: tuple[int, ...]

    def expand(self) -> tuple[int, ...]:
        if self.shared_bytes == 0:
            return self.suffixes
        shift = 8 * (8 - self.shared_bytes)
        return tuple((self.prefix << shift) | suffix for suffix in self.suffixes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_bytes": self.shared_bytes,
            "prefix": self.prefix,
            "count": len(self.suffixes),
        }


@dataclass(frozen=True)
class GridCodeCompression:
    """无损前缀压缩表示：``expand()`` 必须能还原 sorted unique 码。"""

    groups: tuple[GridCodeGroup, ...]
    code_count: int

    def expand(self) -> tuple[int, ...]:
        out: list[int] = []
        for group in self.groups:
            out.extend(group.expand())
        return tuple(out)

    @property
    def raw_bytes(self) -> int:
        return self.code_count * 8

    @property
    def compressed_bytes(self) -> int:
        total = 0
        for group in self.groups:
            total += 8  # prefix 表示口径：按 8 字节整数计
            total += len(group.suffixes) * (8 - group.shared_bytes)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_count": self.code_count,
            "group_count": len(self.groups),
            "raw_bytes": self.raw_bytes,
            "compressed_bytes": self.compressed_bytes,
            "groups": [group.to_dict() for group in self.groups],
        }


def _common_prefix_bytes(first: int, last: int) -> int:
    """sorted 序列的首/末码 → 整个序列的公共前缀字节数（0..7）。"""

    diff = first ^ last
    if diff == 0:
        raise ValueError("_common_prefix_bytes 需要至少两个不同的码")
    return (64 - diff.bit_length()) // 8


def compress_grid_codes(codes: Iterable[int]) -> GridCodeCompression:
    """把 sorted unique 码集合压成"高字节前缀 + 低字节后缀"分组（无损）。

    分组按最高字节（X 字段的高位）聚合——GeoSOT 相邻格网的高位相同，
    这是最自然的局部性；不做任何有损近似，``expand()`` 逐元素可还原。
    """

    ordered = sorted(set(_validated(codes)))
    groups: list[GridCodeGroup] = []
    index = 0
    while index < len(ordered):
        run_start = index
        top = ordered[index] >> 56
        while index < len(ordered) and (ordered[index] >> 56) == top:
            index += 1
        run = ordered[run_start:index]
        if len(run) == 1:
            groups.append(GridCodeGroup(0, 0, (run[0],)))
            continue
        shared = _common_prefix_bytes(run[0], run[-1])
        if shared == 0:
            groups.append(GridCodeGroup(0, 0, tuple(run)))
            continue
        shift = 8 * (8 - shared)
        mask = (1 << shift) - 1
        groups.append(
            GridCodeGroup(shared, run[0] >> shift, tuple(code & mask for code in run))
        )
    return GridCodeCompression(groups=tuple(groups), code_count=len(ordered))


def partition_grid_codes(
    codes: Iterable[int], *, num_buckets: int = 16
) -> tuple[tuple[int, ...], ...]:
    """按 64 位码的最高 log2(num_buckets) 位分桶（无损、确定性）。

    - ``num_buckets`` 必须是 2 的幂——非 2 的幂直接报错，不用取模近似分布，
      那会把"高位相同"的空间局部性弄乱；
    - 返回全部桶（含空桶），顺序即桶号；并集 == sorted unique 输入；
    - 本函数只做**划分**：候选集剪枝（§10）未实现，不在此丢弃任何码。
    """

    if num_buckets < 1 or (num_buckets & (num_buckets - 1)):
        raise ValueError(f"num_buckets 必须是 2 的幂，实得 {num_buckets!r}")
    ordered = sorted(set(_validated(codes)))
    bits = num_buckets.bit_length() - 1
    buckets: list[list[int]] = [[] for _ in range(num_buckets)]
    if bits == 0:
        buckets[0] = ordered
    else:
        shift = 64 - bits
        for code in ordered:
            buckets[code >> shift].append(code)
    return tuple(tuple(bucket) for bucket in buckets)
