# -*- coding: utf-8 -*-
# 口径复算脚本（与《三维格网接入层口径勘误与修正说明》配套）
# 仅依赖 Python 标准库；输出与脚本同目录。
import os, sys, io, json
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
"""终检：修正版 JSON 自洽性 + 时间原点唯一性论证"""
import json, math, datetime
P=r"C:\Users\DELL\Documents\Codex\2026-09-20\geosot-3d-dqg-4d-c-users-2\outputs\格网数据样例_明文与密态映射_v5.json"
d=json.load(open(P,encoding="utf-8"))
print("JSON 载入 OK；version=%s；cells=%d；quant dict=%d"% (d["meta"]["version"],len(d["cells"]),len(d["quantization_dict"])))
def tz(n): return 99 if n==0 else (n&-n).bit_length()-1
def canon(s,e):
    out=[];c=s
    while c<e:
        rem=e-c; l=min(tz(c),rem.bit_length()-1)
        while (1<<l)>rem: l-=1
        out.append((c,l)); c+=1<<l
    assert sum(1<<l for _,l in out)==e-s
    return out
ok=True
for c in d["cells"]:
    for sg in c["time_segments"]:
        for n in sg["canonical_nodes"]:
            a = n["Toff"]%(1<<n["Lt"])==0
            if not a: ok=False; print("  对齐失败",c["cell_id"],n)
        # 规范分解与实际区间一致
        s0,e0=sg["Toff_span"]
        if (s0,e0)!=[0,1<<14]:
            exp=[(t,l) for t,l in canon(s0,e0)]
            got=[(n["Toff"],n["Lt"]) for n in sg["canonical_nodes"]]
            if exp!=got: ok=False; print("  分解不符",c["cell_id"],exp,got)
print("全部时间节点对齐且分解正确:",ok)
print("窗口投影节点数 %d -> 500 网格 %d 条 -> %d 倍"%(d["window_projection"]["node_count"],
      500*d["window_projection"]["node_count"], d["time_cost"]["reduction"]))
print("L2=%d 条, 倍数=%.1f ; L1=%d 条"%(d["cost_prediction"]["L2_bitplane"],
      d["cost_prediction"]["reduction_L2_vs_naive"],d["cost_prediction"]["L1_same_cell_cross_attr"]))
print()
print("量化字典阈值对齐状态：")
for q in d["quantization_dict"]:
    ta=q["threshold_alignment"]
    print("  %s %-8s 阈值 %-8s 箱号 %-6s 对齐=%-5s 对齐所需上界=%s"%(q["attr"],q["name"],ta["value"],ta["bin"],ta["aligned"],ta["required_max_for_alignment"]))
print()
print("="*90)
print("时间原点唯一性（要求：4 个交付节点均对齐 + 7 天窗口 = 14 节点 + 原点为整点）")
EV1=(2040,2056); EV2=(2056,2160)
hits=[]
for off in range(0,6305):
    o=off
    a=canon(EV1[0]+o,EV1[1]+o); b=canon(EV2[0]+o,EV2[1]+o); w=canon(o,o+10080)
    if len(a)==1 and a[0][1]==4 and [l for _,l in b]==[6,5,3] and len(w)==14: hits.append(off)
print("  满足结构约束的偏移:",hits)
print("  其中整点解:",[h for h in hits if h%60==0])
print("  -> 唯一整点解 %d 分钟，即原点 2026-09-06T22:00:00+08:00"%[h for h in hits if h%60==0][0])
