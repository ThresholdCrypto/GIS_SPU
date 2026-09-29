# -*- coding: utf-8 -*-
# 口径复算脚本（与《三维格网接入层口径勘误与修正说明》配套）
# 仅依赖 Python 标准库；输出与脚本同目录。
import os, sys, io, json
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
"""时间树原点反解：哪个原点能同时复现 (a) 交付的 64+32+8 结构 (b) 7天窗口=14节点 (c) 720x"""
def tz(n): return 99 if n==0 else (n&-n).bit_length()-1
def canon(s,e):
    out=[];c=s
    while c<e:
        rem=e-c; l=min(tz(c),rem.bit_length()-1)
        while (1<<l)>rem: l-=1
        out.append((c,l)); c+=1<<l
    assert sum(1<<l for _,l in out)==e-s
    return out

# 交付结构（相对窗口起点）: 事件1 单节点 Lt=4；事件2 = Lt(6,5,3)
ev1=(2040,2056); ev2=(2056,2160)
win=(0,10080)

print("目标：(a) 事件1 -> 单节点 Lt=4  (b) 事件2 -> Lt=(6,5,3)  (c) 窗口 -> 14 节点  (d) 10080/节点数=720")
print()
hits=[]
for off in range(-2000,2001):          # 原点相对窗口起点的偏移（分钟）
    s1=canon(ev1[0]+off, ev1[1]+off)
    s2=canon(ev2[0]+off, ev2[1]+off)
    sw=canon(win[0]+off, win[1]+off)
    a = (len(s1)==1 and s1[0][1]==4)
    b = ([l for _,l in s2]==[6,5,3])
    c = (len(sw)==14)
    if a and b and c:
        hits.append(off)
print("同时满足 (a)(b)(c) 的原点偏移：", hits)
print()
for off in hits[:4]:
    print("偏移 %+d 分钟（原点 = 窗口起点 %s %d 分钟）" % (off, "前" if off>0 else "后", abs(off)))
    print("   事件1 ->", canon(ev1[0]+off, ev1[1]+off))
    print("   事件2 ->", [(t,l) for t,l in canon(ev2[0]+off, ev2[1]+off)])
    print("   窗口  -> %d 节点 -> 500网格x%d = %d 条 ; 720x 校验 %d/10080 -> %s" %
          (len(canon(win[0]+off,win[1]+off)), len(canon(win[0]+off,win[1]+off)),
           500*len(canon(win[0]+off,win[1]+off)), 10080//len(canon(win[0]+off,win[1]+off)),
           "MATCH" if 10080//len(canon(win[0]+off,win[1]+off))==720 else "no"))
    print()
print("="*100)
print("结论校验：原点 = 窗口起点前 120 分钟 = 2026-09-06T22:00:00+08:00")
OFF=120
T=[("C-0001 seg1 10:00-10:16",2040,4),("C-0001 seg2a",2056,6),("C-0001 seg2b",2120,5),("C-0001 seg2c",2152,3)]
for nm,t,l in T:
    nt=t+OFF
    print("  %-24s 交付 Toff=%-5d -> 修正 Toff=%-5d  Lt=%d  对齐=%s" % (nm,t,nt,l,nt%(1<<l)==0))
print("  交付 Toff=0/Lt=14 的整窗常量 -> 窗口 [120,10200) 规范分解 %d 节点，非单节点" % len(canon(120,10200)))
print("  但：(0, Lt=14) 作为'全地址空间常量'仍成立；若按窗口裁剪则为 14 节点")
print()
print("  修正后的事件2 分解复核：", canon(2056+OFF,2160+OFF), "-> Lt=(6,5,3) = 64+32+8 分钟 ✓")
print("  24h 语义核对：Toff=2160 -> 2160/60 = %d h 相对原点；原点 22:00 + 36h = 2026-09-08 10:00 ✓" % (2160//60))
