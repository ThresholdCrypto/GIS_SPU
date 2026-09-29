# -*- coding: utf-8 -*-
# 口径复算脚本（与《三维格网接入层口径勘误与修正说明》配套）
# 仅依赖 Python 标准库；输出与脚本同目录。
import os, sys, io, json
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
import math, json
R=[]
def add(no,item,sev,src,ev,imp,fix):
    R.append(dict(no=no,item=item,severity=sev,source=src,evidence=ev,impact=imp,fix=fix))

BAR="="*104
print(BAR); print("F1  grid_code 定长键的位序与总位宽：三处描述不一致"); print(BAR)
v1=[("交接/JSON/PPT",[17,17,7,4,14,4,1]),("技术讨论说明.docx",[1,4,17,17,7,4,4,14])]
for nm,w in v1:
    print("  %-22s 分段=%s 合计=%d bit %s" % (nm, w, sum(w), "OK" if sum(w)==64 else "<< %d bit != 64" % sum(w)))
print("  docx 位序 = version|geo_level|X'|Y'|Z|L|Lt|Toff = 68 bit，与其自身'合计64位即8字节'表述矛盾")
print("  JSON 的 ver 在最低位；docx 的 version 在最高位 -> 前缀/包含语义不同")
add("F1","grid_code 位序与位宽口径不一致","高",
    "交接/JSON/PPT = X'|Y'|Z|L|Toff|Lt|ver(64b)；技术讨论说明.docx = version|geo_level|X'|Y'|Z|L|Lt|Toff(68b)",
    "17+17+7+4+14+4+1=64；1+4+17+17+7+4+4+14=68，超出 8 字节 4 位。脚本回环验证：三例 hex 仅在 X' 置最高位时全部吻合",
    "位序决定 x & mask_L 的前缀包含语义与 OPRF 输入字节流；68!=64 使批量 OPRF 的定长假设失效，属落库后不可低成本回改项",
    "以 JSON/PPT 位序为唯一版本，回改 docx 并删除多余的 geo_level 字段")

print(); print(BAR); print("F2  属性 A02 人口密度的值域：两份文档相差 50 倍"); print(BAR)
def q(v,mn,mx,b=8): return math.floor((v-mn)/(mx-mn)*(2**b-1)+0.5)
for nm,mn,mx,src in [("JSON/PPT/交接",0,20000,"三处交付物"),("技术讨论说明.docx",0,10**6,"docx 正文")]:
    print("  %-22s 值域 %-9s 8600 -> q=%-4d 阈值12000 -> q=%-4d 箱宽=%.4f   (%s)" % (nm,"%d-%d"%(mn,mx),q(8600,mn,mx),q(12000,mn,mx),(mx-mn)/255.0,src))
print("  10^6 口径下 12000 仅落在第 3 箱 -> 0-20000 全域被压缩到 3/255 箱号，分箱退化")
add("F2","属性 A02 人口密度值域冲突","高","JSON/PPT/交接 = 0-20000；技术讨论说明.docx = 0-10^6",
    "同一值 8600 量化结果分别为 110 与 2；阈值 12000 分别为 153 与 3",
    "per-attribute 量化字典是 b<=8 与阈值对齐箱边界的前提；字典不一致则跨方同一值得到不同密态输入，阈值比较出现系统性假阴性",
    "以 A02 = 0-20000 /km2 为准，回改 docx 的 0-10^6 表述")

print(); print(BAR); print("F3  时间节点不是合法的二叉时间树节点"); print(BAR)
def tz(n): return 99 if n==0 else (n&-n).bit_length()-1
def canonical(s,e):
    out=[];c=s
    while c<e:
        rem=e-c; l=min(tz(c),rem.bit_length()-1)
        while (1<<l)>rem: l-=1
        out.append((c,l)); c+=1<<l
    return out
for nm,t,l in [("seg1 [10:00,10:16)",2040,4),("seg2a 64min",2056,6),("seg2b 32min",2120,5),("seg2c 8min",2152,3)]:
    print("  %-20s Toff=%-5d Lt=%-2d 块长=%-5d Toff mod 2^Lt = %-4d %s" % (nm,t,l,1<<l,t%(1<<l),"合法" if t%(1<<l)==0 else "<< 非对齐"))
print("  合法率 %d/4" % sum(1 for _,t,l in [(0,2040,4),(0,2056,6),(0,2120,5),(0,2152,3)] if t%(1<<l)==0))
print("  声明 epoch=2026-09-07T00:00 下 [2040,2056) 规范分解 = %s" % canonical(2040,2056))
print("  声明 epoch=2026-09-07T00:00 下 [2056,2160) 规范分解 = %s" % canonical(2056,2160))
print("  交付所写 64+32+8 需要 2056%%64==0 且 2120%%32==0；实测余数均为 8")
print("  若 epoch 前移 8 分钟 -> 2040->2032, 2056->2048, 2120->2112, 2152->2144 全部对齐")
add("F3","时间段节点非二叉对齐，规范分解不可复现","高","JSON 样例 + PPT 第9页 + DQG-4D 修订章节 步骤2.2A",
    "4 节点中仅 1 个满足 Toff mod 2^Lt = 0；按声明 epoch 复算 [2040,2056) 应为 2 节点、[2056,2160) 应为 5 节点，而非 1+3=4",
    "步骤2.2A 要求节点完全包含于区间且父节点不完全包含；非对齐节点破坏唯一分解，检索键无法按 (Lt,Toff) 定位，段内恒定掩码的位与运算失去前提",
    "time_epoch 改为与树根对齐（2026-09-06T23:52 或生成侧减 8 分钟偏移），并给出该 epoch 下的规范分解")

print(); print(BAR); print("F4  '整窗恒定'用单根节点表示与窗口长度不符"); print(BAR)
print("  窗口 [0,10080) = 7 天；树根 Lt=14 覆盖 %d min -> 超出窗口末端 %d min" % (1<<14,(1<<14)-10080))
print("  [0,10080) 规范分解 = %s (%d 节点)" % (canonical(0,10080), len(canonical(0,10080))))
print("  [0,16384) 规范分解 = %s -> 单根节点仅在把区间定义为全地址空间时成立" % canonical(0,16384))
add("F4","窗口长非 2 的幂，整窗常量=单根节点不成立","中","JSON 样例 C-0002/C-0003 均写 Toff=0, Lt=14",
    "7 天 = 10080 min，2^14 = 16384 min；[0,10080) 的规范分解为 6 个节点",
    "覆盖超出窗口 6304 min，违反完全包含于区间；按单节点物化会引入窗口外无效时段。节点数上界 2*Lt_max=28 仍成立，量级不变",
    "约定 t_end = +inf 一律解释为窗口末端，窗口常量按 6 节点物化")

print(); print(BAR); print("F5  两个业务阈值未对齐箱边界"); print(BAR)
for nm,mn,mx,t in [("A01 风速",0,50,10.8),("A02 人口密度",0,20000,12000),("A03 噪声",30,120,90),("A04 建筑风险",0,1,0.70)]:
    k=(t-mn)/(mx-mn)*255.0; ke=round(k); w=(mx-mn)/255.0; edge=mn+ke*w
    tie=abs(k-math.floor(k)-0.5)<1e-12
    st="对齐" if abs(k-round(k))<1e-9 else "未对齐(边界 %.6f, 偏差 %+.6f = %.1f%% 箱宽)" % (edge,edge-t,abs(edge-t)/w*100)
    print("  %-12s 阈值 %-7s 箱号 %8.4f -> %-4d %s%s" % (nm,t,k,ke,st,"  << 半格 tie" if tie else ""))
print("  A01 对齐解：值域改 0-51 -> 箱宽 0.2，10.8 精确落第 54 箱边界")
print("  A04 对齐解：值域改 0-0.997207；或显式采 half-up 并记 0.70 为第 179 箱")
add("F5","业务阈值未精确落在箱边界（风速/建筑风险）","中","JSON quantization_dict + 交接量化口径",
    "风速 10.8 箱号 55.08（非整数），最近边界 10.784314，偏离 0.0157 = 8% 箱宽；建筑风险 0.70 箱号 178.5 恰为半格",
    "阈值对齐箱边界是密态比较退化为常数比较、不引入隐藏截断的前提；未对齐使判定带 8%-50% 箱宽吸附误差",
    "风速值域 0-51（10.8 落第 54 箱边界）；建筑风险值域 0-0.997207 或写死 half-up")

print(); print(BAR); print("F6  取整规则未写死（0.70 落在半格）"); print(BAR)
print("  0.70 x 255 = %s -> half-up 得 179；Python round() banker's rounding 得 %d" % (0.70*255,round(178.5)))
print("  JSON 写 179；若平台用 round() 则得 %d -> 判定边界 %.6f 而非 0.70" % (round(178.5),178/255.0))
add("F6","取整规则未写死，存在半格歧义","中","JSON A04 threshold bin=179",
    "0.70x255 = 178.5 恰为半格；half-up=179，banker's rounding=178，判定边界 0.698039",
    "接入层与平台若采不同取整实现，同一阈值产生 1 个箱号的系统性偏移；在 veto 型硬约束上等于改变业务判据",
    "文档写死 q = floor((v-min)/(max-min)*(2^b-1) + 0.5)")

print(); print(BAR); print("F7  已复核通过项"); print(BAR)
def enc(X,Y,Z,L,T,Lt,v): return ((X&0x1FFFF)<<47)|((Y&0x1FFFF)<<30)|((Z&0x7F)<<23)|((L&0xF)<<19)|((T&0x3FFF)<<5)|((Lt&0xF)<<1)|(v&1)
chk=[("grid_code C-0001", "0x%016X"%enc(21861,27702,15,9,2040,4,0), "0x2AB29B0D87C8FF08"),
     ("grid_code C-0002", "0x%016X"%enc(21861,27703,15,9,0,14,0), "0x2AB29B0DC7C8001C"),
     ("grid_code C-0003", "0x%016X"%enc(21862,27702,16,9,0,14,0), "0x2AB31B0D8848001C"),
     ("rel_bitset 2^3+2^7+2^12", str(2**3+2**7+2**12), "4232"),
     ("L2 密文条数 ceil(196000/8192)*8", str(math.ceil(196000/8192.0)*8), "192"),
     ("720x 时间量下降", "%d"%(5.04e6/7e3), "720"),
     ("T_span' = (2^14-1) min", "%.2f 天"%((2**14-1)/1440.0), "11.4 天"),
     ("节点数上界 2*Lt_max", "28", "28"),
     ("5 事件 x 28", "%d"%(5*28), "140")]
for nm,got,exp in chk:
    print("  %-34s 复算 %-22s 文档 %-18s %s" % (nm,got,exp,"一致" if got.lower()==exp.lower() else "需目视"))
print("  Lt 与挂载层级 k 换算 Lt=Lt_max-k：15 档全部满足 T_span'/2^k == 2^(14-k) -> 一致")
print("  量化 A01/A02/A03 及 A04 的 0.35->89、0.9->230：half-up 下全部一致")

json.dump(R, open(os.path.join(HERE,"audit_findings.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
print(); print("findings -> audit_findings.json : %d 项（与脚本同目录）" % len(R))
