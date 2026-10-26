# -*- coding: utf-8 -*-
"""五份英雄决策树独立复核（修正口径后）。
检查：体积 / 英雄覆盖 / 节点数与顺序 / 分支三分行完整 / 状态标签定义与复用 /
      ASCII markdown 符号 / 评价性词 / 不可观测前提 / 标签链跨节点引用。
"""
import re, os, json

BASE = r"<本机目录>"

LANES = {
    "01-clash-lane-heroes.txt": ["夏侯惇","吕布","关羽","马超","铠","老夫子","狂铁","夏洛特","蒙恬","曹操",
        "司空震","芈月","亚连","姬小满","达摩","花木兰","哪吒","盘古","杨戬","橘右京","白起","廉颇",
        "项羽","猪八戒","程咬金","刘禅","钟无艳","元歌","兰陵王","大司命","墨子","刘邦","牛魔"],
    "02-farm-lane-heroes.txt": ["百里守约","敖隐","后羿","莱西奥","公孙离","戈娅","黄忠","伽罗","虞姬",
        "狄仁杰","孙尚香","马可波罗","蒙犽","鲁班七号","李元芳","孙权","艾琳","蚩奼","卢雅那","元射",
        "光信","苍"],
    "03-mid-lane-heroes.txt": ["海月","王维","大乔","甄姬","小乔","沈梦溪","女娲","嫦娥","不知火舞","西施",
        "貂蝉","弈星","上官婉儿","杨玉环","诸葛亮","武则天","扁鹊","妲己","安琪拉","王昭君","米莱狄",
        "嬴政","干将莫邪","周瑜","姜子牙","金蝉","张良","海诺","司空震"],
    "04-jungle-lane-heroes.txt": ["赵云","澜","娜可露露","裴擒虎","镜","橘右京","阿古朵","孙悟空","韩信",
        "百里玄策","露娜","云缨","铠","李白","曜","暃","大司命","六耳","马超","宫本武藏","杨戬","曹操",
        "司马懿","盘古","哪吒","云中君","梦奇","阿轲","司空震","苍","周瑜","夏侯惇"],
    "05-roam-lane-heroes.txt": ["盾山","鲁班大师","元辅","少司缘","苏烈","牛魔","太乙真人","张飞","大乔",
        "墨子","张良","东皇太一","蔡文姬","瑶","庄周","廉颇","钟馗","刘禅","朵莉亚","孙膑","金蝉",
        "鬼谷子","空空儿","赵怀真","刘邦","杨玉环","桑启","大禹","明世隐","甄姬","弈星"],
}

NODES = ["0:00 节点", "2:00 节点", "4:00 节点", "6:00 节点", "10:00 节点", "12:00 之后"]

# 真正的主观评价词（机制术语如『无敌帧』不计入）
BANNED = ["最强", "必选", "版本答案", "最重要", "划算", "很赚", "碾压", "无脑", "闭眼选",
          "完胜", "必ban", "必 Ban", "神级", "垃圾", "轻松", "随意", "吊打", "血赚"]
# 不可观测 / 结果类前提用词
UNOBS = ["劣势", "领先", "经济差", "等级差", "评分", "局势", "占优", "节奏好", "逆风", "顺风", "崩盘"]

report = {}
all_ok = True
for fname, heroes in LANES.items():
    p = os.path.join(BASE, fname)
    raw = open(p, "rb").read()
    text = raw.decode("utf-8")
    lines = text.split("\n")
    prob = []

    if raw.startswith(b"\xef\xbb\xbf"):
        prob.append("含 BOM")
    if "\r\n" in text:
        prob.append("含 CRLF")

    md = [(i + 1, l) for i, l in enumerate(lines)
          if re.match(r"^\s*[|#*>`]", l) or re.search(r"[`]", l)]
    if md:
        prob.append("ASCII markdown 符号 %d 处（首例 L%d）" % (len(md), md[0][0]))

    bh = [(i + 1, w, l.strip()) for i, l in enumerate(lines) for w in BANNED if w in l]
    if bh:
        prob.append("主观评价词 %d 处（首例 L%d [%s]）" % (len(bh), bh[0][0], bh[0][1]))

    un = [(i + 1, l.strip()) for i, l in enumerate(lines)
          if "前提：" in l and any(w in l for w in UNOBS)]
    if un:
        prob.append("不可观测前提 %d 处（首例 L%d: %s）" % (len(un), un[0][0], un[0][1][:60]))

    node_idx = [i for i, l in enumerate(lines) if l.strip() in NODES]
    if len(node_idx) != len(heroes) * 6:
        prob.append("节点数 %d ≠ %d" % (len(node_idx), len(heroes) * 6))
    for g in range(len(node_idx) // 6):
        seq = [lines[node_idx[g * 6 + k]].strip() for k in range(6)]
        if seq != NODES:
            prob.append("第 %d 组节点顺序错: %s" % (g, seq))
            break

    hero_cnt = {h: sum(1 for l in lines if l.strip() == h) for h in heroes}
    miss = [h for h, v in hero_cnt.items() if v == 0]
    dup = [h for h, v in hero_cnt.items() if v > 1]
    if miss:
        prob.append("缺失英雄: " + ",".join(miss))
    if dup:
        prob.append("重复英雄: " + ",".join(dup))

    mal = no_tag = 0
    def is_cont(l):
        # 续行：以 │ 开头（可带缩进）或以 3 个以上空格开头
        return re.match(r"^\s*│", l) or re.match(r"^\s{3,}\S", l)
    for i, l in enumerate(lines):
        if re.match(r"^\s*[├└]─\s*前提：", l):
            blk, j = [], i + 1
            while j < len(lines) and lines[j].strip() and is_cont(lines[j]) \
                    and not re.match(r"^\s*\d+:\d\d", lines[j]):
                blk.append(lines[j]); j += 1
            body = "\n".join(blk)
            if "动作：" not in body or "后果：" not in body:
                mal += 1
            cons = [b for b in blk if "后果：" in b]
            if not cons or not re.search(r"【[^】]+】\s*$", cons[-1].strip()):
                no_tag += 1
    if mal:
        prob.append("三分支不完整 %d 个" % mal)
    if no_tag:
        prob.append("后果缺状态标签 %d 个" % no_tag)

    defined = set(re.findall(r"后果：.*?【([^】]+)】", text))
    used = set(re.findall(r"前提：【([^】]+)】", text))
    if used - defined:
        prob.append("引用未定义标签 %d 个" % len(used - defined))

    # 标签链：每个节点产出的标签是否在下一节点被引用
    chain_ok = 0
    chain_total = 0
    n_groups = len(node_idx) // 6
    for g in range(n_groups - 1):
        cur = "\n".join(lines[node_idx[g * 6]:node_idx[g * 6 + 6]])
        end = node_idx[g * 6 + 12] if g * 6 + 12 < len(node_idx) else len(lines)
        nxt = "\n".join(lines[node_idx[g * 6 + 6]:end])
        tags = set(re.findall(r"后果：.*?【([^】]+)】", cur))
        refs = set(re.findall(r"前提：【([^】]+)】", nxt))
        chain_total += 1
        if tags & refs:
            chain_ok += 1

    report[fname] = {
        "bytes": len(raw), "approx_pages": round(len(raw) / 2000.0, 1), "lines": len(lines),
        "heroes": "%d/%d" % (sum(1 for v in hero_cnt.values() if v == 1), len(heroes)),
        "nodes": len(node_idx), "tags_defined": len(defined), "tags_used": len(used),
        "node_chain_reuse": "%d/%d" % (chain_ok, chain_total),
        "problems": prob,
    }
    if prob:
        all_ok = False

out = []
for k, v in report.items():
    out.append("%-28s %7d B  %6.1f 页  英雄 %s  节点 %3d  标签 %3d/%3d  节点链 %s"
               % (k, v["bytes"], v["approx_pages"], v["heroes"], v["nodes"],
                  v["tags_used"], v["tags_defined"], v["node_chain_reuse"]))
    for p in v["problems"]:
        out.append("      ! " + p)
tot = sum(v["bytes"] for v in report.values())
out.append("-" * 78)
out.append("合计 %d 字节，约 %.1f 页（按 2000 字节/页）" % (tot, tot / 2000.0))
out.append("复核结论：%s" % ("全部通过" if all_ok else "存在问题，见上"))
txt = "\n".join(out)
open(os.path.join(BASE, "verify-all-lanes.txt"), "w", encoding="utf-8").write(txt)
print(txt)
