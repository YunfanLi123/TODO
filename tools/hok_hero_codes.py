"""英雄代码表 —— 英雄中文名 <-> 规则 id 里的稳定代码。

为什么要有这张表
----------------
英雄层规则的 id 必须能区分英雄，但攻略原文里只有中文名。
id 里塞中文会让 URL 和日志都不好看，而用「英雄在文件里的序号」
更糟：原文加一个英雄，后面所有英雄的序号全变，
已经入库的规则 id 就跟新生成的对不上，库里会同时留着两套。

所以这里维护一份**显式**的代码表：代码只跟英雄名绑定，
原文增删不影响已有英雄的 id。

代码取自英雄名拼音首字母，冲突时人工挑一个不冲突的写法
（表尾的 :func:`verify_hero_codes` 会守住唯一性）。

这张表同时给界面用：模式库页按英雄筛选、标注页的英雄下拉，
都要「英雄名 + 代码」这一对。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

#: 英雄名 -> 代码。
HERO_CODES: Dict[str, str] = {
    # -- 对抗路 --------------------------------------------------------
    "夏侯惇": "XHD",
    "吕布": "LV",
    "关羽": "GY",
    "马超": "MC",
    "铠": "KAI",
    "老夫子": "LFZ",
    "狂铁": "KT",
    "夏洛特": "XLT",
    "蒙恬": "MT",
    "曹操": "CC",
    "司空震": "SKZ",
    "芈月": "MY",
    "亚连": "YL",
    "姬小满": "JXM",
    "达摩": "DM",
    "花木兰": "HMQ",
    "哪吒": "NZ",
    "盘古": "PG",
    "杨戬": "YJ",
    "橘右京": "JRJ",
    "白起": "BQ",
    "廉颇": "LP",
    "项羽": "XY",
    "猪八戒": "ZBJ",
    "程咬金": "CJ",
    "刘禅": "LS",
    "钟无艳": "ZWY",
    "元歌": "YG",
    "兰陵王": "LLW",
    "大司命": "DSM",
    "墨子": "MZ",
    "刘邦": "LB",
    "牛魔": "NM",
    # -- 发育路 --------------------------------------------------------
    "百里守约": "BLYY",
    "敖隐": "AY",
    "后羿": "HY",
    "莱西奥": "LXA",
    "公孙离": "GSL",
    "戈娅": "GYA",
    "黄忠": "HZ",
    "伽罗": "GL",
    "虞姬": "YJI",
    "狄仁杰": "DRJ",
    "孙尚香": "SSX",
    "马可波罗": "MKBL",
    "蒙犽": "MYN",
    "鲁班七号": "LBQH",
    "李元芳": "LYF",
    "孙权": "SQ",
    "艾琳": "AL",
    "蚩奼": "CZ",
    "卢雅那": "LYN",
    "元射": "YS",
    "光信": "GX",
    "苍": "CANG",
    # -- 中路 ----------------------------------------------------------
    "海月": "HAIY",
    "王维": "WW",
    "大乔": "DQ",
    "甄姬": "ZJ",
    "小乔": "XQ",
    "沈梦溪": "SMX",
    "女娲": "NW",
    "嫦娥": "CE",
    "不知火舞": "BZHW",
    "西施": "XS",
    "貂蝉": "DC",
    "弈星": "YX",
    "上官婉儿": "SGWE",
    "杨玉环": "YYH",
    "诸葛亮": "ZGL",
    "武则天": "WZT",
    "扁鹊": "BQUE",
    "妲己": "DJ",
    "安琪拉": "AQL",
    "王昭君": "WZJ",
    "米莱狄": "MLD",
    "嬴政": "YZ",
    "干将莫邪": "GJMY",
    "周瑜": "ZY",
    "姜子牙": "JZY",
    "金蝉": "JC",
    "张良": "ZL",
    "海诺": "HN",
    # -- 打野 ----------------------------------------------------------
    "赵云": "ZYUN",
    "澜": "LAN",
    "娜可露露": "NKLL",
    "裴擒虎": "PQH",
    "镜": "JING",
    "阿古朵": "AGD",
    "孙悟空": "SWK",
    "韩信": "HX",
    "百里玄策": "BLXC",
    "露娜": "LN",
    "云缨": "YY",
    "李白": "LBAI",
    "曜": "YAO",
    "暃": "FEI",
    "六耳": "LUE",
    "宫本武藏": "GMWC",
    "司马懿": "SMY",
    "云中君": "YZJ",
    "梦奇": "MQ",
    "阿轲": "AK",
    # -- 游走 ----------------------------------------------------------
    "盾山": "DS",
    "鲁班大师": "LBDS",
    "元辅": "YF",
    "少司缘": "SSY",
    "苏烈": "SL",
    "太乙真人": "TYZR",
    "张飞": "ZF",
    "朵莉亚": "DLY",
    "孙膑": "SB",
    "钟馗": "ZK",
    "鬼谷子": "GGZ",
    "空空儿": "KKE",
    "赵怀真": "ZHZ",
    "大禹": "DAYU",
    "明世隐": "MSY",
    "东皇太一": "DHY",
    "蔡文姬": "CWJ",
    "瑶": "YAO2",
    "庄周": "ZZ",
    "桑启": "SQ2",
}


def hero_code(name: str) -> str:
    """英雄名 -> 代码。查不到就退回英雄名的十六进制摘要。

    退路不是好设计，但**必须有**：没有它的话，攻略里冒出一个
    新英雄，生成器就会崩在生成阶段 —— 那时候库里已经有一半规则
    写进去了。摘要至少能保证生成不中断，代码表再补上即可。
    """
    code = HERO_CODES.get(name)
    if code:
        return code
    return "H" + "".join(f"{ord(ch):02X}" for ch in name)[:8]


def verify_hero_codes() -> List[str]:
    """核对代码表：不能有重名代码，也不能有重名英雄。"""
    problems: List[str] = []
    seen: Dict[str, str] = {}
    for name, code in HERO_CODES.items():
        if code in seen:
            problems.append(f"代码重复：{code}（{seen[code]} 与 {name}）")
        seen[code] = name
        if not code.isascii() or not code.isupper():
            problems.append(f"代码必须是 ASCII 大写：{name} -> {code}")
    return problems


def hero_codes_by_name() -> List[Tuple[str, str]]:
    """按代码排序的 (英雄名, 代码) 列表。"""
    return sorted(HERO_CODES.items(), key=lambda kv: kv[1])


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    issues = verify_hero_codes()
    if issues:
        print("✗ 英雄代码表有问题：")
        for p in issues:
            print("  -", p)
        raise SystemExit(1)
    print(f"✓ 英雄代码表 {len(HERO_CODES)} 条，代码唯一")
