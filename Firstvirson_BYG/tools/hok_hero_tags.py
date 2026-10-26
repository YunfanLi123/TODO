"""官方英雄属性表：**加载**、**自证**与**派生标记**。

这份表的定位
------------
它回答的是「这个英雄**是什么**」—— 定位是坦克还是射手、能打哪几条线。
它**不**回答「这一局该做什么」—— 那是攻略原文的活（见 §8）。

两者的分工不能混：操作模式只能从攻略原文来（唯一真源），
英雄属性是游戏客观数据，官方英雄资料里有，机器可读、抄不错。

派生标记
--------
``定位`` 与 ``推荐分路`` 是官方给的；下面这些是从它们**机械推导**的，
推导规则写死在这里，不靠人拍脑袋：

- ``output``   输出位 = 射手 / 法师 / 刺客（阵容完整性检查要数这个）
- ``flexible`` 摇摆位 = 推荐分路 >= 2 条（能打两条线，隐藏真实意图用）
- ``frontline`` 前排 = 坦克 / 战士
- ``damage_*`` 伤害类型 —— 官方资料里**没有**，不做（不猜）

刻意**不做**的标记：回复型、开团型、硬控、消耗型、连招型、体系归属。
这些是攻略里的**定性描述**，不是能查到的客观属性 ——
硬编一张表就是替玩家下判断，判错了玩家还找不到原因。
需要这些事实的 BP 规则，按项目惯例列入未入库清单并写明理由。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(ROOT, "app", "seed", "sources", "hok_hero_tags")
MANIFEST_PATH = os.path.join(SOURCE_DIR, "manifest.json")
RAW_PATH = os.path.join(SOURCE_DIR, "herolist.json")
TAGS_PATH = os.path.join(ROOT, "app", "seed", "hok_hero_tags.json")

#: 官方 ``hero_type`` -> 定位中文。
ROLE_LABELS: Dict[int, str] = {
    1: "战士",
    2: "法师",
    3: "坦克",
    4: "刺客",
    5: "射手",
    6: "辅助",
}

#: 官方 ``roles`` -> 分路中文。
LANE_LABELS: Dict[str, str] = {
    "1": "对抗路",
    "2": "打野",
    "3": "中路",
    "4": "发育路",
    "5": "游走",
}

#: 定位 -> 系统内部编码（写进事实值里）。
ROLE_CODES: Dict[str, str] = {
    "战士": "warrior",
    "法师": "mage",
    "坦克": "tank",
    "刺客": "assassin",
    "射手": "marksman",
    "辅助": "support",
}

#: 分路 -> 系统内部编码（与 ``app/adapters/honor_of_kings.py`` 的 position 一致）。
LANE_CODES: Dict[str, str] = {
    "对抗路": "side",
    "打野": "jungle",
    "中路": "mid",
    "发育路": "carry",
    "游走": "roam",
}

#: 算「输出位」的定位。
OUTPUT_ROLES = ("射手", "法师", "刺客")
#: 算「前排」的定位。
FRONTLINE_ROLES = ("坦克", "战士")

#: 攻略里的**简称** -> 官方全名。
#:
#: 攻略用简称、官方用全名，两边对不上就会「认不出这个英雄」——
#: 而认不出的后果是**静默失效**：算阵容时把它当空气，
#: 于是「己方有没有坦克」这类结论直接算错，还查不出原因。
#:
#: 这三条都对着两边的原文核过：
#:
#: - 六耳：官方叫「心魔六耳」（孙悟空命格，打野 / 对抗路）
#: - 元射：官方叫「元流之子(射手)」（射手，发育路）
#: - 元辅：官方叫「元流之子(辅助)」（辅助，游走）
NAME_ALIASES: Dict[str, str] = {
    "六耳": "心魔六耳",
    "元射": "元流之子(射手)",
    "元辅": "元流之子(辅助)",
}

#: 攻略里有、官方静态表里**还没有**的英雄。
#:
#: 官方那份 ``herolist.json`` 是静态快照，新英雄会晚一步。
#: 这类英雄的属性按**攻略原文的定位行**补上（来源标 ``guide``），
#: 而不是等官方表更新 —— 少一个英雄的定位，整套阵容的结论就少一块。
#:
#: 每条都要写清出处，不是拍脑袋填的。
GUIDE_ONLY_ROLES: Dict[str, Dict] = {
    # 攻略原文：「定位：走发育路的战士型英雄，一技能短蓄取霸体后接雾行千山……」
    "光信": {"role": "战士", "lanes": ["发育路"], "note": "攻略原文的定位行；官方静态表尚未收录"},
}


def load_manifest() -> Dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sources() -> List[str]:
    """核对原始数据的 sha256。返回问题列表，空列表表示一致。"""
    problems: List[str] = []
    manifest = load_manifest()
    for entry in manifest["files"]:
        path = os.path.join(SOURCE_DIR, entry["file"])
        if not os.path.exists(path):
            problems.append(f"官方英雄资料缺失：{entry['file']}")
            continue
        actual = sha256_of(path)
        if actual != entry["sha256"]:
            problems.append(
                f"官方英雄资料被改动：{entry['file']}"
                f"（清单 {entry['sha256'][:12]}… / 实际 {actual[:12]}…）"
            )
    return problems


def load_raw() -> List[Dict]:
    """官方原始英雄列表。"""
    with open(RAW_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_tags(raw: Optional[List[Dict]] = None) -> Dict[str, Dict]:
    """从官方原始数据派生每个英雄的标记。

    简称（``元射``）在表里**同时**留一份 —— 录 BP 时玩家打的是简称，
    认不出来就成了静默失效。
    """
    raw = raw if raw is not None else load_raw()
    out: Dict[str, Dict] = {}

    def make(name: str, role: str, lanes: List[str], title: str,
             source: str, note: str = "") -> Dict:
        return {
            "name": name,
            "title": title,
            "role": role,
            "role_code": ROLE_CODES.get(role, ""),
            "lanes": lanes,
            "lane_codes": [LANE_CODES[l] for l in lanes if l in LANE_CODES],
            # 派生标记（推导规则见模块文档）
            "output": role in OUTPUT_ROLES,
            "frontline": role in FRONTLINE_ROLES,
            "tank": role == "坦克",
            "mage": role == "法师",
            "assassin": role == "刺客",
            "marksman": role == "射手",
            "support": role == "辅助",
            "flexible": len(lanes) >= 2,
            "source": source,
            "note": note,
        }

    for h in raw:
        role = ROLE_LABELS.get(int(h.get("hero_type") or 0), "")
        lanes = [LANE_LABELS[c] for c in str(h.get("roles") or "").split("|") if c in LANE_LABELS]
        out[h["cname"]] = make(h["cname"], role, lanes, h.get("title") or "", "official")

    # 官方表里有、攻略用简称的：别名也留一份，指向同一份属性。
    for alias, full in NAME_ALIASES.items():
        if full in out:
            tag = dict(out[full])
            tag["note"] = f"攻略简称，官方名「{full}」"
            out[alias] = tag

    # 攻略里有、官方表还没有的：按攻略原文的定位行补。
    for name, spec in GUIDE_ONLY_ROLES.items():
        if name in out:
            continue
        out[name] = make(name, spec.get("role", ""), list(spec.get("lanes") or []),
                         "", "guide", spec.get("note", ""))

    return out



def load_tags() -> Dict[str, Dict]:
    """生成的英雄属性表（``app/seed/hok_hero_tags.json``）。"""
    with open(TAGS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def comp_summary(heroes: List[str], tags: Optional[Dict[str, Dict]] = None) -> Dict:
    """按一份英雄名单，算出阵容的结构性事实。

    给 BP 规则当触发条件用：己方/对方阵容里有没有坦克、有几个输出位、
    有没有摇摆位。

    认不出的英雄**不猜** —— 跳过并在 ``unknown`` 里列出来。
    """
    tags = tags if tags is not None else load_tags()
    out = {
        "count": 0,
        "tank": 0,
        "output": 0,
        "assassin": 0,
        "mage": 0,
        "marksman": 0,
        "support": 0,
        "frontline": 0,
        "flexible": 0,
        "unknown": [],
        "heroes": [],
    }
    for name in heroes:
        name = (name or "").strip()
        if not name:
            continue
        tag = tags.get(name)
        if tag is None:
            out["unknown"].append(name)
            continue
        out["count"] += 1
        out["heroes"].append(name)
        for key in ("tank", "output", "assassin", "mage", "marksman",
                    "support", "frontline", "flexible"):
            if tag.get(key):
                out[key] += 1
    return out


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    import sys

    issues = verify_sources()
    if issues:
        print("✗ 官方英雄资料自检失败：")
        for p in issues:
            print("  -", p)
        raise SystemExit(1)
    tags = build_tags()
    print(f"✓ 官方英雄资料 sha256 一致，派生 {len(tags)} 个英雄的标记")
