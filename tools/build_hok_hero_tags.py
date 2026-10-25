"""生成 ``app/seed/hok_hero_tags.json``（官方英雄属性表）。

用法：
    python tools/build_hok_hero_tags.py            # 生成
    python tools/build_hok_hero_tags.py --check    # 只校验

为什么单独一层
--------------
英雄属性是**游戏客观数据**（定位、分路），来自官方英雄资料；
操作模式是**攻略**（这一分钟该做什么），来自攻略原文。
两者的真源不同，链路也不同，所以生成器也分开 ——
免得哪天有人说「规则库是从官方接口生成的」，把两件事搅在一起。

生成产物仍然**不许手改**：``tests/test_hero_mode_library.py`` 里有测试
会重跑生成器逐条比对，手改立刻红。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from tools.hok_hero_tags import (          # noqa: E402
    TAGS_PATH,
    build_tags,
    load_raw,
    load_tags,
    verify_sources,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = ap.parse_args()

    issues = verify_sources()
    if issues:
        print("✗ 官方英雄资料自检失败：")
        for i in issues:
            print("  -", i)
        return 1

    raw = load_raw()
    tags = build_tags(raw)

    # 与攻略英雄名单对账：攻略里出现的英雄，官方表里都得认得。
    # 认不得说明两边对不上（改名？新英雄？），BP 规则会拿着不存在的英雄
    # 去算阵容，算出来的结论是假的。
    roster_path = os.path.join(ROOT, "app", "seed", "hok_heroes.json")
    missing: List[str] = []
    if os.path.exists(roster_path):
        with open(roster_path, "r", encoding="utf-8") as fh:
            roster = json.load(fh)
        for lane, block in (roster.get("lanes") or {}).items():
            for name in block.get("heroes") or []:
                if name not in tags:
                    missing.append(name)

    if missing:
        print(f"✗ 攻略里的 {len(missing)} 个英雄官方表里没有：{missing[:20]}")
        return 1

    if not args.check:
        with open(TAGS_PATH, "w", encoding="utf-8") as fh:
            json.dump(tags, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"✓ 写入 {TAGS_PATH}（{len(tags)} 个英雄）")
    else:
        old = load_tags()
        if old != tags:
            print("✗ 种子里的英雄属性表与官方数据对不上（重新生成）")
            return 1
        print(f"✓ 校验通过（{len(tags)} 个英雄）")
    return 0


from typing import List  # noqa: E402  （放在后面只是为了让上面的 import 块干净）


if __name__ == "__main__":
    raise SystemExit(main())
