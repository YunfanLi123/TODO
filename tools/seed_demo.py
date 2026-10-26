"""把示例对局灌进**正在跑的服务**（走它自己的 API，路径和真实用户一致）。

用法::

    python tools/seed_demo.py http://127.0.0.1:8000

为什么要走 API 而不是直接写库：入库 + 分析这一步走的也是产品代码，
验的就是这条路径本身 —— 直接写库的话，种子库加载、规则匹配、
报告生成这三段都没被跑到。

王者会灌两局：一局**没填英雄**（判分路通则），一局**选了赵云**
（判英雄层专属规则）。两局摆在一起，能看出选了英雄之后结论
变得多具体。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo_data import build_demo_matches  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
GAME = sys.argv[2] if len(sys.argv) > 2 else "hok"


def _post(url: str, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print("HTTP 错误:", exc.code)
        print(exc.read().decode("utf-8", "replace")[:2000])
        raise SystemExit(1)


def main() -> int:
    # 先把规则库灌上 —— 没有规则库时入库会直接 409
    seed = _post(f"{BASE}/api/games/{GAME}/rules/seed", {})
    print(f"规则库：收 {seed.get('accepted_count', 0)} 条，"
          f"拒 {seed.get('rejected_count', 0)} 条，"
          f"退役 {len(seed.get('retired', []))} 条")

    for payload in build_demo_matches(GAME):
        out = _post(f"{BASE}/api/matches?player_id=local", payload)
        hero = payload.get("player", {}).get("hero") or "（未选）"
        print()
        print(f"match_id   : {out['match_id']}   英雄：{hero}")
        print(f"gap_count  : {out['gap_count']}")
        print(f"report_url : {out['report_url']}")
        print(f"stats      : {json.dumps(out.get('stats'), ensure_ascii=False)}")
        for f in (out.get("lanes") or {}).get("findings", []):
            print(f"  · [{f.get('mode_id')}] {f.get('title')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
