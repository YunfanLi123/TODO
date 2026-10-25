"""把**官方英雄定位表**下载入库（字节级），并生成 sha256 清单。

用法：python tools/import_hok_hero_tags.py

这份数据与攻略原文是两种东西
--------------------------
攻略原文（``hok_s45_hero_guides/``、``hok_s45_bp/``）是**操作模式**的唯一真源：
「这一分钟该做什么」只能从攻略来。

但**英雄本身的属性**（定位是坦克还是射手、能打哪几条线）是**游戏客观数据**，
官方王者官网的英雄资料里就有。它不是操作模式，也不是谁写出来的攻略，
所以单独一份来源、单独一份清单 —— 别把两件事混成一句「攻略是唯一真源」。

为什么下载而不是手抄
------------------
123 个英雄 × 定位 + 分路，手抄必然出错，而出错的后果是**静默失效**：
把镜的定位写成射手，BP 规则就会在不该触发的时候指控玩家。
官方接口是机器可读的，抄不错。

口径
----
``https://pvp.qq.com/web201605/js/herolist.json``（王者荣耀官网英雄资料）：

- ``cname`` 英雄名
- ``hero_type`` 定位：1 战士 / 2 法师 / 3 坦克 / 4 刺客 / 5 射手 / 6 辅助
- ``roles`` 推荐分路：1 对抗路 / 2 打野 / 3 中路 / 4 发育路 / 5 游走，
  多条用 ``|`` 分隔（能打两条以上分路的「摇摆位」就是这么认出来的）
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_DIR = os.path.join(ROOT, "app", "seed", "sources", "hok_hero_tags")
RAW_NAME = "herolist.json"
MANIFEST = "manifest.json"

URL = "https://pvp.qq.com/web201605/js/herolist.json"
#: 官方接口偶尔会挂，加个明显的 UA 与重试次数。
TIMEOUT = 30
RETRIES = 3


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> bytes:
    last = None
    for _ in range(RETRIES):
        try:
            req = urllib.request.Request(
                URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            return urllib.request.urlopen(req, timeout=TIMEOUT).read()
        except Exception as exc:            # noqa: BLE001
            last = exc
    raise SystemExit(f"✗ 官方英雄资料下载失败：{last}")


def main() -> int:
    os.makedirs(DEST_DIR, exist_ok=True)
    raw = download()

    dst = os.path.join(DEST_DIR, RAW_NAME)
    with open(dst, "wb") as fh:
        fh.write(raw)

    data = json.loads(raw.decode("utf-8"))
    manifest = {
        "source": "王者荣耀官网 · 英雄资料",
        "url": URL,
        "fetched_at": "2026-10-22",
        "note": (
            "英雄的客观属性（定位 / 推荐分路）。操作模式仍以攻略原文为唯一真源，"
            "这份数据只回答「这个英雄是什么定位、能打哪条线」。"
        ),
        "files": [
            {
                "file": RAW_NAME,
                "bytes": os.path.getsize(dst),
                "sha256": sha256_of(dst),
                "hero_count": len(data),
            }
        ],
    }
    with open(os.path.join(DEST_DIR, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"✓ {RAW_NAME}  {len(data)} 个英雄  "
          f"{os.path.getsize(dst)} bytes  sha256 {manifest['files'][0]['sha256'][:12]}…")
    print(f"✓ 清单写入 {os.path.join(DEST_DIR, MANIFEST)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
