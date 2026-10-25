"""视频链路的端到端校验：上传 → 扫描 → 报告 → 通读文案。

用法::

    python tools/verify_video_pipeline.py <录屏路径> [--base http://127.0.0.1:8850]

它做的是**真人录像**跑一遍完整链路，然后把报告页面上所有人能看到的
文字抠出来体检：

- 有没有英文词（游戏自己的说法 BUFF、C 位不算）；
- 有没有攻略内部的状态记号「状态【…】」；
- 有没有模糊词与空话（「尽量」「必要时」「稳扎稳打」…）；
- 有没有规则 ID 泄漏。

流程里**没有标注页**：分路、胜负、BP、每段操作都由模型从画面读，
扫完直接 finalize 出报告。

跑之前要先把服务起起来、并且已经载入规则库::

    .\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8850
    Invoke-WebRequest -Uri http://127.0.0.1:8850/api/games/hok/rules/seed -Method POST
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tools.hok_wording import lint  # noqa: E402

#: 游戏自己就在用的说法（王者界面里写的就是「红BUFF」「核心C位」），
#: 不算「英文标签」。
ALLOWED_WORDS = {"buff", "BUFF", "C", "CD"}

#: 扫描产出的每一段叫什么 —— MOBA 叫时间片，自走棋叫回合。
UNIT_BY_PRODUCT = {
    "moba_slice": "时间片",
    "phase": "阶段抽帧",
    "round": "回合",
}

#: 报告页里必须出现的四段（用户要的结构）。
REQUIRED_PARTS = (
    "对局时间区间",
    "原视频被建议部分片段",
    "原视频操作的文字描述",
    "根据模式库建议操作的描述",
)


def upload(client: httpx.Client, base: str, path: str) -> dict:
    """上传录像，等扫描完成，返回任务摘要。"""
    name = os.path.basename(path)
    with open(path, "rb") as fh:
        r = client.post(f"{base}/api/videos/upload",
                        files={"file": (name, fh, "video/mp4")},
                        data={"game": "hok"},
                        timeout=3600)
    r.raise_for_status()
    info = r.json()
    print(f"上传完成：{info['match_id']}（{info['message']}）")

    job_id = info["job_id"]
    import time

    last = ""
    while True:
        time.sleep(5)
        s = client.get(f"{base}/api/videos/jobs/{job_id}", timeout=60).json()
        line = f"{s.get('status')} {s.get('stage_text', '')} {s.get('progress', 0)}"
        if line != last:
            last = line
            print(f"  {line}")
        if s.get("status") in ("done", "failed", "error"):
            if s.get("status") != "done":
                raise SystemExit(f"扫描失败：{s.get('error')}")
            return s


def finalize(client: httpx.Client, base: str, match_id: str) -> dict:
    """扫完 → 建对局 → 出报告。

    **中间没有人工那一步**：分路、胜负、BP、每段操作都是模型从画面读的
    （用户原话：「我只要模型去读出报告，其余一切步骤都省去」）。
    读不出来的那几类记成缺失 → 内核判「无法判定」，**不回头让人填**。
    """
    r = client.post(f"{base}/api/videos/finalize",
                    json={"match_id": match_id}, timeout=600)
    r.raise_for_status()
    return r.json()


def visible_text(html: str) -> list:
    """把页面上人能看到的文字抠出来。"""
    body = re.sub(r"<script.*?</script>", " ", html, flags=re.S)
    body = re.sub(r"<style.*?</style>", " ", body, flags=re.S)
    text = re.sub(r"<[^>]+>", "\n", body)
    out = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) >= 4:
            out.append(line)
    return out


def audit(lines: list) -> list:
    """体检：返回问题清单。"""
    problems = []
    for line in lines:
        for w in re.findall(r"[A-Za-z_][A-Za-z0-9_\-]{1,}", line):
            if w in ALLOWED_WORDS:
                continue
            problems.append(f"英文词「{w}」：{line[:60]}")
        if "【" in line or "】" in line:
            problems.append(f"状态记号：{line[:60]}")
        for issue in lint(line):
            problems.append(f"{issue}：{line[:60]}")
        if re.search(r"HOK-(MODE|HERO|BP)-", line):
            problems.append(f"规则 ID 泄漏：{line[:60]}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="录屏文件路径")
    ap.add_argument("--base", default="http://127.0.0.1:8850")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print("找不到录像：", args.video)
        return 1

    with httpx.Client(timeout=httpx.Timeout(3600.0)) as client:
        job = upload(client, args.base, args.video)
        summary = job["summary"]
        match_id = summary["match_id"]

        print(f"\n产物类型：{summary.get('product')}"
              f"（{UNIT_BY_PRODUCT.get(summary.get('product'), '片段')}）")
        print(f"切片数：{summary.get('round_count')}")
        print(f"扫描说明：{(summary.get('scan') or {}).get('note')}")

        prefill = summary.get("prefill") or {}
        print(f"模型读出：分路={prefill.get('position') or '—'} "
              f"胜负={prefill.get('result') or '—'} 英雄={prefill.get('hero') or '—'}")

        data = finalize(client, args.base, match_id)
        print(f"\n判定统计：{json.dumps(data.get('stats'), ensure_ascii=False)}")
        print(f"结论条数：{data.get('gap_count')}")

        report_id = data.get("match_id") or match_id
        html = client.get(f"{args.base}/api/matches/{report_id}/report",
                          timeout=120).text
        lines = visible_text(html)

        print("\n===== 报告正文")
        for line in lines:
            print("   ", line[:110])

        missing = [p for p in REQUIRED_PARTS if p not in html]
        problems = audit(lines)
        print("\n===== 体检")
        if missing:
            for m in missing:
                print(f"   ✗ 缺少结构：{m}")
        else:
            print("   ✓ 四部分结构齐全")
        if problems:
            for p in problems[:20]:
                print("   ✗", p)
        else:
            print("   ✓ 没有英文词 / 状态记号 / 模糊词 / 规则 ID")
        return 1 if (missing or problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
