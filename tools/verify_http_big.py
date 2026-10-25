"""真实 HTTP 上传一个超过 500MB 的文件，确认限制真的放开了。

只测后端的 ``save_upload_stream`` 是不够的 —— 那跳过了 multipart 解析，
而 starlette 的 ``max_part_size`` 恰恰卡在这一层。必须走完整的 HTTP 链路
才能证明「真能传」而不只是「代码里写着能传」。
"""

from __future__ import annotations

import io
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAILURES = []


def check(cond, msg):
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILURES.append(msg)


def main() -> int:
    # 必须先 import app.main —— 放开 multipart 上限的正是在那里执行的。
    # 先检查再导入会看到「没放开」的假象（踩过）。
    from app import db as dbmod

    p = os.path.join(ROOT, "data", "verify_http_big.db")
    if os.path.exists(p):
        os.remove(p)
    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(p)

    from fastapi.testclient import TestClient

    from app.main import app
    from starlette.formparsers import MultiPartParser

    from app.collection.video_upload import MAX_UPLOAD_BYTES

    print("\n[0] 检查限制配置（已 import app.main）")
    print(f"      MAX_UPLOAD_BYTES = {MAX_UPLOAD_BYTES / 1024 ** 3:.0f} GB")
    check(MAX_UPLOAD_BYTES >= 2 * 1024 ** 3, "代码里的上限 ≥ 2GB")
    check(getattr(MultiPartParser.__init__, "_mavis_patched", False),
          "starlette 的 max_part_size 已放开")

    client = TestClient(app)
    client.post("/api/games/tft/rules/seed")

    print("\n[1] 造一个 >500MB 的**有效** mp4")
    # 不能用随机字节 —— 那不是视频，OpenCV 打不开，只能验到「过了解析层」
    # 这一层，验不到「能真的传完并落盘」。
    # 做法：先生成一段有效 mp4，再在**末尾**追加垃圾字节。
    # mp4 的 moov 在开头，尾部追加不影响读取。
    big = os.path.join(ROOT, "data", "sample", "http600.mp4")
    os.makedirs(os.path.dirname(big), exist_ok=True)
    if os.path.exists(big):
        os.remove(big)

    import sys as _s
    _s.path.insert(0, ROOT)
    from make_sample_video import build as build_small

    build_small(big)                       # 有效视频
    valid_size = os.path.getsize(big)
    pad = 620 * 1024 * 1024 - valid_size
    with open(big, "ab") as fh:            # 尾部填充到 620MB
        block = os.urandom(8 * 1024 * 1024)
        left = pad
        while left > 0:
            fh.write(block[: min(len(block), left)])
            left -= min(len(block), left)
    size = os.path.getsize(big)
    print(f"      有效视频 {valid_size / 1024:.0f} KB + 尾部填充 = "
          f"{size / 1024 ** 2:.0f} MB")
    check(size > 500 * 1024 ** 2,
          f"文件确实超过旧的 500MB 限制（{size / 1024 ** 2:.0f} MB）")

    print("\n[2] 走完整 HTTP 上传链路")
    # 用 hok，不用 tft：tft 的种子资料已删（模块留着，资料后补），
    # 没有操作模式库时上传接口会直接 409 —— 那是设计上的正确行为，
    # 但会让这个脚本测不到「大文件上传」本身。
    client.post("/api/games/hok/rules/seed")
    t0 = time.time()
    with open(big, "rb") as fh:
        r = client.post(
            "/api/videos/upload",
            files={"file": ("http600.mp4", fh, "video/mp4")},
            data={"game": "hok"},
        )
    dt = time.time() - t0
    print(f"      HTTP {r.status_code} · {dt:.1f}s · "
          f"{size / 1024 ** 2 / dt:.0f} MB/s")

    if r.status_code == 413:
        check(False, "被 starlette 拒绝（413）—— max_part_size 没放开")
        return 1
    if r.status_code != 200:
        check(False, f"上传失败（{r.status_code}）：{r.text[:200]}")
        return 1

    data = r.json()
    check(True, f"上传成功，返回任务号 {data.get('job_id')}")
    check(data["video"]["size"] > 500 * 1024 ** 2,
          f"落盘大小 {data['video']['size'] / 1024 ** 2:.0f} MB 完整保留")
    check(data["video"]["size"] == size, "字节数完全一致，没有截断")
    check(data["video"]["duration_sec"] > 0,
          f"OpenCV 能打开（时长 {data['video']['duration_sec']:.1f}s）"
          f" —— 证明字节没被破坏")

    print("\n[3] 超上限时会被拒绝")
    # 用一个很小的 max_bytes 模拟超限，确认报错清楚
    from app.collection import video_upload as vu

    import asyncio

    class FakeUpload:
        filename = "x.mp4"

        def __init__(self, path):
            self._fh = open(path, "rb")

        async def read(self, n=-1):
            return self._fh.read(1024 * 1024)

    try:
        asyncio.run(vu.save_upload_stream(
            FakeUpload(big), ROOT, "toolimit", max_bytes=100 * 1024 * 1024
        ))
        check(False, "超限未被拦截")
    except vu.VideoProcessingError as e:
        check("超过上限" in str(e), f"超限报错清楚：{str(e)[:60]}…")

    leftovers = [f for f in os.listdir(vu.video_store_dir(ROOT))
                 if f.startswith("toolimit.")]
    check(not leftovers, f"半截文件已清理（残留 {leftovers}）")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"失败 {len(FAILURES)} 项：")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("大文件 HTTP 上传验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
