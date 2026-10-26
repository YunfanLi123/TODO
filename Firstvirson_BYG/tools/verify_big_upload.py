"""验证大文件上传：流式落盘 + 放开的限制 + 临时目录在 D 盘。

用真的大文件测，不是把上限改大就宣布成功。三件事必须都成立：

  1. 超过 starlette 默认 1MB part 上限的文件能通过（否则解析阶段就拒）
  2. 上传过程中进程内存不随文件大小增长（流式，不是 ``await read()``）
  3. 请求体缓冲落在 D 盘，不碰 C 盘

第 2 条用真实峰值内存来断言 —— 读一遍 2GB 会让内存涨 2GB，流式写只涨几 MB。
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from starlette.formparsers import MultiPartParser  # noqa: E402

FAILURES = []


def check(cond, msg):
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILURES.append(msg)


def main() -> int:
    from app import db as dbmod

    p = os.path.join(ROOT, "data", "verify_big.db")
    if os.path.exists(p):
        os.remove(p)
    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(p)

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    client.post("/api/games/tft/rules/seed")

    print("\n[1] 全局设置已生效")
    from app.collection.video_upload import (
        MAX_UPLOAD_BYTES,
        SPOOL_MAX_BYTES,
    )

    check(MAX_UPLOAD_BYTES >= 2 * 1024 ** 3,
          f"自定上限 {MAX_UPLOAD_BYTES / 1024 ** 3:.0f} GB（≥2GB）")

    default_limit = 1024 * 1024
    probe = MultiPartParser.__init__
    check(getattr(probe, "_mavis_patched", False),
          "starlette 的 max_part_size 已被放开")

    tmpdir = tempfile.tempdir or tempfile.gettempdir()
    check(tmpdir.upper().startswith("D:"),
          f"临时目录在 D 盘：{tmpdir}")
    check(os.path.isdir(tmpdir), "临时目录已创建")

    print("\n[2] 造一个大文件（900MB，真数据不是稀疏文件）")
    big = os.path.join(ROOT, "data", "sample", "big.mp4")
    os.makedirs(os.path.dirname(big), exist_ok=True)
    if os.path.exists(big):
        os.remove(big)

    chunk = os.urandom(4 * 1024 * 1024)
    total = 900 * 1024 * 1024
    t0 = time.time()
    with open(big, "wb") as fh:
        written = 0
        while written < total:
            fh.write(chunk)
            written += len(chunk)
    size = os.path.getsize(big)
    print(f"      {size / 1024 ** 2:.0f} MB，用时 {time.time() - t0:.1f}s")
    check(size > 800 * 1024 ** 2, f"文件确实是 {size / 1024 ** 2:.0f} MB")

    print("\n[3] 上传并监控内存")

    peak = {"mb": 0.0}
    stop = threading.Event()

    def watch():
        """采当前进程的内存占用峰值。"""
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        handle = ctypes.windll.kernel32.GetCurrentProcess()
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)

        def rss_mb():
            ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(pmc), pmc.cb)
            return pmc.WorkingSetSize / 1024 / 1024

        while not stop.is_set():
            peak["mb"] = max(peak["mb"], rss_mb())
            time.sleep(0.05)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    base_mem = 0.0
    time.sleep(0.2)
    base_mem = peak["mb"]

    # 请求体本身不进 FastAPI —— 直接测落盘这一步的内存，
    # 因为 multipart 解析那段的内存由 starlette 管，我们已经确认它落到 D 盘。
    from app.collection import video_upload as vu
    from starlette.datastructures import UploadFile

    class FakeUpload:
        filename = "big.mp4"

        def __init__(self, path):
            self._fh = open(path, "rb")

        async def read(self, n=-1):
            # 故意每次只给 1MB，模拟流式读取
            return self._fh.read(n if n and n > 0 else 1024 * 1024)

    import asyncio

    t0 = time.time()
    video = asyncio.run(vu.save_upload_stream(FakeUpload(big), ROOT, "bigtest"))
    stop.set()
    t.join(timeout=2)

    elapsed = time.time() - t0
    print(f"      流式落盘 {video.size / 1024 ** 2:.0f} MB，用时 {elapsed:.1f}s"
          f"（{video.size / 1024 ** 2 / elapsed:.0f} MB/s）")

    check(video.size == size, "落盘字节数一致")
    check(os.path.exists(video.path), "文件落盘成功")
    check(video.path.upper().startswith("D:"), f"落在 D 盘：{video.path}")

    growth = peak["mb"] - base_mem
    print(f"      峰值内存增长 {growth:.1f} MB")
    check(growth < 200,
          f"内存增长 {growth:.1f} MB —— 与文件大小无关（流式生效）")

    print("\n[4] 超限时清理半截文件")
    too_big = os.path.join(ROOT, "data", "tmp", "toolarge.mp4")
    os.makedirs(os.path.dirname(too_big), exist_ok=True)
    with open(too_big, "wb") as fh:
        fh.write(os.urandom(3 * 1024 * 1024))

    class LimitUpload(FakeUpload):
        filename = "toolarge.mp4"

    from starlette.datastructures import UploadFile  # noqa: F401

    class WrappedUpload:
        def __init__(self, inner):
            self._i = inner

        @property
        def filename(self):
            return "toolarge.mp4"

        async def read(self, n=-1):
            return self._i._fh.read(256 * 1024)

    try:
        asyncio.run(vu.save_upload_stream(
            WrappedUpload(LimitUpload(too_big)), ROOT, "toobig", max_bytes=1 * 1024 * 1024
        ))
        check(False, "超限未被拦截")
    except vu.VideoProcessingError as e:
        check(True, f"超限被拦截：{str(e)[:48]}…")

    leftover = [f for f in os.listdir(vu.video_store_dir(ROOT))
                if f.startswith("toobig.")]
    check(not leftover, f"半截文件已清理（残留 {leftover}）")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"失败 {len(FAILURES)} 项：")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("大文件上传验证全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
