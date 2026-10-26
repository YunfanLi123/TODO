"""视频上传与回合门控。

这一层做的是整条链路里**唯一能真正自动化**的那部分：

    上传视频 → 逐帧扫描 → 读时钟定时间线 → 按信息密度切时间片 → 抽关键帧 → 产出识别骨架

为什么只做这部分
--------------
玩家的**操作**（这一回合是存钱还是追星）从画面上几乎读不出来 —— 存钱就是
「什么都没点」，追星是「点升级再点两次」。这两者在几帧画面里的差别微乎其微，
靠视觉猜出来的动作会十有八九是错的，而规则的 ``expect`` 判定的恰恰就是动作。
猜错动作会直接得出「你该存钱」这种颠倒的结论，比不判更糟。

所以分工是：

  系统负责：**回合怎么切、每回合在第几秒、血量这种纯视觉量**
  玩家负责：**我当时做了什么**

这不是妥协。手工录 20 局对局要几小时，上传视频再逐回合勾选，
从几小时降到十几分钟 —— 而「我当时做了什么」本来也只有玩家知道。

血条可以直接读
--------------
血量是 HUD 上的一条填充条，**填充比例就是血量百分比**，像素级就能算准，
不需要 OCR、不需要模型。这是这套方案里唯一能给出高置信度自动值的量。
金币和等级是数字，没有 OCR 引擎就诚实地标成待补，而不是猜一个。
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import clock
from . import match_sync

from .video import (
    DetectedAction,
    DetectedFact,
    RecognitionResult,
    RecognizedRound,
)


class VideoProcessingError(RuntimeError):
    pass


#: 上传体积上限。
#:
#: 一局对局的录屏动辄 1~2 GB（1080p / 30fps / 40 分钟），500 MB 那个上限
#: 等于大部分真实录屏都传不进来。
#:
#: 但真正卡住上传的其实不是这里的数字 —— 是 starlette 的
#: ``max_part_size`` 默认只有 **1 MB**，它会在请求解析阶段就把大文件拒掉，
#: 根本走不到这个检查。两者都要放开，见 :func:`raise_multipart_limit`。
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024      # 4 GB

#: 真正开始用磁盘之前，内存里最多缓冲这么多。
#: 超过它 starlette 才把请求体落到临时文件 —— 所以临时目录必须在 D 盘。
SPOOL_MAX_BYTES = 1024 * 1024                  # 1 MB

ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv"}

#: 流式落盘的块大小。1 MB 足够，又不会让单次 read 分配过大内存。
CHUNK_BYTES = 1024 * 1024


def raise_multipart_limit(limit_bytes: int = MAX_UPLOAD_BYTES) -> None:
    """放开 starlette 的单 part 上限，并把临时目录挪到 D 盘。

    这两件事都必须做，且都容易被忽略：

    1. starlette 的 ``max_part_size`` 默认 **1 MB**。设成 2 GB 上限却不管它，
       请求会在解析阶段就被拒，用户看到的错误还特别莫名其妙。
    2. 请求体超过 :data:`SPOOL_MAX_BYTES` 后会被写到 ``tempfile`` 指定的目录。
       Windows 上那是 ``<本机目录><user>\\AppData\\Local\\Temp`` ——
       一个只想在 D 盘工作的项目，会因为上传视频而写满 C 盘。
    """
    import os as _os
    import tempfile

    from starlette.formparsers import MultiPartParser

    original = MultiPartParser.__init__

    def patched(self, headers, stream, *args, **kwargs):
        kwargs.setdefault("max_part_size", limit_bytes)
        original(self, headers, stream, *args, **kwargs)

    if not getattr(MultiPartParser.__init__, "_mavis_patched", False):
        MultiPartParser.__init__ = patched
        MultiPartParser.__init__._mavis_patched = True


def redirect_tempdir(base_dir: str) -> str:
    """把 Python 的临时目录指到用户数据目录下。

    只影响本进程。对这个项目来说这是对的行为 —— 视频上传是最吃磁盘的
    操作，让它落在系统盘既没道理也容易出事。
    """
    import tempfile

    target = os.path.join(data_root(base_dir), "tmp")
    os.makedirs(target, exist_ok=True)
    tempfile.tempdir = target
    return target

#: 扫描的采样间隔（秒），不是帧数。
#:
#: 按时间采样是对的单位：一局王者十几分钟，按 0.3 秒采一次就够，
#: 采样量直接决定耗时（2GB 的录屏扫一遍要几分钟）。
GATE_SAMPLE_SEC = 0.3


def temp_dir_hint(base_dir: str) -> str:
    """当前临时目录的位置 —— 用于在接口里如实告知，不藏着。"""
    import tempfile

    return tempfile.tempdir or tempfile.gettempdir()


@dataclass
class ScanConfig:
    """扫描参数。

    只剩采样间隔 —— 切分单位是**时间片**（王者的一局是连续的一整场，
    没有「回合」可切），所以不需要「边界阈值 / 最短回合间隔」那一套。
    """

    #: 采样间隔（秒）
    sample_sec: float = GATE_SAMPLE_SEC


def adapt_sample_sec(duration_sec: float) -> float:
    """按时长调整采样间隔。

    短视频可以采得密一点（找边界更准），长视频必须放宽，
    否则一个 2GB 的录屏能扫好几分钟。这个放宽是**有依据**的：
    回合切换本来就是低频事件，采样再稀也漏不掉边界，
    但采样量直接决定了耗时。
    """
    if duration_sec <= 0:
        return GATE_SAMPLE_SEC
    if duration_sec <= 300:          # 5 分钟以内
        return 0.15
    if duration_sec <= 1800:         # 30 分钟以内
        return GATE_SAMPLE_SEC
    if duration_sec <= 3600:
        return 0.5
    return 0.8


@dataclass
class UploadedVideo:
    """已落盘的视频。"""

    video_id: str
    path: str
    original_name: str
    size: int
    duration_sec: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    stored_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "original_name": self.original_name,
            "size": self.size,
            "size_mb": round(self.size / 1024 / 1024, 1),
            "duration_sec": round(self.duration_sec, 1),
            "duration_min": round(self.duration_sec / 60, 1),
            "fps": round(self.fps, 1),
            "resolution": f"{self.width}x{self.height}",
            "frame_count": self.frame_count,
            "stored_at": self.stored_at,
        }


# --------------------------------------------------------------------------
# 存储
# --------------------------------------------------------------------------


def data_root(base_dir: str = "") -> str:
    """所有用户数据的根目录。

    优先用 ``MATCH_DATA_DIR``（exe 启动器会把它指到 exe 旁边的
    ``MatchAnalyzerData``），这样数据库、视频、临时文件**全在一处** ——
    分散在两个目录里，用户备份时会漏掉一半。
    """
    env = os.environ.get("MATCH_DATA_DIR")
    if env:
        return env
    return base_dir or os.getcwd()
def video_store_dir(base_dir: str = "") -> str:
    d = os.path.join(data_root(base_dir), "videos")
    os.makedirs(d, exist_ok=True)
    return d


def _target_path(base_dir: str, video_id: str, filename: str) -> str:
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise VideoProcessingError(
            f"不支持的文件格式 {suffix or '(无扩展名)'}，"
            f"支持：{', '.join(sorted(ALLOWED_SUFFIXES))}"
        )
    return os.path.join(video_store_dir(base_dir), video_id + suffix)


async def save_upload_stream(
    upload, base_dir: str, video_id: str, max_bytes: int = MAX_UPLOAD_BYTES
) -> UploadedVideo:
    """把上传流**边读边写**到磁盘，不整份进内存。

    为什么必须流式：2 GB 的录屏用 ``await file.read()`` 读进来，
    光这一个变量就能把进程内存吃穿，然后被系统直接杀掉 ——
    表现是「上传到一半就没反应了」，而不是一个像样的报错。
    分块写还有个附带好处：超限时能立刻删掉已写的部分，不留半截垃圾文件。

    ``upload`` 是 FastAPI 的 ``UploadFile``，只需 ``read`` / ``filename``。
    """
    filename = getattr(upload, "filename", "") or "video.mp4"
    path = _target_path(base_dir, video_id, filename)

    written = 0
    try:
        with open(path, "wb") as out:
            while True:
                chunk = await upload.read(CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise VideoProcessingError(
                        f"视频超过上限 {max_bytes / 1024 ** 3:.1f} GB，"
                        f"已传 {written / 1024 ** 3:.2f} GB。"
                        f"可以截取对局片段后再上传。"
                    )
                out.write(chunk)
    except Exception:
        # 半截文件必须清掉 —— 留着既占磁盘，下次列目录还会误导人
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        raise

    if written == 0:
        os.remove(path)
        raise VideoProcessingError("上传内容为空")

    return UploadedVideo(
        video_id=video_id,
        path=path,
        original_name=filename,
        size=written,
        stored_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def save_upload(
    filename: str, data: bytes, base_dir: str, video_id: str
) -> UploadedVideo:
    """内存版落盘，只用于测试与小文件（合成视频、样例）。"""
    path = _target_path(base_dir, video_id, filename)

    if len(data) > MAX_UPLOAD_BYTES:
        raise VideoProcessingError(
            f"视频 {len(data) / 1024 ** 3:.2f} GB 超过上限 "
            f"{MAX_UPLOAD_BYTES / 1024 ** 3:.1f} GB"
        )
    if not data:
        raise VideoProcessingError("上传内容为空")

    with open(path, "wb") as fh:
        fh.write(data)

    return UploadedVideo(
        video_id=video_id,
        path=path,
        original_name=filename,
        size=len(data),
        stored_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def load_video(base_dir: str, video_id: str) -> UploadedVideo:
    d = video_store_dir(base_dir)
    for f in os.listdir(d):
        if f.startswith(video_id + "."):
            path = os.path.join(d, f)
            return UploadedVideo(
                video_id=video_id,
                path=path,
                original_name=f,
                size=os.path.getsize(path),
                stored_at=datetime.fromtimestamp(os.path.getmtime(path))
                .strftime("%Y-%m-%d %H:%M"),
            )
    raise VideoProcessingError(f"视频 {video_id} 不存在")


def delete_video(base_dir: str, video_id: str) -> bool:
    d = video_store_dir(base_dir)
    removed = False
    for f in os.listdir(d):
        if f.startswith(video_id + ".") or f.startswith(video_id + "_"):
            target = os.path.join(d, f)
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            else:
                os.remove(target)
            removed = True
    return removed


# --------------------------------------------------------------------------
# 视频探针与回合门控
# --------------------------------------------------------------------------


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:                # pragma: no cover
        raise VideoProcessingError(
            "缺少 opencv-python-headless。请执行：\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install opencv-python-headless numpy"
        ) from exc
    return cv2


def imwrite_u(path: str, img, quality: int = 85) -> bool:
    """Unicode 安全的写图。

    必须用它而不是 ``cv2.imwrite``：OpenCV 在 Windows 上走系统 ANSI 接口，
    路径里有中文（本项目就放在 ``D:\\大三上\\...``）时，``imwrite`` 不报错、
    只在控制台打一条 ``can't open file for writing: permission denied`` 然后
    **静默失败** —— 文件根本没落盘。随后读取方全部读空，表现为
    「模型一窗画面都没收到、每段都没有操作结论、报告缩略图黑屏」，
    而界面上没有任何错误提示。

    改成先在内存编码成 JPEG 字节，再用 Python 的 ``open`` 写 ——
    Python 的文件 API 对 Unicode 路径是原生支持的。
    """
    import numpy as np

    cv2 = _require_cv2()
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return False
    with open(path, "wb") as fh:
        fh.write(np.asarray(buf).tobytes())
    return True


def imread_u(path: str, flags: int = 1):
    """Unicode 安全的读图（对应 :func:`imwrite_u`）。

    ``cv2.imread`` 同样在中文路径下读不到、返回 ``None``。
    用 Python 读出字节再 ``cv2.imdecode`` 解码，绕开 ANSI 路径限制。
    """
    import numpy as np

    cv2 = _require_cv2()
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


# --------------------------------------------------------------------------
# 取景归一：旋转 + 裁黑边
# --------------------------------------------------------------------------
#
# 手机录屏有两层「摆不正」的问题，两层都要在分析前解决，
# 否则后面全是错的，而且错得很隐蔽：
#
# 1. **容器里的旋转元数据**。手机横过来录，容器会记一句
#    「显示时要转 90°」（orientation=90）。OpenCV 的 FFmpeg 后端
#    **不转** —— 解出来的每一帧都是躺着的。后果：
#    - 报告里的截图没人看得懂；
#    - 时钟模板永远对不上（槽位是按画面方向标定的），于是时间线整个读不出来；
#    - 回合边界的灰度差也被稀释（画面里一大半是黑的）。
# 2. **黑边**。竖屏录屏里游戏只占中间一条，其余是黑的。
#    同样稀释灰度差，回合边界几乎检不出来。
#
# 所以这里按容器元数据把帧转正，再把「整段视频里始终是黑的」那部分裁掉。
# 只裁**始终**黑的地方：某一帧偶尔黑一下的区域会保留 ——
# 那是画面内容，裁掉就等于把游戏画面裁坏了。

#: 判定「这块是黑边」的灰度下限。低于这个值算黑。
BLACK_LEVEL = 18

#: 内容包围盒的采样帧数（沿时间均匀取）。
BAND_SAMPLES = 7

#: 采样时缩到多大来算包围盒。够细就行，目的是快，不是精确。
BAND_PROBE_W = 240

#: 打开视频最多试几次（第一次失败时会等一下再试）。
OPEN_RETRY = 3


def open_capture(path: str):
    """打开视频，返回 ``(cap, 旋转角度)``。

    旋转角度取自容器的旋转元数据（``CAP_PROP_ORIENTATION_META``），
    表示「显示时需要顺时针转多少度」。

    打不开会**重试**几次
    ------------------
    实测在同一台机器上，同一个路径连续打开 sometimes 会第一次失败、
    第二次就成功（文件在即时通讯软件 的接收目录里，偶尔被安全软件扫到）。
    一次失败就报「无法打开视频」会把用户直接卡死 —— 而文件其实好好的，
    所以这里重试几次再放弃。
    """
    cv2 = _require_cv2()
    cap = None
    for attempt in range(OPEN_RETRY):
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            break
        cap.release()
        cap = None
        if attempt + 1 < OPEN_RETRY:
            time.sleep(0.4 * (attempt + 1))
    if cap is None:
        raise VideoProcessingError(f"无法打开视频：{path}")
    rot = 0
    if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
        try:
            rot = int(round(float(cap.get(cv2.CAP_PROP_ORIENTATION_META)))) % 360
        except (TypeError, ValueError):
            rot = 0
    return cap, rot


def orient_frame(cv2, frame, rot: int):
    """把画面转正。``rot`` 是要转的角度（0/90/180/270，顺时针）。"""
    if frame is None or not rot:
        return frame
    if rot == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def content_band(video: "UploadedVideo", samples: int = BAND_SAMPLES,
                 rotation: int = 0):
    """整段视频里**始终**是黑的那部分：返回 ``(x0, y0, x1, y1)`` 或 ``None``。

    取多个采样帧的「亮区」**交集**再求包围盒 ——
    只有从头到尾都黑的地方才算黑边；某一帧亮过的位置一律保留。

    交集中空、或包围盒几乎等于整帧，就不裁（返回 None）。

    ``rotation`` 是先把画面转正的角度：黑边的位置跟画面朝向有关，
    所以要先转再量。
    """
    cv2 = _require_cv2()
    import numpy as np

    cap, rot = open_capture(video.path)
    if not cap.isOpened():
        cap.release()
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return None

    inter = None
    frame_w = frame_h = 0
    for i in range(samples):
        idx = int(total * (i + 0.5) / samples)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total - 1))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = orient_frame(cv2, frame, rotation)
        h, w = frame.shape[:2]
        frame_w, frame_h = w, h
        scale = BAND_PROBE_W / max(1, w)
        probe = cv2.resize(
            frame, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
        mask = gray > BLACK_LEVEL
        inter = mask if inter is None else (inter & mask)
    cap.release()

    if inter is None or not inter.any() or not frame_w:
        return None
    ys, xs = np.where(inter)
    ph, pw = inter.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    # 裁掉的太少就不用裁
    if (x1 - x0) >= pw * 0.98 and (y1 - y0) >= ph * 0.98:
        return None
    # 换算回转正后的原图坐标
    sx = frame_w / pw
    sy = frame_h / ph
    return (
        max(0, int(x0 * sx)),
        max(0, int(y0 * sy)),
        min(frame_w, int(x1 * sx)),
        min(frame_h, int(y1 * sy)),
    )


def _corner_features(img) -> Dict[str, Tuple[float, float, int]]:
    """四个角的特征：``(边密度, 亮斑占比, 圆斑数)``。

    角的名字用「上左 / 上右 / 下左 / 下右」。
    """
    cv2 = _require_cv2()
    import numpy as np

    H, W = img.shape[:2]
    out: Dict[str, Tuple[float, float, int]] = {}
    for fy, fyname in ((0, "上"), (1, "下")):
        for fx, fxname in ((0, "左"), (1, "右")):
            x0 = 0 if fx == 0 else int(W * 0.70)
            y0 = 0 if fy == 0 else int(H * 0.66)
            x1 = x0 + int(W * 0.30)
            y1 = y0 + int(H * 0.34)
            reg = img[y0:y1, x0:x1]
            if reg.size == 0:
                out[f"{fyname}{fxname}"] = (0.0, 0.0, 0)
                continue
            gray = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
            edges = float(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)).mean())
            bright = float((gray > 165).mean())
            mask = (gray > 165).astype(np.uint8)
            n, _lab, stats, _c = cv2.connectedComponentsWithStats(mask)
            blobs = 0
            for i in range(1, n):
                a = int(stats[i, cv2.CC_STAT_AREA])
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                if a < 40 or bw < 6 or bh < 6:
                    continue
                if 0.6 <= bw / max(1, bh) <= 1.6:
                    blobs += 1
            out[f"{fyname}{fxname}"] = (edges, bright, blobs)
    return out


#: 角名 -> HUD 里的位置。默认按王者荣耀的布局。
#:
#: 小地图在左上：满屏细线，边密度最高；
#: 技能按钮在右下：亮斑最多、圆形按钮最多；
#: 摇杆在左下：一个很大的光滑圆，边密度最低。
DEFAULT_HUD_LAYOUT = {
    "minimap": "上左",
    "skills": "下右",
    "joystick": "下左",
}


def detect_orientation(canvas, layout: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """判定画面该转多少度才正。

    为什么不用容器里的旋转元数据
    --------------------------
    实测遇到过一个手机录屏：容器写着「显示时顺时针转 90°」，
    可按它转过来，画面是**上下颠倒**的（小地图跑到右下角、摇杆跑到右下）。
    元数据写错了，而界面只会显示「没读到时钟」，看不出真正的原因是画面转反了。

    为什么不用时钟判方向
    ------------------
    时钟模板是标定值，换机型、换比例、换清晰度就未必对得上；
    低清晰度的录屏（数字只有三四个像素高）更是完全读不出来。
    方向判错一次，整局的截图全都得倒着看。

    所以改用**画面布局**判：这个游戏的 HUD 布局是固定的
    （小地图在左上、技能按钮在右下、摇杆在左下），
    哪个方向摆过去符合这个布局，哪个方向就是正的。
    这个判据与分辨率、比例、清晰度都无关。

    返回 ``{"rotation": 角度, "score": 分数, "features": ...}``。
    """
    cv2 = _require_cv2()
    lay = dict(DEFAULT_HUD_LAYOUT)
    lay.update(layout or {})

    best: Optional[Dict[str, Any]] = None
    for rotation in (0, 90, 180, 270):
        cand = orient_frame(cv2, canvas, rotation)
        feats = _corner_features(cand)
        if not feats:
            continue
        score = _hud_layout_score(feats, lay)
        if best is None or score > best["score"]:
            best = {"rotation": rotation, "score": score, "features": feats}

    return best or {"rotation": 0, "score": 0.0, "features": {}}


def _hud_layout_score(feats: Dict[str, Tuple[float, float, int]],
                      lay: Dict[str, str]) -> float:
    """画面布局符不符合这个游戏的 HUD 布局（分数越高越像）。

    三个判据各对应 HUD 的一个特征：

    - **小地图**在左上角 —— 一小块塞满细线的地图，边密度最高；
    - **技能按钮**在右下角 —— 亮斑最多、圆形按钮最多；
    - **摇杆**在左下角 —— 一个很大的光滑圆，边密度最低。
    """
    if not feats:
        return 0.0
    edges = {k: v[0] for k, v in feats.items()}
    bright = {k: v[1] for k, v in feats.items()}
    blobs = {k: v[2] for k, v in feats.items()}

    def argmax(d):
        return max(d, key=lambda k: d[k]) if d else ""

    def argmin(d):
        return min(d, key=lambda k: d[k]) if d else ""

    score = 0.0
    if argmax(edges) == lay["minimap"]:
        score += 1.0
    skills = lay["skills"]
    if argmax(bright) == skills or argmax(blobs) == skills:
        score += 1.0
    if argmin(edges) == lay["joystick"]:
        score += 1.0
    score += 0.3 * (
        (edges[lay["minimap"]] - max(v for k, v in edges.items() if k != lay["minimap"]))
        / 100.0
    )
    return score


def hud_ready(feats: Dict[str, Tuple[float, float, int]],
              lay: Dict[str, str]) -> bool:
    """这一帧的 HUD 是不是已经就位（也就是已经在对局里了）。

    判据：**小地图在左上、摇杆在左下**。
    选与 BP、选人、加载、结算这些「对局前」的界面的区别是：
    对局里的画面左下角压着一个很大的摇杆圆（边密度最低），
    左上角压着满屏细线的小地图（边密度最高）；而对局前的界面没有这两样。
    """
    if not feats:
        return False
    edges = {k: v[0] for k, v in feats.items()}
    if not edges:
        return False
    argmax = max(edges, key=lambda k: edges[k])
    argmin = min(edges, key=lambda k: edges[k])
    return argmax == lay["minimap"] and argmin == lay["joystick"]


#: 摇杆区亮像素占比的**滚动中位数**窗口（秒）。
#:
#: 短窗不够：打架时左下角会有技能特效，一两秒就亮起来。
#: 实测（聊天软件转发的王者荣耀录屏，480x217 画布）：
#: 逐帧值在对局期间波动很大（均值 0.009、中位 0.004，但 95 分位到 0.036）；
#: 而**任意 30 秒窗口的中位数**在对局期间始终 ≤ 0.012、
#: 在 BP 与加载期间始终 ≥ 0.024 —— 中位数把那些瞬间全抹掉了。
JOYSTICK_MEDIAN_SEC = 10.0

#: 摇杆区（左下）在对局里的亮像素占比上限（中位数之后比）。
#:
#: 对局里左下角压着一个**大而光滑的摇杆圆**，整块是暗的；
#: BP、选人、加载界面的左下角是**队友头像列表** —— 一堆亮图标。
#: 实测中间是干净的界：对局段 89.6% 的采样点 ≤ 0.02，BP 段只有 1.5%。
JOYSTICK_BRIGHT_MAX = 0.02


def in_match(feats: Dict[str, Tuple[float, float, int]],
             lay: Dict[str, str],
             joy_bright: Optional[float] = None) -> bool:
    """这一帧是不是已经**在对局里**了。

    判据只有一条：**左下角是暗的**（摇杆）。

    ``joy_bright`` 可以传入**平滑后**的摇杆区亮像素占比 ——
    扫描时用它把瞬间的特效挡在外面（见 :data:`JOYSTICK_BRIGHT_MAX`）。

    为什么不再加「左上角边密度最高」那一条
    --------------------------------------
    方向判定已经用过它了（:func:`detect_orientation`），但那是**整段视频**
    投票出来的方向，稳；逐帧去要求「左上角必须是边密度最高的角」不稳 ——
    实测对局期间有 15% 的帧里不是（技能特效一炸就把别的角顶上去），
    那会让连续段断成碎段，起点就测不准了。

    摇杆那一块一整局都是暗的，不随画面变。
    """
    if not feats:
        return False
    if joy_bright is None:
        joy_bright = feats.get(lay["joystick"], (0.0, 1.0, 0))[1]
    return float(joy_bright) <= JOYSTICK_BRIGHT_MAX



def _vote_rotation(choices: List[Dict[str, Any]]) -> int:
    """方向投票：多数说了算，票数相同比总分。

    为什么不能「得分最高的那一帧说了算」
    ----------------------------------
    实测（720p 录屏）：BP 选人界面那一帧的布局得分 2.03，比所有对局帧
    （1.93~1.97）都高 —— BP 界面里没有小地图，「左上角边密度最高」这种
    判据在它身上纯属巧合，却赢下了单帧最高分，于是**整个方向判反**，
    画布里小地图在右下、摇杆在右上，什么都读不对。
    多数票把这种巧合帧冲掉。
    """
    tally: Dict[int, int] = {}
    for c in choices:
        tally[c["rotation"]] = tally.get(c["rotation"], 0) + 1
    return max(
        sorted(tally),
        key=lambda r: (tally[r],
                        sum(c["score"] for c in choices if c["rotation"] == r)),
    )


def detect_view(video: "UploadedVideo", samples: int = 5,
                hud_layout: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """判定录屏的**方向**与**内容画布**，顺带找时钟的锚点。

    方向用画面布局判（见 :func:`detect_orientation`），
    时钟位置用搜索找（见 :func:`app.collection.clock.detect_anchor`）。

    判不出来（画面里没有时钟、清晰度不够）就返回空表，
    调用方退回容器元数据。
    """
    cv2 = _require_cv2()

    cap, meta_rot = open_capture(video.path)
    if not cap.isOpened():
        cap.release()
        return {}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return {}

    # 没有声明 HUD 布局的游戏不做方向判定 ——
    # 用别的游戏的布局去判，判出来的一定是错的。
    if not hud_layout:
        cap.release()
        return {}

    # 逐个采样帧各投一票，最后**多数**说了算。
    #
    # 为什么不是「得分最高的那一帧说了算」
    # --------------------------------------
    # 实测（720p 录屏，12.5 分钟）：BP 选人界面那一帧的布局得分 2.03，
    # 比所有对局帧（1.93~1.97）都高 —— 因为 BP 界面里没有小地图，
    # 「左上角边密度最高」这种判据在它身上纯属巧合，却赢下了单帧最高分。
    # 而对局帧之间也不完全一致（有的认 90°、有的认 270°）。
    #
    # 多数票把 BP 帧的巧合冲掉（对局帧占多数），票数相同再比总分。
    choices: List[Dict[str, Any]] = []
    for i in range(samples):
        idx = int(total * (i + 0.5) / samples)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total - 1))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        # 先把手机屏幕那块（黑边之外）裁出来
        screen = _crop_content(frame)
        if screen is None:
            continue
        best_for_frame: Optional[Dict[str, Any]] = None
        for base in (0, 90, 180, 270):
            cand = orient_frame(cv2, screen, base)
            h, w = cand.shape[:2]
            # 手机竖着录横屏游戏时，屏幕区域本身是竖的；
            # 转到横着的那两个方向才可能是游戏画布。
            if w <= h:
                continue
            orient = detect_orientation(cand, hud_layout)
            total_rot = (base + orient["rotation"]) % 360
            # **最终画布必须是横屏** —— 王者是横屏游戏。
            # 手机竖着录的时候，游戏在画面里是**躺着**的（整屏转 90°）：
            # 不把总旋转算完再判一次，就会把躺着的游戏当成正的，
            # 于是画布里小地图在右下、摇杆在右上，什么都读不对。
            final = orient_frame(cv2, screen, total_rot)
            fh, fw = final.shape[:2]
            if fw <= fh:
                continue
            if (best_for_frame is None
                    or orient["score"] > best_for_frame["score"]):
                best_for_frame = {"rotation": total_rot,
                                  "score": orient["score"]}
        if best_for_frame:
            choices.append(best_for_frame)
    cap.release()

    if not choices:
        return {}

    # 多数票；票数相同比总分（再相同取较小的旋转角，纯粹为了结果可复现）。
    rotation = _vote_rotation(choices)
    tally: Dict[int, int] = {}
    for c in choices:
        tally[c["rotation"]] = tally.get(c["rotation"], 0) + 1
    return {
        "rotation": rotation,
        "orientation_score": max(c["score"] for c in choices
                                 if c["rotation"] == rotation),
        "meta_rotation": meta_rot,
        "orientation_votes": tally,
    }


def _crop_content(frame):
    """裁掉整块黑边，返回内容区域。"""
    cv2 = _require_cv2()
    import numpy as np

    if frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray > BLACK_LEVEL
    if not mask.any():
        return frame
    ys, xs = np.where(mask)
    pad = 3
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(frame.shape[1], int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(frame.shape[0], int(ys.max()) + pad + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return frame
    return frame[y0:y1, x0:x1]


#: 取景信息缓存（按 video_id）。判方向要抽帧 + 搜索，别处重复算。
_VIEW_CACHE: Dict[str, Dict[str, Any]] = {}


def video_view(video: "UploadedVideo",
              hud_layout: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """这个视频的取景信息：方向、黑边、时钟锚点。

    判一次、存起来：方向判定要抽帧搜索，重复算一遍要几十秒。
    """
    key = getattr(video, "video_id", "") or ""
    if key and key in _VIEW_CACHE:
        return _VIEW_CACHE[key]

    cv2 = _require_cv2()
    cap, meta_rot = open_capture(video.path)
    meta_rot = meta_rot if cap.isOpened() else 0
    cap.release()

    view = detect_view(video, hud_layout=hud_layout)
    rotation = int(view.get("rotation", meta_rot))
    band = content_band(video, rotation=rotation)
    anchor = _find_anchor(video, rotation)
    out = {
        "rotation": rotation,
        "meta_rotation": meta_rot,
        "band": band,
        "anchor": anchor,
        "orientation_score": view.get("orientation_score", 0.0),
        "orientation_votes": view.get("orientation_votes"),
        "detected": bool(view),
    }
    if key:
        _VIEW_CACHE[key] = out
    return out


def _find_anchor(video: "UploadedVideo", rotation: int,
                 samples: int = 5) -> Optional[Dict[str, Any]]:
    """在摆正的画布上找时钟的位置。

    方向已经判出来了，所以这里只试这一个方向 ——
    搜索本身会挑到别的位置的相似纹理上，方向一限定就稳了。
    """
    cv2 = _require_cv2()

    cap, _meta = open_capture(video.path)
    if not cap.isOpened():
        cap.release()
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return None

    band = content_band(video, rotation=rotation)
    best: Optional[Dict[str, Any]] = None
    for i in range(samples):
        idx = int(total * (i + 0.5) / samples)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total - 1))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        canvas = crop_to_band(orient_frame(cv2, frame, rotation), band)
        anchor = clock.detect_anchor(canvas, rotations=(0,))
        if not anchor:
            continue
        if best is None or anchor["score"] > best["score"]:
            best = anchor
    cap.release()
    return best


def crop_to_band(frame, band):
    """按内容包围盒裁帧。``band`` 为 None 时原样返回。"""
    if not band or frame is None:
        return frame
    x0, y0, x1, y1 = band
    if x1 - x0 < 8 or y1 - y0 < 8:
        return frame
    return frame[y0:y1, x0:x1]


def probe(video: UploadedVideo) -> UploadedVideo:
    """读取视频基本信息（时长、帧率、分辨率）。

    分辨率报的是**显示方向**的：容器带旋转元数据时，画面要转过来看，
    报原始像素尺寸会让人以为视频是躺的。
    """
    cv2 = _require_cv2()
    cap, rot = open_capture(video.path)
    if not cap.isOpened():
        raise VideoProcessingError(
            f"无法打开视频：{video.original_name}。"
            f"可能是编码不受支持，建议转成 H.264 编码的 mp4。"
        )
    video.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    video.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if rot in (90, 270):
        width, height = height, width
    video.width = width
    video.height = height
    if video.fps > 0:
        video.duration_sec = video.frame_count / video.fps
    cap.release()
    return video


#: 时钟探测的采样间隔（秒）。
#:
#: 走「按阶段抽帧」这条路时**不需要**灰度差 —— 王者的一局是连续的一整场，
#: 按灰度差切「回合」切出来的全是镜头移动和技能特效（实测这一局切出
#: 80 个「回合」）。这里只要读时钟，而时钟一秒才跳一次，2 秒采一次
#: 就够拟合时间线了。
#:
#: 采样疏了就可以用 seek：seek 每次的代价是「从上一个关键帧解起」，
#: 与采样间隔**无关**；顺序解码的代价则是「整段视频解一遍」，
#: 与间隔也无关但总量巨大。所以间隔越大，seek 越划算 ——
#: 实测见 :func:`probe_clock` 的说明。
CLOCK_PROBE_SEC = 2.0


def probe_clock(
    video: UploadedVideo,
    sample_sec: float = CLOCK_PROBE_SEC,
    progress=None,
    hud_layout: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """只读时钟：稀疏采样 + seek。返回 ``(时钟信息, 统计)``。

    为什么可以稀疏
    --------------
    时钟一秒才跳一次，读它的目的是拿到「视频秒 ↔ 游戏秒」这条直线，
    不是逐帧追踪。直线的拟合精度取决于**点的分布**，不取决于点数 ——
    2 秒一个点、一局 12 分钟就是 360 个点，远超拟合所需的 8 个。

    为什么用 seek 而不是顺序解码
    --------------------------
    实测（真实录屏，前 240 秒，采样间隔 2 秒）：

        seek       56.5 秒   ← 最快
        顺序 grab  110.8 秒
        顺序 read  153.3 秒

    而同样这 120 秒视频，采样间隔 0.3 秒时顺序 grab 反而最快（46.0 秒
    对 seek 的 150.3 秒）。两处结论不矛盾：seek 的单次代价固定，
    顺序解码的代价正比于**总帧数**；间隔越密，顺序解码摊到的帧越少。

    于是有了分工：MOBA 只要时钟 → 稀疏 + seek；
    要按信息密度切时间片 → 必须密集 → 顺序解码。

    读不出就返回 ``ok=False``，调用方据此回退到密集扫描那套。
    """
    cv2 = _require_cv2()

    cap, meta_rot = open_capture(video.path)
    if not cap.isOpened():
        raise VideoProcessingError(f"无法打开视频：{video.original_name}")

    view = video_view(video, hud_layout=hud_layout)
    band = view.get("band")
    anchor = view.get("anchor")
    rotation = int(view.get("rotation", meta_rot))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if fps > 0 else video.duration_sec
    max_idx = (total - 1) if total else int(duration * fps)

    step_frames = max(1, int(round(sample_sec * fps)))
    samples: List = []
    scanned = 0
    idx = 0
    #: 逐帧的小地图候选 —— 跨帧串成英雄轨迹（见 app/collection/skilltrack.py）。
    #:
    #: **顺带采，不另开一趟解码**：这一趟本来就要 seek 扫完整局，
    #: 每帧多裁一次小地图、多跑一次连通域，实测每帧约 8 毫秒。
    track_cands: List[Tuple[float, List[Any]]] = []
    geo = None
    try:
        from . import minimap as mm_mod

        geo = mm_mod.MapGeometry.load()
    except Exception:                                   # noqa: BLE001
        geo = None
    #: 「对局中」的逐点记录 —— 用来跟读数拟合出的起点互验
    hud_flags: List[Tuple[float, bool]] = []
    joy_hist: deque = deque(maxlen=max(3, int(JOYSTICK_MEDIAN_SEC / max(sample_sec, 0.05))))
    lay = dict(DEFAULT_HUD_LAYOUT)
    lay.update(hud_layout or {})
    while idx <= max_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        scanned += 1
        t = idx / fps
        frame = crop_to_band(orient_frame(cv2, frame, rotation), band)
        if anchor:
            game_s, conf = clock.read_clock_anchored(frame, anchor)
        else:
            game_s, conf = clock.read_clock_frame(frame)
        if game_s is not None:
            samples.append(
                clock.ClockSample(t=t, game_s=game_s, confidence=conf)
            )
        feats = _corner_features(frame)
        joy_hist.append(float(feats[lay["joystick"]][1]))
        hud_flags.append((t, in_match(feats, lay, joy_bright=median(joy_hist))))
        if geo is not None:
            try:
                mini = mm_mod.minimap_crop(frame, geo)
                # **宽松阈值**：靠形状（空心弧）认玩家，不靠亮度 ——
                # 实测玩家标记的绿度峰值有时只有 22，而同帧的绿色地形能到 116。
                cands = (mm_mod.find_hero_candidates(mini, ratio=0.15)
                         if mini is not None else [])
                track_cands.append((t, cands))
            except Exception:                           # noqa: BLE001
                pass
        idx += step_frames
        if progress and total:
            progress(min(1.0, idx / max(total - 1, 1)), "scanning")

    cap.release()
    track = clock.build_clock_track(samples)
    info = track.to_dict()
    # 两条来源互验：读数拟合出的起点 vs 画面检测出的起点
    # 两条来源互验：读数拟合的起点 vs 画面检测的起点。
    # **时钟优先** —— 那是游戏自己的钟；HUD 检测只是读不到时钟时的兜底
    # （泉水那几秒画面是亮的，摇杆判据会失效，起点会判晚，实测晚了一分半）。
    hud_match = match_sync.detect_match_start(hud_flags)
    match = hud_match
    if track.ok and track.game_start_t is not None:
        match = match_sync.MatchStart(
            t=float(track.game_start_t),
            source="clock",
            ok=True,
            run_sec=hud_match.run_sec,
            reason=(f"对局起点由时钟读数定：局内 0:00 对应视频第 "
                    f"{float(track.game_start_t):.0f} 秒"),
        )
    cross = match_sync.cross_check(track.game_start_t, hud_match.t)
    note = (
        f"按每 {sample_sec:.2f}s 采样读时钟"
        + (f"；{info['reason']}" if info.get("reason") else "")
    )
    if cross and not cross["consistent"]:
        note += f"；{cross['note']}"
    stats = {
        "scanned_samples": scanned,
        "total_frames": total,
        "fps": round(fps, 2),
        "sample_sec": sample_sec,
        "duration_sec": round(duration, 1),
        "decode_mode": "seek",
        "note": note,
        "match_start_t": round(match.t, 1) if match.ok else None,
        "match_start_reason": match.reason,
        "match_start_cross_check": cross,
        "clock": info,
    }
    # 英雄轨迹：跨帧串起来（操作模式项的探测器靠它）
    if track_cands:
        try:
            from . import skilltrack as st

            points, diag = st.build_track(track_cands)
            # **缺口也要存**（只有 t、没有 nx/ny）：覆盖率要按
            # 「这一段本来该有多少个采样点」算，才看得出有多少读不出来。
            rows = []
            for (t, _c), p in zip(track_cands, points):
                rows.append(p.to_dict() if p is not None
                            else {"t": round(t, 2)})
            stats["hero_track"] = rows
            stats["hero_track_diag"] = diag
        except Exception as exc:                      # noqa: BLE001
            stats["hero_track"] = []
            stats["hero_track_error"] = f"{type(exc).__name__}: {exc}"
    return info, stats


#: HUD「就位」要连续多少个采样点才算数。
#:
#: BP 界面偶尔也会撞上「小地图在左上、摇杆在左下」这个图案，
#: 但对局里的小地图与摇杆是一路都在的 —— 连续成立才是真的。
HUD_READY_STREAK = 6

#: 判定「画面在动」的灰度差下限（缩略图均值差）。
#:
#: 用来回答两个问题：哪一段是**对局中**（画面一直在变），
#: 哪一段**信息密度高**（变得剧烈 = 打架、转场、抢资源）。
ACTIVITY_DIFF = 2.0

#: 切片密度用的时间桶（秒）。
#:
#: 一分钟一桶：规则的**时间窗本来就是分钟粒度**的
#: （``between [0, 4]`` 涵盖第 0~4 分钟），切片按分钟分桶，
#: 切片才谈得上「落在哪条规则的时间窗里」。
DENSITY_BUCKET_SEC = 60.0

#: 默认切片预算。一局 15 分钟切三十来段，够密也不至于让人填到手软。
DENSITY_BUDGET = 32

#: 切片密度（每分钟多少片）—— **上限**，不是定数。
#:
#: 预算不是拍脑袋的常数，而是**按时长算的密度**。一局 15 分钟该有多少片，
#: 取决于这局有多长。
#:
#: 为什么上限是每分钟 10 片（每 6 秒一片）：攻略说的那些判断点
#: （刷野、转线、开龙、带线）在一局里出现的间隔是**几秒到十几秒**，
#: 一次龙坑团也就十来秒。每分钟切 2 片的话，一局只能采样 30 个瞬间，
#: 真正落在有效时刻上的不到两成 —— 剩下的时间全在赶路和等冷却。
#:
#: 这是**上限**：实际切多少由事件聚类决定（同一场团战切一片就够），
#: 密集的时段自然低于上限，安静的时段靠补齐。
#:
#: 代价要知道：15 分钟一局就是 150 帧，每帧都要过一次分析 ——
#: 这就是自动识别层那笔账。
DENSITY_PER_MINUTE = 10.0

#: 切片密度的下限（再短的对局也至少切这些）。
DENSITY_MIN_BUDGET = 12

#: 推算局内时间时，在对局段起点上加的秒数。
#:
#: 对局段起点 = 画面开始剧烈变化的地方 ≈ 加载结束、对局开始。
#: 所以这个值取 0；留成常量是为了将来遇到「录屏在开打前还有一段动画」
#: 时有地方调。
MATCH_LOAD_SEC = 0.0


def hud_region(img, corner: str):
    """HUD 某个角所在的画面区域（左上 / 右上 / 左下 / 右下）。

    小地图就在左上角。它是**固定的一小块**（不随镜头移动），
    里面的变化 = 玩家在地图上的移动 —— 也就是攻略说的那些操作
    （刷野、转线、开龙、带线）。拿它当「有事发生」的信号，
    比整幅画面的灰度差准得多：后者主要是镜头移动和技能特效，
    玩家走两步画面就变，切出来的片段可能全在赶路。
    """
    h, w = img.shape[:2]
    if corner == "上左":
        return img[0:int(h * 0.34), 0:int(w * 0.30)]
    if corner == "上右":
        return img[0:int(h * 0.34), int(w * 0.70):]
    if corner == "下左":
        return img[int(h * 0.66):, 0:int(w * 0.30)]
    return img[int(h * 0.66):, int(w * 0.70):]


def _dense_scan(video: UploadedVideo, config: Optional[ScanConfig],
                progress=None,
                hud_layout: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """顺序解码 + 采样：一次扫描拿到 **变化序列 + 时钟读数**。

    两个产物共用一趟解码 —— 分别扫两遍要花两倍时间，而录像动辄 2 GB。
    """
    cv2 = _require_cv2()
    import numpy as np

    cfg = config or ScanConfig()
    cap, meta_rot = open_capture(video.path)
    if not cap.isOpened():
        raise VideoProcessingError(f"无法打开视频：{video.original_name}")

    # 先判方向：容器里的旋转元数据**不可信**（实测遇到过写反的）。
    # 用画面内容判 —— HUD 布局摆过去对不对得上。
    #
    # 取景信息（方向 + 黑边 + 时钟锚点）一次算好、缓存起来，
    # 后面每帧只管按它裁、按它转。
    view = video_view(video, hud_layout=hud_layout)
    rotation = int(view.get("rotation", meta_rot))
    anchor = view.get("anchor")
    band = view.get("band")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if fps > 0 else video.duration_sec

    if not cfg.sample_sec:                       # 允许调用方显式覆盖
        cfg.sample_sec = adapt_sample_sec(duration)

    points: List[Tuple[float, float]] = []       # (视频秒, 灰度差)
    diffs: List[float] = []
    last_thumb = None
    scanned = 0
    max_idx = total - 1 if total else int(duration * fps)

    clock_samples: List = []
    #: 时钟跳字得到的采样（读不出数字时用）。
    tick_samples: List = []
    tick_state = None
    ticks = 0
    #: HUD 就位的时刻 = 对局开始（时钟组件出现 = 0:00）。
    hud_start_t: Optional[float] = None
    hud_streak = 0
    #: 逐个采样点的「HUD 是否就位」。扫描结束后由
    #: :mod:`app.collection.match_sync` 取**最长连续成立段**的起点 ——
    #: BP 界面偶尔会撞上对局的 HUD 图案，但只撞一小段。
    hud_flags: List[Tuple[float, bool]] = []
    #: 摇杆区亮像素占比的历史（滚动中位数滤波用）
    joy_hist: deque = deque(maxlen=max(5, int(JOYSTICK_MEDIAN_SEC / max(cfg.sample_sec, 0.05))))
    lay = dict(DEFAULT_HUD_LAYOUT)
    lay.update(hud_layout or {})

    step_frames = max(1, int(round(cfg.sample_sec * fps)))
    idx = 0
    next_sample = 0
    #: 画布尺寸 —— 用来判断计时器数字够不够大（见 clock.too_small_to_read）
    canvas_h = 0
    canvas_w = 0
    frame_w = 0
    frame_h = 0
    while idx <= max_idx:
        if idx == next_sample:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            scanned += 1

            shot = crop_to_band(orient_frame(cv2, frame, rotation), band)
            canvas_h = int(shot.shape[0])
            canvas_w = int(shot.shape[1])
            if not frame_w:
                rotated = orient_frame(cv2, frame, rotation)
                frame_h, frame_w = int(rotated.shape[0]), int(rotated.shape[1])

            # 变化信号取自**小地图所在的那一块**（HUD 的左上角），
            # 不是整幅画面 —— 后者主要是镜头移动与技能特效，
            # 玩家走两步画面就变，切出来的片段会全在赶路。
            # 小地图是固定的一小块，它的变化 = 玩家在地图上的移动。
            region = hud_region(shot, lay["minimap"])
            thumb = cv2.cvtColor(
                cv2.resize(region, (16, 16), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY,
            ).astype(np.float32)

            t = idx / fps
            if anchor:
                game_s, game_conf = clock.read_clock_anchored(shot, anchor)
            else:
                game_s, game_conf = clock.read_clock_frame(shot)
            if game_s is not None:
                clock_samples.append(
                    clock.ClockSample(t=t, game_s=game_s, confidence=game_conf)
                )

            # -- 对局起点：HUD 就位的逐点记录 --------------------------------
            #
            # 计时器只在对局中才有，所以「对局 HUD 就位」就是局内 0:00 的
            # 候选。这里记录**每一个**采样点的就位情况，扫描结束后由
            # :mod:`app.collection.match_sync` 取「最长连续成立段」的起点。
            #
            # 这与手机型号、录屏软件、分辨率都无关 —— 判的是画面布局。
            feats = _corner_features(shot)
            # 摇杆区亮像素占比：取最近几个采样点的**中位数**，
            # 免得技能特效扫过左下角的那一两帧把连续段打断。
            joy_hist.append(float(feats[lay["joystick"]][1]))
            ready = in_match(feats, lay, joy_bright=median(joy_hist))
            hud_flags.append((t, ready))
            if ready:
                hud_streak += 1
                # 必须**连续**成立：BP 界面偶尔也会撞上这个图案，
                # 但对局里的小地图与摇杆是一路都在的。
                if hud_streak >= HUD_READY_STREAK and hud_start_t is None:
                    hud_start_t = t
            else:
                hud_streak = 0
            if anchor and hud_start_t is not None:
                mask = clock.clock_mask(shot, anchor)
                if mask is not None:
                    if clock.mask_changed(tick_state, mask):
                        ticks += 1
                    tick_state = mask
                    tick_samples.append(
                        clock.ClockSample(t=t, game_s=ticks, confidence=0.6)
                    )

            if last_thumb is not None:
                # 灰度差 = 这一刻画面变化了多少 = **信息密度**。
                # 打架、转场、抢资源的时刻变化剧烈，安静刷野的时刻变化小。
                d = float(np.abs(thumb - last_thumb).mean())
                diffs.append(d)
                points.append((t, d))

            last_thumb = thumb
            next_sample = idx + step_frames
        else:
            # 跳过的帧只 grab 不 retrieve
            if not cap.grab():
                break
        idx += 1
        if progress and total:
            progress(min(1.0, idx / max(total - 1, 1)), "scanning")

    cap.release()

    track = clock.build_clock_track(clock_samples)
    if not track.ok and tick_samples:
        # 直接读数字失败（录屏清晰度不够）→ 用跳字计时
        tick_track = clock.build_clock_track(tick_samples)
        if tick_track.ok:
            track = tick_track

    # 对局起点：取「HUD 连续就位」里**最长的一段**的起点。
    # 每段视频进入对局的时刻不同，必须逐段测 —— 写死或按比例推都会错。
    hud_match = match_sync.detect_match_start(hud_flags)
    match = hud_match
    if track.ok and track.game_start_t is not None:
        # **时钟读出来了，它就是对局起点** —— 那是游戏自己的钟，
        # 不会比任何画面判据差。
        #
        # HUD 检测只是读不到时钟时的兜底：泉水那几秒画面是亮的，
        # 「摇杆区是暗的」判据会失效，起点就判晚（实测晚了一分半：
        # 实况里 207 秒已经在泉水里，HUD 判据要等到 371 秒才认）。
        match = match_sync.MatchStart(
            t=float(track.game_start_t),
            source="clock",
            ok=True,
            run_sec=hud_match.run_sec,
            reason=(f"对局起点由时钟读数定：局内 0:00 对应视频第 "
                    f"{float(track.game_start_t):.0f} 秒"),
        )
    cross = match_sync.cross_check(track.game_start_t, hud_match.t)
    # 这份录屏**够不够自动识别** —— 说清楚差在哪，玩家才知道要重录到什么清晰度。
    feas = clock.recognition_feasibility(canvas_w, canvas_h)
    # 录屏规格：横屏 + 游戏画面占满全屏（用户定的标准）。
    spec = recording_spec(canvas_w, canvas_h, frame_w or canvas_w, frame_h or canvas_h)

    if not diffs:
        note = "视频太短或画面几乎无变化，没有可分析的片段"
    elif not track.ok:
        why = track.reason
        if not feas.get("ok"):
            # 不够清晰 → 说清差在哪（画布多高、数字多少像素、要多少）
            why = feas["note"] + "，阶段要手填"
        elif canvas_h and clock.too_small_to_read(canvas_h):
            why = clock.legibility_note(canvas_h) + "，阶段要手填"
        note = (f"{note_base(cfg, diffs)}；{why}")
    else:
        note = (f"{note_base(cfg, diffs)}；{track.reason}")
    if match.ok:
        note += f"；{match.reason}"
    if not spec["ok"]:
        note += f"；录屏规格：{spec['note']}"

    stats = {
        "scanned_samples": scanned,
        "total_frames": total or idx,
        "fps": round(fps, 2),
        "sample_sec": cfg.sample_sec,
        "duration_sec": round(duration, 1),
        "diff_median": round(sorted(diffs)[len(diffs) // 2], 2) if diffs else None,
        "diff_max": round(max(diffs), 2) if diffs else None,
        "note": note,
        # HUD 就位的时刻 = 对局开始的时刻（时钟组件一出现就是 0:00）。
        # 读不到时钟时，它比「按画面活动推算」准得多。
        "hud_start_t": round(hud_start_t, 1) if hud_start_t is not None else None,
        # 对局起点（局内 0:00 对应的视频秒）。逐段视频单独检测。
        "match_start_t": round(match.t, 1) if match.ok else None,
        "match_start_source": match.source if match.ok else "",
        "match_start_reason": match.reason,
        "match_start_run_sec": round(match.run_sec, 1),
        # 这份录屏**够不够自动识别**（看的是游戏画布，不是容器分辨率）。
        "recognition": feas,
        # 录屏规格：横屏 + 游戏画面占满全屏。
        "spec": spec,
        # 时钟读数与 HUD 检测两条来源的对照。
        "match_start_cross_check": cross,
        "decode_mode": "sequential",
        # 时钟读数随扫描一起返回：阶段划分全靠它，读不出来就必须说清楚
        "clock": track.to_dict(),
    }
    return {"points": points, "stats": stats}


def activity_scan(video: UploadedVideo, config: Optional[ScanConfig] = None,
                  progress=None,
                  hud_layout: Optional[Dict[str, str]] = None
                  ) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """扫描出**画面变化序列**（MOBA 按信息密度切时间片时用）。

    返回 ``(变化序列, 统计)``。变化序列是 ``[(视频秒, 灰度差), ...]``。
    """
    out = _dense_scan(video, config, progress, hud_layout)
    return out["points"], out["stats"]


def activity_span(points: List[Tuple[float, float]],
                  cover: float = 0.9) -> Optional[Tuple[float, float]]:
    """活动最集中的那一段 —— 那就是对局段。

    录屏的三段结构很清楚：开头是选人与加载（画面动一动就停），
    中间是**连续的一整场**（画面一直在变），结尾是结算（也停了）。
    所以对局段 = 变化量最集中的那个窗口。

    为什么用「累计 90%」而不是「最长的一段连续有动静」
    --------------------------------------------------
    对局中间会有安静的时刻：原地等技能、回城、暂停看装备。
    那些时刻一长（实测有一段 20 秒），「连续段」就被它截断，
    于是对局段只认出前半截 —— 后半局整局没进切片。
    累计口径不看连续性：只要把 90% 的变化量装进窗口里，
    中间夹着几秒安静都不影响。

    「变化量」用平滑后的灰度差：单帧的尖峰（一次技能特效）不算数。
    """
    if not points:
        return None

    # 平滑：相邻几个采样点取平均，抹掉单帧尖峰。
    window = 3
    smoothed: List[Tuple[float, float]] = []
    for i, (t, d) in enumerate(points):
        lo = max(0, i - window)
        hi = min(len(points), i + window + 1)
        avg = sum(p[1] for p in points[lo:hi]) / (hi - lo)
        smoothed.append((t, avg))

    weights = [max(0.0, d) for _, d in smoothed]
    total = sum(weights)
    if total <= 0:
        return None
    target = total * cover

    # 滑动窗口：找装下 90% 变化量的最短窗口。
    best: Optional[Tuple[float, float]] = None
    acc = 0.0
    left = 0
    for right, w in enumerate(weights):
        acc += w
        while acc - weights[left] >= target and left < right:
            acc -= weights[left]
            left += 1
        if acc >= target:
            span = (smoothed[left][0], smoothed[right][0])
            if best is None or (span[1] - span[0]) < (best[1] - best[0]):
                best = span
    if best is None:
        best = (smoothed[0][0], smoothed[-1][0])
    if (best[1] - best[0]) < 45.0:
        return None
    return best


def density_slices(points: List[Tuple[float, float]],
                   budget: int = DENSITY_BUDGET,
                   bucket_sec: float = DENSITY_BUCKET_SEC,
                   span: Optional[Tuple[float, float]] = None,
                   weights: Optional[Dict[int, float]] = None,
                   game_start_t: Optional[float] = None,
                   intervals: Optional[Sequence[Tuple[int, int]]] = None,
                   ) -> List[float]:
    """切片：**按模式项存在的时间区间切**，区间内切够密、且不冗余。

    口径（用户定的）
    ----------------
    > 「对模式项存在的所有时间区间内，对对局视频切很多片，
    > 要包含所有有效信息，但不冗余。然后每个时间区间的切片去匹配模式项」

    所以分桶的单位是**模式项的时间区间**（从规则库倒推），
    不是「每一分钟」—— 区间才是判定单位，证据就该按区间组织。

    两个信号各管一件事
    ------------------
    - **规则密度**（``weights``）决定**这个区间切几片**：
      区间里能判出的规则越多，切得越密。一条规则都没有的区间不切 ——
      那里切了也判不出任何东西，预算浪费在空气上。
    - **画面变化**决定**区间里切在哪一刻**：
      打架、转场、抢资源的时刻画面变化最剧烈，那一帧才是有事发生的那一帧。

    桶 = **一个模式项时间区间**，边界按 ``game_start_t`` 对齐 ——
    桶号就是局内分钟号，密度直接查得到，不会出现「桶算的是视频第几分钟、
    密度查的是局内第几分钟」这种错位。

    没有密度表（``weights`` 为空）时退化成「每区间一片」：
    宁可少切，也不要在没有依据的地方多切。
    """
    if not points:
        return []
    if span:
        t0, t1 = span
        pts = [(t, d) for t, d in points if t0 <= t <= t1]
    else:
        pts = list(points)
    if not pts:
        return []

    # 桶的起点对齐到局内分钟：桶号 = 局内分钟号
    origin = game_start_t if game_start_t is not None else pts[0][0]
    if intervals:
        return _interval_slices(pts, intervals, weights, budget, origin,
                                bucket_sec)
    return _minute_slices(pts, weights, budget, origin, bucket_sec)


def _interval_slices(pts: List[Tuple[float, float]],
                     intervals: Sequence[Tuple[int, int]],
                     weights: Optional[Dict[int, float]],
                     budget: int,
                     origin: float,
                     bucket_sec: float) -> List[float]:
    """按**模式项的时间区间**切片。

    区间内切多少片：按该区间能判出的规则条数分（区间越长、规则越多，
    切得越密）。区间内切在哪：画面变化的峰值，且彼此**拉开间距** ——
    同一场团战切出七八片就是冗余，规则看的是「这段有没有做那件事」。
    """
    buckets: List[Tuple[int, int, List[Tuple[float, float]]]] = []
    for lo, hi in intervals:
        items = [(t, d) for t, d in pts
                 if lo <= int((t - origin) // bucket_sec) <= hi]
        if items:
            buckets.append((lo, hi, items))
    if not buckets:
        return []

    activity = [sum(d for _, d in items) for _lo, _hi, items in buckets]
    max_act = max(activity) if activity else 0.0

    # 权重 = 区间内的规则密度总和 × 活动系数
    #
    # 密度总和只算**这一局里实际出现**的那些分钟 ——
    # 「12:00 之后」那个区间的名义范围到第 60 分钟，
    # 可这局只打到第 14 分钟，按名义范围算会把它的权重抬高好几倍。
    w: Dict[int, float] = {}
    for i, (lo, hi, items) in enumerate(buckets):
        minutes = {int((t - origin) // bucket_sec) for t, _d in items}
        base = sum(float((weights or {}).get(m, 0.0)) for m in minutes)
        factor = 0.7 + 0.3 * (activity[i] / max_act if max_act else 0.0)
        w[i] = base * factor

    quota = _allocate(w, list(range(len(buckets))), budget)
    times: List[float] = []
    for i, (lo, hi, items) in enumerate(buckets):
        k = quota.get(i, 0)
        if k <= 0:
            continue
        # 落点按**事件**取：一次团战切一片，不在同一件事上切好几片。
        times.extend(_pick_peaks(items, k, 0.0))
    times.sort()
    return times


def _pick_peaks(items: List[Tuple[float, float]], k: int,
                gap: float) -> List[float]:
    """从一个区间里挑 ``k`` 片：**按事件聚类**，一次事件切一片。

    「包含所有有效信息」与「不冗余」这两条要同时满足，靠的是分清
    「同一件事的多个瞬间」和「不同的几件事」：

    - 画面连续剧烈变化的那一段 = **一件事**（一次团战、一次转场）；
      里面切多片是冗余 —— 规则看的是「这段有没有做那件事」。
    - 分开的几段 = **几件事**；每件都该有一片，漏掉一件就是漏掉一条结论。

    所以先按时间把显著变化点聚成事件，再**每个事件取一片**（峰值）。
    事件多于配额时，按事件的总变化量取前 ``k`` 个 ——
    预算不够时，宁可漏变化小的那些事，也不能在一件事上切好几片。
    """
    if not items:
        return []
    # 显著阈值用**均值**而不是中位数：变化量是双峰的（安静 vs 事件），
    # 中位数会正好落在两峰之间或事件峰上，阈值定高了一个事件都认不出。
    mean = sum(d for _t, d in items) / len(items)
    threshold = max(mean * 1.2, 0.5)

    # 显著变化点 → 按时间聚成事件
    events: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = []
    last_t: Optional[float] = None
    for t, d in sorted(items):
        if d < threshold:
            continue
        if last_t is not None and (t - last_t) > 8.0:
            events.append(cur)
            cur = []
        cur.append((t, d))
        last_t = t
    if cur:
        events.append(cur)

    if not events:
        events = [[(t, d)] for t, d in sorted(items, key=lambda p: -p[1])[:k]]

    # 每个事件取变化量最大的那一刻
    peaks = [(max(ev, key=lambda p: p[1])[0], sum(d for _t, d in ev))
             for ev in events]
    peaks.sort(key=lambda p: -p[1])
    picked = [t for t, _w in peaks[:k]]

    # 事件不够填满配额时，用剩下的变化点补齐（拉开间距，别又切到一件事上）。
    if len(picked) < k:
        span = max(1.0, items[-1][0] - items[0][0])
        gap = span / (k + 1)
        for t, _d in sorted(items, key=lambda p: -p[1]):
            if len(picked) >= k:
                break
            if all(abs(t - p) >= gap for p in picked):
                picked.append(t)
    picked.sort()
    return picked


def _minute_slices(pts: List[Tuple[float, float]],
                   weights: Optional[Dict[int, float]],
                   budget: int,
                   origin: float,
                   bucket_sec: float) -> List[float]:
    """老路：按局内分钟分桶（没有模式项时间区间时的退路）。"""
    buckets: Dict[int, List[Tuple[float, float]]] = {}
    for t, d in pts:
        idx = int((t - origin) // bucket_sec)
        buckets.setdefault(idx, []).append((t, d))
    if not buckets:
        return []

    order = sorted(buckets)
    # 每个桶的画面活动量（这一分钟里的变化总量）
    activity = {i: sum(d for _, d in buckets[i]) for i in order}
    max_act = max(activity.values()) if activity else 0.0

    # 权重 = 规则密度 × 活动系数
    #
    # 规则密度是主项（决定「这一分钟能判出多少条结论」），
    # 活动系数只做调节：同一分钟里画面变化剧烈的地方多切一刀，
    # 但不让它盖过规则密度（规则密度为 0 的分钟，活动再大也不切 ——
    # 那里根本没有规则能判）。
    w: Dict[int, float] = {}
    for i in order:
        base = float((weights or {}).get(i, 0.0))
        factor = 0.7 + 0.3 * (activity[i] / max_act if max_act else 0.0)
        w[i] = base * factor

    # 只切有规则的分钟（密度为 0 的分钟跳过）
    covered = [i for i in order if (weights or {}).get(i, 0.0) > 0]
    if not covered:
        covered = order          # 没有密度表就全都切，每分钟一片

    quota = _allocate(w, covered, budget)

    times: List[float] = []
    for i in order:
        k = quota.get(i, 0)
        if k <= 0:
            continue
        items = sorted(buckets[i], key=lambda p: p[1], reverse=True)
        picked: List[float] = []
        for t, _ in items:
            # 落点之间至少隔 3 秒，免得切出一堆同一秒的帧
            if all(abs(t - p) >= 3.0 for p in picked):
                picked.append(t)
            if len(picked) >= k:
                break
        times.extend(picked)

    times.sort()
    return times


def _allocate(weights: Dict[int, float], covered: List[int],
              budget: int) -> Dict[int, int]:
    """把预算分给各个分钟。

    用**最高平均法**（D'Hondt）：每轮把下一个名额给「当前平均分最低」的
    那个分钟。它的好处是**单调** —— 权重大的分钟一定拿得多，不会像
    最大余数法那样在权重接近时被零头抹平（那正是密度驱动最怕的事：
    第 4 分钟比第 0 分钟密 38%，正是要体现在切片数上）。

    每个覆盖到的分钟**保底一片**（全程都要有），权重为 0 的分钟不参与。
    """
    quota: Dict[int, int] = {i: 0 for i in covered}
    if not covered or budget <= 0:
        return quota

    # 权重全为 0 时退化成平均分
    total_w = sum(weights[i] for i in covered)
    if total_w <= 0:
        for i in covered:
            quota[i] = 1
        extra = budget - len(covered)
        if extra > 0:
            base, rem = divmod(extra, len(covered))
            for i in covered:
                quota[i] += base
            for i in covered[:rem]:
                quota[i] += 1
        return quota

    for _ in range(budget):
        # 平均分 = 已得 / 权重；给平均分最低的那个再加一片
        pick = min(covered, key=lambda i: (quota[i] + 1) / weights[i])
        quota[pick] += 1
    return quota


def moba_slice_product(
    video: UploadedVideo,
    points: List[Tuple[float, float]],
    base_dir: str,
    phase_windows: Dict[str, List[int]],
    budget: int = DENSITY_BUDGET,
    progress=None,
    hud_layout: Optional[Dict[str, str]] = None,
    weights: Optional[Dict[int, float]] = None,
    game_start_t: Optional[float] = None,
    analysis_windows: Optional[Sequence[Any]] = None,
    hero_track: Optional[List[Any]] = None,
    game: str = "",
    lane: str = "",
) -> Optional[Dict[str, Any]]:
    """MOBA 的产物：**按信息密度切时间片**。

    什么时候走这条路
    --------------
    - 能读时钟 → 按阶段抽帧（:func:`sample_by_phase`），那是最好的情况；
    - 读不到时钟（机型没标定、录屏是竖屏带黑边…）→ 走这条。

    为什么不用「回合边界」那套
    ----------------------
    王者荣耀的一局是**连续的一整场**：镜头一直在动、技能特效一直在闪，
    按灰度差切出来的「回合」全是镜头移动，跟规则库的时间窗毫无关系。
    实测一局王者切出 80 个「回合」，而真正值得判断的时刻不到 30 个。

    所以这里的切片单位是**时间**，落点按画面变化密度分配：
    打架、转场、抢资源的时段多切几刀，安静时段少切。

    局内时间是**推算**的（对局段起点 + 加载时长），
    标注页里可以改 —— 规则的时间窗是分钟粒度的，差一分钟就判到另一批规则上，
    所以这个值只当默认值，不当答案。
    """
    span = activity_span(points)
    if not span:
        return None
    # 对局起点：优先用 **HUD 就位**的时刻（时钟组件一出现就是 0:00），
    # 没有就用画面活动的起点推。HUD 那个准得多 —— 活动会被镜头移动、
    # 技能特效带偏（实测能差 20 秒）。
    if game_start_t is None:
        game_start_t = span[0] + MATCH_LOAD_SEC
    match_end_t = span[1]

    # 切片：**按模式项存在的时间区间分桶**，区间内切够密且不冗余。
    # 桶 = 一个局内分钟（边界对齐到局内分钟，桶号就是分钟号），
    # 预算按**规则密度**分，桶内落点取画面变化的峰值。
    slice_intervals = None
    if analysis_windows:
        from .window_analysis import segments as _segments
        slice_intervals = [(w.lo, w.hi)
                           for w in _segments(analysis_windows)]
    # 预算 = **按对局时长算的密度**（每分钟 6 片），不小于默认预算。
    match_minutes = max(0.0, (match_end_t - game_start_t) / 60.0)
    total_budget = max(budget,
                       int(math.ceil(match_minutes * DENSITY_PER_MINUTE)),
                       DENSITY_MIN_BUDGET)
    # 切片范围：**对局起点 → 对局段结束**。
    #
    # 不能直接用活动段：活动段的起点会偏晚（开局那几分钟画面变化不大，
    # 小地图也动得少），用它当起点会把「开局 0 ~ 1 分钟」那批规则的
    # 时段整段切掉 —— 那是 237 条规则，不是零星几条。
    cut_span = (game_start_t, span[1])
    times = density_slices(points, budget=total_budget, span=cut_span,
                           weights=weights, game_start_t=game_start_t,
                           intervals=slice_intervals)
    if not times:
        return None

    specs = []
    for t in times:
        minute = (t - game_start_t) / 60.0
        # 阶段按**整分钟**判：规则的窗口是分钟粒度的（early 是 0~4 分，
        # mid 是 5~10 分），拿 4.8 分去比就会掉进两个窗口之间的空档。
        minute_i = int(minute)
        phase = ""
        for name, window in sorted(phase_windows.items(),
                                   key=lambda kv: kv[1][0]):
            lo, hi = int(window[0]), int(window[1])
            if lo <= minute_i <= hi:
                phase = name
                break
        specs.append({"phase": phase, "t": t, "minute": minute})

    shots = extract_frames_at(
        video, [(s["phase"], s["t"]) for s in specs], base_dir, "slice",
        progress=progress, hud_layout=hud_layout,
    )
    rounds: List[Dict[str, Any]] = []
    for i, (spec, shot) in enumerate(zip(specs, shots)):
        minute = spec["minute"]
        rounds.append({
            "round_no": i + 1,
            "phase": spec["phase"],
            "game_minute": round(minute, 2) if minute >= 0 else None,
            "timestamp": shot["timestamp"],
            "image": shot["image"],
            "image_url": shot["image_url"],
            # 帧是确定的：它就是那一刻的画面。**时间**是推算的，
            # 不准这件事记在时间事实的置信度上（见 build_skeleton），
            # 不能记在回合门控置信度上 ——
            # 门控置信度低于门槛会让整段回合被判为缺失，
            # 于是这一局一条规则都不触发，报告看起来只是「没结论」。
            "confidence": 1.0,
            "time_source": "estimate",
        })

    # 窗口层：**分析单元是时间窗，不是时间片**。
    #
    # 窗从规则库倒推（每条规则的 time_min between）—— 那是规则的时间口径，
    # 不是我拍脑袋定的分段。窗里装着那段时间的切片（证据），
    # 一次综合分析给出这一窗的结论，窗内的规则共用它。
    window_bundles: List[Dict[str, Any]] = []
    if analysis_windows:
        from .window_analysis import bundle_slices, segments as _segments
        segs = _segments(analysis_windows)
        # 这一局的**键位档位**：先挑好，再让每个窗用同一套坐标。
        hud_regions = _hud_regions_for(video, base_dir, rounds)
        pending: List[Tuple[Any, Dict[str, Any]]] = []
        for b in bundle_slices(rounds, analysis_windows):
            d = b.to_dict()
            seg = None
            for s in segs:
                if s.contains(b.window.lo) and s.contains(b.window.hi):
                    seg = s
                    break
            d["segment_key"] = seg.key if seg else b.window.key
            d["segment_label"] = seg.label() if seg else b.window.label()
            # **自动识别**：把这一窗的若干帧一起交给识别器，
            # 出一份事实（局面类）+ 识别器认出的动作与证据。
            # 读不出的写进 missing，不猜。
            pending.append((b, d))
        # 逐窗读并行：一局 8 个窗串行要好几分钟（模型调用是网络等待）。
        window_bundles = _read_window_bundles(
            video, base_dir, pending, hud_regions, hero_track,
            float(game_start_t or 0.0), progress, game=game, lane=lane)

    return {
        "rounds": rounds,
        "windows": window_bundles,
        "bp": [],
        "match_start_t": round(game_start_t, 1),
        "match_end_t": round(match_end_t, 1),
        "match_minutes": round((match_end_t - game_start_t) / 60.0, 1),
        "slice_budget": budget,
        "density_span": [round(span[0], 1), round(span[1], 1)],
        # 这一局切了哪些局内分钟、每分钟切了几片 —— 报告里说得清，
        # 用户才可能发现「某个该判的分钟没切到」。
        "slice_minutes": sorted({int((t - game_start_t) // 60.0) for t in times}),
    }


#: 录屏规格：用户定的标准。
#:
#: 「横屏，游戏画面占满全屏」——
#: 竖着录（游戏在画面里躺着）、或者画面只占中间一小条（两边大片黑边），
#: 都会把可识别的像素砍掉一大半。标准写在这里，扫描时逐条核对。
#:
#: - **横屏**：游戏画布宽 > 高（王者荣耀是横屏游戏）；
#: - **占满全屏**：内容区面积 ≥ 画面的 90%（剩下的是黑边）。
FULLSCREEN_MIN_RATIO = 0.90


def recording_spec(canvas_w: int, canvas_h: int,
                  frame_w: int, frame_h: int) -> Dict[str, Any]:
    """这段录屏**符不符合标准**，差在哪。

    用户定的标准：「横屏，游戏画面占满全屏」。

    这里只管**规格**（横屏 / 满屏）；**清晰度**是另一件事，
    由 :func:`app.collection.clock.recognition_feasibility` 判 ——
    两件事分开，才不会在扫描说明里把同一句话讲三遍。
    """
    land = canvas_w >= canvas_h
    area = float(canvas_w * canvas_h)
    full = float(max(1, frame_w * frame_h))
    ratio = area / full
    problems: List[str] = []
    if not land:
        problems.append(f"画面是竖的（{canvas_w}×{canvas_h}）")
    if ratio < FULLSCREEN_MIN_RATIO:
        problems.append(
            f"游戏画面只占 {ratio * 100:.0f}%，其余是黑边"
            f"（需要 ≥ {FULLSCREEN_MIN_RATIO * 100:.0f}%）")
    return {
        "ok": not problems,
        "landscape": land,
        "fullscreen": ratio >= FULLSCREEN_MIN_RATIO,
        "fill_ratio": round(ratio, 3),
        "canvas": [canvas_w, canvas_h],
        "frame": [frame_w, frame_h],
        "recognition": clock.recognition_feasibility(canvas_w, canvas_h),
        "problems": problems,
        "note": (
            f"符合标准：横屏 {canvas_w}×{canvas_h}，游戏画面占满全屏"
            if not problems else "；".join(problems)),
    }


def quick_spec(video: "UploadedVideo",
               hud_layout: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """**上传当场**核对录屏规格：横屏 + 游戏画面占满全屏 + 够不够识别。

    只抽几帧（``detect_view`` 已经做过一次取景，代价很小），
    目的是让用户**上传完立刻**知道这段录屏能不能用 ——
    而不是白等 18 分钟扫完才发现画面只有 217 像素宽。
    """
    try:
        # 用 video_view（不是 detect_view）：它还给出**黑边**，
        # 没有黑边就不知道游戏画面在整帧里占多大。
        view = video_view(video, hud_layout=hud_layout or DEFAULT_HUD_LAYOUT)
    except Exception:                              # noqa: BLE001
        return {"ok": False, "problems": ["取景失败，识别不了"],
                "note": "取景失败，这段录屏识别不了"}
    rot = int(view.get("rotation", 0))
    band = view.get("band")
    cap, _ = open_capture(video.path)
    canvas = None
    raw_w = raw_h = 0
    if cap.isOpened():
        raw_w = int(cap.get(cv2_of().CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2_of().CAP_PROP_FRAME_HEIGHT))
        ok, frame = cap.read()
        cap.release()
        if ok:
            canvas = crop_to_band(orient_frame(cv2_of(), frame, rot), band)
    if canvas is None:
        return {"ok": False, "problems": ["读不出画面"],
                "note": "读不出画面，这段录屏识别不了"}
    ch, cw = canvas.shape[:2]
    # **转正后**的整帧尺寸（黑边就比在这里）
    if rot in (90, 270):
        frame_w, frame_h = raw_h, raw_w
    else:
        frame_w, frame_h = raw_w, raw_h
    return recording_spec(cw, ch, frame_w or cw, frame_h or ch)


def cv2_of():
    """取 cv2（延迟导入，顶层不依赖 opencv）。"""
    return _require_cv2()


#: 逐窗读画的并发数。
#:
#: 每一窗要做的是「重抽帧 → 合成 HUD 面板 → 问模型」。前两步是本地计算，
#: 第三步是**网络等待** —— 所以并发收益很大（一局 8 个窗串行要好几分钟）。
#:
#: 4 是保守值：模型调用有速率限制，一次放太多会被限流，
#: 限流之后反而更慢（要等退避）。
WINDOW_READ_CONCURRENCY = 4


def _read_window_bundles(video: "UploadedVideo", base_dir: str,
                         pending: Sequence[Tuple[Any, Dict[str, Any]]],
                         regions=None,
                         hero_track: Optional[List[Any]] = None,
                         game_start_t: float = 0.0,
                         progress=None,
                         game: str = "",
                         lane: str = "") -> List[Dict[str, Any]]:
    """并行地把每个分析窗交给识别器，返回填好 ``facts`` 的窗列表。

    ``pending`` 是 ``[(bundle, dict), ...]``：bundle 是窗口对象，
    dict 是已经写好 ``segment_key`` / ``segment_label`` 的骨架。

    ``game`` / ``lane`` 决定这一窗拿到的**操作模式清单** ——
    模型要拿画面去对攻略，前提是它手里有攻略原文（见
    :func:`app.rules.checklist.window_checklist`）。``lane`` 已知时
    清单只给这一条线，短一大截，模型也不会对着别的线的打法乱指。

    **为什么能并发**
    ----------------
    每一窗自己开自己的 ``VideoCapture``、自己抽帧、自己读文件，
    不共享任何可变状态；模型那一段是网络等待，并发直接省时间。

    **并发数为什么封顶**
    ------------------
    模型调用有速率限制。放太多会被限流，限流之后要等退避，
    反而比串行更慢。四路是实测下来的保守值。

    **读不出来的那一窗**
    --------------------
    记缺失、写原因，**不猜**，也**不中断其他窗** ——
    一窗读不出来只说明那一窗判不了，不说明整局白跑。

    取景信息（方向/黑边）在这里**先算一遍**：那一步要抽帧搜索，
    四条线程同时算会各算一遍，白白多花几十秒。
    """
    if not pending:
        return []

    # 取景信息先算一遍，把缓存焐热（:func:`video_view` 按 video_id 缓存）。
    try:
        video_view(video)
    except Exception:                              # noqa: BLE001
        pass

    gs = float(game_start_t or 0.0)

    def _one(item):
        b, d = item
        window = b.window
        segment_key = d.get("segment_key") or window.key
        try:
            facts = _recognize_window(
                video, base_dir, b, segment_key, regions, hero_track,
                # 轨迹切片按**整窗**给（局内分钟 → 视频秒）——
                # 这一条路的时间是推算的，但窗的范围必须完整，
                # 不然轨迹上的点不够，探测器算不出运动特征。
                t0=gs + window.lo * 60.0,
                t1=gs + (window.hi + 1) * 60.0,
                game=game, lane=lane,
            )
        except Exception as exc:                   # noqa: BLE001
            facts = {
                "facts": {},
                "missing": ["画面"],
                "notes": [f"这一窗读不出来：{exc}"],
            }
        return d, facts

    out: List[Dict[str, Any]] = []
    done = 0
    total = len(pending)
    with ThreadPoolExecutor(max_workers=WINDOW_READ_CONCURRENCY) as pool:
        for d, facts in pool.map(_one, pending):
            d["facts"] = facts
            out.append(d)
            done += 1
            if progress:
                progress(done / total, "reading")
    return out


def _recognize_window(video: "UploadedVideo", base_dir: str, bundle,
                       segment_key: str, regions=None,
                       hero_track: Optional[List[Any]] = None,
                       t0: Optional[float] = None,
                       t1: Optional[float] = None,
                       game: str = "",
                       lane: str = "") -> Dict[str, Any]:
    """对**一窗的若干帧**做一次综合分析，返回这一窗识别出来的事实。

    识别器有**两套**（见 :mod:`app.collection.settings`）：

    - ``vlm``（默认）：把这一窗重抽一遍帧，交给多模态模型读
      （:mod:`app.collection.vlm_recognizer`）；
    - ``cv``：早先那套几何/颜色探测器。用户判过它这条路失败，
      所以默认关着，只作为对照与兜底保留。

    两套产出的形状**完全一样**（facts/actions/missing/notes/
    probe_actions/probe_moments/absent），上层不区分。

    ``hero_track`` 是**整局**的英雄轨迹（:mod:`app.collection.skilltrack`
    的行）。``t0``/``t1`` 是这一窗在原视频里的秒数区间 ——
    探测器靠它把轨迹切出这一段。

    **区间要按整窗给，不能只按抽到的那几帧给**：
    阶段抽帧是「一分钟一帧」，一窗两帧之间可能隔着好几分钟 ——
    只切两帧之间的那一小段，轨迹上的点不够，探测器算不出运动特征，
    于是整窗都判不了。
    """
    from . import settings

    if settings.recognizer_kind() == "cv":
        return _recognize_window_cv(video, base_dir, bundle, segment_key,
                                     regions, hero_track, t0, t1)
    out = _recognize_window_vlm(video, base_dir, bundle, segment_key,
                                 regions, t0, t1, game=game, lane=lane)
    out["segment_key"] = segment_key
    return out


#: 喂给模型的帧密度：每 6 秒一帧，最多 16 帧。
#:
#: 阶段抽帧是「一分钟一帧」，一窗只有两三帧 —— 那点画面看不出
#: 「在野区来回」还是「路过野区」。每 6 秒一帧才能看出移动，
#: 也让窗内的关键动作（打龙、转线、开团）更不容易恰好落在两帧之间漏掉。
#: 代价是每窗帧更多、模型输入更大，所以上限仍卡在 VLM_MAX_FRAMES。
VLM_FRAME_PERIOD_SEC = 6.0
VLM_MIN_FRAMES = 4
VLM_MAX_FRAMES = 16


def _recognize_window_vlm(video: "UploadedVideo", base_dir: str, bundle,
                           segment_key: str, regions=None,
                           t0: Optional[float] = None,
                           t1: Optional[float] = None,
                           game: str = "",
                           lane: str = "") -> Dict[str, Any]:
    """按窗重抽帧 → 合成 HUD 面板 → 交给模型读。"""
    from .hud_panel import build_panel
    from .vlm_recognizer import FrameInput, VlmRecognizer, WindowInput

    window = bundle.window
    lo, hi = int(window.lo), int(window.hi)
    if t0 is None or t1 is None or t1 <= t0:
        stamps = [float(s.get("timestamp") or 0.0) for s in bundle.slices]
        if not stamps:
            return {"facts": {}, "missing": ["画面"], "notes": ["这一窗没有画面"]}
        t0, t1 = min(stamps), max(stamps)

    try:
        cv2 = _require_cv2()
        import numpy as np
    except Exception:                              # noqa: BLE001
        return {"facts": {}, "missing": ["画面"], "notes": ["识别器不可用"]}

    # 1) 这一窗重抽帧（密度按窗口时长算）
    span = float(t1 - t0)
    count = int(max(VLM_MIN_FRAMES, min(VLM_MAX_FRAMES, round(span / VLM_FRAME_PERIOD_SEC))))
    times = [t0 + span * (i + 0.5) / count for i in range(count)]
    prefix = f"vlm-{window.key}"
    shots = extract_frames_at(
        video, [(f"第 {i + 1} 帧", t) for i, t in enumerate(times)],
        base_dir, prefix)

    frames_dir = os.path.join(video_store_dir(base_dir), video.video_id + "_frames")
    frames: List[FrameInput] = []
    for s in shots:
        rel = s.get("image")
        if not rel:
            continue
        path = os.path.join(frames_dir, rel)
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        frames.append(FrameInput(t=float(s.get("timestamp") or 0.0), data=data,
                                 name=str(rel or "")))
    if not frames:
        return {"facts": {}, "missing": ["画面"], "notes": ["这一窗没抽到画面"]}

    # 2) 合成 HUD 面板（小地图/大招/计时器放大拼一起）
    panel = None
    try:
        first = cv2.imdecode(np.frombuffer(frames[0].data, np.uint8),
                             cv2.IMREAD_COLOR)
        panel = build_panel(first, regions)
    except Exception:                              # noqa: BLE001
        panel = None

    # 3) 问模型
    #    局内 0:00 对应的视频秒：t0 落在窗口的第 lo 分钟上。
    game_start_t = float(t0) - lo * 60.0

    # 这一窗的**操作模式清单** —— 攻略原文（前提 / 该做 / 后果）。
    #
    # 模型要当裁判，前提是它手里有尺子。早先这里给的是空的，
    # 于是模型只被当成「读出几个原子事实」的黑盒，剩下的全靠
    # 硬规则去套 —— 而攻略的分支前提在画面里读不出来，
    # 报告于是既空又不敢下结论。
    modes: List[Dict[str, Any]] = []
    if game:
        try:
            from ..rules import checklist
            modes = checklist.window_checklist(game, lo, hi, lane=lane)
        except Exception:                          # noqa: BLE001
            modes = []

    inp = WindowInput(lo=lo, hi=hi, frames=frames, panel=panel,
                      modes=modes, lane=lane,
                      segment_key=segment_key, game_start_t=game_start_t)
    from . import settings

    recognizer = VlmRecognizer(
        data_dir=settings.data_dir(),
        cache_dir=os.path.join(settings.data_dir(), "vlm_cache"))
    return recognizer.analyze(inp)


def _recognize_window_cv(video: "UploadedVideo", base_dir: str, bundle,
                          segment_key: str, regions=None,
                          hero_track: Optional[List[Any]] = None,
                          t0: Optional[float] = None,
                          t1: Optional[float] = None) -> Dict[str, Any]:
    """旧的那套：几何/颜色阈值探测器（默认关着，只作对照与兜底）。"""
    from .recognizer import WindowRecognizer

    try:
        cv2 = _require_cv2()
    except Exception:                              # noqa: BLE001
        return {"facts": {}, "missing": [], "notes": ["识别器不可用"]}
    frames_dir = os.path.join(video_store_dir(base_dir), video.video_id + "_frames")
    images = []
    stamps = []
    for s in bundle.slices:
        rel = s.get("image")
        if not rel:
            continue
        path = os.path.join(frames_dir, rel)
        img = imread_u(path)
        if img is not None:
            images.append(img)
            stamps.append(float(s.get("timestamp") or 0.0))
    if not images:
        return {"facts": {}, "missing": ["画面"], "notes": ["这一窗没有画面"]}
    if t0 is None and stamps:
        t0 = min(stamps)
    if t1 is None and stamps:
        t1 = max(stamps)
    reading = WindowRecognizer(regions).analyze(images, bundle.window,
                                                 track=hero_track, t0=t0, t1=t1)
    out = reading.to_dict()
    out["segment_key"] = segment_key
    return out


def _hud_regions_for(video: "UploadedVideo", base_dir: str,
                     rounds: List[Dict[str, Any]], sample: int = 16):
    """给这一局挑一个**键位档位**，返回配好档位的 HUD 区域表。

    做法：从抽出来的帧里**跨帧稳定**地检出右下角按钮群 ——
    草地上的假圆每帧位置都不同，会被这一步滤掉；真按钮位置固定，留下来。
    再拿这份按钮群跟标定文件里的档位比对：匹配上哪一档就用哪一档的
    大招框，一档都匹配不上就**不读**大招（宁可记缺失，不给随机数）。

    标定文件里没有档位时（老格式），返回默认区域表，行为不变。
    """
    from . import skillbar
    from .recognizer import HudRegions

    regions = HudRegions.load()
    if not regions.profiles or not rounds:
        return regions
    try:
        cv2 = _require_cv2()
    except Exception:                              # noqa: BLE001
        return regions
    frames_dir = os.path.join(video_store_dir(base_dir), video.video_id + "_frames")
    step = max(1, len(rounds) // sample)
    images = []
    for r in rounds[::step][:sample]:
        rel = r.get("image")
        if not rel:
            continue
        img = imread_u(os.path.join(frames_dir, rel))
        if img is not None:
            images.append(img)
    if not images:
        return regions
    try:
        regions.select_profile(skillbar.stable_buttons(images))
    except Exception:                              # noqa: BLE001
        pass
    return regions


def note_base(cfg, diffs: List[float]) -> str:
    """扫描说明的第一句 —— 说清采样口径与画面变化的量级。"""
    return (
        f"按每 {cfg.sample_sec:.2f}s 采样，灰度差中位数 "
        f"{sorted(diffs)[len(diffs) // 2]:.1f}"
    )


# --------------------------------------------------------------------------
# 血条读取 —— 唯一能高置信度自动给出的局面量
# --------------------------------------------------------------------------

#: 血条颜色。默认按「亮绿的填充 + 深色底」找，这是多数游戏 HUD 的配色。
#: 换游戏时调这里就行 —— 阈值集中在配置里，不散落在算法中。
DEFAULT_BAR_COLORS = (
    # (BGR 下界, 上界, 名称)
    ((40, 120, 60), (120, 255, 160), "green"),
    ((30, 90, 200), (90, 200, 255), "red"),
    ((150, 120, 30), (255, 220, 90), "blue"),
)


@dataclass
class BarReading:
    """一条进度条的读数。"""

    ratio: float                  # 0..1 填充比例
    value: Optional[float] = None # 换算成游戏内数值（血量）
    max_value: Optional[float] = None
    confidence: float = 0.0
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None and self.confidence > 0


def read_bar_ratio(
    frame, box: Tuple[int, int, int, int], colors=DEFAULT_BAR_COLORS
) -> BarReading:
    """从帧里读一条水平进度条的填充比例。

    ``box`` 是 ``(x, y, w, h)``，相对坐标 0..1，由调用方按分辨率换算。

    做法：裁出 HUD 区域 → 按颜色掩膜找出填充像素 → 取最右侧填充像素的位置
    作为填充终点 → 除以条宽。这是纯像素统计，没有模型、没有猜测，
    血量这种 0~100 的量能读得很准。
    """
    cv2 = _require_cv2()
    import numpy as np

    x, y, w, h = box
    H, W = frame.shape[:2]
    px, py = int(x * W), int(y * H)
    pw, ph = max(1, int(w * W)), max(1, int(h * H))
    if px + pw > W or py + ph > H:
        return BarReading(0.0, reason="区域超出画面范围")

    roi = frame[py:py + ph, px:px + pw]
    if roi.size == 0:
        return BarReading(0.0, reason="区域为空")

    ratio = 0.0
    matched = ""
    for lo, hi, name in colors:
        lo_arr = np.array(lo, dtype=np.uint8)
        hi_arr = np.array(hi, dtype=np.uint8)
        mask = cv2.inRange(roi, lo_arr, hi_arr)
        frac = float((mask > 0).mean())
        if frac < 0.02:
            continue
        # 找出每一行里最后一个填充像素的位置
        cols = np.where(mask.any(axis=0))[0]
        if cols.size == 0:
            continue
        ratio = (cols.max() + 1) / float(pw)
        matched = name
        break

    if not matched:
        return BarReading(
            0.0,
            reason="未找到血条颜色的像素 —— 区域可能取错，或该 HUD 用的是渐变/描边样式",
        )

    # 一条只剩 3% 的残血条和一条刚开局的满血条，长度是一样的。
    # 只有当读取结果明确落在「不可能」的两端之外才降置信度。
    conf = 0.9
    if ratio < 0.02:
        conf = 0.6
    elif ratio > 0.99:
        conf = 0.85

    return BarReading(ratio=round(ratio, 4), confidence=conf,
                      reason=f"按 {matched} 色掩膜测得")


# --------------------------------------------------------------------------
# 抽帧 + 产出识别骨架
# --------------------------------------------------------------------------


def vu_clock_minutes(clock_info: Dict[str, Any]):
    """由扫描返回的时钟信息，造一个「视频秒 -> 游戏内分钟」的函数。

    读不到时钟就返回 None —— 调用方据此知道**不能**换算，
    于是阶段要手填。按视频秒硬套一个比例凑出时间来，是这套东西里
    最容易犯也最难发现的错：每一条带时间窗的模式都会判到错误的阶段，
    而报告上只是「没触发」。
    """
    if not clock_info or not clock_info.get("ok"):
        return None
    start = clock_info.get("game_start_t")
    if start is None:
        return None
    return lambda video_t: max(0.0, (video_t - float(start)) / 60.0)


# --------------------------------------------------------------------------
# 按阶段均匀抽帧 —— MOBA 的扫描产物
# --------------------------------------------------------------------------

#: 每个阶段抽几帧。
#:
#: 抽帧数是个取舍，两头都有代价：
#:
#:   抽得少   漏掉「这一整段做反了」的概率大 —— 而这正是要报的东西。
#:   抽得多   要标注的格子多，而且每多一格就多一次「这一下算不算偏离」
#:            的争议（同一阶段里玩家本来就会干不一样的事）。
#:
#: 6 帧是折中：一局 12 分钟的前期（5 分钟）有 6 个落点，约每 50 秒一个，
#: 足以看出「一直在刷野」还是「一直在支援」。
PHASE_SAMPLES = 6

#: BP 段抽几帧。
#:
#: BP 段是**对局开始之前**的那一段：征召、加载、选人、最终对阵。
#: 它的价值是让用户看清「我方和敌方各选了谁」—— 英雄层模式要靠这个才
#: 能触发，而对位优劣本来就只能由人判断。
#:
#: 取 10 帧的依据是真实录屏的实测分布（0–200 秒）：征召在 0–20 秒、
#: 地图加载 25 秒、选人 40–130 秒、3D 加载 170 秒、最终对阵 195 秒。
#: 均匀取 10 帧正好每 20 秒一个，把这几段都盖住。
BP_SAMPLES = 10


def _even_times(lo: float, hi: float, n: int) -> List[float]:
    """在 ``[lo, hi]`` 里等距取 ``n`` 个时刻，落在**格子中心**。

    取格子中心而不是端点，是因为端点会正好压在阶段边界上：
    早期阶段的最后一帧落在第 5.0 分钟，那一分钟按规则已经算中期了 ——
    于是同一帧在「前期」和「中期」各出现一次，标注页上两行一模一样，
    用户会以为系统抽重复了。

    ``n <= 1`` 时取中点。区间退化（lo == hi）时也返回该点，不返回空 ——
    空列表会让「这一阶段抽不到帧」看起来像「这一阶段不存在」。
    """
    if hi <= lo:
        return [max(0.0, lo)]
    n = max(1, n)
    span = (hi - lo) / n
    return [lo + span * (i + 0.5) for i in range(n)]


def extract_frames_at(
    video: UploadedVideo,
    times: List[Tuple[str, float]],
    base_dir: str,
    prefix: str,
    progress=None,
    hud_layout: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """在给定时刻各抽一帧存成 jpg。

    ``times`` 是 ``[(名字, 视频秒), ...]``。返回的每一项带上了
    ``image_url``，调用方不用再自己拼路径 —— 拼路径的逻辑只有一处，
    就不会出现「有的图能显示有的点开是 404」。

    这里用 seek 而不是顺序解码：一次 seek 约 0.5 秒，而要抽的帧只有
    二三十张。顺序解码整段视频要六分钟，为了二三十张图不值得。
    """
    cv2 = _require_cv2()

    out_dir = os.path.join(video_store_dir(base_dir), video.video_id + "_frames")
    os.makedirs(out_dir, exist_ok=True)

    cap, meta_rot = open_capture(video.path)
    if not cap.isOpened():
        raise VideoProcessingError(f"无法打开视频：{video.original_name}")
    view = video_view(video, hud_layout=hud_layout)
    band = view.get("band")
    rotation = int(view.get("rotation", meta_rot))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    shots: List[Dict[str, Any]] = []
    total = len(times) or 1
    for i, (label, t) in enumerate(times):
        target = max(0, int(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, img = cap.read()
        name = f"{prefix}-{i + 1:03d}.jpg"
        if ok and img is not None:
            img = crop_to_band(orient_frame(cv2, img, rotation), band)
            # Unicode 安全写图：中文路径下 cv2.imwrite 会静默失败，
            # 导致发给模型的帧一张都落不了盘（见 imwrite_u 说明）。
            imwrite_u(os.path.join(out_dir, name), img, quality=85)
        else:
            name = ""
        shots.append({
            "label": label,
            "timestamp": round(t, 2),
            "image": name,
            "image_url": (
                f"/api/videos/{video.video_id}/frames/{name}" if name else ""
            ),
        })
        if progress:
            progress((i + 1) / total, "extracting")
    cap.release()
    return shots


def sample_by_phase(
    video: UploadedVideo,
    clock_info: Dict[str, Any],
    base_dir: str,
    phase_windows: Dict[str, List[int]],
    phase_samples: int = PHASE_SAMPLES,
    bp_samples: int = BP_SAMPLES,
    analysis_windows: Optional[List[Any]] = None,
    hero_track: Optional[List[Any]] = None,
    progress=None,
    game: str = "",
    lane: str = "",
) -> Optional[Dict[str, Any]]:
    """产出 MOBA 的扫描产物：**BP 帧序列 + 按阶段均匀抽帧**。

    为什么不再按「回合边界」切
    ----------------------
    王者的一局是**连续的一整场**：灰度差被镜头移动、技能特效、推塔特效
    反复触发，切出来的「回合」和操作模式的时间窗毫无关系。
    规则库判的是「前期 / 中期 / 后期」，所以抽帧也该按阶段来 ——
    抽帧落点与规则库的时间窗对齐，两边才说同一件事。

    阶段窗口从适配器取（``phase_time_windows``），不另写一份：
    两份窗口迟早会只改一处，然后出现「帧标中期、内核判前期」的错位。

    抽帧数
    ------
    每个阶段 :data:`PHASE_SAMPLES` 帧，在该阶段的**有效区间**内等距取。
    「有效区间」比规则里的窗口多一分钟：规则窗口是 ``between [0, 4]``，
    闭区间，涵盖第 0~4 分钟；而 ``time_min`` 取的是整分，所以第 4 分钟
    实际覆盖 4:00–4:59。窗口写 [0,4] 说的是「0–4 分这一段」，
    抽帧必须按 [0,5) 来铺，否则第 4~5 分钟会一个落点都没有。

    读不到时钟就返回 None
    -------------------
    返回 None 表示「不能按阶段抽」—— 调用方据此回退到**按信息密度切时间片**
    那套（局内时间变成推算值，标注页可改）。按视频秒硬凑一个阶段是
    最容易犯也最难发现的错：每一条带时间窗的模式都会判到错误的阶段，
    而报告上只是「没触发」。
    """
    minutes_of = vu_clock_minutes(clock_info)
    if minutes_of is None:
        return None
    start_t = float(clock_info["game_start_t"])

    duration = video.duration_sec or 0.0
    if duration <= 0:
        return None

    # 对局的结束点：时钟**最后读到**的那一刻。录屏往往在对局结束后还继续
    # （结算画面、退出大厅），那段不是对局，抽进去只会标出无意义的帧。
    # 时钟读数表是按采样时刻存的，取最后一个能读到的时刻。
    readable = sorted(clock_info.get("readable_times") or [])
    match_end_t = readable[-1] if readable else duration
    match_end_t = min(match_end_t, duration)
    if match_end_t <= start_t:
        return None

    # -- BP 帧序列：对局开始之前的那一段 ----------------------------------
    bp_times: List[Tuple[str, float]] = []
    if start_t >= 5.0:
        for i, t in enumerate(_even_times(0.0, start_t, bp_samples)):
            bp_times.append((f"BP {i + 1}", t))

    # -- 按阶段均匀抽帧 ---------------------------------------------------
    # 规则的时间窗是**分钟粒度**的（``between [0, 4]`` 涵盖第 0~4 分钟），
    # 所以抽帧也按分钟来：一分钟一帧，落点取该分钟的中间。
    #
    # 为什么不按「阶段长度平均分」：前期是第 0~4 分钟共 5 分钟，平均分成 6 份
    # 会出现「第 2 分钟取两帧、第 5 分钟一帧都没有」这种落点，而 5 分钟取 5 帧
    # 正好每分钟一帧，判定时每个分钟都被覆盖到。
    match_minutes = (match_end_t - start_t) / 60.0
    round_specs: List[Dict[str, Any]] = []
    for phase, window in sorted(phase_windows.items(), key=lambda kv: kv[1][0]):
        lo, hi = int(window[0]), int(window[1])
        # 闭区间窗口覆盖第 lo..hi 分钟，也就是游戏内的 [lo, hi+1) 分钟
        m0 = float(lo)
        m1 = min(float(hi) + 1.0, match_minutes)
        if m1 <= m0:
            continue
        minute_count = m1 - m0
        # 一分钟一帧，但最多 phase_samples 帧 —— 后期可能有二十多分钟，
        # 逐分钟抽会让人填几十格，标注就变成了苦役。
        n = max(1, min(phase_samples, int(math.ceil(minute_count))))
        for g in _even_times(m0, m1, n):
            round_specs.append({"phase": phase, "t": start_t + g * 60.0})

    if not round_specs:
        return None

    shots = extract_frames_at(
        video, [(r["phase"], r["t"]) for r in round_specs],
        base_dir, "phase", progress=progress,
    )
    rounds: List[Dict[str, Any]] = []
    for i, (spec, shot) in enumerate(zip(round_specs, shots)):
        game_min = minutes_of(spec["t"])
        rounds.append({
            "round_no": i + 1,
            "phase": spec["phase"],
            "game_minute": round(game_min, 2) if game_min is not None else None,
            "timestamp": shot["timestamp"],
            "image": shot["image"],
            "image_url": shot["image_url"],
            # 阶段是按时钟**算**出来的，不是猜的，置信度给满。
            "confidence": 1.0,
        })

    bp_shots = extract_frames_at(video, bp_times, base_dir, "bp") if bp_times else []

    # -- 分析窗：把阶段帧按**时间窗**分组，每窗做一次综合分析 ----------
    #
    # 为什么按阶段抽了帧还要再分窗：阶段帧是**证据**（哪几分钟有事发生），
    # 分析单元是**时间窗**（一条规则的时间范围）。规则判的是窗，
    # 报告也按窗给结论。少了这一步，窗口就只剩一个时间戳，
    # 局面类事实（大招就绪、击杀差、位置）一个都读不出来 ——
    # 那些规则于是永远判不了，而报告上只是「没触发」。
    window_bundles: List[Dict[str, Any]] = []
    if analysis_windows:
        from .window_analysis import bundle_slices, segments as _segments
        segs = _segments(analysis_windows)
        hud_regions = _hud_regions_for(video, base_dir, rounds)
        pending: List[Tuple[Any, Dict[str, Any]]] = []
        for b in bundle_slices(rounds, analysis_windows):
            d = b.to_dict()
            seg = None
            for s in segs:
                if s.contains(b.window.lo) and s.contains(b.window.hi):
                    seg = s
                    break
            d["segment_key"] = seg.key if seg else b.window.key
            d["segment_label"] = seg.label() if seg else b.window.label()
            pending.append((b, d))
        # 逐窗读并行（理由同 :func:`_read_window_bundles`）。
        # 轨迹切片按**整窗**给：阶段抽帧是一分钟一帧，
        # 只切两帧之间那一小段的话，点不够，探测器算不出运动特征。
        window_bundles = _read_window_bundles(
            video, base_dir, pending, hud_regions, hero_track,
            float(clock_info.get("game_start_t") or 0.0), progress,
            game=game, lane=lane)

    return {
        "rounds": rounds,
        "windows": window_bundles,
        "bp": bp_shots,
        "match_start_t": round(start_t, 1),
        "match_end_t": round(match_end_t, 1),
        "match_minutes": round((match_end_t - start_t) / 60.0, 1),
        "phase_samples": phase_samples,
    }


def build_skeleton(
    video: UploadedVideo,
    shots: List[Dict[str, Any]],
    game: str,
    game_version: str,
    match_id: str,
    windows: Optional[List[Dict[str, Any]]] = None,
) -> RecognitionResult:
    """产出识别骨架 —— 有窗口、有画面，**没有编造的数值**。

    分析单元是**时间窗**：窗内做一次综合分析，窗里的规则共用这份结论。
    时间片降级成**证据**（``slices``），报告里「看这一段」定位到窗内那一片。

    刻意不给每一窗塞默认血量/金币：那不是识别结果，是编造。
    骨架里只有「第几窗、覆盖哪几分钟、画面长这样」这些确凿信息，
    局面数值留给血条读取和用户标注。
    """
    rounds: List[RecognizedRound] = []
    if windows:
        rounds = _window_rounds(windows, game)
    else:
        rounds = _slice_rounds(shots)
    from . import settings
    from .vlm_recognizer import PROMPT_VERSION

    kind = settings.recognizer_kind()
    version = PROMPT_VERSION if kind == "vlm" else "cv-legacy"
    return RecognitionResult(
        match_id=match_id,
        game=game,
        game_version=game_version,
        frames=[],
        rounds=rounds,
        stats={"rounds": len(rounds), "source": video.video_id},
        video_id=video.video_id,
        # **哪套识别器、哪一版**：报告要写出来。
        # 换一版识别器结论就可能不同，读者得知道这局是哪套得出的。
        recognizer_id=kind,
        recognizer_version=version,
    )


def _window_rounds(windows: List[Dict[str, Any]],
                   game: str) -> List[RecognizedRound]:
    """按**时间窗**建骨架记录。

    每条记录 = 一个判定单元（一条规则的时间范围），
    事实里带的是**窗的起始分钟** —— 规则的时间条件按这个值判，
    落在窗里的规则就一定能触发（窗的起点必在它自己的范围内）。
    """
    out: List[RecognizedRound] = []
    for i, w in enumerate(windows):
        slices = list(w.get("slices") or [])
        if not slices:
            continue
        first = slices[0]
        lo = int(w.get("lo", 0))
        facts = [
            DetectedFact(
                name="time_min",
                value=lo,
                # 时间是**测**出来的（对局起点由 HUD 就位检出），
                # 但起点本身有误差，所以不是满分。
                confidence=0.8,
                method="window",
            )
        ]
        # 自动识别层给的事实：读出来就带上，读不出的**不写**
        # （不写 = 缺失 = 内核判「无法判定」，而不是猜一个值）。
        reading = (w.get("facts") or {})
        for fname, item in (reading.get("facts") or {}).items():
            facts.append(
                DetectedFact(
                    name=fname,
                    value=item.get("value"),
                    confidence=float(item.get("confidence", 0.0)),
                    method="auto",
                )
            )
        # 探测器认出来的**动作**：这一段玩家做过哪几件事。
        # 没有它，内核的「期望动作」比对会拿到空列表 → 判成「该做而没做」，
        # 于是报告里出现一条玩家没下过的结论。
        actions = [
            DetectedAction(
                kind=str(a.get("kind") or ""),
                value=a.get("value"),
                label=str(a.get("label") or ""),
                confidence=float(a.get("confidence", 1.0)),
                source=str(a.get("source") or "probe"),
            )
            for a in (reading.get("probe_actions") or [])
        ]
        out.append(
            RecognizedRound(
                stage=i + 1,
                phase=1,
                label_override=w.get("label") or f"第 {lo} 分钟",
                time_index=float(lo),
                video_time=float(first.get("timestamp", 0.0)),
                image=first.get("image", ""),
                facts=facts,
                actions=actions,
                probe_moments=list(reading.get("probe_moments") or []),
                probe_absent=list(reading.get("absent") or []),
                # 模型对攻略清单的裁决（已过硬校验，见
                # :mod:`app.collection.vlm_review`）。
                review=list(reading.get("review") or []),
                slices=slices,
                segment_key=str(w.get("segment_key") or ""),
                gate_confidence=1.0,
            )
        )
    return out


def _slice_rounds(shots: List[Dict[str, Any]]) -> List[RecognizedRound]:
    """没有窗口信息时的老路：一片一条记录。"""
    rounds: List[RecognizedRound] = []
    for s in shots:
        # 有游戏内时间就用它，没有就退回视频秒 —— 后者只是顺序坐标，
        # 不是「第几分钟」，所以标注页必须让人自己补，不要当成答案。
        minute = s.get("game_minute")
        # 时间从哪来：读时钟（准）还是推算（准不准要看估得对不对）。
        time_source = s.get("time_source") or "clock"
        facts: List[DetectedFact] = []
        if minute is not None:
            # 局内时间必须进事实：适配器归一化时从 ``facts["time_min"]``
            # 取局内时间，取不到就把每一回合都当成第 0 分钟并标成缺失 ——
            # 于是所有带时间窗的模式一条都不触发，报告看起来只是「没结论」。
            #
            # 推算出来的分钟**照样要采信**（低于 0.30 才会被丢掉），
            # 只是要标成低置信度，让报告提醒用户核对。
            facts.append(
                DetectedFact(
                    name="time_min",
                    value=int(minute),
                    confidence=1.0 if time_source == "clock" else 0.4,
                    method=time_source,
                )
            )
        rounds.append(
            RecognizedRound(
                stage=s["round_no"],
                phase=1,
                label_override=(
                    f"第 {int(minute)} 分钟" if minute is not None
                    else f"第 {s['round_no']} 回合"
                ),
                time_index=float(minute if minute is not None else s["timestamp"]),
                # 原视频里的秒数。报告的「看这一段」靠它定位 ——
                # 与 time_index（游戏内分钟）是两回事，混用会跳到录屏开头。
                video_time=float(s.get("timestamp", 0.0)),
                image=s.get("image", ""),
                facts=facts,            # 留给血条读取 / 用户标注
                actions=[],
                gate_confidence=s["confidence"],
            )
        )
    return rounds


def summarize(video: UploadedVideo, scan: Dict[str, Any],
               shots: List[Dict[str, Any]],
               bp: Optional[List[Dict[str, Any]]] = None,
               product: str = "moba_slice") -> Dict[str, Any]:
    """把扫描结果整理成给用户看的说明。

    ``product`` 说明这次扫出来的是哪一套产物：

        ``phase``      BP 帧序列 + 按阶段均匀抽帧（读得到时钟时）
        ``moba_slice`` 按信息密度切时间片（读不到时钟时）
    """
    return {
        "video": video.to_dict(),
        "scan": scan,
        "product": product,
        "round_count": len(shots),
        "rounds": shots,
        "bp": bp or [],
        "bp_count": len(bp or []),
    }
