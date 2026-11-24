"""视频上传、画面归一与抽帧。

这一层负责**几何与搬运**，不负责「看懂对局」：

    上传视频 → 判录屏方向 → 裁黑边 → 抽帧落盘

看懂对局是 :mod:`app.recognition2` 的事（切段 → 逐段读画面 → 裁决）。
这里只提供它需要的两样东西：

- **画面归一**（:func:`detect_view` / :func:`video_view` /
  :func:`crop_to_band`）：手机录屏可能横竖屏颠倒、上下有黑边 ——
  不归一，模型看到的是错的画面，认英雄会认不出来；
- **抽帧落盘**（:func:`extract_frames_at` / :func:`extract_bp_frames`）：
  证据帧要真的存下来，报告里的「看这一段」才点得开。

为什么不再有「读时钟」「切时间片」那几套
--------------------------------------
那是 v1 的做法，两处都不准：

- **OpenCV 模板匹配读钟** —— 实测在这段录屏上 5 次读错 4 次
  （02:38 读成 02:16、12:13 读成 02:43…）。现在时间轴由**模型逐帧读钟**
  （:mod:`app.recognition2.clockaxis`）；
- **按规则窗口切段** —— 画面里发生的事只要不落在攻略的时间边界上，
  就归不进任何一段，于是「这段没识别到明确操作」。现在分段由
  **画面变化**驱动（:mod:`app.recognition2.sampler`）。

两条老路都删掉了，不是「留着当备用」—— 留着就会有人去调它，
而它给出的答案看着像样、其实不对。
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from . import clock


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

    #: 只读前 N 段（调试用）。0 = 整局。
    #:
    #: 为什么留这个开关
    #: --------------
    #: 一局读下来要花好几次模型调用；只为了验证「链路通不通」就整局跑一遍，
    #: 既慢又烧额度。加了它，调试时可以只读前几段 —— **但它只影响
    #: 读画面那一层**，不影响切段、裁决与综述。
    limit_segments: int = 0


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
    """这个视频的取景信息：方向、黑边。

    判一次、存起来：方向判定要抽帧搜索，重复算一遍要几十秒。

    **这里不再找时钟锚点** —— 局内时刻现在由**模型逐帧读**出来
    （:mod:`app.recognition2.clockaxis`）。OpenCV 模板匹配读钟那条路
    实测 5 次读错 4 次，已经撤掉了；留着锚点搜索只是白花时间。
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
    out = {
        "rotation": rotation,
        "meta_rotation": meta_rot,
        "band": band,
        "orientation_score": view.get("orientation_score", 0.0),
        "orientation_votes": view.get("orientation_votes"),
        "detected": bool(view),
    }
    if key:
        _VIEW_CACHE[key] = out
    return out



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


# --------------------------------------------------------------------------
# 抽帧
# --------------------------------------------------------------------------


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


#: BP 阶段抽几帧。
BP_SAMPLES = 10


def extract_bp_frames(video: UploadedVideo, base_dir: str,
                      start_t: Optional[float] = None,
                      samples: int = BP_SAMPLES,
                      progress=None) -> List[Dict[str, Any]]:
    """抽对局开始**之前**的那一段（BP 阶段）的帧。

    为什么单独一个函数
    ------------------
    v1 那套「按阶段均匀抽帧」把 BP 帧和局内帧混在一个函数里，
    而局内那一半是**按规则窗口**切的（那正是要被撤掉的东西）。
    现在局内帧由 :mod:`app.recognition2.sampler` 按**画面变化**切，
    BP 帧只需要「对局开始之前那几秒」—— 单独抽，单独命名。

    ``start_t`` 是「对局在这一段视频里的第几秒」。它定不下来
    （模型读钟失败）就**不抽** —— 抽出来的多半是加载画面，
    让模型去读「谁禁了谁」只会得到一堆编造的答案。
    """
    if start_t is None:
        return []
    start = float(start_t)
    if start < 5.0:
        # 对局几乎就是从视频开头开始的 —— 没有 BP 段。
        return []
    times = [("BP %d" % (i + 1), t)
             for i, t in enumerate(_even_times(0.0, start, samples))]
    return extract_frames_at(video, times, base_dir, "bp", progress=progress)






