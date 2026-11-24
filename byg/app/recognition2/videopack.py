"""**视频打包层**：把一局录屏变成「模型能直接看」的形态。

为什么要有它
------------
「让模型直接看视频」有两道**硬门槛**，都在体积上：

1. MiniMax 直传视频上限 **50 MB**，请求体上限 **64 MB**，
   而 base64 还要把体积撑大三分之一；
2. Files API（``purpose=video_understanding``）上限 **512 MB**，
   但要先上传，传完引用 ``mm_file://{file_id}``。

而一局王者录屏实测：1280×574 / 59.5 fps / 18.8 分钟 = **749 MB** ——
两条路都超。所以必须先把视频**压到能传的大小**。

本机**没有 ffmpeg**，只有 OpenCV —— 它能编 MPEG-4 Part 2（``mp4v``），
编不了 H.264（缺 openh264 的 DLL）。所以打包出来的片子是 mp4v。

**这个风险要说明白**：对方能不能解 mp4v，本机没额度、验不了。
真跑第一次就会知道 —— 解不了会报 400，错误信息会带出来。

**画面归一（转正 / 裁黑边）要在这里做**
--------------------------------------
旧链路是「抽帧 → 每帧转正裁边 → 给模型」；现在整段视频直接给模型，
转正就得**烙进视频里**。所以只要有 ``view``（转正 + 裁边），
打包时就不能「原样复制」，必须重编码。

**时间轴会漂**
--------------
重编码后帧率变了，模型报的「第 N 秒」指的是**打包后**的视频秒，
不是原视频秒。漂移很小（重编码用浮点步进，误差 < 0.2%），
但一局 19 分钟累积下来也有几秒 —— 规则的时间窗是分钟级，几秒无伤，
但仍然要**换算回去**，不能拿打包后的秒数直接当原视频秒数。
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

#: 打包后的**采样帧率**（每秒写几帧）。
#:
#: 为什么不是原片的 59.5 fps：模型那边有自己的采样密度
#: （``fps`` 参数，0.2~5），我们把片子做密了只是白占体积。
#: 实测 1280×574 @ 2 fps 的 mp4v ≈ 130 MB / 19 分钟。
DEFAULT_PACK_FPS = 2.0

#: 切片（裁决用）的采样帧率 —— 片段短，可以密一点，看得清动作。
DEFAULT_CLIP_FPS = 4.0

#: 重编码用的编码器。
#:
#: 只有 mp4v：本机 OpenCV 缺 openh264 的 DLL，``avc1`` 开不了。
_CODEC = "mp4v"


def h264_available() -> bool:
    """本机能不能编 H.264。

    没有 ffmpeg / openh264 时是 False —— 那种机器只能编 mp4v，
    而 MiniMax 这类收视频的接口会拒 mp4v，这时该改走抽帧。
    """
    try:
        import cv2
        import os
        import tempfile

        path = os.path.join(tempfile.gettempdir(), "_dsh_h264_probe.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc("a", "v", "c", "1"),
                                 2.0, (64, 64))
        ok = bool(writer.isOpened())
        writer.release()
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        return ok
    except Exception:                                  # noqa: BLE001
        return False


def _remove_quiet(path: str) -> None:
    """删一个临时文件 —— 删不掉就算了，别把整段打包带崩。"""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


@dataclass
class VideoInfo:
    """一个视频文件的基本信息。"""

    path: str = ""
    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    size: int = 0
    codec: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "duration": round(float(self.duration), 2),
            "fps": round(float(self.fps), 3),
            "width": int(self.width),
            "height": int(self.height),
            "size": int(self.size),
            "codec": self.codec,
        }


@dataclass
class PackResult:
    """打包结果。"""

    path: str = ""
    #: ``copy``（原样复制，编码不变）/ ``transcode``（重编码）。
    mode: str = ""
    #: 实际写进去的编码（``mp4v`` = MPEG-4 Part 2）。
    codec: str = ""
    fps: float = 0.0
    width: int = 0
    height: int = 0
    duration: float = 0.0
    size: int = 0
    #: 原视频的时长（``duration`` 是打包后的）。
    source_duration: float = 0.0
    #: 原视频的起点（切片用；整局打包一般是 0）。
    source_start: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def time_scale(self) -> float:
        """打包后的秒 → 原视频秒的**比例**。

        重编码会改变实际帧率（步进取整），所以模型报的秒数要乘这个
        比例才回到原视频。整局打包一般接近 1.0。
        """
        if self.duration <= 0:
            return 1.0
        return float(self.source_duration) / float(self.duration)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "mode": self.mode,
            "codec": self.codec,
            "fps": round(float(self.fps), 3),
            "width": int(self.width),
            "height": int(self.height),
            "duration": round(float(self.duration), 2),
            "size": int(self.size),
            "source_duration": round(float(self.source_duration), 2),
            "time_scale": round(self.time_scale, 5),
        }


# ---------------------------------------------------------------- 基础


def probe(path: str) -> VideoInfo:
    """读一个视频的基本信息。读不出来就返回**空信息**（不猜）。"""
    import cv2

    info = VideoInfo(path=path)
    try:
        info.size = int(os.path.getsize(path))
    except OSError:
        return info
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return info
    try:
        info.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        info.codec = "".join(chr((fourcc >> (8 * i)) & 0xFF)
                             for i in range(4)).strip("\x00 ")
        if info.duration <= 0 and info.fps > 0 and frames > 0:
            info.duration = frames / info.fps
        if info.duration <= 0 and frames > 0 and info.fps <= 0:
            info.duration = 0.0
    finally:
        cap.release()
    return info


def _writer(cv2, path: str, fps: float, size: Tuple[int, int]):
    """开一个写视频的句柄。**开不了就返回 None**（别静默写空文件）。"""
    fourcc = cv2.VideoWriter_fourcc(*_CODEC)
    writer = cv2.VideoWriter(path, fourcc, float(fps), (int(size[0]), int(size[1])))
    if not writer.isOpened():
        writer.release()
        return None
    return writer


def _transcode(src: str, dst: str, start: float, end: Optional[float],
               fps: float, scale: float = 1.0,
               transform: Optional[Callable[[Any], Any]] = None
               ) -> VideoInfo:
    """把 ``src[start:end]`` 重编码成 ``dst``。

    **时间轴不能漂** —— 所以按浮点步进取帧，而不是「每 N 帧取一张」
    （后者取整误差会累积：59.5 fps 每 30 帧取一张 → 实际 1.983 fps，
    一局 19 分钟就差 8 秒）。

    写完把**写进去多少帧**记下来，用它算真实时长。
    """
    import cv2

    src_info = probe(src)
    out = VideoInfo(path=dst, codec=_CODEC, fps=float(fps))
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        cap.release()
        return out
    try:
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(start) * 1000.0)
        src_fps = float(src_info.fps or 30.0) or 30.0
        width = int(src_info.width or 0)
        height = int(src_info.height or 0)
        if width <= 0 or height <= 0:
            return out
        # 先取一帧、**先归一再定尺寸** ——
        # 顺序反了的话，转正后的画面（宽高互换）会被缩放回原尺寸，
        # 旋转就被抹掉了（实测踩过）。
        ok, first = cap.read()
        if not ok or first is None:
            return out
        if transform is not None:
            try:
                first = transform(first)
            except Exception:                              # noqa: BLE001
                pass
        if first is None:
            return out
        height, width = first.shape[:2]
        if scale != 1.0:
            width = max(2, int(width * scale) // 2 * 2)
            height = max(2, int(height * scale) // 2 * 2)
        writer = _writer(cv2, dst, fps, (width, height))
        if writer is None:
            return out
        stride = src_fps / max(0.1, float(fps))
        acc = 0.0
        written = 0
        try:
            frame = first
            while True:
                if end is not None and start + written * (1.0 / fps) > end:
                    break
                if frame is not None:
                    if transform is not None:
                        try:
                            frame = transform(frame)
                        except Exception:                  # noqa: BLE001
                            pass
                    if frame is None:
                        continue
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height))
                    writer.write(frame)
                    written += 1
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                acc += 1.0
                if acc < stride:
                    frame = None
                    continue
                acc -= stride
        finally:
            writer.release()
        out.width = width
        out.height = height
        out.duration = written / max(0.1, float(fps))
    finally:
        cap.release()
    try:
        out.size = int(os.path.getsize(dst))
    except OSError:
        out.size = 0
    return out


# ---------------------------------------------------------------- 打包


def pack_video(path: str, out_dir: str, *,
               max_bytes: int = 480 * 1024 * 1024,
               fps: float = DEFAULT_PACK_FPS,
               transform: Optional[Callable[[Any], Any]] = None,
               stem: str = "match") -> PackResult:
    """把整局视频压成「能传」的形态。

    两条路
    ------
    - **原样复制**：体积已经够小、且画面不需要归一 → 直接复制。
      好处是**编码不变**（原片是 H.264 就还是 H.264），
      对方解码一定没问题 —— 这是最稳的一种。
    - **重编码**：体积超了，或画面要转正/裁边 → 用 mp4v 重编码，
      帧率逐档往下压，直到进得了体积上限。

    ``max_bytes`` 是**成品**的上限。调用方按「直传还是走 Files」给不同值。

    **要切片段就别用这里** —— 用 :func:`cut_clip`（裁决与认英雄都走它）：
    它按时间窗切、算得出「片段秒 ↔ 原视频秒」的换算，体积天然就小。
    """
    os.makedirs(out_dir, exist_ok=True)
    src = probe(path)
    result = PackResult(source_duration=src.duration)
    if src.duration <= 0 or src.width <= 0:
        result.notes.append("打不开这个视频（或者它不是视频文件）。")
        return result

    dst = os.path.join(out_dir, "%s.mp4" % stem)
    need_transform = transform is not None

    if not need_transform and src.size <= max_bytes:
        try:
            shutil.copyfile(path, dst)
        except OSError as exc:
            result.notes.append("复制视频失败：%s" % exc)
            return result
        result.path = dst
        result.mode = "copy"
        result.codec = src.codec or "(原样)"
        result.fps = src.fps
        result.width = src.width
        result.height = src.height
        result.duration = src.duration
        result.size = src.size
        return result

    # 重编码：帧率与分辨率逐档往下压。
    profiles = [(fps, 1.0), (fps * 0.75, 1.0), (fps * 0.55, 1.0),
                (fps * 0.55, 0.8), (fps * 0.4, 0.7), (fps * 0.3, 0.6)]
    best: Optional[PackResult] = None
    best_path = ""
    written: List[str] = []
    for idx, (out_fps, scale) in enumerate(profiles):
        # **每档写自己的文件**。
        #
        # 六档都写同一个 dst 时，磁盘上留下的是**最后一档**，
        # 而 best 可能是另一档 —— 于是 pack.size / duration / fps
        # 描述的不是磁盘上那个文件，``time_scale`` 算错，
        # 模型报的秒整体漂移（现在整局打包 scale≈1，漂得少；
        # 一旦不是整局就是整局偏移）。
        tmp = "%s.p%d.mp4" % (dst, idx)
        info = _transcode(path, tmp, 0.0, None, out_fps, scale, transform)
        if info.size <= 0:
            _remove_quiet(tmp)
            continue
        written.append(tmp)
        cand = PackResult(
            path=tmp, mode="transcode", codec=_CODEC, fps=out_fps,
            width=info.width, height=info.height, duration=info.duration,
            size=info.size, source_duration=src.duration)
        if best is None or cand.size < best.size:
            best = cand
            best_path = tmp
        if cand.size <= max_bytes:
            break
    if best is None:
        for tmp in written:
            _remove_quiet(tmp)
        result.notes.append("重编码没成功（OpenCV 开不了写视频的句柄）。")
        return result
    # **把最佳那一档落到最终路径**，其余删掉。
    if os.path.abspath(best_path) != os.path.abspath(dst):
        os.replace(best_path, dst)
    for tmp in written:
        if os.path.abspath(tmp) != os.path.abspath(dst):
            _remove_quiet(tmp)
    best.path = dst
    if best.size > max_bytes:
        best.notes.append(
            "压到最小还是 %d MB，超过上限 %d MB —— 只能分段上传。"
            % (best.size // 1024 // 1024, max_bytes // 1024 // 1024))
    if need_transform:
        best.notes.append("画面已按录屏的实际方向转正、裁掉黑边。")
    if best.codec != src.codec:
        best.notes.append(
            "视频重编码过（%s → %s）：本机没有 H.264 编码器，"
            "只能用 MPEG-4 Part 2。" % (src.codec or "原片", best.codec))
    return best


def cut_clip(path: str, start: float, end: float, out_path: str, *,
             fps: float = DEFAULT_CLIP_FPS,
             transform: Optional[Callable[[Any], Any]] = None) -> VideoInfo:
    """切出一段视频（裁决用）。

    片段短（几十秒），压出来的文件很小 → 可以**内联**传给模型，
    不用先上传。
    """
    start = max(0.0, float(start))
    end = max(start + 0.5, float(end))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    info = _transcode(path, out_path, start, end, fps, 1.0, transform)
    info.source_start = start
    return info


# ---------------------------------------------------------------- 缩略图


def save_thumbnails(path: str, times: List[float], frames_dir: str, *,
                    prefix: str = "seg", index: int = 0,
                    transform: Optional[Callable[[Any], Any]] = None,
                    quality: int = 80) -> List[Optional[Tuple[str, float]]]:
    """在指定的几个时刻各存一张图（**报告里的证据缩略图**）。

    命名沿用旧链路的格式 ``seg%04d_f%d_%07.1f.jpg`` —— 有别的代码
    按文件名反推时间，保持一致就不用改那些地方。

    **返回列表与 ``times`` 一一对应**（存不下来的那个位置是 ``None``）。

    为什么不能「存不下来的就跳过」
    ------------------------------
    调用方是按**下标**把图挂回段落上的（``zip(beats, thumbs)``）。
    跳过会让后面每一段都拿到**前一段的图** —— 报告上「看这一段」
    跳到错误的秒数，证据和结论对不上。模型报的秒数超过录像时长时
    就会踩到（读帧失败、写图失败都会跳过）。
    所以这里**位置留着**，让调用方知道「这一段没有图」。

    **这跟「抽帧给模型看」是两回事**：模型判读用的是**视频**，
    这些图只是报告里让人能一眼看到「说的是哪一秒」。
    """
    from ..collection.video_upload import imwrite_u

    if not times:
        return []
    import cv2

    os.makedirs(frames_dir, exist_ok=True)
    out: List[Optional[Tuple[str, float]]] = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return out
    try:
        for i, t in enumerate(times, start=1):
            try:
                t = float(t)
            except (TypeError, ValueError):
                out.append(None)
                continue
            if t < 0:
                out.append(None)
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                out.append(None)
                continue
            if transform is not None:
                try:
                    frame = transform(frame)
                except Exception:                          # noqa: BLE001
                    pass
            actual = t
            try:
                actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            except Exception:                              # noqa: BLE001
                pass
            name = "%s%04d_f%d_%07.1f.jpg" % (prefix, index, i, actual)
            if imwrite_u(os.path.join(frames_dir, name), frame, quality=quality):
                out.append((name, float(actual)))
            else:
                out.append(None)
    finally:
        cap.release()
    return out
