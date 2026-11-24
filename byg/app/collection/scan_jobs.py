"""后台扫描任务。

为什么要后台
------------
一局对局的录屏动辄 2 GB，扫完要几十秒。放在请求里同步做，用户盯着一个
不动的进度条等，还以为卡死了；服务器端则是这个连接一直被占着。

所以：上传和扫描拆开。上传完立刻返回一个 ``job_id``，扫描在后台线程里跑，
前端轮询进度。这样进度是**真的**（真在跑，不是假动画），失败了也能
看到原因，而不是一个卡住的请求。

跑的是哪条链路
--------------
**只有 v2**（:mod:`app.recognition2`）::

    切段 → 逐段读画面 → 定对局级事实 → 筛候选 → 模型裁决 → 全局综述

老的那套「按规则窗口切段 + 逐窗枚举动作」的识别链路
（``vlm_recognizer`` / ``vlm_review`` / ``probes`` / ``window_analysis`` /
``realign``）**已经删掉了**。

为什么删
--------
它的产物是**动作码** —— 而那套码只覆盖得到模式库词表里的动作，
读不出来的段一律显示「这段没识别到明确操作」。更要命的是它按
**规则窗口**切段：画面里发生的事只要不落在攻略的时间边界上，
就归不进任何一段。v2 是**画面驱动**的：发生了就看得见。

线程安全的考虑
--------------
扫描只碰文件系统、OpenCV 与模型 API，**不碰数据库**。数据库操作留在请求
线程里（标注提交、分析），所以不需要跨线程传 session。
"""

from __future__ import annotations

import contextvars
import os
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from . import video_upload as vu
from ..recognition2 import prefill
from ..recognition2.pipeline import PipelineConfig, run_pipeline

#: 任务状态
PENDING = "pending"
SCANNING = "scanning"
EXTRACTING = "extracting"
DONE = "done"
FAILED = "failed"

STAGE_TEXT = {
    PENDING: "排队中",
    SCANNING: "正在分析画面变化",
    EXTRACTING: "正在读画面",
    DONE: "完成",
    FAILED: "失败",
}


@dataclass
class ScanJob:
    """一次扫描任务。"""

    job_id: str
    match_id: str
    video: vu.UploadedVideo
    game: str
    game_version: str
    #: **玩家自己填的英雄 / 分路** —— 上传前填的，识别层以它为准。
    hero: str = ""
    position: str = ""
    status: str = PENDING
    progress: float = 0.0
    error: str = ""
    #: 进度条上的阶段说明。
    #:
    #: 默认按状态给一句话，但那句话未必合身：王者没有「回合」，
    #: 那一步在做的是「分析画面变化」。所以这一句是**可覆盖**的。
    stage_text: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    finished_at: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, **kw: Any) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "match_id": self.match_id,
            "game": self.game,
            "game_version": self.game_version,
            "status": self.status,
            "stage_text": self.stage_text or STAGE_TEXT.get(self.status, self.status),
            "progress": round(self.progress, 3),
            "error": self.error,
            "summary": self.summary,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class JobRunner:
    """后台任务执行器。

    用普通线程而不是进程池：扫描是 IO + 数值计算密集，GIL 只在纯 Python
    片段上起作用，而 OpenCV 解码与 numpy 运算都会释放 GIL，线程已经够用，
    还能直接共享内存里的状态（进度、错误）。
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, ScanJob] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, match_id: str, video: vu.UploadedVideo,
               game: str, game_version: str,
               hero: str = "", position: str = "") -> ScanJob:
        job = ScanJob(job_id, match_id, video, game, game_version,
                      hero=hero, position=position)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[ScanJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def all_jobs(self) -> List[ScanJob]:
        with self._lock:
            return list(self._jobs.values())

    def run_async(self, job: ScanJob, base_dir: str,
                  cfg: Optional[vu.ScanConfig] = None,
                  on_done: Optional[Callable[[ScanJob], None]] = None) -> None:
        # **带上当前的上下文副本**：线程默认不继承 contextvar，
        # 而识别链要读的东西（API Key 所在的设备配置）就挂在上面 ——
        # 丢了就读不到 Key，报告会是空的，而且看不出是哪一步丢了。
        ctx = contextvars.copy_context()
        t = threading.Thread(
            target=ctx.run, args=(self._run, job, base_dir, cfg, on_done),
            daemon=True
        )
        t.start()

    # ------------------------------------------------------------------ 主流程

    def _run(self, job: ScanJob, base_dir: str,
             cfg: Optional[vu.ScanConfig], on_done) -> None:
        try:
            job.set(status=SCANNING, progress=0.02,
                    stage_text=STAGE_TEXT[SCANNING])

            # **画面归一**：手机录屏可能横竖屏颠倒、上下有黑边。
            # 不归一，模型看到的就是歪的画面 —— 认英雄会认不出来，
            # 读行为会读成别的动作。转正与裁黑边的口径必须与识别层一致。
            view = vu.video_view(job.video, hud_layout=self._hud_layout(job.game))

            # 证据帧的落盘目录 —— **必须**是报告里那张 URL 指的地方，
            # 否则「看这一段」点开全是 404。
            frames_dir = os.path.join(
                vu.video_store_dir(base_dir), job.video.video_id + "_frames")
            os.makedirs(frames_dir, exist_ok=True)
            #: 打包视频 / 裁决片段的落盘目录。
            video_dir = os.path.join(
                vu.video_store_dir(base_dir), job.video.video_id + "_video")
            os.makedirs(video_dir, exist_ok=True)

            duration = float(getattr(job.video, "duration_sec", 0.0) or 0.0)
            #: 读到的对局级事实（英雄/分路/胜负）—— 回调里写进来，
            #: 摘要要用。
            facts_box: Dict[str, Any] = {}

            def _resolve_facts(start_t: Optional[float]) -> Dict[str, Any]:
                facts = prefill.resolve_match_facts(
                    job.video, base_dir, start_t, duration)
                facts_box.update(facts)
                return facts

            def _progress(stage: str, frac: float) -> None:
                job.set(progress=max(0.02, min(0.99, float(frac))),
                        status=SCANNING if frac < 0.8 else EXTRACTING,
                        stage_text=stage)

            from ..collection import settings as _settings

            result = run_pipeline(job.video.path, PipelineConfig(
                data_dir=_settings.data_dir(),
                frames_dir=frames_dir,
                video_dir=video_dir,
                game=job.game,
                # 玩家上传前填的身份 —— 识别层与提示词都以它为准。
                hero=job.hero,
                position=job.position,
                view=view,
                resolve_facts=_resolve_facts,
                on_progress=_progress,
            ))

            meta = result.get("meta") or {}
            notes = list(result.get("notes") or [])
            rounds = list(result.get("rounds") or [])

            # -- BP：对局开始**之前**的那一段 --------------------------------
            # BP 帧要落在对局起点之前，所以得等起点定下来才能抽。
            start_t = meta.get("match_start_t")
            bp: Dict[str, Any] = {}
            # **BP 可能已经在 facts 里读过了** —— 认英雄要拿禁用名单，
            # 那一步在链路里就调过一次 ``resolve_match_facts``，
            # 这里别再读一遍（那是一次模型调用，不便宜）。
            if isinstance(facts_box.get("bp"), dict) and facts_box["bp"]:
                bp = facts_box["bp"]
            elif start_t:
                try:
                    vu.extract_bp_frames(job.video, base_dir, float(start_t))
                    bp = prefill.read_bp(job.video, base_dir)
                except Exception:                      # noqa: BLE001
                    bp = {}
            if bp:
                facts_box["bp"] = bp

            summary = {
                "video": job.video.to_dict(),
                # 报告层要的就是这个形状（每段带模型裁决与模型原话）。
                "rounds": rounds,
                "round_count": len(rounds),
                "match_id": job.match_id,
                "game": job.game,
                "game_version": job.game_version,
                # 模型读出来的**整体情况**（英雄/分路/胜负/BP）。
                "prefill": facts_box,
                # **整局综述 +「攻略之外的问题」**（observer 那一步）。
                #
                # 2026-11-13 之前**没传**：那一步花了模型钱算出来，
                # 到这一步就丢了 —— 报告正文于是只剩「模式库判出来的偏离」
                # 与血线/死亡条目；模式库没覆盖到的那些问题一条都上不了屏，
                # 而用户要的恰恰是「这一局到底哪里有问题」。
                "overview": result.get("overview"),
                # 留痕：没读出来的段、判不了的原因、模型调不通的原因。
                # 空表最容易被读成「打得没问题」—— 那这里必须说清楚。
                "notes": notes,
                "match_start_t": start_t,
                "meta": meta,
            }

            job.set(
                progress=1.0, summary=summary,
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            # 顺序要紧：**先把结果交出去（挂到会话上），再标完成**。
            #
            # 前端一看到 done 就去建对局。顺序反了的话，那会儿会话还没挂上，
            # 建对局拿到的是「解析结果已失效」，用户白等一场。
            if on_done:
                on_done(job)
            job.set(status=DONE)

        except Exception as exc:                  # noqa: BLE001
            job.set(
                status=FAILED,
                error=f"{type(exc).__name__}: {exc}",
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            traceback.print_exc()
            if on_done:
                on_done(job)

    # ------------------------------------------------------------------ 配置

    @staticmethod
    def _hud_layout(game: str) -> Dict[str, str]:
        """该游戏的 HUD 布局；没有就返回空表。

        采集层用它判**录屏方向**：手机录屏的旋转元数据不一定对
        （实测遇到过写反的），方向判错的话整局的截图全都倒着看。
        布局从适配器取，不在采集层写死一份。
        """
        try:
            from ..adapters.base import get_adapter

            adapter = get_adapter(game)
        except Exception:                          # noqa: BLE001
            return {}
        return dict(getattr(adapter, "hud_layout", {}) or {})

    def cleanup(self, keep: int = 50) -> int:
        """丢掉过老的已完成任务，防止内存无限增长。"""
        with self._lock:
            done = [j for j in self._jobs.values() if j.status in (DONE, FAILED)]
            done.sort(key=lambda j: j.created_at)
            removed = 0
            for job in done[: max(0, len(done) - keep)]:
                self._jobs.pop(job.job_id, None)
                removed += 1
            return removed


#: 进程级单例
RUNNER = JobRunner()
