"""后台扫描任务。

为什么要后台
------------
一局对局的录屏动辄 2 GB，扫完要几十秒。放在请求里同步做，用户盯着一个
不动的进度条等，还以为卡死了；服务器端则是这个连接一直被占着。

所以：上传和扫描拆开。上传完立刻返回一个 ``job_id``，扫描在后台线程里跑，
前端轮询进度。这样进度是**真的**（真在跑，不是假动画），失败了也能
看到原因，而不是一个卡住的请求。

线程安全的考虑
--------------
扫描只碰文件系统与 OpenCV，**不碰数据库**。数据库操作留在请求线程里
（标注提交、分析），所以不需要跨线程传 session。
"""

from __future__ import annotations

import threading
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from . import video_upload as vu

#: 任务状态
PENDING = "pending"
SCANNING = "scanning"
EXTRACTING = "extracting"
DONE = "done"
FAILED = "failed"

STAGE_TEXT = {
    PENDING: "排队中",
    SCANNING: "正在分析画面变化",
    EXTRACTING: "正在截取关键帧",
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
               game: str, game_version: str) -> ScanJob:
        job = ScanJob(job_id, match_id, video, game, game_version)
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
        t = threading.Thread(
            target=self._run, args=(job, base_dir, cfg, on_done), daemon=True
        )
        t.start()

    def _run(self, job: ScanJob, base_dir: str,
             cfg: Optional[vu.ScanConfig], on_done) -> None:
        try:
            job.set(status=SCANNING, progress=0.02)

            eff_cfg = cfg or vu.ScanConfig()
            eff_cfg.sample_sec = vu.adapt_sample_sec(job.video.duration_sec)

            def on_scan(frac: float, _stage: str) -> None:
                # 扫描占 80%，抽帧占 20%
                job.set(progress=0.02 + frac * 0.8, status=SCANNING)

            def on_ex(frac: float, _stage: str) -> None:
                job.set(progress=0.85 + frac * 0.14, status=EXTRACTING)

            phase_windows = self._phase_windows(job.game)
            hud_layout = self._hud_layout(job.game)
            if phase_windows:
                # 王者没有「回合」：这一步在做的是「分析画面变化」。
                job.set(stage_text="正在分析画面变化")
            else:
                job.set(stage_text=STAGE_TEXT[SCANNING])

            # -- 一、探时钟 -------------------------------------------------
            # 读得到时钟就按阶段抽帧（最准）；读不到就按信息密度切时间片。
            # 两条路都需要时钟探测，所以只探一次。
            clock_track: Dict[str, Any] = {}
            scan: Dict[str, Any] = {}
            if phase_windows:
                clock_track, scan = vu.probe_clock(
                    job.video, progress=on_scan, hud_layout=hud_layout
                )

            # -- 二、产出 ----------------------------------------------------
            if not phase_windows:
                # 这个产品只做王者荣耀。别的游戏没有「时间片」这套切法，
                # 与其悄悄按自走棋的回合切一遍（切出来的东西玩家根本没法用），
                # 不如直接说清楚。
                raise RuntimeError(
                    f"{job.game} 还没有视频解析：目前只支持王者荣耀的对局录屏。"
                )
            if clock_track.get("ok"):
                job.set(status=EXTRACTING, progress=0.85,
                         stage_text="正在读画面")
                # **先读分路**：清单要按玩家的线裁剪（见
                # :func:`app.rules.checklist.window_checklist`），
                # 分路得在读窗**之前**知道。读分路只需要开局那几帧，
                # 不依赖抽帧结果，所以排在最前面。
                identity = self._read_identity(
                    job, base_dir, clock_track.get("game_start_t"), scan)
                product = self._phase_product(
                    job, base_dir, clock_track, scan, on_ex,
                    lane=(identity.get("position") or ""),
                    hero=(identity.get("hero") or ""),
                )
            else:
                # 读不到时钟（机型没标定、录屏是竖屏带黑边…）。
                #
                # 这里**不**退回自走棋的「回合边界」那套：王者没有回合，
                # 按灰度差切出来的「回合」全是镜头移动，跟规则库的时间窗
                # 毫无关系 —— 实测一局王者切出 80 个「回合」，真正值得判断
                # 的时刻不到 30 个。
                #
                # 改成按**信息密度**切时间片：打架、转场、抢资源的时段
                # 多切几刀，安静时段少切。
                points, scan = vu.activity_scan(
                    job.video, eff_cfg, progress=on_scan, hud_layout=hud_layout
                )
                job.set(status=EXTRACTING, progress=0.85,
                         stage_text="正在读画面")
                identity = self._read_identity(
                    job, base_dir, self._guess_start_t(points, scan), scan)
                product = self._moba_product(
                    job, base_dir, points, scan, on_ex,
                    lane=(identity.get("position") or ""),
                    hero=(identity.get("hero") or ""),
                )

            shots = product["rounds"]
            bp = product.get("bp") or []

            # 身份读取是独立的一次 VLM 投票；它可能因 HUD 遮挡而保守地
            # 留空。但窗级识别已经看到了整局的位置，不能丢掉这份跨窗证据。
            # 例如本局 6 段都在野区，是「打野」的强证据，不该仍给出
            # 「分路未知」并跳过全部规则。
            # ``read_lane_and_result`` 以字面量 ``unknown`` 表示未识别；
            # 它和空字符串一样都必须进入窗口证据兜底，不能把它当成分路。
            identity_position = str((identity or {}).get("position") or "").strip()
            if identity_position in ("", "unknown"):
                inferred = self._infer_lane_from_rounds(shots)
                if inferred:
                    identity = dict(identity or {})
                    identity["position"] = inferred
                    identity["position_source"] = "window_positions"

            # 读**这一局的整体情况**：分路、胜负、BP 禁用与选人。
            #
            # 这三样以前是让人一格一格填的（用户原话：「难道还要我自己去选？」）。
            # 现在由模型从画面里读 —— 读不出来的就留空，**不猜**。
            prefill = self._read_prefills(job, base_dir, product,
                                          lane_info=identity)

            summary = vu.summarize(
                job.video, scan, shots, bp=bp, product=product["kind"]
            )
            for key in ("match_start_t", "match_end_t", "match_minutes",
                       "slice_minutes", "match_start_reason",
                       "match_start_cross_check", "windows"):
                if key in product:
                    summary[key] = product[key]
            # 起点是**测**出来的，理由要跟着走 —— 用户看到「第 125 秒」，
            # 得知道这是怎么来的（也能据此判断要不要手动改）。
            for key in ("match_start_reason", "match_start_cross_check"):
                if key in (scan or {}):
                    summary[key] = scan[key]
            # 分析单元是时间窗 —— 页面上是「逐段标注」，不是「逐时间片标注」。
            # 显示的是**分析段**的个数（段里的判定单元共用一份答案）。
            seg_keys = {w.get("segment_key") or w.get("key")
                        for w in (product.get("windows") or [])}
            summary["window_count"] = len(seg_keys)
            summary["match_id"] = job.match_id
            summary["game"] = job.game
            summary["game_version"] = job.game_version
            # 模型读出来的**整体情况**（分路/胜负/BP）。
            # 跟着摘要走，后面的「建对局」那一步直接拿来用 ——
            # 中间不再有让人填的一页。
            summary["prefill"] = prefill

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

    @staticmethod
    def _infer_lane_from_rounds(rounds: List[Any]) -> str:
        """从多个时间窗的已识别位置推断稳定分路。

        仅在至少 3 段、且单一位置占 60% 以上时采用。"野区"是
        打野职责的强证据；其它路线也必须是模型明确读到的路线名，
        不会把河道、泉水或一次游走硬归进某条线。
        """
        aliases = {"野区": "jungle", "中路": "mid", "对抗路": "side",
                   "发育路": "carry"}
        values: List[str] = []
        for rnd in rounds or []:
            for fact in getattr(rnd, "facts", []) or []:
                if getattr(fact, "name", "") != "position":
                    continue
                mapped = aliases.get(str(getattr(fact, "value", "") or "").strip())
                if mapped:
                    values.append(mapped)
        if len(values) < 3:
            return ""
        lane, count = Counter(values).most_common(1)[0]
        return lane if count / len(values) >= 0.60 else ""

    def _phase_product(self, job: ScanJob, base_dir: str,
                       clock_track: Dict[str, Any], scan: Dict[str, Any],
                       on_ex, lane: str = "", hero: str = "") -> Dict[str, Any]:
        """MOBA 的产物：BP 帧序列 + 按阶段均匀抽帧。"""
        result = vu.sample_by_phase(
            job.video, clock_track, base_dir,
            self._phase_windows(job.game), progress=on_ex,
            # 分析单元：时间窗，从规则库倒推。阶段帧是**证据**，
            # 窗才是判定单元 —— 不分窗就没有局面类事实，规则判不了。
            analysis_windows=self._analysis_windows(job.game),
            # 英雄轨迹：时钟探测那一趟顺带采的（见 vu.probe_clock）
            hero_track=self._hero_track(scan),
            # 分路：清单按它裁剪。模型要拿画面去对攻略，
            # 得先知道「玩家打的是哪条线」。
            game=job.game, lane=lane, hero=hero,
        )
        if not result or not result.get("rounds"):
            raise RuntimeError("按阶段抽帧没有产出任何帧")
        result["kind"] = "phase"
        return result

    @staticmethod
    def _hero_track(scan: Dict[str, Any]) -> List[Any]:
        """扫描阶段采到的英雄轨迹（原始行）。

        轨迹在扫描结果里存的是普通字典（要落库、要进摘要）。
        **带缺口的行也留着**（只有 t、没有 nx/ny）——
        覆盖率要按「这一段本来该有多少个采样点」算。
        """
        rows = (scan or {}).get("hero_track") or []
        return list(rows)

    def _moba_product(self, job: ScanJob, base_dir: str,
                      points: List[Any], scan: Dict[str, Any],
                      on_ex, lane: str = "", hero: str = "") -> Dict[str, Any]:
        """MOBA 读不到时钟时的产物：按信息密度切时间片。"""
        result = vu.moba_slice_product(
            job.video, points, base_dir,
            self._phase_windows(job.game),
            budget=self._slice_budget(job.game),
            progress=on_ex,
            hud_layout=self._hud_layout(job.game),
            weights=self._rule_weights(job.game),
            # 对局起点：优先用「HUD 连续就位」测出来的（逐段视频单独测，
            # 每段录屏进入对局的时刻都不同）；没有就退回「第一次就位」的时刻。
            game_start_t=((scan or {}).get("match_start_t")
                          or (scan or {}).get("hud_start_t")),
            # 分析单元：时间窗，从规则库倒推。
            analysis_windows=self._analysis_windows(job.game),
            # 英雄轨迹（时钟探测那一趟顺带采的）
            hero_track=self._hero_track(scan),
            game=job.game, lane=lane, hero=hero,
        )
        if not result or not result.get("rounds"):
            raise RuntimeError("按信息密度切时间片没有产出任何帧")
        result["kind"] = "moba_slice"
        return result

    @staticmethod
    def _guess_start_t(points: List[Any], scan: Dict[str, Any]) -> Optional[float]:
        """读不到时钟时**推算**对局起点。

        与 :func:`app.collection.video_upload.moba_slice_product` 用的是
        同一套算法与同一个常量 —— 两边各算一次，迟早会算出两个起点，
        于是「模型读的是第 4 分钟那几帧、规则判的是第 5 分钟」。
        """
        start = ((scan or {}).get("match_start_t")
                 or (scan or {}).get("hud_start_t"))
        if start:
            return float(start)
        span = vu.activity_span(points)
        if not span:
            return None
        return float(span[0]) + vu.MATCH_LOAD_SEC

    def _read_identity(self, job: ScanJob, base_dir: str,
                       start_t: Optional[float], scan: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
        """读「玩家打哪条线」+「这局赢没赢」。

        为什么排在读窗**之前**
        ----------------------
        模型要拿画面去对攻略，前提是它手里有**这一条线**的攻略原文。
        分路晚一步读到，窗就读成了「各条线的大杂烩」——
        模型会对着别的线的打法乱指。

        读不出来就是 ``unknown`` —— **不猜**。猜错位置会让整局每一处
        判定都指向另一条线的攻略，比读不到更糟。
        """
        from . import settings

        if settings.recognizer_kind() != "vlm":
            return {}
        if settings.credentials_for("vlm") is None:
            return {}
        if start_t is None:
            return {}
        try:
            from .vlm_prefill import infer_lane_from_track, read_lane_and_result
            lane_track = infer_lane_from_track(
                (scan or {}).get("hero_track") or [], float(start_t))
            return read_lane_and_result(
                job.video, vu.video_store_dir(base_dir), float(start_t),
                float(getattr(job.video, "duration_sec", 0.0) or 0.0),
                position_hint=str(lane_track.get("position") or "")) | {
                    "lane_track": lane_track
                }
        except Exception:                          # noqa: BLE001
            return {}

    def _read_prefills(self, job: ScanJob, base_dir: str,
                       product: Dict[str, Any],
                       lane_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """读这一局的**整体情况**：分路、胜负、BP 禁用与选人。

        这三样以前是让人一格一格填的（用户原话：「难道还要我自己去选？」）。
        现在由模型从画面里读 —— 读不出来的**留空，不猜**。

        分路、英雄与胜负通常**已经读过了**（读窗之前那趟，见
        :meth:`_read_identity`），
        那一趟的结果直接复用 —— 同一个模型调用读两遍是白花钱。
        """
        from . import settings
        from .vlm_prefill import read_bp_steps, read_lane_and_result

        if settings.recognizer_kind() != "vlm":
            return {}
        if settings.credentials_for("vlm") is None:
            return {}
        video = job.video
        store = vu.video_store_dir(base_dir)
        start_t = product.get("match_start_t")
        duration = float(getattr(video, "duration_sec", 0.0) or 0.0)
        out: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_lane = None
            if not lane_info:
                f_lane = pool.submit(read_lane_and_result, video, store,
                                     start_t, duration)
            f_bp = pool.submit(read_bp_steps, video, store)
            lane = lane_info or {}
            if f_lane is not None:
                try:
                    lane = f_lane.result() or {}
                except Exception:                      # noqa: BLE001
                    lane = {}
            try:
                bp = f_bp.result()
            except Exception:                          # noqa: BLE001
                bp = {}
        if isinstance(lane, dict):
            out.update(lane)
        if isinstance(bp, dict) and bp:
            out["bp"] = bp
        return out

    @staticmethod
    def _analysis_windows(game: str) -> List[Any]:
        """这个游戏的**分析单元**：逐分钟复盘节点。

        节点仍从规则库倒推（每条规则的 ``time_min between``），但会把
        长窗口展开到每个有效分钟：6~9 分钟会分析 6、7、8、9 四次。
        这样报告能指出具体哪一分钟该转线、该参团或该回撤，而不是只给
        一段笼统的中期总结。
        王者 S45 实测：1681 条规则用 8 种窗，合成 6 个分析段。

        从种子库读，不碰数据库（与 :meth:`_rule_weights` 同理）。
        """
        try:
            from ..rules.density import seed_windows
            from .window_analysis import checkpoint_windows
            return checkpoint_windows(seed_windows(game))
        except Exception:                          # noqa: BLE001
            return []

    @staticmethod
    def _rule_weights(game: str) -> Dict[int, float]:
        """每个局内分钟能判出多少条规则（归一化后的权重）。

        切片预算按它分：密度大的分钟多切几刀，一条规则都没有的分钟不切 ——
        那里切了也判不出任何东西。

        从种子库读，不碰数据库：扫描跑在后台线程，按约定不碰库
        （见本模块的文档字符串）。密度只是个形状提示，用种子算与用库里的
        版本算相差无几 —— 库里的版本本来就是种子灌进去的。
        """
        try:
            from ..rules.density import density_weights, seed_density
        except Exception:                          # noqa: BLE001
            return {}
        try:
            return density_weights(seed_density(game))
        except Exception:                          # noqa: BLE001
            return {}

    @staticmethod
    def _slice_budget(game: str) -> int:
        """时间片的数量预算。默认 :data:`video_upload.DENSITY_BUDGET`。

        适配器可以覆盖（``moba_slice_budget``）—— 一局十五分钟切成多少段，
        取决于这个游戏「值得判断的时刻」有多密。
        """
        try:
            from ..adapters.base import get_adapter

            adapter = get_adapter(game)
        except Exception:                          # noqa: BLE001
            return vu.DENSITY_BUDGET
        return int(
            getattr(adapter, "moba_slice_budget", vu.DENSITY_BUDGET)
            or vu.DENSITY_BUDGET
        )

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

    @staticmethod
    def _phase_windows(game: str) -> Dict[str, List[int]]:
        """该游戏的阶段时间窗；没有就返回空表。

        从适配器取，不在采集层写死一份 MOBA 的窗口 —— 采集层不该知道
        「前期是 0 到 4 分钟」这种事，那是规则库与判定层的事。
        """
        try:
            from ..adapters.base import get_adapter

            adapter = get_adapter(game)
        except Exception:                          # noqa: BLE001
            return {}
        return dict(getattr(adapter, "phase_time_windows", {}) or {})

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
