"""视频上传链路端到端验证。

用 :mod:`make_sample_video` 生成的合成视频走完整条链路：

    上传 → 后台扫描（模型读画面） → 建对局 → 报告

合成视频的「正确答案」是已知的（几个回合、每回合血量多少），
所以每一步都能断言，而不是靠肉眼看。

**没有标注页了**：分路、胜负、BP、每段操作都由模型从画面读，
扫完直接 finalize 出报告（用户原话：「我只要模型去读出报告，
其余一切步骤都省去」）。

为了让脚本**不花钱、不联网**，这里起一个**本地 mock 模型服务**：
它按提示词的形状回不同的答案 —— 读分路与胜负的、读 BP 的、
读窗内动作的，各答各的。真实模型答得更好，但这三层接线一模一样。
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: 数据目录要指到自己的地方：视频、抽帧、模型配置全在这儿。
#: 指到项目根的 data/ 会污染用户自己的库与配置。
VERIFY_DIR = os.path.join(ROOT, "data", "verify_video_flow")
os.makedirs(VERIFY_DIR, exist_ok=True)
os.environ["MATCH_DATA_DIR"] = VERIFY_DIR

sys.path.insert(0, ROOT)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient          # noqa: E402

from app import db as dbmod                       # noqa: E402
from app.main import app                          # noqa: E402

from make_sample_video import build                    # noqa: E402

FAILURES = []

#: 合成视频要多少回合。
#:
#: 采集层认定「对局段」至少 45 秒（``activity_span``）——
#: 太短的片子会被判成「不像一局对局」。每回合 2 秒 + 0.5 秒转场，
#: 30 回合约 75 秒，够过那道门槛。
#:
#: 但**要跑进 4:00 / 6:00 那两个节点**（攻略在这两处给的是最多打法的地方，
#: 结论最值得看），游戏内时长就得够 7 分钟 —— 合成视频的时钟就是视频
#: 时长，没法快进。170 回合约 7 分 05 秒。
SAMPLE_ROUNDS = 170


# ---------------------------------------------------------------- mock 模型


#: 窗内那一段的回答：模型看到「在野区刷野」。
WINDOW_ANSWER = {
    "actions": [{
        "kind": "objective_action", "value": "farm_jungle",
        "frames": [1, 2], "why": "小地图在野区来回移动", "confidence": 0.85,
    }],
    "position": "野区",
    "in_combat": "false",
    "ult_button": "ready",
    "clock_readings": [],
    "unknown": [],
}

#: 分路与胜负的回答（开局几帧 + 结尾几帧）。
PREFILL_ANSWER = {
    "position": "jungle", "result": "win",
    "why": "一直在野区刷野，结算画面写着胜利",
}

#: BP 的回答：一手都不写 —— 读不出来就不写，不编。
BP_ANSWER = {"steps": [], "my_missing_frontline": False}


class _ModelHandler(BaseHTTPRequestHandler):
    """按提示词的形状回不同的答案。

    窗内那一段还**按分钟变化**：前期只刷野，中期开始开团 ——
    这样才能真的触发「该控龙/该开团」那批规则的偏离判定，
    报告里的时间线、分路结论、原视频定位才有东西可验。
    """

    received = []

    def log_message(self, *args):                # noqa: D102
        pass

    def do_POST(self):                           # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        text = " ".join(
            str(b.get("text") or "") for m in body.get("messages") or []
            for b in (m.get("content") or []) if isinstance(b, dict))
        type(self).received.append(text)
        if "BP 阶段" in text:
            answer = BP_ANSWER
        elif '"result": "win|lose|unknown"' in text:
            answer = PREFILL_ANSWER
        else:
            answer = _window_answer(text)
        payload = json.dumps({
            "choices": [{"message": {"content": json.dumps(answer,
                                                           ensure_ascii=False)}}],
            "usage": {"input_tokens": 10, "output_tokens": 10},
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _window_answer(prompt: str) -> dict:
    """窗内那一段的答案：**按局内分钟**变。

    这 mock 扮演的是一个**称职的多模态模型**：它看着画面读出玩家做了什么，
    再逐条对攻略清单给判断 —— 前提看得见成立就说不符合，看不见就说看不出来。
    真实模型答得更好，但这三层接线一模一样。

    剧本（局内分钟 -> 这一窗的判断）：

    - **0:00**：玩家在野区点视野精灵，攻略要求反向开局刷野 → 前提可见
      （敌方红开），判**偏离**。
    - **2:00~4:00**：玩家在刷野。这一段攻略的分支前提是「均势 / 劣势」，
      画面里读不出来 → 判**看不出来**（不纠正）。
    - **4:00**：玩家在刷野，攻略这会儿要求带线推塔（塔的护盾没了、
      敌方打野不在我方半区，画面里看得见）→ 判**偏离**。
    - **6:00~10:00**：玩家在刷野，攻略这会儿要求趁敌方减员开团
      （敌方两人阵亡看得见）→ 判**偏离**。
    - 其余：判**做对了 / 看不出来**。
    """
    import re

    m = re.search(r"局内第\s*(\d+)\s*分", prompt)
    lo = int(m.group(1)) if m else 0

    # 这一窗玩家做了什么（动作清单）。
    if lo == 0:
        answer = {
            "actions": [{
                "kind": "objective_action", "value": "clear_vision_sprite",
                "frames": [1], "why": "在野区点掉了河道里的视野精灵",
                "confidence": 0.8,
            }],
            "position": "野区",
            "in_combat": "false",
            "ult_button": "ready",
            "clock_readings": [],
            "unknown": [],
        }
    else:
        answer = dict(WINDOW_ANSWER)
        if lo >= 2:
            answer["actions"] = list(WINDOW_ANSWER["actions"]) + [{
                "kind": "teamfight_action", "value": "engage",
                "frames": [1], "why": "几个英雄挤在一起，有技能特效",
                "confidence": 0.8,
            }]
            answer["in_combat"] = "true"

    # 逐条对清单给判断。清单里每条模式的 id 都以「【id】」开头。
    ids = re.findall(r"【([A-Za-z0-9_\-]+)】", prompt)
    answer["review"] = []
    if not ids:
        return answer

    #: 这一窗的判断剧本：
    #: (判第几条, 第几支, 前提, 是否符合, 玩家在做什么,
    #:  看到了什么, 凭什么)
    #:
    #: 「判第几条」不能随便挑 —— 清单里每条模式有自己的**适用时机**
    #: （这条规则是在玩家做哪类操作时才拿来判）。玩家在刷野的时候，
    #: 「带线推塔」那条的适用时机是站位类动作，对不上，不该被指控；
    #: 而「龙坑最后一刻交惩」「人齐不先开团」那几条的适用时机里
    #: 就有刷野 —— 才轮得到它们说话。
    #:
    #: 「第几支」同理：攻略一个节点往往给好几支（「敌方红开就己方蓝开，
    #: 敌方蓝开就己方红开」），指认了哪支，报告才给得出**这一局该怎么做**
    #: 的那一句，而不是一张让玩家自己配对的菜单。
    script = {
        0: (0, 2, "visible", "no", "在野区点河道视野精灵",
            "敌方蓝开，我方红开",
            "红开方向没错，错在开局先去点河道精灵，没起手打红 BUFF"),
        2: (0, 0, "unknown", "unclear", "在刷野",
            "分不清这会儿是均势还是劣势",
            "这一段攻略的分支要看是均势还是劣势，画面里读不出来"),
        4: (1, 1, "visible", "no", "一直在红 buff 野区刷怪",
            "小地图上龙坑图标在闪，龙坑里有我方视野",
            "龙快刷了却还在红 buff 刷怪，该去占龙坑、惩击留到最后一刻再交"),
        6: (1, 1, "visible", "no", "还在野区刷怪",
            "小地图上五个人都在，敌方一个没少",
            "人齐不先开团，该用龙和塔逼对方动"),
    }
    idx, branch, premise, matched, did, evidence, why = script.get(
        lo, (0, 0, "unknown", "unclear", "在刷野", "画面里看不出该走哪一支",
            "这一段看不出该走哪一支"))
    if idx >= len(ids):
        idx = 0
    answer["review"] = [{
        "mode_id": ids[idx],
        "premise": premise,
        "branch": branch,
        "matched": matched,
        "frames": [1, 2],
        "evidence": evidence,
        "did": did,
        "why": why,
        "confidence": 0.8,
    }]
    return answer


def start_mock_model():
    """起一个本地模型服务，返回它的完整端点地址。"""
    _ModelHandler.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[0], server.server_address[1]
    return server, f"http://{host}:{port}/v1/chat/completions"


def check(cond, msg):
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILURES.append(msg)


def isolate():
    path = os.path.join(VERIFY_DIR, "verify.db")
    if os.path.exists(path):
        os.remove(path)
    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(path)
    # **清掉识别的缓存**：缓存键是「提示词」，提示词没变但 mock 的答案变了时，
    # 旧答案会被原样端上来 —— 那测的就不是这一轮的 mock 了。
    cache = os.path.join(VERIFY_DIR, "vlm_cache")
    if os.path.isdir(cache):
        for name in os.listdir(cache):
            os.remove(os.path.join(cache, name))


def wait_for_job(client, status_url: str, timeout: float = 900.0):
    """轮询后台扫描任务直到完成，返回最终的 summary。

    顺带验了三件事：进度在动（不是卡在 0）、状态是 done 而不是 failed、
    中途拿到的进度确实介于 0 和 1 之间 —— 一个永远返回 100% 的假进度
    条比没有进度条更糟。
    """
    import time

    t0 = time.time()
    progresses = []
    last = None
    while time.time() - t0 < timeout:
        r = client.get(status_url)
        if r.status_code != 200:
            return None
        j = r.json()
        progresses.append(j["progress"])
        last = j
        if j["status"] == "done":
            print(f"      进度变化 {len(set(progresses))} 个不同值 · "
                  f"用时 {time.time() - t0:.1f}s")
            return j["summary"]
        if j["status"] == "failed":
            print(f"      扫描失败：{j.get('error')}")
            return None
        time.sleep(0.2)
    if last:
        print(f"      超时，最后状态 {last['status']} @ {last['progress']:.0%}")
    return None


def main() -> int:
    isolate()

    print("\n[0] 准备合成视频")
    sample = os.path.join(VERIFY_DIR, "verify.mp4")
    info = build(sample, rounds=SAMPLE_ROUNDS)
    data = open(sample, "rb").read()
    check(len(data) > 0,
          f"合成视频 {len(data) / 1024:.0f} KB（{info['rounds']} 回合）")

    server, endpoint = start_mock_model()
    print(f"      mock 模型服务 {endpoint}")

    # 只填一个 key 就能配好 —— 端点、协议、模型名都由预设推出来。
    from app.collection import settings

    settings.save_slot("vlm", provider="custom", api_key="mock-key",
                       endpoint=endpoint, protocol="openai", model="mock-vlm")
    check(settings.credentials_for("vlm") is not None, "模型配置写进去了")

    client = TestClient(app)
    client.post("/api/games/hok/rules/seed")

    print("\n[1] 封面与上传页可打开")
    r = client.get("/")
    check(r.status_code == 200, f"封面页 {r.status_code}")
    check("不断向被验证的模式优化拟合" in r.text, "封面标题")
    check("探索玩法" in r.text and 'href="/upload"' in r.text, "封面有进入按钮")
    # 背景是第三方开源项目的银河，走 iframe 引入。所以这里要验的是
    # **资源真的取得到**，而不是页面上有没有 requestAnimationFrame ——
    # 动画在 iframe 里面，封面 HTML 里本来就该没有。
    check('class="galaxy-bg"' in r.text, "封面挂了银河背景")
    g = client.get("/static/galaxy/index.html")
    check(g.status_code == 200, f"银河页 {g.status_code}")
    check("<canvas" in g.text, "银河页有 canvas")

    r = client.get("/upload")
    check(r.status_code == 200, f"上传页 {r.status_code}")
    check("上传对局视频" in r.text, "上传页有上传入口")
    check("/static/app.css" in r.text, "上传页引入了应用样式")
    check("modelKey" not in r.text, "上传页不再收 API Key（配置在 /model）")

    r = client.get("/model")
    check(r.status_code == 200, f"模型页 {r.status_code}")
    check("看画面的模型" in r.text, "模型页有「看画面的模型」")

    print("\n[2] 上传 + 后台扫描")
    # 版本不由前端给。规则库里最新的那个由后端自己取 ——
    # 传错版本不会报错，只会让后面拿到的报告是空的。
    latest = client.get("/api/games/hok/versions").json()["latest"]
    with open(sample, "rb") as fh:
        r = client.post(
            "/api/videos/upload",
            files={"file": ("verify.mp4", fh, "video/mp4")},
            data={"game": "hok"},
        )
    check(r.status_code == 200, f"上传成功（{r.status_code}）")
    if r.status_code != 200:
        print(r.text[:300])
        return 1

    handed = r.json()
    check("job_id" in handed, f"立即返回任务号 {handed.get('job_id')}")
    check("status_url" in handed, "给了轮询地址")
    check(handed["video"]["duration_sec"] > 0, "上传阶段就拿到了视频元信息")

    # 上传是异步的 —— 这正是要验的：立刻返回，不阻塞
    print(f"      {handed['message']}")

    print("\n[2b] 轮询扫描进度")
    summary = wait_for_job(client, handed["status_url"], timeout=900)
    if summary is None:
        check(False, "扫描任务未在超时内完成")
        return 1

    check(summary.get("game_version") == latest,
          f"扫描用的版本是规则库里最新的那个（{latest}），"
          f"实际 {summary.get('game_version')}")

    found = summary["round_count"]
    print(f"      扫描到 {found} 个回合（真值 {SAMPLE_ROUNDS}）")
    print(f"      {summary['scan']['note']}")
    check(found >= 1, f"切出了 {found} 个时间片")
    check(summary["scan"]["scanned_samples"] < summary["scan"]["total_frames"],
          f"采样量远小于总帧数（{summary['scan']['scanned_samples']}"
          f"/{summary['scan']['total_frames']}）")

    check(summary["video"]["fps"] > 0, "读到帧率")
    check(summary["video"]["duration_sec"] > 0, "读到时长")
    check(summary["video"]["resolution"] == "640x360", "读到分辨率")

    print("\n[3] 抽帧")
    shots = summary["rounds"]
    with_img = [s for s in shots if s["image"]]
    check(len(with_img) >= max(1, found - 1),
          f"{len(with_img)}/{found} 个回合截到了图")
    check(all(0 < s["confidence"] <= 1 for s in shots), "边界置信度都在 (0,1]")
    first = shots[0]
    img = client.get(first["image_url"])
    check(img.status_code == 200, f"帧图片可取（{len(img.content)} 字节）")
    check(img.headers["content-type"].startswith("image/"), "帧图片 MIME 正确")

    print("\n[4] 模型读出来的东西")
    prefill = summary.get("prefill") or {}
    check(prefill.get("position") == "jungle",
          f"分路从画面里读出来了（{prefill.get('position')}）")
    check(prefill.get("result") == "win",
          f"胜负从画面里读出来了（{prefill.get('result')}）")
    check(_ModelHandler.received, "确实问过模型")

    print("\n[5] 建对局 → 分析（中间没有人工那一步）")
    match_id = summary["match_id"]
    r = client.post("/api/videos/finalize", json={"match_id": match_id})
    check(r.status_code == 200, f"建对局成功（{r.status_code}）")
    if r.status_code != 200:
        print(r.text[:400])
        return 1
    out = r.json()
    print(f"      统计 {json.dumps(out['stats'], ensure_ascii=False)}")
    check(out["stats"]["triggered"] > 0,
          f"规则触发了判定（{out['stats']['triggered']} 次）")
    check(out["gap_count"] > 0, f"定位到 {out['gap_count']} 处差距")

    # 幂等：再来一次是同一局
    again = client.post("/api/videos/finalize", json={"match_id": match_id}).json()
    check(again["match_id"] == out["match_id"], "重复 finalize 是同一局")

    print("\n[6] 报告")
    r = client.get(out["report_url"])
    check(r.status_code == 200, f"报告页 {r.status_code}")
    # 报告正文现在只有一节「操作模式分析」：分路结论与逐回合差距合成了
    # ``article.mode-item``（老的 timeline / lanes 两节已经并进去了）。
    check('id="analysis"' in r.text, "有「操作模式分析」这一节")
    check("mode-item" in r.text, "有结论条目")
    # 报告**不解释**结论的来源（用户明确要求去掉这类描述性文字）
    check("从画面里读出来的" not in r.text, "报告又在解释结论的来源了")
    check("mock-vlm" not in r.text, "报告上不该摆模型名")
    # 四部分结构：时间区间 / 原视频片段 / 操作描述 / 建议
    for part in ("对局时间区间", "原视频被建议部分片段", "原视频操作的文字描述",
                 "根据模式库建议操作的描述"):
        check(part in r.text, f"结论里有「{part}」")

    rj = client.get(out["report_url"].replace("/report", "/report.json"))
    rep = rj.json()
    check(len(rep["gaps"]) == out["gap_count"], "报告与返回一致")
    # 分路结论要说清「你在做什么 / 该做什么 / 怎么改」
    lanes = rep.get("lanes") or {}
    findings = lanes.get("findings") or []
    check(findings, f"给出分路结论（{len(findings)} 条）")
    for f in findings:
        check(bool(f.get("should_do")), f"{f['mode_id']} 没写应该做什么")
        check(bool(f.get("how_to_fix")), f"{f['mode_id']} 没写怎么改")
        check(f.get("lane") == "jungle", f"{f['mode_id']} 分路不对")
    problems = [f for f in findings
                if f["verdict"] in ("systemic", "never", "occasional")]
    check(problems, f"标出问题段（{len(problems)} 段）")
    check(any(f.get("actually_doing") for f in problems),
          "问题段说得出玩家当时在做什么")
    check(problems and problems[0].get("actually_doing"),
          f"说得出当时在做什么（{problems[0].get('actually_doing') if problems else '—'}）")
    print(f"      分路 {lanes['lanes'][0]['lane_label']}，"
          f"问题 {lanes['problem_count']} 段")
    for f in problems[:3]:
        print(f"        [{f['phase_label']}] {f['verdict_label']} "
              f"{f['title'][:26]}")
        print(f"          你在做 {f['actually_doing'] or '—'} → "
              f"该做 {f['should_do']}")
    for g in rep["gaps"][:3]:
        print(f"      {g['round_label']} {g['rule_title'][:26]}")
        print(f"        该做 {g['expected_text']} | 实际 {g['actual_text']}")

    print("\n[6b] 出错的那一段能不能定位回原视频")
    # 这一局来自真实上传，所以 video_id 与每回合的秒数都在对局里
    check(lanes.get("video_id"), f"对局记下了原视频（{lanes.get('video_id')}）")
    check(lanes.get("video_url", "").endswith("/file"), "给出了原视频地址")
    shots = rep.get("lanes", {}).get("video_id") and rep["lanes"].get("findings")
    with_shots = [f for f in (shots or []) if f.get("moments")]
    check(with_shots, f"结论带上了出错时刻（{len(with_shots)} 条）")
    if with_shots:
        m0 = with_shots[0]["moments"][0]
        check(m0["time"] > 0, f"时刻是秒数（{m0['time']}）")
        check(m0["image_url"].startswith("/api/videos/"),
              f"截图地址指向原视频目录（{m0['image_url']}）")
    # 片段播放器：缩略图是 poster，地址是原视频，点一下会跳到出错那一秒
    check('class="clip"' in r.text, "报告里有出错片段的播放器")
    check('class="clip-vid"' in r.text, "播放器是个 <video>")
    check("poster=" in r.text, "缩略图是那一帧的截图")
    check("clip-play" in r.text, "有「播放这一段」按钮")
    check("vid.currentTime = start" in r.text, "播放器会跳到出错那一秒")
    check('data-t="' in r.text, "片段带着出错那几秒的起点")

    print("\n[7] 帧图片与视频可回放")
    vid = summary["video"]["video_id"]
    check(client.get(f"/api/videos/{vid}/frames/{first['image']}").status_code == 200,
          "帧图片按 video_id 也能取")
    rv = client.get(f"/api/videos/{vid}/file")
    check(rv.status_code == 200, f"原视频可回放（{len(rv.content)} 字节）")

    print("\n[8] 错误处理")
    r = client.post("/api/videos/upload",
                    files={"file": ("x.txt", b"not a video", "text/plain")},
                    data={"game": "hok"})
    check(r.status_code == 422, f"非视频文件被拒（{r.status_code}）")

    r = client.post("/api/videos/upload",
                    files={"file": ("v.mp4", data, "video/mp4")},
                    data={"game": "dota"})
    check(r.status_code == 422, f"未知游戏被拒（{r.status_code}）")

    r = client.get("/videos/no-such-match/annotate")
    check(r.status_code == 410, "标注页已下线（410，不是 404）")

    server.shutdown()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"失败 {len(FAILURES)} 项：")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("视频上传链路验证全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
