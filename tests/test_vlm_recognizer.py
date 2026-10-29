"""多模态识别层：客户端、提示词契约、解析、降级。

盯的是几件**错了也不会报错**的事：

1. 请求体的形状对不对 —— 端点拼错会 404，图片块形状错会被拒；
2. 模型答了**词表外**的取值时，要**丢掉并记一条**，不能塞进对局数据；
3. 模型答不出来时，要记**缺失**（内核判「无法判定」），不能当成「没做」；
4. 时间戳是**我们自己**换算的（帧的秒数），不是模型说的；
5. 没凭据 / 调不通 → 这一窗读不出来，**不许**回退到旧探测器的结论。

这里用一个**本地 mock 模型服务**把整条链路钉住 —— 测试不花钱、不联网，
也不需要用户先配 key。
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from app.collection import vlm_client, vlm_recognizer
from app.collection.vlm_recognizer import FrameInput, WindowInput


# ---------------------------------------------------------------- mock 服务


class _Handler(BaseHTTPRequestHandler):
    """回一个固定的 Anthropic 形状回答；把收到的请求体存起来供断言。"""

    answer: Dict[str, Any] = {}
    received: List[Dict[str, Any]] = []
    status: int = 200

    def log_message(self, *args):                # noqa: D102
        pass

    def do_POST(self):                           # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).received.append({
            "body": body,
            "path": self.path,
            "headers": dict(self.headers),
        })
        payload = json.dumps({
            "content": [{"type": "text",
                         "text": json.dumps(type(self).answer, ensure_ascii=False)}],
            "usage": {"input_tokens": 1234, "output_tokens": 56},
        }).encode("utf-8")
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def mock_server():
    _Handler.answer = {}
    _Handler.received = []
    _Handler.status = 200
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        # 端点**原样**给出去 —— 客户端一个字都不加（见 build_request）。
        yield f"http://{host}:{port}/anthropic/v1/messages", _Handler
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------- 客户端


def test_the_endpoint_is_used_verbatim():
    """端点**一个字都不加** —— 加路径是踩过坑的地方。

    以前客户端自己拼 ``/v1/messages``、``/chat/completions``，
    结果少写一段路径就 404（DeepSeek 的官方 base 不带 ``/v1``）。
    现在完整地址由供应方预设备好，客户端直接用。
    """
    creds = vlm_client.Credentials(
        api_key="k", endpoint="https://api.minimaxi.com/anthropic/v1/messages")
    url, _headers, _body = vlm_client.build_request("MiniMax-M3", [], creds)
    assert url == "https://api.minimaxi.com/anthropic/v1/messages", url

    creds = vlm_client.Credentials(
        api_key="k", endpoint="https://api.deepseek.com/chat/completions",
        protocol="openai")
    url, _h, _b = vlm_client.build_request("deepseek-flash", [], creds)
    assert url == "https://api.deepseek.com/chat/completions", url


def test_the_two_protocols_use_different_shapes():
    blocks = [vlm_client.text_block("hi"),
              vlm_client.image_block(b"\xff\xd8\xff", "image/jpeg")]
    anth_creds = vlm_client.Credentials(
        api_key="k", endpoint="https://x/anthropic/v1/messages",
        protocol="anthropic")
    _u, _h, anth = vlm_client.build_request(
        "MiniMax-M3", blocks, anth_creds, system="S")
    assert anth["system"] == "S"
    assert anth["thinking"] == {"type": "disabled"}
    kinds = [b["type"] for b in anth["messages"][0]["content"]]
    assert kinds == ["text", "image"]

    oai_creds = vlm_client.Credentials(
        api_key="k", endpoint="https://x/v1/chat/completions", protocol="openai")
    _u, headers, oai = vlm_client.build_request(
        "MiniMax-M3", blocks, oai_creds, system="S")
    assert headers["Authorization"] == "Bearer k"
    assert oai["messages"][0]["role"] == "system"
    kinds = [b["type"] for b in oai["messages"][1]["content"]]
    assert kinds == ["text", "image_url"]


def test_the_thinking_field_name_depends_on_the_provider():
    """「关思考」的字段名**每家不一样**：百炼叫 ``enable_thinking``。

    写死一个，Qwen 那条就不认 —— 模型开始长篇大论，我们的 JSON 解析
    直接失败，而报错还看不出是哪错了。
    """
    bailian = vlm_client.Credentials(
        api_key="k", endpoint="https://x/compatible-mode/v1/chat/completions",
        protocol="openai", thinking_field="enable_thinking")
    _u, _h, body = vlm_client.build_request("qwen-vl-max", [], bailian)
    assert body["enable_thinking"] is False
    assert "thinking" not in body

    deepseek = vlm_client.Credentials(
        api_key="k", endpoint="https://x/chat/completions", protocol="openai",
        thinking_field="thinking")
    _u, _h, body = vlm_client.build_request("deepseek-flash", [], deepseek)
    assert body["thinking"] == {"type": "disabled"}


def test_credentials_come_from_env_then_app_config(tmp_path, monkeypatch):
    """环境变量盖过配置文件；都没配就**没有凭据**（不猜、不编）。"""
    from app.collection import settings

    monkeypatch.setenv("MATCH_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("MATCH_VLM_API_KEY", raising=False)
    monkeypatch.delenv("MATCH_VLM_ENDPOINT", raising=False)
    assert vlm_client.find_credentials("vlm") is None

    monkeypatch.setenv("MATCH_VLM_API_KEY", "env-key")
    monkeypatch.setenv("MATCH_VLM_ENDPOINT", "https://x/chat/completions")
    creds = vlm_client.find_credentials("vlm")
    assert creds and creds.api_key == "env-key"
    assert creds.endpoint == "https://x/chat/completions"

    monkeypatch.delenv("MATCH_VLM_API_KEY", raising=False)
    monkeypatch.delenv("MATCH_VLM_ENDPOINT", raising=False)
    settings.save_slot("vlm", provider="deepseek", api_key="file-key")
    creds = vlm_client.find_credentials("vlm")
    assert creds and creds.api_key == "file-key"
    assert creds.endpoint.startswith("https://")
    assert creds.model == "deepseek-flash"


def test_the_answer_json_is_parsed_out_of_the_code_fence():
    res = vlm_client.VlmResult(text='```json\n{"a": 1}\n```')
    assert res.json() == {"a": 1}
    res = vlm_client.VlmResult(text="not json")
    with pytest.raises(vlm_client.VlmError) as exc:
        res.json()
    assert exc.value.kind == "bad_json"


# ---------------------------------------------------------------- 提示词契约


def test_the_prompt_only_offers_closed_options():
    inp = WindowInput(lo=2, hi=4, frames=[
        FrameInput(t=302.5, data=b"x"), FrameInput(t=317.5, data=b"y")])
    prompt = vlm_recognizer.build_prompt(inp)
    # 动作清单必须是词表里的值
    vocab = vlm_recognizer.action_vocabulary()
    assert vocab, "词表空了 —— 模型就没有封闭选项可选"
    for kind, values in vocab.items():
        assert kind in prompt
        for code in values:
            assert code in prompt
    # 帧号与时间要写清楚（时间戳我们自己换算）
    assert "第 1 帧" in prompt and "第 2 帧" in prompt
    assert "302.5" in prompt
    # 必须要求「看不清就说不知道」
    assert "unknown" in prompt
    # 不许模型**自己编**攻略 —— 它只能拿画面去对给它的攻略原文。
    assert "不要猜" in vlm_recognizer.SYSTEM_PROMPT


def test_the_prompt_carries_the_guide_for_the_model_to_judge():
    """**攻略原文要进提示词** —— 模型当裁判，前提是它手里有尺子。

    早先这一层是空的：模型只被当成「读出几个原子事实」的黑盒，
    剩下的全靠硬规则去套 —— 而攻略的分支前提在画面里读不出来，
    报告于是既空又不敢下结论。

    清单里带的是**原文**（前提 / 该做 / 后果），不是我们的转述。
    """
    from app.rules import checklist

    items = checklist.window_checklist("hok", 4, 4, lane="jungle")
    assert items, "4:00 打野那一窗没有清单 —— 模式库出问题了"
    inp = WindowInput(lo=4, hi=4, frames=[FrameInput(t=1.0, data=b"x")],
                      modes=items, lane="jungle")
    prompt = vlm_recognizer.build_prompt(inp)
    for item in items:
        assert item["id"] in prompt
        assert item["premise"][:12] in prompt
    # 分路已知 → 清单只给这一条线，模型不会对着别的线的打法说话。
    assert "打野" in prompt
    assert len(items) < len(checklist.window_checklist("hok", 4, 4))


def test_the_prompt_asks_the_model_to_judge_the_premise():
    """要问「前提在画面里看不看得见成立」—— 这是纠正的前提。

    前提读不出来时，玩家做的可能就是对的（当时塔还有护盾，
    他退塔下才是对的）。模型必须能回答「看不出来」。
    """
    inp = WindowInput(lo=4, hi=4, frames=[FrameInput(t=1.0, data=b"x")])
    prompt = vlm_recognizer.build_prompt(inp)
    assert "premise" in prompt
    assert "matched" in prompt
    assert "visible" in prompt
    # 帧号是硬要求：没有画面依据的结论不给。
    assert "frames" in prompt
    assert "预期动作" not in prompt


def test_window_sampling_keeps_continuous_evidence_pairs():
    """控制持续很短，抽帧必须同时有连续证据和全段覆盖。"""
    from app.collection.video_upload import VLM_BURST_GAP_SEC, vlm_window_times

    times = vlm_window_times(0.0, 120.0)
    assert len(times) == 24
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert sum(g <= VLM_BURST_GAP_SEC + 0.01 for g in gaps) >= 12
    assert times[0] > 0 and times[-1] < 120


def test_window_sampling_adds_dense_combat_bursts_at_activity_events():
    """高活动切片要换成 0.25 秒连续帧，控制链才有可复查的前后顺序。"""
    from app.collection.video_upload import (
        VLM_COMBAT_FRAME_GAP_SEC, vlm_window_times,
    )

    times = vlm_window_times(0.0, 120.0, event_times=[30.0, 90.0])
    near_30 = [t for t in times if 29.0 <= t <= 31.0]
    near_90 = [t for t in times if 89.0 <= t <= 91.0]
    assert len(near_30) >= 6 and len(near_90) >= 6
    assert max(b - a for a, b in zip(near_30, near_30[1:])) <= VLM_COMBAT_FRAME_GAP_SEC + 0.01


# ---------------------------------------------------------------- 解析


def _vocab():
    return vlm_recognizer.action_vocabulary()


def test_values_outside_the_vocabulary_are_dropped():
    answer = {
        "actions": [
            {"kind": "objective_action", "value": "farm_jungle",
             "frames": [1], "why": "在野区", "confidence": 0.8},
            {"kind": "objective_action", "value": "杀穿全场",
             "frames": [1], "why": "很猛"},
            {"kind": "别的种类", "value": "farm_jungle", "frames": [1]},
        ],
        "position": "野区",
    }
    parsed = vlm_recognizer.parse_answer(answer, _vocab())
    assert [a["value"] for a in parsed.actions] == ["farm_jungle"]
    assert len(parsed.notes) == 2, parsed.notes


def test_actions_without_frames_are_dropped():
    answer = {"actions": [{"kind": "objective_action", "value": "farm_jungle",
                           "why": "在野区"}]}
    parsed = vlm_recognizer.parse_answer(answer, _vocab())
    assert not parsed.actions
    assert parsed.notes


def test_actions_without_visual_evidence_are_dropped():
    """动作会影响规则判定，模型没有写看到了什么就不能入库。"""
    answer = {"actions": [{"kind": "objective_action", "value": "farm_jungle",
                           "frames": [1], "confidence": 0.99}]}
    parsed = vlm_recognizer.parse_answer(answer, _vocab())
    assert not parsed.actions
    assert any("画面依据" in note for note in parsed.notes)


def test_the_position_must_be_one_of_the_enumerated_ones():
    parsed = vlm_recognizer.parse_answer({"position": "敌方野区"}, _vocab())
    assert parsed.position == "unknown"
    parsed = vlm_recognizer.parse_answer({"position": "河道"}, _vocab())
    assert parsed.position == "河道"


# ---------------------------------------------------------------- 端到端


def test_the_whole_window_reads_end_to_end(mock_server):
    base, handler = mock_server
    handler.answer = {
        "actions": [{
            "kind": "objective_action", "value": "farm_jungle",
            "frames": [1, 2], "why": "小地图在红 buff 野区来回",
            "confidence": 0.85,
        }],
        "position": "野区",
        "in_combat": "false",
        "ult_button": "ready",
        "clock_readings": [{"frame": 1, "text": "02:08"}],
        "unknown": ["teamfight_action"],
    }
    creds = vlm_client.Credentials(api_key="k", endpoint=base,
                                   protocol="anthropic")
    rec = vlm_recognizer.VlmRecognizer(creds=creds)
    inp = WindowInput(lo=2, hi=4, game_start_t=175.0, frames=[
        FrameInput(t=302.5, data=b"\xff\xd8\xffaaa"),
        FrameInput(t=317.5, data=b"\xff\xd8\xffbbb"),
    ])
    out = rec.analyze(inp)

    # 形状与旧识别器一致
    for key in ("facts", "actions", "missing", "notes", "probe_actions",
                "probe_moments", "absent"):
        assert key in out, key
    # 动作进了事实与动作表
    assert out["facts"]["objective_action"]["value"] == "farm_jungle"
    assert out["actions"]["objective_action"] > 0
    assert out["probe_actions"][0]["source"] == "vlm"
    # 时间戳是**我们自己**换算的（帧的秒数），不是模型说的
    assert out["probe_moments"][0]["t"] == pytest.approx(302.5)
    # 局面事实
    assert out["facts"]["position"]["value"] == "野区"
    assert out["facts"]["has_ult_ready"]["value"] is True
    # 模型说读不出来的那一类 → 记缺失（内核会判「无法判定」）
    assert "teamfight_action" in out["missing"]
    assert out["facts"]["teamfight_action"]["value"] is None
    # 没提到、也没说读不出来的那一类 → 同样记缺失
    assert "positioning_action" in out["missing"]
    # 帧确实发出去了
    body = handler.received[0]["body"]
    kinds = [b["type"] for b in body["messages"][0]["content"]]
    assert kinds == ["text", "image", "image"]


def test_a_clock_mismatch_is_reported(mock_server):
    """模型读到的计时器与我们算的对不上 → 要说（起点可能偏了）。"""
    base, handler = mock_server
    handler.answer = {
        "actions": [], "position": "unknown",
        "clock_readings": [{"frame": 1, "text": "05:00"}],
    }
    creds = vlm_client.Credentials(api_key="k", endpoint=base)
    rec = vlm_recognizer.VlmRecognizer(creds=creds)
    out = rec.analyze(WindowInput(lo=2, hi=4, game_start_t=175.0, frames=[
        FrameInput(t=302.5, data=b"x")]))
    assert any("起点可能偏了" in n for n in out["notes"]), out["notes"]


def test_without_credentials_the_window_is_simply_unreadable():
    """没凭据 → 这一窗读不出来，**不许**编结论。"""
    rec = vlm_recognizer.VlmRecognizer(creds=None, data_dir="/nonexistent-dir")
    out = rec.analyze(WindowInput(lo=2, hi=4, frames=[
        FrameInput(t=1.0, data=b"x")]))
    assert out["facts"] == {}
    assert out["missing"], "读不出来必须记成缺失"
    assert out["notes"]


def test_the_answer_is_cached(mock_server, tmp_path):
    """同一窗再问一次，不该再花钱。"""
    base, handler = mock_server
    handler.answer = {"actions": [], "position": "野区"}
    creds = vlm_client.Credentials(api_key="k", endpoint=base)
    rec = vlm_recognizer.VlmRecognizer(creds=creds,
                                       cache_dir=str(tmp_path / "cache"))
    inp = WindowInput(lo=2, hi=4, frames=[FrameInput(t=1.0, data=b"x")])
    rec.analyze(inp)
    rec.analyze(inp)
    assert len(handler.received) == 1, "缓存没生效"


def test_cache_identity_changes_with_the_actual_model_and_endpoint():
    """换模型/端点不能复用旧答案；生产路径的 self.model 常为空。"""
    inp = WindowInput(lo=2, hi=4, frames=[FrameInput(t=1.0, data=b"x")])
    rec = vlm_recognizer.VlmRecognizer()
    a = rec._cache_key(inp, model="model-a", provider="https://a.example/v1")
    b = rec._cache_key(inp, model="model-b", provider="https://a.example/v1")
    c = rec._cache_key(inp, model="model-a", provider="https://b.example/v1")
    assert len({a, b, c}) == 3


def test_hero_mode_is_included_only_for_the_recognized_hero():
    """英雄身份必须传进窗口清单，不能只裁剪分路后丢掉英雄层。"""
    from app.rules import checklist

    generic = checklist.window_checklist("hok", 0, 4, lane="side")
    hero = checklist.window_checklist("hok", 0, 4, lane="side", hero="哪吒")
    assert hero and len(hero) > len(generic)
    assert any(item["id"].startswith("HOK-HERO-") for item in hero)
    assert not any(item["id"].startswith("HOK-HERO-") for item in generic)


# ---------------------------------------------------------------- 接线


def test_the_recognizer_is_pluggable(monkeypatch, tmp_path):
    """``recognizer=cv`` 时走旧探测器，``vlm`` 时走模型。"""
    from app.collection import settings, video_upload as vu

    monkeypatch.setenv("MATCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MATCH_RECOGNIZER", "cv")
    assert settings.recognizer_kind() == "cv"
    monkeypatch.setenv("MATCH_RECOGNIZER", "vlm")
    assert settings.recognizer_kind() == "vlm"
    monkeypatch.setenv("MATCH_RECOGNIZER", "乱写的")
    assert settings.recognizer_kind() == "vlm", "认不出的值要退回默认，不能瞎传"
