"""模型配置页：**只填一个 API Key** 就能配好。

盯的是几件「错了也不报错」的事：

1. 端点、协议、「关思考」那个字段叫什么、默认模型名 ——
   全是**供应方的事**，得由预设带出来。用户填就会填错，
   而且填错了报错往往看不出是哪错了。
2. 密钥**只写不回** —— 读接口只给掩码。回显等于把密钥发到浏览器里。
3. 「能不能看图」要**实测**：DeepSeek 与百炼都有视觉型号，
   按供应方断定会错。测法是带一张 64px 的小图问颜色。
4. 「测试连接」**不写盘** —— 测就只是测，顺手把配置改了是会咬人的。

这里用一个**本地 mock 模型服务**把整条链路钉住 —— 测试不花钱、不联网。
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------- mock 服务


class _Handler(BaseHTTPRequestHandler):
    """回一个固定形状的回答；把收到的请求体存起来供断言。"""

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
        payload = json.dumps(type(self).answer).encode("utf-8")
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def mock_model():
    _Handler.answer = {"choices": [{"message": {"content": "红色"}}]}
    _Handler.received = []
    _Handler.status = 200
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}/v1/chat/completions", _Handler
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------- 测试环境


@pytest.fixture
def client(tmp_path, monkeypatch):
    """独立临时库 + 独立数据目录的 TestClient。

    数据目录要单独给一份：配置写在 ``config.yaml`` 里，
    跟数据库不是一回事。
    """
    from fastapi.testclient import TestClient

    from app import db as dbmod
    from app.main import app

    dbmod._engine = None
    dbmod._SessionFactory = None
    dbmod.get_engine(str(tmp_path / "model_config.db"))
    monkeypatch.setenv("MATCH_DATA_DIR", str(tmp_path / "data"))
    for name in ("MATCH_VLM_API_KEY", "MATCH_VLM_ENDPOINT", "MATCH_VLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    yield TestClient(app)

    dbmod._engine = None
    dbmod._SessionFactory = None


# ---------------------------------------------------------------- 读


def test_the_config_endpoint_never_hands_back_the_key(client):
    """密钥只写不回。读接口给的是掩码。"""
    from app.collection import settings

    settings.save_slot("vlm", provider="deepseek", api_key="sk-supersecret1234")

    r = client.get("/api/model/config")
    assert r.status_code == 200
    body = r.text
    assert "sk-supersecret1234" not in body, "密钥被原样发回浏览器了"

    slot = r.json()["slots"]["vlm"]
    assert slot["has_key"] is True
    assert "sk-supersecret1234" not in json.dumps(slot, ensure_ascii=False)
    assert slot["key_masked"].startswith("sk-s")
    assert "supersecret1234" not in slot["key_masked"]


def test_the_provider_catalog_carries_the_full_endpoint(client):
    """供应方清单里必须有**完整端点** —— 客户端一个字都不加。"""
    providers = {p["id"]: p for p in client.get("/api/model/config").json()["providers"]}
    assert "qwen" in providers and "deepseek" in providers
    for pid in ("qwen", "deepseek"):
        endpoint = providers[pid]["endpoint"]
        assert endpoint.startswith("https://")
        assert endpoint.endswith("/chat/completions")
    # DeepSeek 官方 base **不带 /v1** —— 少写一段就是 404
    assert "/v1/" not in providers["deepseek"]["endpoint"]


# ---------------------------------------------------------------- 存


def test_only_the_api_key_is_needed_to_configure_a_provider(client):
    """只填 key + 供应方就该配好：端点、协议、模型名全由预设推出来。"""
    r = client.post("/api/model/config", json={
        "slot": "vlm", "provider": "qwen", "api_key": "sk-abc123456789",
    })
    assert r.status_code == 200, r.text[:400]
    slot = r.json()["slots"]["vlm"]
    assert slot["has_key"] is True
    assert slot["provider"] == "qwen"
    assert slot["protocol"] == "openai"
    assert slot["endpoint"].endswith("/compatible-mode/v1/chat/completions")
    assert slot["model"], "模型名没自动填上"

    # 真的能用：凭据拿得到，端点就是预设里那个
    from app.collection import settings

    creds = settings.credentials_for("vlm")
    assert creds is not None
    assert creds.api_key == "sk-abc123456789"
    assert creds.endpoint == slot["endpoint"]
    assert creds.thinking_field == "enable_thinking", "百炼的关思考字段名不一样"


def test_saving_without_a_key_keeps_the_stored_one(client):
    """界面不回显明文，所以改模型名时 key 框是空的 —— 空 = 不改动。"""
    client.post("/api/model/config", json={
        "slot": "vlm", "provider": "deepseek", "api_key": "sk-keepme123456",
    })
    r = client.post("/api/model/config", json={
        "slot": "vlm", "provider": "deepseek", "model": "deepseek-v4-pro",
    })
    assert r.status_code == 200, r.text[:400]
    slot = r.json()["slots"]["vlm"]
    assert slot["model"] == "deepseek-v4-pro"
    assert slot["has_key"] is True, "空 key 把已存的密钥冲掉了"

    from app.collection import settings

    assert settings.credentials_for("vlm").api_key == "sk-keepme123456"


def test_a_custom_provider_without_an_endpoint_is_refused(client):
    """自定义供应方要填**完整请求地址** —— 只填域名必然 404。"""
    r = client.post("/api/model/config", json={
        "slot": "vlm", "provider": "custom", "api_key": "sk-x",
    })
    assert r.status_code == 422
    assert "请求地址" in r.json()["detail"]


def test_a_provider_without_a_model_name_is_refused(client):
    """模型名不能空 —— 官方要求复制控制台上的完整 ID。"""
    from app.collection import model_presets

    r = client.post("/api/model/config", json={
        "slot": "vlm", "provider": "custom", "api_key": "sk-x",
        "endpoint": "https://example.com/v1/chat/completions",
        "model": "",
    })
    # 自定义没有默认模型 → 必须用户自己填
    assert r.status_code == 422
    assert model_presets.resolve("custom", "", "vlm")["model"] == ""


# ---------------------------------------------------------------- 测试连接


def test_the_test_endpoint_measures_whether_images_can_be_seen(client, mock_model):
    """带一张纯红小图问颜色 —— 答对才算能看图。"""
    endpoint, handler = mock_model
    r = client.post("/api/model/test", json={
        "slot": "vlm", "provider": "custom", "api_key": "sk-test",
        "endpoint": endpoint, "protocol": "openai", "model": "mock-vlm",
    })
    assert r.status_code == 200, r.text[:400]
    data = r.json()
    assert data["ok"] is True
    assert data["can_see_images"] is True, data
    # 请求里确实带了图
    body = handler.received[0]["body"]
    kinds = [b["type"] for b in body["messages"][-1]["content"]]
    assert "image_url" in kinds


def test_a_model_that_cannot_see_images_says_so(client, mock_model):
    """答不出颜色 → 明确说「多半读不了画面」，而不是含糊地「通了」。"""
    endpoint, handler = mock_model
    handler.answer = {"choices": [{"message": {"content": "我不知道"}}]}
    r = client.post("/api/model/test", json={
        "slot": "vlm", "provider": "custom", "api_key": "sk-test",
        "endpoint": endpoint, "protocol": "openai", "model": "mock-vlm",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["can_see_images"] is False
    assert "读不了画面" in data["note"]


def test_the_test_endpoint_does_not_write_the_config(client, mock_model):
    """「测试连接」不写盘 —— 测就只是测。

    顺手把配置改了的按钮是会咬人的：用户测完改主意，
    配置却已经被换掉了。
    """
    endpoint, _handler = mock_model
    r = client.post("/api/model/test", json={
        "slot": "vlm", "provider": "custom", "api_key": "sk-test",
        "endpoint": endpoint, "protocol": "openai", "model": "mock-vlm",
    })
    assert r.status_code == 200
    slot = client.get("/api/model/config").json()["slots"]["vlm"]
    assert slot["has_key"] is False, "测试顺手把 key 存进去了"
    assert slot["provider"] != "custom"


def test_the_test_endpoint_reports_why_it_failed(client):
    """不通要说清是哪一类 —— 「调用失败」没法处理。"""
    r = client.post("/api/model/test", json={
        "slot": "vlm", "provider": "custom", "api_key": "sk-test",
        "endpoint": "http://127.0.0.1:9/never", "protocol": "openai",
        "model": "mock-vlm",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is False
    assert data["kind"]
    assert data["error"]


# ---------------------------------------------------------------- 页面


def test_the_model_page_exists_and_has_both_slots(client):
    r = client.get("/model")
    assert r.status_code == 200
    html = r.text
    assert "看画面的模型" in html
    assert "对账措辞的模型" in html
    assert "/api/model/config" in html
    assert "/api/model/test" in html


def test_the_navigation_has_the_model_entry(client):
    """侧边栏要有一项「模型」—— 配置有自己的页面。"""
    html = client.get("/upload").text
    assert '/model' in html
    assert "模型" in html


def test_the_upload_page_has_only_a_one_line_model_status(client):
    """上传页**不再有模型输入框** —— 那里只留一行状态加一个链接。

    摆一个输入框会让人以为「不填就走不下去」，而它其实是
    「不填就没有自动结论，报告照出」。两件事不一样。
    """
    html = client.get("/upload").text
    assert 'id="modelKey"' not in html, "上传页还在收 API Key"
    assert 'id="modelKeySave"' not in html
    assert 'id="modelState"' in html, "上传页没有那一行模型状态"
    assert "model-hint" not in html, "上传页还在摆解释性小字"
