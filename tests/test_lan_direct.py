"""局域网直连（手机直传）：开关与地址探测。

为什么要有这个
--------------
玩家问「不通过聊天软件怎么把手机上的录屏传到电脑」。答案是：
**手机浏览器直接传到跑在电脑上的服务** —— 不经任何中转，
也就不会被聊天软件重编码（聊天软件转发会把 1080p 压到 217 像素宽，
计时器数字只剩 3~4 像素，信息直接丢了）。

但默认**不能**监听局域网：同网段的别的设备不该看到你的对局。
所以这是一个要主动开的口子，开的方式必须明确、可测。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "scripts"))

import pytest  # noqa: E402

from scripts import launcher  # noqa: E402


def test_lan_is_off_by_default(monkeypatch):
    """默认只监听本机 —— 局域网里的别的设备不该看到你的对局。"""
    monkeypatch.delenv(launcher.LAN_ENV, raising=False)
    assert launcher.lan_enabled([]) is False
    assert launcher.lan_enabled(["--port", "8000"]) is False


def test_lan_can_be_turned_on_by_flag(monkeypatch):
    monkeypatch.delenv(launcher.LAN_ENV, raising=False)
    assert launcher.lan_enabled(["--lan"]) is True


def test_lan_can_be_turned_on_by_env(monkeypatch):
    """打包后双击 exe 没法传命令行参数，环境变量是那时候唯一顺手的开关。"""
    monkeypatch.setenv(launcher.LAN_ENV, "1")
    assert launcher.lan_enabled([]) is True
    monkeypatch.setenv(launcher.LAN_ENV, "0")
    assert launcher.lan_enabled([]) is False


def test_lan_addresses_are_ipv4_and_not_loopback():
    """打印出来的地址必须是手机能访问的局域网地址，不能是 127.0.0.1。"""
    addrs = launcher.lan_addresses()
    assert isinstance(addrs, list)
    for ip in addrs:
        assert not ip.startswith("127.")
        assert ip.count(".") == 3
    # 拿不到地址也不能崩 —— 启动器还要继续跑（只是没地方提示手机连哪）
    assert isinstance(addrs, list)


def test_the_env_var_name_is_documented():
    """环境变量名要写在文档里，否则用户根本知道有这回事。"""
    assert launcher.LAN_ENV in Path(
        APP_ROOT / "README.md"
    ).read_text("utf-8"), "README 里没写局域网直传的开关"
