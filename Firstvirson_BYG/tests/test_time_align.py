"""「对局起点校正」：改一个数，各段**重读一遍**。

流程改成了全自动之后，**页面上那个控件没有了** —— 上传完直接出报告，
中间不再有让人填的一页。但服务端那套「重读」的机制留着：
它是「以后要加人工覆盖时」成套语义的一部分（改起点 → 重新抽帧 →
重新识别 → 结论跟着变），不是随手写的临时代码。

所以这里盯的只剩**服务端那半边**：

1. 起点一改，原来那批帧显示的局内时刻本来就是错的，
   在前端挪一下数字等于拿旧画面说新话 —— 平移是不行的。
2. 服务端要真的**重读**：重新抽帧、重新识别，不是挪数字。
   （重读的机制见 ``tests/test_realign.py``，这里盯接线。）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _src(rel: str) -> str:
    return (APP_ROOT / rel).read_text("utf-8")


def test_the_server_rereads_on_correction():
    """校正要交给服务端重读，不能只挪数字。"""
    api = _src("app/collection/video_api.py")
    assert "_apply_start_correction" in api, "校正路径没了"
    assert "realign.realign_windows" in api, "校正没有走重读模块"

    realign = _src("app/collection/realign.py")
    assert "realign_windows" in realign
