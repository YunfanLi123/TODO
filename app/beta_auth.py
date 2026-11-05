"""内测版身份与邀请码。

密码只保存 PBKDF2 哈希；会话由 Starlette 的签名 Cookie 承载。生产环境必须
设置 ``BETA_SESSION_SECRET`` 和 ``BETA_INVITE_CODE``，不要把它们提交进仓库。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
from typing import Optional

from .db import User

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
ITERATIONS = 260_000


def enabled() -> bool:
    return os.environ.get("BETA_AUTH", "").strip().lower() in {"1", "true", "yes"}


def session_secret() -> str:
    """开发环境可启动；正式内测若未设置密钥则拒绝启用登录。"""
    value = os.environ.get("BETA_SESSION_SECRET", "").strip()
    if enabled() and not value:
        raise RuntimeError("内测模式必须设置 BETA_SESSION_SECRET")
    return value or "local-development-only-not-for-public-use"


def player_id(user_id: str) -> str:
    return f"user-{user_id}"


def normalize_email(value: str) -> str:
    return value.strip().lower()


def validate_registration(email: str, password: str, invite_code: str) -> Optional[str]:
    if not EMAIL_RE.fullmatch(email):
        return "请输入有效的邮箱地址。"
    if len(password) < 8:
        return "密码至少需要 8 位。"
    expected = os.environ.get("BETA_INVITE_CODE", "").strip()
    if not expected:
        return "服务尚未配置内测邀请码。"
    if not hmac.compare_digest(invite_code.strip(), expected):
        return "邀请码不正确。"
    return None


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        ITERATIONS, base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, count, encoded_salt, encoded_digest = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(encoded_salt.encode("ascii"))
        expected = base64.urlsafe_b64decode(encoded_digest.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(count))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def make_user(email: str, password: str, display_name: str = "") -> User:
    return User(id=secrets.token_urlsafe(18), email=normalize_email(email),
                password_hash=hash_password(password), display_name=display_name.strip()[:80])
