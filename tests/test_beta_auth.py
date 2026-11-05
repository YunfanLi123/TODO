from app import beta_auth


def test_password_is_hashed_and_verified():
    encoded = beta_auth.hash_password("safe-password")
    assert "safe-password" not in encoded
    assert beta_auth.verify_password("safe-password", encoded)
    assert not beta_auth.verify_password("wrong-password", encoded)


def test_registration_requires_the_invite_code(monkeypatch):
    monkeypatch.setenv("BETA_INVITE_CODE", "invite-only")
    assert beta_auth.validate_registration("user@example.com", "password123", "wrong")
    assert beta_auth.validate_registration("not-an-email", "password123", "invite-only")
    assert beta_auth.validate_registration("user@example.com", "password123", "invite-only") is None
