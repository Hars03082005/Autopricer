"""Supabase JWT verification.

The issuer test is the important one here. A token signed with a valid key but
issued by a *different* Supabase project verifies its signature successfully, so
without an explicit issuer check any other project's users would be accepted as
users of this one.
"""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from backend.auth import require_admin_token, verify_token
from backend.config import Settings

PROJECT_URL = "https://project.supabase.co"
OTHER_PROJECT_URL = "https://someone-elses-project.supabase.co"
SECRET = "test-jwt-secret-value-at-least-32-chars"
USER_ID = "3f8c1a44-0c5e-4f2a-9b1e-2d7c6a5b4e33"


def make_token(
    *,
    secret: str = SECRET,
    issuer: str = f"{PROJECT_URL}/auth/v1",
    audience: str = "authenticated",
    subject: str | None = USER_ID,
    expires_in: int = 3600,
    algorithm: str = "HS256",
    **extra,
) -> str:
    now = int(time.time())
    claims = {
        "iss": issuer,
        "aud": audience,
        "iat": now,
        "exp": now + expires_in,
        "email": "dealer@example.com",
        "role": "authenticated",
        **extra,
    }
    if subject is not None:
        claims["sub"] = subject
    return jwt.encode(claims, secret, algorithm=algorithm)


def hs256_settings(**overrides) -> Settings:
    base = {
        "SUPABASE_URL": PROJECT_URL,
        "SUPABASE_ANON_KEY": "anon-key",
        "SUPABASE_JWT_SECRET": SECRET,
    }
    base.update(overrides)
    return Settings.load(base)


class TestTokenVerification:
    def test_valid_token_is_accepted(self):
        user = verify_token(make_token(), hs256_settings())
        assert user.id == USER_ID
        assert user.email == "dealer@example.com"
        assert user.role == "authenticated"

    def test_access_token_is_retained_for_rls_scoped_queries(self):
        token = make_token()
        user = verify_token(token, hs256_settings())
        assert user.access_token == token

    def test_access_token_is_not_in_repr(self):
        """Guards against the token reaching logs via an exception trace."""
        user = verify_token(make_token(), hs256_settings())
        assert user.access_token not in repr(user)

    def test_token_from_another_project_is_rejected(self):
        token = make_token(issuer=f"{OTHER_PROJECT_URL}/auth/v1")
        with pytest.raises(HTTPException) as exc:
            verify_token(token, hs256_settings())
        assert exc.value.status_code == 401
        assert "different project" in exc.value.detail

    def test_expired_token_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            verify_token(make_token(expires_in=-10), hs256_settings())
        assert exc.value.status_code == 401
        assert "expired" in exc.value.detail.lower()

    def test_wrong_audience_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            verify_token(make_token(audience="anon"), hs256_settings())
        assert exc.value.status_code == 401

    def test_token_signed_with_the_wrong_secret_is_rejected(self):
        token = make_token(secret="a-completely-different-secret-value")
        with pytest.raises(HTTPException) as exc:
            verify_token(token, hs256_settings())
        assert exc.value.status_code == 401

    def test_token_without_subject_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            verify_token(make_token(subject=None), hs256_settings())
        assert exc.value.status_code == 401

    def test_unsigned_token_is_rejected(self):
        """alg=none must never be accepted."""
        token = jwt.encode(
            {"sub": USER_ID, "aud": "authenticated", "iss": f"{PROJECT_URL}/auth/v1",
             "exp": int(time.time()) + 60},
            key="",
            algorithm="none",
        )
        with pytest.raises(HTTPException) as exc:
            verify_token(token, hs256_settings())
        assert exc.value.status_code == 401

    def test_garbage_token_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            verify_token("not-a-jwt-at-all", hs256_settings())
        assert exc.value.status_code == 401


class TestServiceMisconfiguration:
    def test_missing_supabase_url_reports_503_not_401(self):
        """A server-side gap is not the caller's fault, so it must not read as 401."""
        settings = Settings.load({})
        with pytest.raises(HTTPException) as exc:
            verify_token(make_token(), settings)
        assert exc.value.status_code == 503

    def test_hs256_token_without_configured_secret_reports_503(self):
        settings = Settings.load(
            {"SUPABASE_URL": PROJECT_URL, "SUPABASE_ANON_KEY": "anon-key"}
        )
        with pytest.raises(HTTPException) as exc:
            verify_token(make_token(), settings)
        assert exc.value.status_code == 503
        assert "SUPABASE_JWT_SECRET" in exc.value.detail


class TestAdminGuard:
    def _creds(self, token: str) -> HTTPAuthorizationCredentials:
        return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    def test_refused_when_switching_is_disabled(self, env):
        env(ALLOW_RUNTIME_VARIANT_SWITCH="false", ADMIN_API_TOKEN=None)
        with pytest.raises(HTTPException) as exc:
            require_admin_token(self._creds("x" * 40))
        assert exc.value.status_code == 403
        assert "disabled" in exc.value.detail

    def test_wrong_token_is_refused(self, env):
        env(ALLOW_RUNTIME_VARIANT_SWITCH="true", ADMIN_API_TOKEN="a" * 40)
        with pytest.raises(HTTPException) as exc:
            require_admin_token(self._creds("b" * 40))
        assert exc.value.status_code == 403

    def test_missing_token_is_refused(self, env):
        env(ALLOW_RUNTIME_VARIANT_SWITCH="true", ADMIN_API_TOKEN="a" * 40)
        with pytest.raises(HTTPException) as exc:
            require_admin_token(None)
        assert exc.value.status_code == 401

    def test_correct_token_passes(self, env):
        env(ALLOW_RUNTIME_VARIANT_SWITCH="true", ADMIN_API_TOKEN="a" * 40)
        assert require_admin_token(self._creds("a" * 40)) is None
