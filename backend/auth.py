"""JWT Verification."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.config import Settings, get_settings

log = logging.getLogger("priceref.auth")

_EXPECTED_AUDIENCE = "authenticated"

_SUPPORTED_ASYMMETRIC_ALGORITHMS = ("ES256", "RS256", "ES384", "RS384", "ES512", "RS512")

_bearer_scheme = HTTPBearer(auto_error=False, description="Supabase access token")

_jwks_client: jwt.PyJWKClient | None = None
_jwks_client_url: str | None = None


@dataclass(frozen=True)
class AuthenticatedUser:
    """Verified Caller."""

    id: str
    email: str | None
    role: str
    claims: dict[str, Any]
    access_token: str = field(default="", repr=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _get_jwks_client(settings: Settings) -> jwt.PyJWKClient:
    """Get JWKS Client."""
    global _jwks_client, _jwks_client_url
    if _jwks_client is None or _jwks_client_url != settings.jwks_url:
        _jwks_client = jwt.PyJWKClient(
            settings.jwks_url,
            cache_keys=True,
            lifespan=600,
            max_cached_keys=16,
            timeout=int(settings.request_timeout_seconds),
        )
        _jwks_client_url = settings.jwks_url
    return _jwks_client


def reset_jwks_cache() -> None:
    """Reset JWKS Cache."""
    global _jwks_client, _jwks_client_url
    _jwks_client = None
    _jwks_client_url = None


def verify_token(token: str, settings: Settings) -> AuthenticatedUser:
    """Verify Token."""
    if not settings.auth_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on this server (SUPABASE_URL unset).",
        )

    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise _unauthorized(f"Malformed token: {exc}") from exc

    algorithm = header.get("alg")

    if algorithm == "HS256":
        if not settings.supabase_jwt_secret:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Token is HS256-signed but SUPABASE_JWT_SECRET is not set. "
                    "Supply it, or migrate the project to asymmetric JWT keys."
                ),
            )
        key: Any = settings.supabase_jwt_secret
        algorithms = ["HS256"]
    elif algorithm in _SUPPORTED_ASYMMETRIC_ALGORITHMS:
        try:
            key = _get_jwks_client(settings).get_signing_key_from_jwt(token).key
        except jwt.PyJWTError as exc:
            raise _unauthorized(f"Token signing key could not be resolved: {exc}") from exc
        except Exception as exc:  # network / malformed JWKS document
            log.warning("JWKS fetch failed for %s: %s", settings.jwks_url, exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Unable to reach the identity provider to verify the token.",
            ) from exc
        algorithms = [algorithm]
    else:
        raise _unauthorized(f"Unsupported token algorithm: {algorithm!r}")

    try:
        claims = jwt.decode(
            token,
            key=key,
            algorithms=algorithms,
            audience=_EXPECTED_AUDIENCE,
            issuer=f"{settings.supabase_url}/auth/v1",
            options={"require": ["exp", "sub", "aud", "iss"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise _unauthorized("Token has expired.") from exc
    except jwt.InvalidAudienceError as exc:
        raise _unauthorized("Token audience is not 'authenticated'.") from exc
    except jwt.InvalidIssuerError as exc:
        raise _unauthorized("Token was issued by a different project.") from exc
    except jwt.PyJWTError as exc:
        raise _unauthorized(f"Token verification failed: {exc}") from exc

    subject = claims.get("sub")
    if not subject:
        raise _unauthorized("Token has no subject claim.")

    return AuthenticatedUser(
        id=str(subject),
        email=claims.get("email"),
        role=str(claims.get("role") or "authenticated"),
        claims=claims,
        access_token=token,
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthenticatedUser:
    """Get Current User."""
    if credentials is None or not credentials.credentials:
        raise _unauthorized("Missing bearer token.")
    return verify_token(credentials.credentials, get_settings())


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthenticatedUser | None:
    """Get Optional User."""
    if credentials is None or not credentials.credentials:
        return None
    return verify_token(credentials.credentials, get_settings())


def require_admin_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """Require Admin Token."""
    import secrets

    settings = get_settings()

    if not settings.allow_runtime_variant_switch:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Runtime variant switching is disabled. Each replica would switch "
                "independently and serve a different model from its peers. Deploy "
                "a new revision with a different ACTIVE_VARIANT_ID instead."
            ),
        )

    if credentials is None or not credentials.credentials:
        raise _unauthorized("Missing admin bearer token.")

    if not secrets.compare_digest(credentials.credentials, settings.admin_api_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin token.",
        )
