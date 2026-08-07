"""Supabase JWT verification.

The browser continues to authenticate directly against Supabase Auth (GoTrue) —
that part was already correct and there is no reason to proxy password flows
through this service. What changed is that the access token it receives is now
presented to *this* API, which verifies it and performs database work on the
user's behalf, instead of the browser writing to Postgres directly.

Two signing schemes are supported, because which one a project uses depends on
when it was created and whether it has migrated:

  * Asymmetric (ES256 / RS256) — current default. Verified against the project's
    published JWKS. No shared secret needs to be deployed.
  * HS256 — legacy shared secret, supplied as SUPABASE_JWT_SECRET.

Both the audience and the issuer are checked. Skipping the issuer check is the
subtle mistake here: without it, a validly-signed token from *any other*
Supabase project would be accepted, since the signature verifies against
whatever key material that project publishes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.config import Settings, get_settings

log = logging.getLogger("priceref.auth")

# Supabase issues access tokens with aud="authenticated" for signed-in users.
_EXPECTED_AUDIENCE = "authenticated"

_SUPPORTED_ASYMMETRIC_ALGORITHMS = ("ES256", "RS256", "ES384", "RS384", "ES512", "RS512")

# auto_error=False so that endpoints can distinguish "no credentials supplied"
# from "credentials supplied but invalid" and phrase the 401 accordingly.
_bearer_scheme = HTTPBearer(auto_error=False, description="Supabase access token")

_jwks_client: jwt.PyJWKClient | None = None
_jwks_client_url: str | None = None


@dataclass(frozen=True)
class AuthenticatedUser:
    """The verified caller. `id` is the Supabase auth.users UUID.

    `access_token` is retained because database calls are made *as this user*
    against PostgREST, so row-level security still applies (see backend/db.py).
    It is excluded from repr so it cannot leak into a logged exception.
    """

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
    """Cached JWKS client.

    Rebuilt only if the configured URL changes (which in practice means tests).
    PyJWKClient does its own key caching, so a request does not fetch per call;
    `lifespan` bounds how long a rotated-out key stays trusted.
    """
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
    """Drop the cached JWKS client. For tests only."""
    global _jwks_client, _jwks_client_url
    _jwks_client = None
    _jwks_client_url = None


def verify_token(token: str, settings: Settings) -> AuthenticatedUser:
    """Verify a Supabase access token and return the caller.

    Raises HTTPException(401) for anything unverifiable, and 503 when the
    service itself is not configured to verify tokens at all — the latter is a
    server fault, not the caller's, so it must not be reported as 401.
    """
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
            # The project still signs with a shared secret but this deployment
            # was not given it. Refusing loudly beats silently rejecting every
            # user as unauthenticated.
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
        # Correctly signed, but by a different Supabase project.
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
    """FastAPI dependency requiring a verified Supabase user.

    Sync rather than async on purpose: JWKS resolution is a blocking HTTP call
    inside PyJWT, so FastAPI running this in its threadpool keeps it off the
    event loop.
    """
    if credentials is None or not credentials.credentials:
        raise _unauthorized("Missing bearer token.")
    return verify_token(credentials.credentials, get_settings())


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthenticatedUser | None:
    """As `get_current_user`, but returns None instead of raising when absent.

    For endpoints that serve guests and signed-in users alike. An invalid token
    still raises: silently treating a bad token as "guest" would hide expiry
    from the client and look like data loss to the user.
    """
    if credentials is None or not credentials.credentials:
        return None
    return verify_token(credentials.credentials, get_settings())


def require_admin_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """Guard for the model-variant administration endpoint.

    A static token rather than a Supabase role check: this is an operator action,
    not a user action, and it must stay usable from CI and from a shell without
    provisioning an application user.
    """
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

    # compare_digest to keep the check constant-time.
    if not secrets.compare_digest(credentials.credentials, settings.admin_api_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin token.",
        )
