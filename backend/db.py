"""Supabase DB Layer."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import HTTPException, status

from backend.config import Settings, get_settings

log = logging.getLogger("priceref.db")

EVALUATIONS_TABLE = "evaluations"
PROFILES_TABLE = "profiles"

_client: httpx.AsyncClient | None = None


async def startup() -> None:
    """DB Startup."""
    global _client
    if _client is not None:
        return

    settings = get_settings()
    if not settings.database_enabled:
        log.info("db | Supabase not configured — persistence endpoints will return 503")
        return

    _client = httpx.AsyncClient(
        base_url=settings.postgrest_url,
        timeout=httpx.Timeout(settings.request_timeout_seconds, connect=5.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        headers={"Accept": "application/json"},
    )
    log.info("db | PostgREST client ready (%s)", settings.postgrest_url)


async def shutdown() -> None:
    """DB Shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _require_client(settings: Settings) -> httpx.AsyncClient:
    if not settings.database_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Persistence is not configured on this server. Valuation "
                "endpoints are unaffected; history is unavailable."
            ),
        )
    if _client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database client is not initialised.",
        )
    return _client


def _user_headers(access_token: str, settings: Settings, *, prefer: str | None = None) -> dict[str, str]:
    headers = {
        "apikey": settings.supabase_anon_key,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _handle_error(response: httpx.Response, operation: str) -> None:
    """Handle DB Error."""
    if response.is_success:
        return

    try:
        body = response.json()
    except ValueError:
        body = {"message": response.text[:500]}

    log.warning(
        "db | %s failed | status=%s code=%s message=%s details=%s",
        operation,
        response.status_code,
        body.get("code"),
        body.get("message"),
        body.get("details"),
    )

    if response.status_code in (401, 403):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not permitted to access this record.",
        )

    if response.status_code == 409:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Record conflicts with an existing row.",
        )

    if response.status_code == 404 or body.get("code") == "42P01":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Database schema is not initialised. Apply the migrations in "
                "supabase/migrations/."
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"Database request failed ({operation}).",
    )


async def _request(
    method: str,
    path: str,
    *,
    access_token: str,
    settings: Settings,
    operation: str,
    params: dict[str, Any] | None = None,
    json_body: Any = None,
    prefer: str | None = None,
) -> httpx.Response:
    client = _require_client(settings)
    try:
        response = await client.request(
            method,
            path,
            params=params,
            json=json_body,
            headers=_user_headers(access_token, settings, prefer=prefer),
        )
    except httpx.TimeoutException as exc:
        log.warning("db | %s timed out: %s", operation, exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Database request timed out ({operation}).",
        ) from exc
    except httpx.HTTPError as exc:
        log.warning("db | %s transport error: %s", operation, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not reach the database ({operation}).",
        ) from exc

    _handle_error(response, operation)
    return response


async def list_evaluations(
    *, user_id: str, access_token: str, limit: int
) -> list[dict[str, Any]]:
    settings = get_settings()
    capped = max(1, min(limit, settings.max_history_rows))
    response = await _request(
        "GET",
        f"/{EVALUATIONS_TABLE}",
        access_token=access_token,
        settings=settings,
        operation="list_evaluations",
        params={
            "select": "*",
            "user_id": f"eq.{user_id}",
            "order": "created_at.desc",
            "limit": str(capped),
        },
    )
    payload = response.json()
    return payload if isinstance(payload, list) else []


async def insert_evaluation(
    *, row: dict[str, Any], access_token: str
) -> dict[str, Any]:
    settings = get_settings()
    response = await _request(
        "POST",
        f"/{EVALUATIONS_TABLE}",
        access_token=access_token,
        settings=settings,
        operation="insert_evaluation",
        json_body=row,
        prefer="return=representation",
    )
    payload = response.json()
    if isinstance(payload, list) and payload:
        return payload[0]
    if isinstance(payload, dict):
        return payload
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Database returned no row for the inserted evaluation.",
    )


async def delete_evaluations(*, user_id: str, access_token: str) -> int:
    settings = get_settings()
    response = await _request(
        "DELETE",
        f"/{EVALUATIONS_TABLE}",
        access_token=access_token,
        settings=settings,
        operation="delete_evaluations",
        params={"user_id": f"eq.{user_id}"},
        prefer="return=representation",
    )
    payload = response.json()
    return len(payload) if isinstance(payload, list) else 0


async def delete_evaluation(*, user_id: str, evaluation_id: str, access_token: str) -> int:
    settings = get_settings()
    response = await _request(
        "DELETE",
        f"/{EVALUATIONS_TABLE}",
        access_token=access_token,
        settings=settings,
        operation="delete_evaluation",
        params={"id": f"eq.{evaluation_id}", "user_id": f"eq.{user_id}"},
        prefer="return=representation",
    )
    payload = response.json()
    return len(payload) if isinstance(payload, list) else 0


async def get_profile(*, user_id: str, access_token: str) -> dict[str, Any] | None:
    settings = get_settings()
    response = await _request(
        "GET",
        f"/{PROFILES_TABLE}",
        access_token=access_token,
        settings=settings,
        operation="get_profile",
        params={"select": "*", "id": f"eq.{user_id}", "limit": "1"},
    )
    payload = response.json()
    if isinstance(payload, list) and payload:
        return payload[0]
    return None


async def upsert_profile(
    *, user_id: str, access_token: str, fields: dict[str, Any]
) -> dict[str, Any]:
    settings = get_settings()
    row = {**fields, "id": user_id}
    response = await _request(
        "POST",
        f"/{PROFILES_TABLE}",
        access_token=access_token,
        settings=settings,
        operation="upsert_profile",
        json_body=row,
        prefer="return=representation,resolution=merge-duplicates",
    )
    payload = response.json()
    if isinstance(payload, list) and payload:
        return payload[0]
    if isinstance(payload, dict):
        return payload
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Database returned no row for the upserted profile.",
    )


async def ping(*, access_token: str) -> bool:
    """DB Ping."""
    settings = get_settings()
    if not settings.database_enabled:
        return False
    try:
        await _request(
            "GET",
            f"/{EVALUATIONS_TABLE}",
            access_token=access_token,
            settings=settings,
            operation="ping",
            params={"select": "id", "limit": "1"},
        )
        return True
    except HTTPException:
        return False
