"""Runtime Configuration."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

log = logging.getLogger("priceref.config")

_DEFAULT_DEV_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
)

_TRUTHY = frozenset({"1", "true", "yes", "on"})


class ConfigError(RuntimeError):
    """Config Error."""


def _as_bool(raw: str | None, *, default: bool = False) -> bool:
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in _TRUTHY


def _as_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration."""

    environment: str = "development"
    cors_allowed_origins: tuple[str, ...] = _DEFAULT_DEV_ORIGINS
    cors_allow_all: bool = False

    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""

    allow_runtime_variant_switch: bool = False
    admin_api_token: str = ""

    active_variant_id: str = ""
    request_timeout_seconds: float = 15.0
    max_history_rows: int = 500

    _warnings: tuple[str, ...] = field(default=(), repr=False)


    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() in {"production", "prod"}

    @property
    def database_enabled(self) -> bool:
        """DB Enabled."""
        return bool(self.supabase_url and self.supabase_anon_key)

    @property
    def auth_enabled(self) -> bool:
        """Auth Enabled."""
        return bool(self.supabase_url)

    @property
    def jwks_url(self) -> str:
        """JWKS URL."""
        return f"{self.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"

    @property
    def postgrest_url(self) -> str:
        """PostgREST URL."""
        return f"{self.supabase_url.rstrip('/')}/rest/v1"

    @classmethod
    def load(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Load Settings."""
        source = os.environ if env is None else env
        warnings: list[str] = []

        environment = (source.get("APP_ENVIRONMENT") or "development").strip()
        is_production = environment.lower() in {"production", "prod"}

        raw_origins = _as_list(source.get("CORS_ALLOWED_ORIGINS"))
        allow_all = "*" in raw_origins

        if allow_all and is_production:
            raise ConfigError(
                "CORS_ALLOWED_ORIGINS='*' is not permitted when "
                "APP_ENVIRONMENT=production. Set an explicit origin allowlist "
                "(comma-separated), e.g. https://app.example.com"
            )

        if allow_all:
            warnings.append(
                "CORS_ALLOWED_ORIGINS='*' — every origin may call this API. "
                "Acceptable locally; set an allowlist before deploying."
            )
            origins: tuple[str, ...] = ()
        elif raw_origins:
            origins = tuple(raw_origins)
        else:
            origins = _DEFAULT_DEV_ORIGINS
            if is_production:
                raise ConfigError(
                    "CORS_ALLOWED_ORIGINS must be set when "
                    "APP_ENVIRONMENT=production."
                )
            warnings.append(
                f"CORS_ALLOWED_ORIGINS unset — defaulting to local dev origins: "
                f"{', '.join(origins)}"
            )

        supabase_url = (source.get("SUPABASE_URL") or "").strip().rstrip("/")
        anon_key = (source.get("SUPABASE_ANON_KEY") or "").strip()
        service_key = (source.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
        jwt_secret = (source.get("SUPABASE_JWT_SECRET") or "").strip()

        if supabase_url and not supabase_url.startswith("https://"):
            if not supabase_url.startswith("http://127.0.0.1") and not supabase_url.startswith(
                "http://localhost"
            ):
                raise ConfigError(
                    f"SUPABASE_URL must use https:// (got {supabase_url!r})."
                )

        if bool(supabase_url) != bool(anon_key):
            warnings.append(
                "Only one of SUPABASE_URL / SUPABASE_ANON_KEY is set — "
                "server-side persistence stays disabled until both are provided."
            )

        if service_key and service_key == anon_key:
            raise ConfigError(
                "SUPABASE_SERVICE_ROLE_KEY and SUPABASE_ANON_KEY are identical — "
                "one of them is wrong. Copy them separately from the Supabase "
                "project API settings."
            )

        allow_switch = _as_bool(source.get("ALLOW_RUNTIME_VARIANT_SWITCH"))
        admin_token = (source.get("ADMIN_API_TOKEN") or "").strip()

        if allow_switch and not admin_token:
            raise ConfigError(
                "ALLOW_RUNTIME_VARIANT_SWITCH is enabled but ADMIN_API_TOKEN is "
                "empty. The variant-activation endpoint mutates the served model, "
                "so it cannot be exposed without a token."
            )
        if allow_switch and is_production:
            warnings.append(
                "ALLOW_RUNTIME_VARIANT_SWITCH is enabled in production. Each "
                "replica switches independently, so replicas will serve different "
                "models until they are all restarted. Prefer redeploying with a "
                "different ACTIVE_VARIANT_ID."
            )

        if admin_token and 0 < len(admin_token) < 32:
            raise ConfigError(
                "ADMIN_API_TOKEN must be at least 32 characters — it is the only "
                "thing standing in front of an endpoint that changes which model "
                "serves production prices."
            )

        return cls(
            environment=environment,
            cors_allowed_origins=origins,
            cors_allow_all=allow_all,
            supabase_url=supabase_url,
            supabase_anon_key=anon_key,
            supabase_service_role_key=service_key,
            supabase_jwt_secret=jwt_secret,
            allow_runtime_variant_switch=allow_switch,
            admin_api_token=admin_token,
            active_variant_id=(source.get("ACTIVE_VARIANT_ID") or "").strip(),
            request_timeout_seconds=float(source.get("SUPABASE_TIMEOUT_SECONDS") or 15.0),
            max_history_rows=int(source.get("MAX_HISTORY_ROWS") or 500),
            _warnings=tuple(warnings),
        )

    def log_summary(self) -> None:
        """Log Config Summary."""
        log.info(
            "config | env=%s cors=%s database=%s auth=%s variant=%s",
            self.environment,
            "*" if self.cors_allow_all else ",".join(self.cors_allowed_origins) or "(none)",
            "enabled" if self.database_enabled else "disabled",
            "enabled" if self.auth_enabled else "disabled",
            self.active_variant_id or "(registry default)",
        )
        for warning in self._warnings:
            log.warning("config | %s", warning)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get Settings."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def reset_settings_cache() -> None:
    """Reset Settings Cache."""
    global _settings
    _settings = None
