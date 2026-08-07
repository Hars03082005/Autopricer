"""Configuration validation.

These are the checks that turn a misconfigured deployment into a startup failure
with a readable message instead of a subtle runtime behaviour change. They are
cheap and need no model artifacts, so they run on every push.
"""

from __future__ import annotations

import pytest

from backend.config import ConfigError, Settings


class TestCors:
    def test_wildcard_is_refused_in_production(self):
        """The original code shipped allow_origins=['*'] unconditionally."""
        with pytest.raises(ConfigError, match="not permitted when"):
            Settings.load({"APP_ENVIRONMENT": "production", "CORS_ALLOWED_ORIGINS": "*"})

    def test_missing_allowlist_is_refused_in_production(self):
        with pytest.raises(ConfigError, match="must be set when"):
            Settings.load({"APP_ENVIRONMENT": "production"})

    def test_wildcard_allowed_in_development_with_a_warning(self):
        settings = Settings.load({"APP_ENVIRONMENT": "development", "CORS_ALLOWED_ORIGINS": "*"})
        assert settings.cors_allow_all is True
        assert any("every origin" in w for w in settings._warnings)

    def test_explicit_allowlist_is_parsed_and_trimmed(self):
        settings = Settings.load(
            {
                "APP_ENVIRONMENT": "production",
                "CORS_ALLOWED_ORIGINS": " https://a.example , https://b.example ",
            }
        )
        assert settings.cors_allowed_origins == ("https://a.example", "https://b.example")
        assert settings.cors_allow_all is False

    def test_development_defaults_to_local_origins(self):
        settings = Settings.load({})
        assert "http://localhost:5173" in settings.cors_allowed_origins


class TestSupabase:
    def test_http_url_is_refused(self):
        """Plain http would put the caller's bearer token on the wire in clear."""
        with pytest.raises(ConfigError, match="must use https"):
            Settings.load({"SUPABASE_URL": "http://project.supabase.co"})

    def test_local_http_is_allowed_for_the_cli_stack(self):
        settings = Settings.load({"SUPABASE_URL": "http://127.0.0.1:54321"})
        assert settings.supabase_url == "http://127.0.0.1:54321"

    def test_identical_anon_and_service_keys_are_refused(self):
        with pytest.raises(ConfigError, match="identical"):
            Settings.load(
                {
                    "SUPABASE_URL": "https://p.supabase.co",
                    "SUPABASE_ANON_KEY": "same-key-value",
                    "SUPABASE_SERVICE_ROLE_KEY": "same-key-value",
                }
            )

    def test_database_enabled_needs_url_and_anon_key(self):
        assert Settings.load({}).database_enabled is False
        assert (
            Settings.load({"SUPABASE_URL": "https://p.supabase.co"}).database_enabled is False
        )
        assert (
            Settings.load(
                {"SUPABASE_URL": "https://p.supabase.co", "SUPABASE_ANON_KEY": "anon"}
            ).database_enabled
            is True
        )

    def test_database_enabled_does_not_require_the_service_role_key(self):
        """User-scoped queries run as the caller so RLS still applies."""
        settings = Settings.load(
            {"SUPABASE_URL": "https://p.supabase.co", "SUPABASE_ANON_KEY": "anon"}
        )
        assert settings.supabase_service_role_key == ""
        assert settings.database_enabled is True

    def test_trailing_slash_is_stripped_so_urls_do_not_double_up(self):
        settings = Settings.load({"SUPABASE_URL": "https://p.supabase.co/"})
        assert settings.postgrest_url == "https://p.supabase.co/rest/v1"
        assert settings.jwks_url == "https://p.supabase.co/auth/v1/.well-known/jwks.json"


class TestAdminSurface:
    def test_variant_switch_requires_a_token(self):
        with pytest.raises(ConfigError, match="ADMIN_API_TOKEN is"):
            Settings.load({"ALLOW_RUNTIME_VARIANT_SWITCH": "true"})

    def test_short_token_is_refused(self):
        with pytest.raises(ConfigError, match="at least 32 characters"):
            Settings.load(
                {"ALLOW_RUNTIME_VARIANT_SWITCH": "true", "ADMIN_API_TOKEN": "short"}
            )

    def test_variant_switch_is_off_by_default(self):
        assert Settings.load({}).allow_runtime_variant_switch is False

    def test_enabling_in_production_warns_about_replica_drift(self):
        settings = Settings.load(
            {
                "APP_ENVIRONMENT": "production",
                "CORS_ALLOWED_ORIGINS": "https://a.example",
                "ALLOW_RUNTIME_VARIANT_SWITCH": "true",
                "ADMIN_API_TOKEN": "x" * 40,
            }
        )
        assert any("replica" in w for w in settings._warnings)

    @pytest.mark.parametrize("raw,expected", [
        ("true", True), ("TRUE", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("0", False), ("", False), ("nonsense", False),
    ])
    def test_boolean_parsing(self, raw, expected):
        settings = Settings.load(
            {"ALLOW_RUNTIME_VARIANT_SWITCH": raw, "ADMIN_API_TOKEN": "x" * 40}
        )
        assert settings.allow_runtime_variant_switch is expected


class TestMisc:
    def test_log_summary_does_not_leak_secrets(self, caplog):
        settings = Settings.load(
            {
                "SUPABASE_URL": "https://p.supabase.co",
                "SUPABASE_ANON_KEY": "anon-key-value",
                "SUPABASE_SERVICE_ROLE_KEY": "super-secret-service-key",
                "SUPABASE_JWT_SECRET": "super-secret-jwt-secret",
            }
        )
        with caplog.at_level("INFO"):
            settings.log_summary()
        logged = caplog.text
        assert "super-secret-service-key" not in logged
        assert "super-secret-jwt-secret" not in logged
        assert "anon-key-value" not in logged

    def test_is_production_recognises_both_spellings(self):
        for value in ("production", "prod", "PRODUCTION"):
            settings = Settings.load(
                {"APP_ENVIRONMENT": value, "CORS_ALLOWED_ORIGINS": "https://a.example"}
            )
            assert settings.is_production is True
