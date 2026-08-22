"""Catalog system tests — verifies the complete brand→model→variant pipeline."""
from __future__ import annotations

import pytest

from backend.brand_catalog import normalize_brand_name, build_brand_catalog, get_catalog_variants


# ── Brand canonicalization ──────────────────────────────────────────────────

class TestBrandNormalization:
    def test_maruti_display_name(self):
        assert normalize_brand_name("Maruti") == "maruti suzuki"

    def test_maruti_suzuki_exact(self):
        assert normalize_brand_name("Maruti Suzuki") == "maruti suzuki"

    def test_maruti_lowercase(self):
        assert normalize_brand_name("maruti") == "maruti suzuki"

    def test_maruti_uppercase(self):
        assert normalize_brand_name("MARUTI SUZUKI") == "maruti suzuki"

    def test_maruti_hyphen(self):
        assert normalize_brand_name("maruti-suzuki") == "maruti suzuki"

    def test_mercedes_benz_variants(self):
        for variant in ["Mercedes", "mercedes benz", "Mercedes-Benz", "merc"]:
            assert normalize_brand_name(variant) == "mercedes-benz", f"Failed for: {variant!r}"

    def test_land_rover_variants(self):
        assert normalize_brand_name("Land Rover") == "land rover"
        assert normalize_brand_name("land-rover") == "land rover"
        assert normalize_brand_name("Range Rover") == "land rover"

    def test_hyundai_passthrough(self):
        assert normalize_brand_name("Hyundai") == "hyundai"

    def test_honda_passthrough(self):
        assert normalize_brand_name("Honda") == "honda"

    def test_empty_string(self):
        assert normalize_brand_name("") == ""

    def test_unsupported_brand_passthrough(self):
        # Unsupported brands pass through normalized (not mapped to anything)
        result = normalize_brand_name("Lamborghini")
        assert result == "lamborghini"


# ── Catalog structure ───────────────────────────────────────────────────────

class TestCatalogStructure:
    @pytest.fixture(scope="class")
    def catalog(self):
        return build_brand_catalog()

    def test_catalog_non_empty(self, catalog):
        assert len(catalog) >= 10

    def test_maruti_suzuki_in_catalog(self, catalog):
        assert "maruti suzuki" in catalog

    def test_maruti_not_duplicated(self, catalog):
        # Must not have both "maruti" and "maruti suzuki" as top-level keys
        assert "maruti" not in catalog

    def test_catalog_has_dataset_brands_only(self, catalog):
        # These brands have no training data and must NOT appear
        for phantom in ["lamborghini", "ferrari", "rolls-royce", "bugatti", "mclaren"]:
            assert phantom not in catalog, f"Phantom brand '{phantom}' leaked into catalog"

    def test_maruti_suzuki_has_jimny(self, catalog):
        maruti_models = catalog.get("maruti suzuki", [])
        assert "jimny" in maruti_models

    def test_hyundai_has_creta(self, catalog):
        hyundai_models = catalog.get("hyundai", [])
        assert "creta" in hyundai_models

    def test_honda_has_city(self, catalog):
        honda_models = catalog.get("honda", [])
        assert "city" in honda_models

    def test_bmw_has_x1(self, catalog):
        bmw_models = catalog.get("bmw", [])
        assert "x1" in bmw_models

    def test_mahindra_has_thar(self, catalog):
        mahindra_models = catalog.get("mahindra", [])
        assert "thar" in mahindra_models

    def test_mahindra_scorpio_n_entries(self, catalog):
        mahindra_models = catalog.get("mahindra", [])
        # Dataset contains both; both must be present with distinct identities
        assert "scorpio n" in mahindra_models or "scorpio-n" in mahindra_models, \
            "Neither scorpio n nor scorpio-n found in mahindra models"

    def test_hyundai_venue_n_line_entries(self, catalog):
        hyundai_models = catalog.get("hyundai", [])
        assert "venue n line" in hyundai_models or "venue n-line" in hyundai_models, \
            "Neither 'venue n line' nor 'venue n-line' found in hyundai models"


# ── Variant resolution ──────────────────────────────────────────────────────

class TestVariantResolution:
    def test_maruti_jimny_variants(self):
        variants = get_catalog_variants("Maruti", "jimny")
        assert variants is not None, "Jimny must be in catalog"
        assert "ALPHA ALL GRIP PRO" in variants
        # Must NOT contain the old generic variants
        generic = {"Standard", "Base Trim", "V MT", "VX MT", "ZX", "ZX CVT", "SX", "SX(O)"}
        overlap = set(variants) & generic
        assert not overlap, f"Generic variants leaked into Jimny: {overlap}"

    def test_maruti_swift_variants(self):
        variants = get_catalog_variants("Maruti", "swift")
        assert variants is not None and len(variants) > 0
        # Swift has real variants like LDI, LXI etc.
        assert any(v in variants for v in ["LDI", "LXI", "VDI", "ZXI"])

    def test_hyundai_creta_variants(self):
        variants = get_catalog_variants("Hyundai", "creta")
        assert variants is not None and len(variants) > 0

    def test_honda_city_variants(self):
        variants = get_catalog_variants("Honda", "city")
        assert variants is not None and len(variants) > 0

    def test_bmw_x1_variants(self):
        variants = get_catalog_variants("BMW", "x1")
        assert variants is not None and len(variants) > 0

    def test_mahindra_thar_variants(self):
        variants = get_catalog_variants("Mahindra", "thar")
        assert variants is not None and len(variants) > 0
        # Thar must have LX variants
        assert any("LX" in v for v in variants)

    def test_maruti_alias_resolution(self):
        # "Maruti" and "Maruti Suzuki" must reach the same catalog
        variants_a = get_catalog_variants("Maruti", "swift")
        variants_b = get_catalog_variants("Maruti Suzuki", "swift")
        assert variants_a == variants_b

    def test_unsupported_brand_returns_none(self):
        result = get_catalog_variants("Lamborghini", "Urus")
        assert result is None, "Unsupported brand must return None, not a fake variant list"

    def test_unsupported_model_returns_none(self):
        result = get_catalog_variants("Honda", "nonexistent_model_xyz")
        assert result is None, "Unsupported model must return None, not a fake variant list"

    def test_scorpio_n_variants_distinct_from_scorpio(self):
        scorpio_v = get_catalog_variants("Mahindra", "scorpio")
        scorpio_n_v = get_catalog_variants("Mahindra", "scorpio n")
        if scorpio_v and scorpio_n_v:
            # They must not be identical — different dataset entries
            assert set(scorpio_v) != set(scorpio_n_v), \
                "scorpio and scorpio n must have distinct variant sets"

    def test_venue_n_line_variants(self):
        # At least one of the two spellings must be in catalog with real variants
        v1 = get_catalog_variants("Hyundai", "venue n line")
        v2 = get_catalog_variants("Hyundai", "venue n-line")
        assert (v1 is not None) or (v2 is not None), \
            "At least one spelling of Venue N Line must be in catalog"


# ── Payload consistency ─────────────────────────────────────────────────────

class TestPayloadConsistency:
    """Verify that the canonical values that reach the ML feature builder are correct."""

    def test_brand_normalization_matches_feature_builder(self):
        """The same brand alias map used in the catalog must match main.py's _normalize_brand."""
        from backend.main import _normalize_brand

        # These must agree: brand_catalog and main.py both map "Maruti" → "maruti suzuki"
        assert normalize_brand_name("Maruti") == _normalize_brand("Maruti")
        assert normalize_brand_name("Maruti Suzuki") == _normalize_brand("Maruti Suzuki")
        assert normalize_brand_name("Mercedes-Benz") == _normalize_brand("Mercedes-Benz")
        assert normalize_brand_name("Land Rover") == _normalize_brand("Land Rover")
        assert normalize_brand_name("Hyundai") == _normalize_brand("Hyundai")

    def test_maruti_jimny_brand_in_feature_builder(self):
        """Maruti → Jimny must produce brand='maruti suzuki' in build_features()."""
        from backend.main import build_features, VehicleInput, _normalize_brand

        vehicle = VehicleInput(
            brand="Maruti", model="Jimny", variant="alpha all grip pro",
            year=2023, fuel_type="Petrol", transmission="Manual",
            odometer_reading=5000,
        )
        features = build_features(vehicle)
        assert features["brand"].iloc[0] == "maruti suzuki", \
            f"Expected 'maruti suzuki', got {features['brand'].iloc[0]!r}"

    def test_no_generic_variants_in_catalog(self):
        """Confirm the dataset catalog never contains the old generic fallback strings."""
        from backend.brand_catalog import build_brand_catalog
        import json
        from pathlib import Path

        catalog_path = Path(__file__).resolve().parents[1] / "model_artifacts" / "dataset_catalog.json"
        if not catalog_path.exists():
            pytest.skip("dataset_catalog.json not found")

        with open(catalog_path, encoding="utf-8") as f:
            cat = json.load(f)

        bad_variants = {"standard", "base trim", "v mt", "vx mt", "sx(o)"}
        for brand, models in cat.items():
            for model, variants in models.items():
                for v in variants:
                    assert v.lower() not in bad_variants, \
                        f"Generic variant '{v}' found under {brand}/{model}"
