"""
Test suite for SolarDatabase (SQLite data storage & persistence).
"""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
import pytest

from src.database import SolarDatabase
from src.feature_extractor import CalibratedActiveRegion
from src.classifier import ClassificationResult, MorphologicalClassInfo, DemonstrationRiskAssessment, DemonstrationRiskFactors
from src.solar_data import ImageMetadata
from src.tracker import TrackHistory, TrackedObservation


@pytest.fixture
def temp_db(tmp_path):
    """Provides a fresh, isolated SQLite database for each test."""
    db_file = tmp_path / "test_solarvision.db"
    db = SolarDatabase(str(db_file))
    return db


def make_dummy_metadata(filename: str = "sdo_test_01.jpg", sha256: str = "abc123456789") -> ImageMetadata:
    return ImageMetadata(
        filename=filename,
        filepath=f"/data/{filename}",
        source_url="https://sdo.gsfc.nasa.gov/assets/test.jpg",
        source_type="NASA_SDO_HMI_CONTINUUM",
        observatory="NASA Solar Dynamics Observatory (SDO)",
        instrument="Helioseismic and Magnetic Imager (HMI)",
        wavelength_channel="Fe I 6173 Å (Visible Continuum)",
        download_timestamp_utc="2024-05-10T00:00:00Z",
        source_last_modified="Fri, 10 May 2024 00:00:00 GMT",
        etag='"etag123"',
        sha256=sha256,
        file_size_bytes=524288,
        image_width=1024,
        image_height=1024,
        channels=3,
        is_authentic_real_data=True,
    )


def make_dummy_region(reg_id: int = 1, lat: float = -16.0, lon: float = 30.0, area_uhem: float = 500.0) -> CalibratedActiveRegion:
    return CalibratedActiveRegion(
        id=reg_id,
        bbox=(100, 150, 60, 45),
        centroid_pixel=(130.0, 172.5),
        heliographic_lat=lat,
        heliographic_lon_cmd=lon,
        heliocentric_angle_deg=35.0,
        cos_theta=0.82,
        projected_area_px=250,
        corrected_area_px=305.0,
        area_uhem=area_uhem,
        umbra_area_uhem=area_uhem * 0.25,
        penumbra_area_uhem=area_uhem * 0.75,
        umbra_penumbra_ratio=0.33,
        longitudinal_extent_deg=8.0,
        latitudinal_extent_deg=5.0,
        spot_count=4,
        mean_contrast=0.48,
        has_penumbra=True,
        is_bipolar=True,
    )


def make_dummy_classification(reg_id: int = 1, code: str = "D", score: float = 72.5) -> ClassificationResult:
    info = MorphologicalClassInfo(
        code=code,
        name=f"Class {code}: Moderate Bipolar Group",
        description="Mature compact bipolar group with penumbra on both ends.",
        typical_lifespan="1 to 2 weeks",
        magnetic_topology="Mature compact bipolar flux group",
        mcintosh_equivalent=f"{code}so",
    )
    factors = DemonstrationRiskFactors(
        area_factor=60.0,
        complexity_factor=55.0,
        penumbra_factor=80.0,
        contrast_factor=75.0,
        compactness_factor=65.0,
    )
    risk = DemonstrationRiskAssessment(
        score=score,
        attention_level="High Attention" if score >= 70.0 else "Moderate Attention",
        factors=factors,
        weights={"area": 0.30, "complexity": 0.25, "penumbra": 0.20, "contrast": 0.15, "compactness": 0.10},
        assumptions=["Geometric complexity proxies emerging magnetic flux"],
    )
    return ClassificationResult(
        region_id=reg_id,
        class_code=code,
        class_name=info.name,
        class_info=info,
        attention_level=risk.attention_level,
        flare_potential=risk.attention_level,
        demonstration_risk=risk,
        rule_trace=["Step 1: Mature penumbra detected at both ends", "Step 2: Longitudinal extent < 10 deg"],
        confidence=0.92,
    )


# ---------------------------------------------------------------------------
# Database Initialization & Schema Tests
# ---------------------------------------------------------------------------

def test_database_initialization_tables(temp_db):
    """Verify all 5 required tables and indexes are created properly."""
    with temp_db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row["name"] for row in cursor.fetchall()}
        
        expected_tables = {"image_metadata", "observations", "active_regions", "tracks", "trajectory_points"}
        assert expected_tables.issubset(tables)


# ---------------------------------------------------------------------------
# Image Metadata & Deduplication Tests
# ---------------------------------------------------------------------------

def test_save_and_retrieve_image_metadata(temp_db):
    """Verify storing image metadata sidecars and retrieving by SHA-256."""
    meta = make_dummy_metadata(filename="sdo_test.jpg", sha256="test_hash_12345")
    img_id = temp_db.save_image_metadata(meta)
    assert img_id > 0

    record = temp_db.get_image_by_hash("test_hash_12345")
    assert record is not None
    assert record["id"] == img_id
    assert record["filename"] == "sdo_test.jpg"
    assert record["observatory"] == "NASA Solar Dynamics Observatory (SDO)"
    assert record["image_width"] == 1024


def test_image_metadata_deduplication(temp_db):
    """Re-inserting identical SHA-256 hash must return existing record ID without duplicates."""
    meta1 = make_dummy_metadata(filename="sdo_file_a.jpg", sha256="duplicate_sha256_abcdef")
    id1 = temp_db.save_image_metadata(meta1)

    meta2 = make_dummy_metadata(filename="sdo_file_b.jpg", sha256="duplicate_sha256_abcdef")
    id2 = temp_db.save_image_metadata(meta2)

    assert id1 == id2
    with temp_db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM image_metadata WHERE sha256 = 'duplicate_sha256_abcdef'")
        assert cursor.fetchone()["count"] == 1


# ---------------------------------------------------------------------------
# Observation & Active Region Persistence Tests
# ---------------------------------------------------------------------------

def test_save_observation_with_regions(temp_db):
    """Verify complete observation persistence including calibrated features and risk factors."""
    meta = make_dummy_metadata()
    img_id = temp_db.save_image_metadata(meta)

    t0 = datetime(2024, 5, 10, 12, 0, 0)
    reg1 = make_dummy_region(reg_id=1, lat=-16.0, lon=32.0, area_uhem=1500.0)
    reg2 = make_dummy_region(reg_id=2, lat=22.0, lon=-10.0, area_uhem=180.0)
    clf1 = make_dummy_classification(reg_id=1, code="F", score=88.5)
    clf2 = make_dummy_classification(reg_id=2, code="H", score=24.0)

    obs_id = temp_db.save_observation(
        filename="sdo_test_01.jpg",
        timestamp=t0,
        center_x=512.0,
        center_y=512.0,
        radius=480.0,
        quiet_sun_intensity=192.5,
        regions=[reg1, reg2],
        classifications=[clf1, clf2],
        tracking_ids=["TRK-001", "TRK-002"],
        image_id=img_id,
        disk_confidence=0.98,
        is_spotless=False,
    )
    assert obs_id > 0

    # Retrieve observation
    obs = temp_db.get_observation_by_id(obs_id)
    assert obs is not None
    assert obs["disk_radius"] == 480.0
    assert obs["active_region_count"] == 2
    assert obs["is_spotless"] == 0

    # Retrieve regions
    regions = temp_db.get_regions_for_observation(obs_id)
    assert len(regions) == 2
    
    # Check AR 1 details
    r1 = regions[0]
    assert r1["tracking_id"] == "TRK-001"
    assert r1["area_uhem"] == pytest.approx(1500.0)
    assert r1["mcintosh_class"] == "F"
    assert r1["demonstration_risk_score"] == pytest.approx(88.5)
    assert r1["attention_level"] == "High Attention"
    assert "Mature penumbra detected" in json.loads(r1["rule_trace_json"])[0]


def test_spotless_solar_disk_observation(temp_db):
    """Verify solar minimum (spotless disk) observations persist cleanly."""
    t0 = datetime(2024, 1, 1, 0, 0, 0)
    obs_id = temp_db.save_observation(
        filename="sdo_spotless.jpg",
        timestamp=t0,
        center_x=512.0,
        center_y=512.0,
        radius=478.0,
        quiet_sun_intensity=190.0,
        regions=[],
        classifications=[],
        is_spotless=True,
    )
    obs = temp_db.get_observation_by_id(obs_id)
    assert obs["is_spotless"] == 1
    assert obs["active_region_count"] == 0
    assert len(temp_db.get_regions_for_observation(obs_id)) == 0


# ---------------------------------------------------------------------------
# Multi-Day Tracking & Trajectory Persistence Tests
# ---------------------------------------------------------------------------

def test_save_and_retrieve_track_history(temp_db):
    """Verify persisting TrackHistory objects and querying trajectories."""
    t0 = datetime(2024, 5, 10, 0, 0)
    t1 = datetime(2024, 5, 11, 0, 0)

    obs0 = TrackedObservation(
        tracking_id="TRK-001",
        observation_time=t0,
        region=None,
        predicted_lon_cmd=32.6,
        actual_lon_cmd=32.6,
        residual_deg=0.0,
        is_new=True,
        predicted_lat=-16.8,
        actual_lat=-16.8,
        area_uhem=1750.0,
        area_change_rate_uhem_per_day=0.0,
        event_note="Emergence (Disk Face)",
    )
    obs1 = TrackedObservation(
        tracking_id="TRK-001",
        observation_time=t1,
        region=None,
        predicted_lon_cmd=45.9,
        actual_lon_cmd=45.6,
        residual_deg=0.34,
        is_new=False,
        predicted_lat=-16.8,
        actual_lat=-16.8,
        area_uhem=2080.0,
        area_change_rate_uhem_per_day=330.0,
        event_note="Tracked Continuity",
    )

    history = TrackHistory(
        track_id="TRK-001",
        birth_time=t0,
        last_seen_time=t1,
        status="Active",
        observations=[obs0, obs1],
    )

    temp_db.save_track_history(history)

    # Query track history
    track_record = temp_db.get_track_history("TRK-001")
    assert track_record is not None
    assert track_record["track_id"] == "TRK-001"
    assert track_record["lifecycle_status"] == "Active"
    assert track_record["observation_count"] == 2
    assert track_record["growth_rate_uhem_per_day"] == pytest.approx(330.0, abs=5.0)
    assert len(track_record["trajectories"]) == 2
    assert track_record["trajectories"][1]["residual_deg"] == pytest.approx(0.34)


# ---------------------------------------------------------------------------
# Catalog & Cascade Deletion Tests
# ---------------------------------------------------------------------------

def test_full_catalog_query(temp_db):
    """Verify rich analytical catalog join across tables."""
    meta = make_dummy_metadata()
    img_id = temp_db.save_image_metadata(meta)

    reg = make_dummy_region(reg_id=1, area_uhem=650.0)
    clf = make_dummy_classification(reg_id=1, code="C", score=55.0)

    obs_id = temp_db.save_observation(
        filename="sdo_catalog_test.jpg",
        timestamp=datetime(2024, 5, 10, 0, 0),
        center_x=512.0,
        center_y=512.0,
        radius=480.0,
        quiet_sun_intensity=190.0,
        regions=[reg],
        classifications=[clf],
        tracking_ids=["TRK-001"],
        image_id=img_id,
    )

    catalog = temp_db.get_full_catalog()
    assert len(catalog) == 1
    item = catalog[0]
    assert item["tracking_id"] == "TRK-001"
    assert item["mcintosh_class"] == "C"
    assert item["area_uhem"] == pytest.approx(650.0)
    assert item["observatory"] == "NASA Solar Dynamics Observatory (SDO)"


def test_cascade_delete_observation(temp_db):
    """Verify deleting an observation cascades to its active regions."""
    reg = make_dummy_region()
    clf = make_dummy_classification()
    obs_id = temp_db.save_observation(
        filename="delete_test.jpg",
        timestamp=datetime.now(),
        center_x=500.0,
        center_y=500.0,
        radius=450.0,
        quiet_sun_intensity=180.0,
        regions=[reg],
        classifications=[clf],
    )
    assert len(temp_db.get_regions_for_observation(obs_id)) == 1

    deleted = temp_db.delete_observation(obs_id)
    assert deleted is True
    assert temp_db.get_observation_by_id(obs_id) is None
    assert len(temp_db.get_regions_for_observation(obs_id)) == 0


def test_clear_database(temp_db):
    """Verify clear_database purges all tables cleanly."""
    temp_db.save_image_metadata(make_dummy_metadata())
    temp_db.save_observation(
        filename="clear_test.jpg",
        timestamp=datetime.now(),
        center_x=500.0,
        center_y=500.0,
        radius=450.0,
        quiet_sun_intensity=180.0,
        regions=[make_dummy_region()],
        classifications=[make_dummy_classification()],
    )
    temp_db.clear_database()
    assert len(temp_db.get_all_observations()) == 0
    assert len(temp_db.get_all_tracks()) == 0
