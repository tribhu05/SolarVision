"""
Test suite for SolarVision unified processing pipeline.
Tests end-to-end integration across ingestion, preprocessing, detection,
classification, tracking, deduplication, and SQLite persistence.
"""

from datetime import datetime
from pathlib import Path
import pytest
import cv2
import numpy as np
import pandas as pd

from src.config import SolarVisionConfig, load_config
from src.database import SolarDatabase
from src.pipeline import SolarVisionPipeline, PipelineResult, SequencePipelineResult


@pytest.fixture
def test_pipeline(tmp_path):
    """Provides a fully initialized SolarVisionPipeline with an isolated SQLite DB."""
    cfg = load_config()
    db_file = tmp_path / "test_pipeline.db"
    db = SolarDatabase(str(db_file))
    pipe = SolarVisionPipeline(config=cfg, db=db)
    return pipe


@pytest.fixture
def sample_sdo_image_path():
    """Path to authentic NASA SDO continuum observation."""
    p = Path("data/sample_images/sdo_hmi_ar3664_20240510.jpg")
    if not p.exists():
        pytest.skip(f"Real SDO sample image not found at {p}")
    return p


@pytest.fixture
def sequence_sdo_image_paths():
    """Paths to authentic 3-day NASA SDO continuum sequence (AR3664)."""
    paths = [
        Path("data/sample_images/sdo_hmi_ar3664_20240510.jpg"),
        Path("data/sample_images/sdo_hmi_ar3664_20240511.jpg"),
        Path("data/sample_images/sdo_hmi_ar3664_20240512.jpg"),
    ]
    for p in paths:
        if not p.exists():
            pytest.skip(f"Real SDO sequence image not found at {p}")
    return paths


# ---------------------------------------------------------------------------
# Single Image Processing Tests
# ---------------------------------------------------------------------------

def test_pipeline_single_real_image(test_pipeline, sample_sdo_image_path):
    """Verify complete end-to-end execution on authentic NASA SDO observation."""
    res = test_pipeline.process_image(sample_sdo_image_path)

    assert res.success is True
    assert res.is_cached is False
    assert res.observation_id > 0
    assert res.image_id is not None
    assert res.solar_disk.is_valid is True
    assert res.solar_disk.radius > 400.0
    assert res.limb_result is not None
    assert res.quiet_sun_intensity > 150.0

    # Detections & Classifications
    assert len(res.regions) > 0
    assert len(res.classifications) == len(res.regions)
    assert len(res.tracked_observations) == len(res.regions)

    # Check top active region (AR3664) features
    top_ar = res.regions[0]
    assert top_ar.area_uhem > 500.0
    assert -25.0 < top_ar.heliographic_lat < -10.0

    # Database Persistence Verification
    db_obs = test_pipeline.db.get_observation_by_id(res.observation_id)
    assert db_obs is not None
    assert db_obs["active_region_count"] == len(res.regions)

    db_regs = test_pipeline.db.get_regions_for_observation(res.observation_id)
    assert len(db_regs) == len(res.regions)
    assert db_regs[0]["mcintosh_class"] in {"D", "E", "F"}
    assert db_regs[0]["demonstration_risk_score"] > 50.0


# ---------------------------------------------------------------------------
# Rerun Idempotency & Deduplication Tests
# ---------------------------------------------------------------------------

def test_pipeline_rerun_idempotency_and_deduplication(test_pipeline, sample_sdo_image_path):
    """
    Rerunning the pipeline on the same observation must identify existing SHA-256 hash,
    return cached results, and avoid creating duplicate records.
    """
    # First execution: computes from scratch
    res1 = test_pipeline.process_image(sample_sdo_image_path, reprocess=False)
    assert res1.success is True
    assert res1.is_cached is False
    obs_id1 = res1.observation_id
    img_id1 = res1.image_id

    # Second execution (default reprocess=False): returns cached observation
    res2 = test_pipeline.process_image(sample_sdo_image_path, reprocess=False)
    assert res2.success is True
    assert res2.is_cached is True
    assert res2.observation_id == obs_id1
    assert res2.image_id == img_id1

    # Verify no duplicate observations or image_metadata were created
    all_obs = test_pipeline.db.get_all_observations()
    assert len(all_obs) == 1

    with test_pipeline.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM image_metadata")
        assert cursor.fetchone()["count"] == 1

    # Third execution with explicit reprocess=True: recalculates and adds new observation session
    res3 = test_pipeline.process_image(sample_sdo_image_path, reprocess=True)
    assert res3.success is True
    assert res3.is_cached is False
    assert res3.observation_id > obs_id1
    assert res3.image_id == img_id1  # Image metadata deduplicated by SHA-256


# ---------------------------------------------------------------------------
# Multi-Day Historical Sequence Processing Tests
# ---------------------------------------------------------------------------

def test_pipeline_multi_day_historical_sequence(test_pipeline, sequence_sdo_image_paths):
    """
    Process 3-day sequential observations of AR3664 (May 10–12, 2024),
    verifying multi-frame kinematic tracking and lifecycle persistence in SQLite.
    """
    seq_res = test_pipeline.process_sequence(sequence_sdo_image_paths, reprocess=True)

    assert seq_res.success is True
    assert seq_res.observation_count == 3
    assert seq_res.total_active_regions_detected > 5
    assert len(seq_res.tracks) > 0

    # Verify persistent track for AR3664 (TRK-001)
    trk1 = seq_res.tracks.get("TRK-001")
    assert trk1 is not None
    assert trk1.observation_count == 3
    assert trk1.status == "Active"
    assert trk1.max_area_uhem > 1500.0
    assert trk1.net_longitude_drift_deg > 20.0  # ~13 deg/day * 2 days = ~26 deg

    # DataFrames generated
    assert isinstance(seq_res.summary_dataframe, pd.DataFrame)
    assert not seq_res.summary_dataframe.empty
    assert "Growth Rate (μHem/day)" in seq_res.summary_dataframe.columns

    assert isinstance(seq_res.trajectory_dataframe, pd.DataFrame)
    assert not seq_res.trajectory_dataframe.empty
    assert "Observed Longitude CMD (deg)" in seq_res.trajectory_dataframe.columns

    # SQLite Database Trajectory Verification
    db_trk1 = test_pipeline.db.get_track_history("TRK-001")
    assert db_trk1 is not None
    assert db_trk1["observation_count"] == 3
    assert len(db_trk1["trajectories"]) == 3


# ---------------------------------------------------------------------------
# In-Memory Array & Error Handling Tests
# ---------------------------------------------------------------------------

def test_pipeline_in_memory_numpy_array(test_pipeline, sample_sdo_image_path):
    """Verify processing directly from an in-memory BGR numpy array."""
    img_bgr = cv2.imread(str(sample_sdo_image_path))
    assert img_bgr is not None

    res = test_pipeline.process_image(img_bgr, observation_time=datetime(2024, 5, 10, 12, 0))
    assert res.success is True
    assert res.observation_id > 0
    assert res.image_id is not None
    assert res.image_metadata.sha256 != ""


def test_pipeline_error_handling_nonexistent_file(test_pipeline):
    """Verify graceful error reporting when image file does not exist."""
    res = test_pipeline.process_image("nonexistent_sdo_file.jpg")
    assert res.success is False
    assert res.observation_id == -1
    assert "does not exist" in res.error_message


def test_pipeline_empty_sequence(test_pipeline):
    """Verify empty sequence handling."""
    seq_res = test_pipeline.process_sequence([])
    assert seq_res.success is False
    assert seq_res.observation_count == 0
