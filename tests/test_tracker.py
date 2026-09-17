"""
Test suite for ActiveRegionTracker (SolarVision multi-observation tracking).
"""

from datetime import datetime, timedelta
import pytest
import numpy as np
import pandas as pd

from src.config import SolarPhysicsConfig, TrackingConfig
from src.tracker import ActiveRegionTracker, TrackedObservation, TrackHistory, TRACKING_DISCLAIMER
from src.feature_extractor import CalibratedActiveRegion


def make_dummy_region(
    reg_id: int = 1,
    lat: float = -15.0,
    lon_cmd: float = -20.0,
    area_uhem: float = 250.0,
) -> CalibratedActiveRegion:
    """Helper to generate a lightweight CalibratedActiveRegion for tracking tests."""
    return CalibratedActiveRegion(
        id=reg_id,
        bbox=(100, 100, 50, 40),
        centroid_pixel=(200.0, 200.0),
        heliographic_lat=lat,
        heliographic_lon_cmd=lon_cmd,
        heliocentric_angle_deg=abs(lon_cmd),
        cos_theta=float(np.cos(np.radians(lon_cmd))),
        projected_area_px=200,
        corrected_area_px=220.0,
        area_uhem=area_uhem,
        umbra_area_uhem=area_uhem * 0.2,
        penumbra_area_uhem=area_uhem * 0.8,
        umbra_penumbra_ratio=0.25,
        longitudinal_extent_deg=5.0,
        latitudinal_extent_deg=4.0,
        spot_count=3,
        mean_contrast=0.45,
        has_penumbra=True,
        is_bipolar=True,
    )


# ---------------------------------------------------------------------------
# Differential Rotation Tests
# ---------------------------------------------------------------------------

def test_differential_rotation_equator_vs_poles():
    """Verify Snodgrass (1984) differential rotation rates across latitudes."""
    tracker = ActiveRegionTracker()
    
    # Equator (lat = 0)
    omega_0 = tracker.compute_differential_rotation(0.0)
    assert omega_0 == pytest.approx(14.713, abs=0.001)
    
    # Mid-latitude (lat = 30)
    omega_30 = tracker.compute_differential_rotation(30.0)
    # 14.713 - 2.396*(0.5)^2 - 1.787*(0.5)^4 = 14.713 - 0.599 - 0.1116875 = 14.002
    assert omega_30 < omega_0
    assert 13.9 < omega_30 < 14.1
    
    # High-latitude (lat = 60)
    omega_60 = tracker.compute_differential_rotation(60.0)
    assert omega_60 < omega_30
    assert 11.5 < omega_60 < 13.0
    
    # Poles (lat = 90)
    omega_90 = tracker.compute_differential_rotation(90.0)
    assert omega_90 == pytest.approx(14.713 - 2.396 - 1.787, abs=0.01)


def test_differential_rotation_hemispheric_symmetry():
    """Verify solar rotation is symmetric across North and South hemispheres."""
    tracker = ActiveRegionTracker()
    omega_north = tracker.compute_differential_rotation(25.0)
    omega_south = tracker.compute_differential_rotation(-25.0)
    assert omega_north == pytest.approx(omega_south, abs=1e-5)


# ---------------------------------------------------------------------------
# Initial Observation Tests
# ---------------------------------------------------------------------------

def test_first_observation_initialization():
    """Initial observation frame should register new tracks and active history."""
    tracker = ActiveRegionTracker()
    t0 = datetime(2024, 5, 10, 12, 0, 0)
    
    r1 = make_dummy_region(reg_id=1, lat=-16.0, lon_cmd=10.0, area_uhem=300.0)
    r2 = make_dummy_region(reg_id=2, lat=22.0, lon_cmd=-70.0, area_uhem=80.0)  # Near East Limb
    
    results = tracker.track_observation([r1, r2], t0)
    
    assert len(results) == 2
    assert results[0].tracking_id == "TRK-001"
    assert results[0].is_new is True
    assert results[0].residual_deg == 0.0
    assert results[0].event_note == "Emergence (Disk Face)"
    
    assert results[1].tracking_id == "TRK-002"
    assert results[1].is_new is True
    assert results[1].event_note == "Emergence (East Limb)"
    
    assert "TRK-001" in tracker.tracks
    assert tracker.tracks["TRK-001"].status == "Active"
    assert tracker.tracks["TRK-001"].observation_count == 1
    assert tracker.tracks["TRK-001"].duration_days == 0.0


# ---------------------------------------------------------------------------
# Kinematic Matching Across Multiple Days
# ---------------------------------------------------------------------------

def test_kinematic_matching_single_region_one_day():
    """Track a single region over 24 hours accounting for rotation."""
    tracker = ActiveRegionTracker()
    t0 = datetime(2024, 5, 10, 12, 0, 0)
    t1 = datetime(2024, 5, 11, 12, 0, 0)  # Exactly 1.0 day
    
    # Latitude -16.0: omega ~ 14.713 - 2.396*sin^2(-16) ~ 14.53 deg/day
    r0 = make_dummy_region(reg_id=1, lat=-16.0, lon_cmd=-10.0, area_uhem=500.0)
    results0 = tracker.track_observation([r0], t0)
    assert results0[0].tracking_id == "TRK-001"
    
    # Next day: actual region is at approx -10.0 + 14.5 = +4.5 deg
    omega = tracker.compute_differential_rotation(-16.0)
    expected_lon = -10.0 + omega * 1.0
    
    r1 = make_dummy_region(reg_id=1, lat=-16.1, lon_cmd=expected_lon + 0.3, area_uhem=550.0)
    results1 = tracker.track_observation([r1], t1)
    
    assert len(results1) == 1
    assert results1[0].tracking_id == "TRK-001"
    assert results1[0].is_new is False
    assert results1[0].residual_deg < 1.0  # Well within kinematic matching threshold
    assert results1[0].area_change_rate_uhem_per_day == pytest.approx(50.0, abs=1.0)
    
    history = tracker.get_track("TRK-001")
    assert history is not None
    assert history.observation_count == 2
    assert history.duration_days == pytest.approx(1.0, abs=0.01)
    assert history.initial_area_uhem == 500.0
    assert history.latest_area_uhem == 550.0
    assert history.growth_rate_uhem_per_day == pytest.approx(50.0, abs=1.0)


def test_three_day_tracking_sequence():
    """Track regions across 3 consecutive daily observations."""
    tracker = ActiveRegionTracker()
    t0 = datetime(2024, 5, 10, 0, 0, 0)
    t1 = datetime(2024, 5, 11, 0, 0, 0)
    t2 = datetime(2024, 5, 12, 0, 0, 0)
    
    # Region 1: Large growing complex at lat -16
    # Region 2: Stable region at lat +25
    r1_day0 = make_dummy_region(reg_id=1, lat=-16.0, lon_cmd=-30.0, area_uhem=1000.0)
    r2_day0 = make_dummy_region(reg_id=2, lat=25.0, lon_cmd=10.0, area_uhem=150.0)
    tracker.track_observation([r1_day0, r2_day0], t0)
    
    omega1 = tracker.compute_differential_rotation(-16.0)
    omega2 = tracker.compute_differential_rotation(25.0)
    
    # Day 1
    r1_day1 = make_dummy_region(reg_id=1, lat=-16.0, lon_cmd=-30.0 + omega1, area_uhem=1200.0)
    r2_day1 = make_dummy_region(reg_id=2, lat=25.1, lon_cmd=10.0 + omega2, area_uhem=140.0)
    tracker.track_observation([r1_day1, r2_day1], t1)
    
    # Day 2
    r1_day2 = make_dummy_region(reg_id=1, lat=-16.0, lon_cmd=-30.0 + 2 * omega1, area_uhem=1500.0)
    r2_day2 = make_dummy_region(reg_id=2, lat=24.9, lon_cmd=10.0 + 2 * omega2, area_uhem=130.0)
    tracker.track_observation([r1_day2, r2_day2], t2)
    
    h1 = tracker.get_track("TRK-001")
    h2 = tracker.get_track("TRK-002")
    
    assert h1.observation_count == 3
    assert h1.duration_days == pytest.approx(2.0, abs=0.01)
    assert h1.max_area_uhem == 1500.0
    assert h1.growth_rate_uhem_per_day == pytest.approx(250.0, abs=5.0)
    
    assert h2.observation_count == 3
    assert h2.duration_days == pytest.approx(2.0, abs=0.01)
    assert h2.growth_rate_uhem_per_day == pytest.approx(-10.0, abs=5.0)


# ---------------------------------------------------------------------------
# Gating and Rejection Tests
# ---------------------------------------------------------------------------

def test_latitude_gating_rejects_mismatch():
    """Regions with different latitudes (> max_lat_diff_deg) must not match."""
    config = TrackingConfig(max_lat_diff_deg=4.0)
    tracker = ActiveRegionTracker(tracking_config=config)
    t0 = datetime(2024, 5, 10, 12, 0)
    t1 = datetime(2024, 5, 11, 12, 0)
    
    r0 = make_dummy_region(lat=10.0, lon_cmd=0.0, area_uhem=200.0)
    tracker.track_observation([r0], t0)
    
    # Next day: region appears at expected longitude (+14.5 deg), but latitude differs by 10 deg
    omega = tracker.compute_differential_rotation(10.0)
    r_wrong_lat = make_dummy_region(lat=20.0, lon_cmd=omega, area_uhem=200.0)
    results1 = tracker.track_observation([r_wrong_lat], t1)
    
    # Should create a new track (TRK-002) and NOT match TRK-001
    assert results1[0].tracking_id == "TRK-002"
    assert results1[0].is_new is True


def test_area_ratio_gating_rejects_drastic_change():
    """Sudden unnatural area changes (> max_area_ratio) must not match."""
    config = TrackingConfig(max_area_ratio=4.0, min_area_ratio=0.25)
    tracker = ActiveRegionTracker(tracking_config=config)
    t0 = datetime(2024, 5, 10, 12, 0)
    t1 = datetime(2024, 5, 11, 12, 0)
    
    r0 = make_dummy_region(lat=-15.0, lon_cmd=0.0, area_uhem=100.0)
    tracker.track_observation([r0], t0)
    
    omega = tracker.compute_differential_rotation(-15.0)
    # Next day: 10x area surge (1000 uHem) -> fails area ratio gating
    r_huge = make_dummy_region(lat=-15.0, lon_cmd=omega, area_uhem=1000.0)
    results1 = tracker.track_observation([r_huge], t1)
    
    assert results1[0].tracking_id == "TRK-002"
    assert results1[0].is_new is True


# ---------------------------------------------------------------------------
# Lifecycle: Limb Rotation & Disappearance Tests
# ---------------------------------------------------------------------------

def test_rotation_over_western_limb():
    """When a region rotates past the western limb (L >= 75 deg), record status properly."""
    config = TrackingConfig(limb_cutoff_deg=75.0)
    tracker = ActiveRegionTracker(tracking_config=config)
    t0 = datetime(2024, 5, 10, 12, 0)
    t1 = datetime(2024, 5, 11, 12, 0)
    
    # Region at L = 68 deg (near western limb)
    r0 = make_dummy_region(lat=0.0, lon_cmd=68.0, area_uhem=300.0)
    tracker.track_observation([r0], t0)
    
    # Next day: predicted position is 68 + 14.7 = 82.7 deg (past limb_cutoff_deg=75.0)
    # In next frame, no regions detected on disk
    tracker.track_observation([], t1)
    
    history = tracker.get_track("TRK-001")
    assert history.status == "Rotated Over Limb (West)"


def test_disappearance_due_to_decay():
    """When a region on the disk face disappears before limb, record as decayed."""
    tracker = ActiveRegionTracker()
    t0 = datetime(2024, 5, 10, 12, 0)
    t1 = datetime(2024, 5, 11, 12, 0)
    
    # Region at center of disk (L = 0 deg)
    r0 = make_dummy_region(lat=10.0, lon_cmd=0.0, area_uhem=100.0)
    tracker.track_observation([r0], t0)
    
    # Next day: disappeared (spot decayed)
    tracker.track_observation([], t1)
    
    history = tracker.get_track("TRK-001")
    assert history.status == "Disappeared (Decayed)"


def test_large_observation_gap_handling():
    """Gaps larger than max_prediction_gap_days mark tracks as stale."""
    config = TrackingConfig(max_prediction_gap_days=3.5)
    tracker = ActiveRegionTracker(tracking_config=config)
    t0 = datetime(2024, 5, 1, 0, 0)
    t1 = datetime(2024, 5, 6, 0, 0)  # 5 days gap (> 3.5 days)
    
    r0 = make_dummy_region(lat=10.0, lon_cmd=-20.0, area_uhem=200.0)
    tracker.track_observation([r0], t0)
    
    r1 = make_dummy_region(lat=10.0, lon_cmd=50.0, area_uhem=200.0)
    results1 = tracker.track_observation([r1], t1)
    
    assert tracker.get_track("TRK-001").status == "Stale (Observation Gap > 3.5d)"
    assert results1[0].tracking_id == "TRK-002"
    assert results1[0].is_new is True


# ---------------------------------------------------------------------------
# Sequence Helper, DataFrames, and Visualizations
# ---------------------------------------------------------------------------

def test_track_sequence_and_dataframes():
    """Verify track_sequence helper, to_dataframe, and get_trajectory_dataframe."""
    tracker = ActiveRegionTracker()
    t0 = datetime(2024, 5, 10, 12, 0)
    t1 = datetime(2024, 5, 11, 12, 0)
    
    omega = tracker.compute_differential_rotation(-16.0)
    seq = [
        (t0, [make_dummy_region(lat=-16.0, lon_cmd=10.0, area_uhem=400.0)]),
        (t1, [make_dummy_region(lat=-16.0, lon_cmd=10.0 + omega, area_uhem=450.0)]),
    ]
    
    tracks = tracker.track_sequence(seq)
    assert len(tracks) == 1
    assert "TRK-001" in tracks
    
    df_summary = tracker.to_dataframe()
    assert isinstance(df_summary, pd.DataFrame)
    assert len(df_summary) == 1
    assert "Track ID" in df_summary.columns
    assert "Growth Rate (μHem/day)" in df_summary.columns
    assert df_summary["Track ID"].iloc[0] == "TRK-001"
    
    df_traj = tracker.get_trajectory_dataframe()
    assert isinstance(df_traj, pd.DataFrame)
    assert len(df_traj) == 2
    assert "Elapsed Days" in df_traj.columns
    assert "Observed Longitude CMD (deg)" in df_traj.columns


def test_plotly_visualizations_render_safely():
    """Verify Plotly charts generate valid figures without errors for empty and active tracks."""
    tracker = ActiveRegionTracker()
    
    # Empty case
    fig_empty_traj = tracker.plot_trajectories_plotly()
    fig_empty_area = tracker.plot_area_evolution_plotly()
    assert fig_empty_traj is not None
    assert fig_empty_area is not None
    
    # Active sequence
    t0 = datetime(2024, 5, 10, 12, 0)
    t1 = datetime(2024, 5, 11, 12, 0)
    omega = tracker.compute_differential_rotation(15.0)
    tracker.track_observation([make_dummy_region(lat=15.0, lon_cmd=-20.0, area_uhem=300.0)], t0)
    tracker.track_observation([make_dummy_region(lat=15.0, lon_cmd=-20.0 + omega, area_uhem=350.0)], t1)
    
    fig_traj = tracker.plot_trajectories_plotly()
    fig_area = tracker.plot_area_evolution_plotly()
    assert fig_traj is not None
    assert fig_area is not None
    assert len(fig_traj.data) > 0
    assert len(fig_area.data) > 0


def test_reset():
    """Verify reset clears internal state."""
    tracker = ActiveRegionTracker()
    t0 = datetime(2024, 5, 10, 12, 0)
    tracker.track_observation([make_dummy_region()], t0)
    assert len(tracker.tracks) == 1
    
    tracker.reset()
    assert len(tracker.tracks) == 0
    assert len(tracker.trajectories) == 0
    assert tracker.next_track_number == 1
    assert tracker.last_observation_time is None
