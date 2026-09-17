"""
SolarVision Multi-Day Solar Active Region Tracking Module.
Tracks solar active regions across multi-temporal observations by compensating
for solar differential rotation kinematics (Snodgrass 1984), maintaining persistent
track history, tracking area evolution and emergence/decay, and handling missing frames.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import SolarPhysicsConfig, TrackingConfig
from src.feature_extractor import CalibratedActiveRegion


TRACKING_DISCLAIMER = (
    "TRACKING DISCLAIMER: This kinematic tracker performs simplified multi-observation "
    "association using the Snodgrass (1984) photospheric differential rotation model and "
    "nearest-neighbor feature gating. Track IDs are heuristic kinematic associations and "
    "MUST NOT be treated as scientifically confirmed NOAA Active Region numbers. True active "
    "region identification requires multi-instrument magnetic polarity verification and manual "
    "curation by NOAA Space Weather Prediction Center (SWPC)."
)


@dataclass
class TrackedObservation:
    """An active region associated with a persistent tracking ID and trajectory point."""
    tracking_id: str
    observation_time: datetime
    region: Any                                      # CalibratedActiveRegion or DetectedRegion
    predicted_lon_cmd: float
    actual_lon_cmd: float
    residual_deg: float
    is_new: bool
    predicted_lat: float = 0.0
    actual_lat: float = 0.0
    area_uhem: float = 0.0
    area_change_rate_uhem_per_day: float = 0.0
    event_note: str = "Tracked Continuity"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracking_id": self.tracking_id,
            "observation_time": self.observation_time.isoformat(),
            "predicted_lon_cmd": round(float(self.predicted_lon_cmd), 2),
            "actual_lon_cmd": round(float(self.actual_lon_cmd), 2),
            "predicted_lat": round(float(self.predicted_lat), 2),
            "actual_lat": round(float(self.actual_lat), 2),
            "residual_deg": round(float(self.residual_deg), 2),
            "area_uhem": round(float(self.area_uhem), 1),
            "area_change_rate_uhem_per_day": round(float(self.area_change_rate_uhem_per_day), 2),
            "is_new": bool(self.is_new),
            "event_note": self.event_note,
        }


@dataclass
class TrackHistory:
    """Persistent chronological lifecycle and trajectory record of a tracked active region."""
    track_id: str
    birth_time: datetime
    last_seen_time: datetime
    status: str                                      # 'Active', 'Rotated Over Limb (West)', 'Disappeared (Decayed)', etc.
    observations: List[TrackedObservation] = field(default_factory=list)

    @property
    def duration_days(self) -> float:
        """Total observed tracking duration in days."""
        if len(self.observations) <= 1:
            return 0.0
        return max(0.0, (self.last_seen_time - self.birth_time).total_seconds() / 86400.0)

    @property
    def observation_count(self) -> int:
        """Number of observation frames this region was detected in."""
        return len(self.observations)

    @property
    def initial_area_uhem(self) -> float:
        """First observed area in millionths of solar hemisphere."""
        return self.observations[0].area_uhem if self.observations else 0.0

    @property
    def latest_area_uhem(self) -> float:
        """Most recent observed area in millionths of solar hemisphere."""
        return self.observations[-1].area_uhem if self.observations else 0.0

    @property
    def max_area_uhem(self) -> float:
        """Peak observed physical area."""
        if not self.observations:
            return 0.0
        return max(obs.area_uhem for obs in self.observations)

    @property
    def growth_rate_uhem_per_day(self) -> float:
        """Overall area growth rate (uHem/day) between initial and latest observations."""
        dur = self.duration_days
        if dur <= 0.01:
            return 0.0
        return (self.latest_area_uhem - self.initial_area_uhem) / dur

    @property
    def mean_latitude(self) -> float:
        """Mean Stonyhurst latitude across all observation frames."""
        if not self.observations:
            return 0.0
        return float(np.mean([obs.actual_lat for obs in self.observations]))

    @property
    def net_longitude_drift_deg(self) -> float:
        """Observed net Stonyhurst longitude migration across the tracking period."""
        if len(self.observations) <= 1:
            return 0.0
        return float(self.observations[-1].actual_lon_cmd - self.observations[0].actual_lon_cmd)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "status": self.status,
            "birth_time": self.birth_time.isoformat(),
            "last_seen_time": self.last_seen_time.isoformat(),
            "duration_days": round(self.duration_days, 2),
            "observation_count": self.observation_count,
            "initial_area_uhem": round(self.initial_area_uhem, 1),
            "latest_area_uhem": round(self.latest_area_uhem, 1),
            "max_area_uhem": round(self.max_area_uhem, 1),
            "growth_rate_uhem_per_day": round(self.growth_rate_uhem_per_day, 2),
            "mean_latitude": round(self.mean_latitude, 2),
            "net_longitude_drift_deg": round(self.net_longitude_drift_deg, 2),
            "observations": [obs.to_dict() for obs in self.observations],
        }


class ActiveRegionTracker:
    """
    Kinematic multi-observation solar active region tracker.
    Utilizes the Snodgrass (1984) differential rotation relation to project
    expected photospheric positions, performing gated nearest-neighbor matching
    incorporating angular position, latitude conservation, and area consistency.
    """

    def __init__(
        self,
        physics_config: Optional[SolarPhysicsConfig] = None,
        tracking_config: Optional[TrackingConfig] = None,
    ):
        self.physics_config = physics_config or SolarPhysicsConfig()
        self.config = tracking_config or TrackingConfig()
        self.next_track_number: int = 1
        
        # Mapping: tracking_id -> List[TrackedObservation] (backwards compatibility)
        self.trajectories: Dict[str, List[TrackedObservation]] = {}
        # Mapping: tracking_id -> TrackHistory
        self.tracks: Dict[str, TrackHistory] = {}
        
        self.last_observation_time: Optional[datetime] = None
        # List of active tracks from immediately preceding frame: [(track_id, region, lon_cmd, lat, area_uhem)]
        self.active_tracks: List[Tuple[str, Any, float, float, float]] = []

    def compute_differential_rotation(self, latitude_deg: float) -> float:
        """
        Compute solar photospheric differential rotation rate in degrees per day (Snodgrass 1984).
        omega(B) = A + B * sin^2(B) + C * sin^4(B)  [deg / day]
        """
        b_rad = np.radians(latitude_deg)
        sin_b = np.sin(b_rad)
        sin2_b = sin_b ** 2
        sin4_b = sin2_b ** 2

        omega = (
            self.physics_config.diff_rot_a
            + self.physics_config.diff_rot_b * sin2_b
            + self.physics_config.diff_rot_c * sin4_b
        )
        return float(omega)

    def track_observation(
        self,
        regions: List[Any],
        observation_time: datetime,
        max_matching_dist_deg: Optional[float] = None,
    ) -> List[TrackedObservation]:
        """
        Associate detections in the current observation with previously tracked active regions.

        Args:
            regions: CalibratedActiveRegion or DetectedRegion objects in this frame
            observation_time: Datetime of this solar observation
            max_matching_dist_deg: Optional runtime override for maximum angular matching distance

        Returns:
            List of TrackedObservation records for all regions in this frame
        """
        max_dist = max_matching_dist_deg or self.config.max_matching_dist_deg
        tracked_results: List[TrackedObservation] = []

        # Extract latitude, longitude, and area for all current regions
        curr_extracted = []
        for r in regions:
            lat = float(getattr(r, "heliographic_lat", 0.0))
            lon = float(getattr(r, "heliographic_lon_cmd", 0.0))
            area = float(getattr(r, "area_uhem", 0.0))
            curr_extracted.append((r, lat, lon, area))

        # Case 1: First observation session or reset state
        if not self.active_tracks or self.last_observation_time is None:
            new_active = []
            for r, lat, lon, area in curr_extracted:
                t_id = f"{self.config.track_id_prefix}-{self.next_track_number:03d}"
                self.next_track_number += 1

                note = "Emergence (East Limb)" if lon <= self.config.east_limb_threshold_deg else "Emergence (Disk Face)"
                obs = TrackedObservation(
                    tracking_id=t_id,
                    observation_time=observation_time,
                    region=r,
                    predicted_lon_cmd=round(lon, 2),
                    actual_lon_cmd=round(lon, 2),
                    predicted_lat=round(lat, 2),
                    actual_lat=round(lat, 2),
                    residual_deg=0.0,
                    is_new=True,
                    area_uhem=round(area, 1),
                    area_change_rate_uhem_per_day=0.0,
                    event_note=note,
                )
                tracked_results.append(obs)
                self.trajectories.setdefault(t_id, []).append(obs)

                history = TrackHistory(
                    track_id=t_id,
                    birth_time=observation_time,
                    last_seen_time=observation_time,
                    status="Active",
                    observations=[obs],
                )
                self.tracks[t_id] = history
                new_active.append((t_id, r, lon, lat, area))

            self.last_observation_time = observation_time
            self.active_tracks = new_active
            return tracked_results

        # Case 2: Subsequent observation
        dt_days = (observation_time - self.last_observation_time).total_seconds() / 86400.0
        if dt_days < 0:
            dt_days = 0.0

        # Check for gap exceeding maximum kinematic prediction threshold
        large_gap = dt_days > self.config.max_prediction_gap_days

        # Predict positions of all previous active tracks
        predicted_previous = []
        for t_id, prev_r, prev_lon, prev_lat, prev_area in self.active_tracks:
            omega = self.compute_differential_rotation(prev_lat)
            pred_lon = prev_lon + omega * dt_days
            pred_lat = prev_lat  # Heliographic latitude is conserved across short timescales
            predicted_previous.append((t_id, pred_lat, pred_lon, prev_area, prev_r))

        matched_prev_indices: Set[int] = set()
        matched_curr_indices: Set[int] = set()
        matches: List[Tuple[int, int, float, float, float, float]] = []

        if not large_gap:
            # Multi-candidate distance matrix
            cost_matrix = []
            for c_idx, (c_r, c_lat, c_lon, c_area) in enumerate(curr_extracted):
                for p_idx, (t_id, p_lat, p_lon, p_area, _) in enumerate(predicted_previous):
                    # 1. Latitude consistency check
                    d_lat = abs(c_lat - p_lat)
                    if d_lat > self.config.max_lat_diff_deg:
                        continue

                    # 2. Angular distance residual
                    d_lon = c_lon - p_lon
                    dist = float(np.hypot(d_lat, d_lon))
                    if dist > max_dist:
                        continue

                    # 3. Area ratio check
                    if p_area > 0 and c_area > 0:
                        ratio = c_area / p_area
                        if not (self.config.min_area_ratio <= ratio <= self.config.max_area_ratio):
                            continue
                        area_penalty = abs(np.log10(ratio)) * 1.5
                    else:
                        area_penalty = 0.0

                    total_cost = dist + area_penalty
                    cost_matrix.append((total_cost, c_idx, p_idx, dist, p_lon, p_lat, p_area))

            # Greedy lowest-cost assignment
            cost_matrix.sort(key=lambda x: x[0])
            for cost, c_idx, p_idx, dist, p_lon, p_lat, p_area in cost_matrix:
                if c_idx not in matched_curr_indices and p_idx not in matched_prev_indices:
                    matched_curr_indices.add(c_idx)
                    matched_prev_indices.add(p_idx)
                    matches.append((c_idx, p_idx, dist, p_lon, p_lat, p_area))

        new_active = []

        # 1. Update matched tracks
        for c_idx, p_idx, dist, pred_lon, pred_lat, prev_area in matches:
            t_id = predicted_previous[p_idx][0]
            c_r, c_lat, c_lon, c_area = curr_extracted[c_idx]

            # Growth rate calculation (uHem / day)
            area_rate = (c_area - prev_area) / max(dt_days, 0.01) if dt_days > 0 else 0.0

            obs = TrackedObservation(
                tracking_id=t_id,
                observation_time=observation_time,
                region=c_r,
                predicted_lon_cmd=round(pred_lon, 2),
                actual_lon_cmd=round(c_lon, 2),
                predicted_lat=round(pred_lat, 2),
                actual_lat=round(c_lat, 2),
                residual_deg=round(dist, 2),
                is_new=False,
                area_uhem=round(c_area, 1),
                area_change_rate_uhem_per_day=round(area_rate, 2),
                event_note="Tracked Continuity",
            )
            tracked_results.append(obs)
            self.trajectories.setdefault(t_id, []).append(obs)

            # Update persistent TrackHistory
            if t_id in self.tracks:
                self.tracks[t_id].observations.append(obs)
                self.tracks[t_id].last_seen_time = observation_time
                self.tracks[t_id].status = "Active"

            new_active.append((t_id, c_r, c_lon, c_lat, c_area))

        # 2. Handle unmatched previous tracks (Appearances, Disappearances, Limb rotation)
        for p_idx, (t_id, p_lat, p_lon, p_area, _) in enumerate(predicted_previous):
            if p_idx not in matched_prev_indices and t_id in self.tracks:
                # Did this active region rotate over the western limb?
                if p_lon >= self.config.limb_cutoff_deg:
                    self.tracks[t_id].status = "Rotated Over Limb (West)"
                elif large_gap:
                    self.tracks[t_id].status = "Stale (Observation Gap > 3.5d)"
                else:
                    self.tracks[t_id].status = "Disappeared (Decayed)"

        # 3. Handle unmatched current regions (New emergences or rotated from east limb)
        for c_idx, (c_r, c_lat, c_lon, c_area) in enumerate(curr_extracted):
            if c_idx not in matched_curr_indices:
                t_id = f"{self.config.track_id_prefix}-{self.next_track_number:03d}"
                self.next_track_number += 1

                note = "Emergence (East Limb)" if c_lon <= self.config.east_limb_threshold_deg else "Emergence (Disk Face)"
                obs = TrackedObservation(
                    tracking_id=t_id,
                    observation_time=observation_time,
                    region=c_r,
                    predicted_lon_cmd=round(c_lon, 2),
                    actual_lon_cmd=round(c_lon, 2),
                    predicted_lat=round(c_lat, 2),
                    actual_lat=round(c_lat, 2),
                    residual_deg=0.0,
                    is_new=True,
                    area_uhem=round(c_area, 1),
                    area_change_rate_uhem_per_day=0.0,
                    event_note=note,
                )
                tracked_results.append(obs)
                self.trajectories.setdefault(t_id, []).append(obs)

                history = TrackHistory(
                    track_id=t_id,
                    birth_time=observation_time,
                    last_seen_time=observation_time,
                    status="Active",
                    observations=[obs],
                )
                self.tracks[t_id] = history
                new_active.append((t_id, c_r, c_lon, c_lat, c_area))

        self.last_observation_time = observation_time
        self.active_tracks = new_active
        return tracked_results

    def track_sequence(
        self,
        sequence: List[Tuple[datetime, List[Any]]],
    ) -> Dict[str, TrackHistory]:
        """Track a sequence of observations chronologically."""
        self.reset()
        for obs_time, regions in sequence:
            self.track_observation(regions, obs_time)
        return self.tracks

    def get_track(self, track_id: str) -> Optional[TrackHistory]:
        """Retrieve the track history object for a given track ID."""
        return self.tracks.get(track_id)

    def to_dataframe(self) -> pd.DataFrame:
        """Export summary metrics of all tracked active regions as a pandas DataFrame."""
        if not self.tracks:
            return pd.DataFrame()

        rows = []
        for t_id, th in self.tracks.items():
            rows.append({
                "Track ID": th.track_id,
                "Lifecycle Status": th.status,
                "First Observed": th.birth_time.strftime("%Y-%m-%d %H:%M"),
                "Last Observed": th.last_seen_time.strftime("%Y-%m-%d %H:%M"),
                "Duration (days)": round(th.duration_days, 2),
                "Frames": th.observation_count,
                "Mean Latitude (deg)": round(th.mean_latitude, 2),
                "Initial Area (μHem)": round(th.initial_area_uhem, 1),
                "Latest Area (μHem)": round(th.latest_area_uhem, 1),
                "Peak Area (μHem)": round(th.max_area_uhem, 1),
                "Growth Rate (μHem/day)": round(th.growth_rate_uhem_per_day, 2),
                "Net Longitude Drift (deg)": round(th.net_longitude_drift_deg, 2),
            })
        return pd.DataFrame(rows)

    def get_trajectory_dataframe(self) -> pd.DataFrame:
        """Export frame-by-frame trajectory points across all tracked regions."""
        rows = []
        for t_id, th in self.tracks.items():
            for obs in th.observations:
                day_offset = (obs.observation_time - th.birth_time).total_seconds() / 86400.0
                rows.append({
                    "Track ID": t_id,
                    "Timestamp": obs.observation_time.strftime("%Y-%m-%d %H:%M"),
                    "Elapsed Days": round(day_offset, 2),
                    "Observed Longitude CMD (deg)": obs.actual_lon_cmd,
                    "Predicted Longitude CMD (deg)": obs.predicted_lon_cmd,
                    "Residual Error (deg)": obs.residual_deg,
                    "Latitude (deg)": obs.actual_lat,
                    "Area (μHem)": obs.area_uhem,
                    "Growth Rate (μHem/day)": obs.area_change_rate_uhem_per_day,
                    "Event Note": obs.event_note,
                })
        return pd.DataFrame(rows)

    def plot_trajectories_plotly(self) -> go.Figure:
        """Generate interactive Plotly trajectory chart showing longitude migration over time."""
        df_traj = self.get_trajectory_dataframe()
        if df_traj.empty:
            fig = go.Figure()
            fig.update_layout(title="No Active Region Trajectories Recorded")
            return fig

        fig = px.line(
            df_traj,
            x="Timestamp",
            y="Observed Longitude CMD (deg)",
            color="Track ID",
            markers=True,
            title="Active Region Photospheric Migration (Snodgrass Differential Rotation)",
            labels={"Observed Longitude CMD (deg)": "Stonyhurst Longitude CMD (°)", "Timestamp": "Observation Timestamp"},
            hover_data=["Latitude (deg)", "Area (μHem)", "Residual Error (deg)", "Event Note"],
        )
        fig.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="Central Meridian")
        fig.update_layout(height=420, template="plotly_white")
        return fig

    def plot_area_evolution_plotly(self) -> go.Figure:
        """Generate interactive Plotly chart tracking physical area (uHem) evolution over time."""
        df_traj = self.get_trajectory_dataframe()
        if df_traj.empty:
            fig = go.Figure()
            fig.update_layout(title="No Area Evolution Data Available")
            return fig

        fig = px.line(
            df_traj,
            x="Timestamp",
            y="Area (μHem)",
            color="Track ID",
            markers=True,
            title="Physical Active Region Area Evolution (Growth & Decay Curves)",
            labels={"Area (μHem)": "Physical Area (μHem)", "Timestamp": "Observation Timestamp"},
            hover_data=["Growth Rate (μHem/day)", "Event Note"],
        )
        fig.update_layout(height=400, template="plotly_white")
        return fig

    def reset(self):
        """Reset tracking state for a new sequence."""
        self.next_track_number = 1
        self.trajectories.clear()
        self.tracks.clear()
        self.last_observation_time = None
        self.active_tracks.clear()

