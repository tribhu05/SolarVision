"""
SQLite database persistence module for SolarVision.
Stores observation telemetry, solar disk geometry, detected active region records,
McIntosh classification rules, demonstration risk scores, and multi-day tracking trajectories.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

from src.classifier import ClassificationResult
from src.feature_extractor import CalibratedActiveRegion
from src.solar_data import ImageMetadata
from src.tracker import TrackHistory, TrackedObservation

logger = logging.getLogger("SolarVision.Database")


class SolarDatabase:
    """
    Manages SQLite storage and retrieval for solar observations, image metadata,
    calibrated active regions, and persistent tracking trajectories.
    """

    def __init__(self, db_path: str = "data/solarvision.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a SQLite connection with foreign keys enabled and row factory set."""
        conn = sqlite3.connect(str(self.db_path), timeout=20.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        """Create database tables and indexes if they do not already exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Image Telemetry & Metadata Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sha256 TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    filepath TEXT,
                    source_url TEXT,
                    source_type TEXT,
                    observatory TEXT,
                    instrument TEXT,
                    wavelength_channel TEXT,
                    download_timestamp_utc TEXT,
                    source_last_modified TEXT,
                    etag TEXT,
                    image_width INTEGER,
                    image_height INTEGER,
                    num_channels INTEGER,
                    file_size_bytes INTEGER,
                    is_authentic_real_data INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL
                );
            """)

            # 2. Solar Observation Sessions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER REFERENCES image_metadata(id) ON DELETE SET NULL,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    disk_center_x REAL NOT NULL,
                    disk_center_y REAL NOT NULL,
                    disk_radius REAL NOT NULL,
                    disk_confidence REAL DEFAULT 1.0,
                    quiet_sun_intensity REAL NOT NULL,
                    active_region_count INTEGER NOT NULL,
                    is_spotless INTEGER DEFAULT 0,
                    pipeline_version TEXT DEFAULT '1.0.0',
                    created_at TEXT NOT NULL
                );
            """)

            # 3. Calibrated Active Regions & Morphological Classifications Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS active_regions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observation_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
                    tracking_id TEXT,
                    region_id TEXT,
                    bbox_x INTEGER NOT NULL,
                    bbox_y INTEGER NOT NULL,
                    bbox_w INTEGER NOT NULL,
                    bbox_h INTEGER NOT NULL,
                    centroid_x REAL NOT NULL,
                    centroid_y REAL NOT NULL,
                    area_pixels INTEGER NOT NULL,
                    perimeter_pixels REAL DEFAULT 0.0,
                    circularity REAL DEFAULT 0.0,
                    equivalent_diameter_px REAL DEFAULT 0.0,
                    mean_intensity REAL DEFAULT 0.0,
                    min_intensity REAL DEFAULT 0.0,
                    contrast REAL DEFAULT 0.0,
                    umbra_area_pixels INTEGER DEFAULT 0,
                    penumbra_area_pixels INTEGER DEFAULT 0,
                    has_penumbra INTEGER DEFAULT 0,
                    spot_count INTEGER DEFAULT 1,
                    scientific_status TEXT DEFAULT 'Candidate Dark Region',
                    detection_confidence REAL DEFAULT 1.0,
                    lat_deg REAL NOT NULL,
                    lon_cmd_deg REAL NOT NULL,
                    heliocentric_angle_deg REAL NOT NULL,
                    cos_theta REAL DEFAULT 1.0,
                    area_uhem REAL NOT NULL,
                    umbra_area_uhem REAL NOT NULL,
                    penumbra_area_uhem REAL NOT NULL,
                    umbra_penumbra_ratio REAL DEFAULT 0.0,
                    longitudinal_extent_deg REAL DEFAULT 0.0,
                    latitudinal_extent_deg REAL DEFAULT 0.0,
                    mcintosh_class TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    attention_level TEXT DEFAULT 'Low Attention',
                    flare_potential TEXT NOT NULL,
                    demonstration_risk_score REAL DEFAULT 0.0,
                    risk_area_factor REAL DEFAULT 0.0,
                    risk_complexity_factor REAL DEFAULT 0.0,
                    risk_penumbra_factor REAL DEFAULT 0.0,
                    risk_contrast_factor REAL DEFAULT 0.0,
                    risk_compactness_factor REAL DEFAULT 0.0,
                    risk_factors_json TEXT,
                    risk_weights_json TEXT,
                    rule_trace_json TEXT NOT NULL,
                    confidence REAL NOT NULL
                );
            """)

            # 4. Multi-Day Persistent Active Region Tracks Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tracks (
                    track_id TEXT PRIMARY KEY,
                    birth_time TEXT NOT NULL,
                    last_seen_time TEXT NOT NULL,
                    lifecycle_status TEXT NOT NULL,
                    observation_count INTEGER NOT NULL DEFAULT 1,
                    initial_area_uhem REAL DEFAULT 0.0,
                    latest_area_uhem REAL DEFAULT 0.0,
                    max_area_uhem REAL DEFAULT 0.0,
                    growth_rate_uhem_per_day REAL DEFAULT 0.0,
                    mean_latitude REAL DEFAULT 0.0,
                    net_longitude_drift_deg REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            # 5. Trajectory Observation Residuals Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trajectory_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id TEXT NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,
                    observation_id INTEGER REFERENCES observations(id) ON DELETE SET NULL,
                    timestamp TEXT NOT NULL,
                    predicted_lon_cmd REAL NOT NULL,
                    actual_lon_cmd REAL NOT NULL,
                    residual_deg REAL NOT NULL,
                    predicted_lat REAL DEFAULT 0.0,
                    actual_lat REAL DEFAULT 0.0,
                    area_uhem REAL DEFAULT 0.0,
                    area_change_rate_uhem_per_day REAL DEFAULT 0.0,
                    is_new INTEGER DEFAULT 0,
                    event_note TEXT DEFAULT 'Tracked Continuity'
                );
            """)

            # 6. Schema Migration for Existing Databases
            self._migrate_schema(cursor)

            # 7. Performance & Integrity Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_img_sha256 ON image_metadata(sha256);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_timestamp ON observations(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_image_id ON observations(image_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_obs_id ON active_regions(observation_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_track_id ON active_regions(tracking_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_status ON tracks(lifecycle_status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_traj_track_id ON trajectory_points(track_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_traj_obs_id ON trajectory_points(observation_id);")

            conn.commit()

    def _migrate_schema(self, cursor: sqlite3.Cursor):
        """Add missing columns to existing tables if database was created with an earlier schema version."""
        # 1. Observations table migration
        cursor.execute("PRAGMA table_info(observations);")
        obs_cols = {row["name"] for row in cursor.fetchall()}
        if obs_cols:
            if "image_id" not in obs_cols:
                cursor.execute("ALTER TABLE observations ADD COLUMN image_id INTEGER REFERENCES image_metadata(id) ON DELETE SET NULL;")
            if "disk_confidence" not in obs_cols:
                cursor.execute("ALTER TABLE observations ADD COLUMN disk_confidence REAL DEFAULT 1.0;")
            if "is_spotless" not in obs_cols:
                cursor.execute("ALTER TABLE observations ADD COLUMN is_spotless INTEGER DEFAULT 0;")
            if "pipeline_version" not in obs_cols:
                cursor.execute("ALTER TABLE observations ADD COLUMN pipeline_version TEXT DEFAULT '1.0.0';")

        # 2. Active regions table migration
        cursor.execute("PRAGMA table_info(active_regions);")
        ar_cols = {row["name"] for row in cursor.fetchall()}
        if ar_cols:
            if "area_pixels" not in ar_cols:
                if "projected_area_px" in ar_cols:
                    try:
                        cursor.execute("ALTER TABLE active_regions RENAME COLUMN projected_area_px TO area_pixels;")
                    except Exception:
                        cursor.execute("ALTER TABLE active_regions ADD COLUMN area_pixels INTEGER DEFAULT 0;")
                        cursor.execute("UPDATE active_regions SET area_pixels = projected_area_px;")
                else:
                    cursor.execute("ALTER TABLE active_regions ADD COLUMN area_pixels INTEGER DEFAULT 0;")

            new_ar_cols = [
                ("region_id", "TEXT"),
                ("perimeter_pixels", "REAL DEFAULT 0.0"),
                ("circularity", "REAL DEFAULT 0.0"),
                ("equivalent_diameter_px", "REAL DEFAULT 0.0"),
                ("mean_intensity", "REAL DEFAULT 0.0"),
                ("min_intensity", "REAL DEFAULT 0.0"),
                ("contrast", "REAL DEFAULT 0.0"),
                ("umbra_area_pixels", "INTEGER DEFAULT 0"),
                ("penumbra_area_pixels", "INTEGER DEFAULT 0"),
                ("has_penumbra", "INTEGER DEFAULT 0"),
                ("spot_count", "INTEGER DEFAULT 1"),
                ("scientific_status", "TEXT DEFAULT 'Candidate Dark Region'"),
                ("detection_confidence", "REAL DEFAULT 1.0"),
                ("cos_theta", "REAL DEFAULT 1.0"),
                ("umbra_penumbra_ratio", "REAL DEFAULT 0.0"),
                ("longitudinal_extent_deg", "REAL DEFAULT 0.0"),
                ("latitudinal_extent_deg", "REAL DEFAULT 0.0"),
                ("attention_level", "TEXT DEFAULT 'Low Attention'"),
                ("demonstration_risk_score", "REAL DEFAULT 0.0"),
                ("risk_area_factor", "REAL DEFAULT 0.0"),
                ("risk_complexity_factor", "REAL DEFAULT 0.0"),
                ("risk_penumbra_factor", "REAL DEFAULT 0.0"),
                ("risk_contrast_factor", "REAL DEFAULT 0.0"),
                ("risk_compactness_factor", "REAL DEFAULT 0.0"),
                ("risk_factors_json", "TEXT"),
                ("risk_weights_json", "TEXT"),
            ]
            for col_name, col_def in new_ar_cols:
                if col_name not in ar_cols:
                    cursor.execute(f"ALTER TABLE active_regions ADD COLUMN {col_name} {col_def};")

    # -------------------------------------------------------------------------
    # Image Metadata Operations
    # -------------------------------------------------------------------------

    def save_image_metadata(self, metadata: ImageMetadata) -> int:
        """
        Save or retrieve image metadata sidecar, enforcing SHA-256 deduplication.

        Returns:
            The image_metadata database ID
        """
        existing = self.get_image_by_hash(metadata.sha256)
        if existing:
            return existing["id"]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO image_metadata (
                    sha256, filename, filepath, source_url, source_type,
                    observatory, instrument, wavelength_channel,
                    download_timestamp_utc, source_last_modified, etag,
                    image_width, image_height, num_channels, file_size_bytes,
                    is_authentic_real_data, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.sha256,
                    metadata.filename,
                    metadata.filepath,
                    metadata.source_url,
                    metadata.source_type,
                    metadata.observatory,
                    metadata.instrument,
                    metadata.wavelength_channel,
                    metadata.download_timestamp_utc,
                    metadata.source_last_modified,
                    metadata.etag,
                    metadata.image_width,
                    metadata.image_height,
                    getattr(metadata, "channels", getattr(metadata, "num_channels", 3)),
                    metadata.file_size_bytes,
                    1 if metadata.is_authentic_real_data else 0,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_image_by_hash(self, sha256: str) -> Optional[Dict[str, Any]]:
        """Fetch image metadata record by SHA-256 hash."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM image_metadata WHERE sha256 = ?", (sha256,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_image_by_id(self, image_id: int) -> Optional[Dict[str, Any]]:
        """Fetch image metadata record by database ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM image_metadata WHERE id = ?", (image_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # -------------------------------------------------------------------------
    # Observation & Active Region Operations
    # -------------------------------------------------------------------------

    def save_observation(
        self,
        filename: str,
        timestamp: datetime,
        center_x: float,
        center_y: float,
        radius: float,
        quiet_sun_intensity: float,
        regions: List[Any],
        classifications: List[ClassificationResult],
        tracking_ids: Optional[List[str]] = None,
        image_id: Optional[int] = None,
        disk_confidence: float = 1.0,
        is_spotless: bool = False,
    ) -> int:
        """
        Save an observation session and all detected active regions.
        Maintains backward compatibility while persisting complete physical
        features, McIntosh classifications, and demonstration risk scores.

        Returns:
            The created observation_id
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO observations (
                    image_id, timestamp, filename, disk_center_x, disk_center_y,
                    disk_radius, disk_confidence, quiet_sun_intensity,
                    active_region_count, is_spotless, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    timestamp.isoformat(),
                    filename,
                    float(center_x),
                    float(center_y),
                    float(radius),
                    float(disk_confidence),
                    float(quiet_sun_intensity),
                    len(regions),
                    1 if is_spotless or len(regions) == 0 else 0,
                    datetime.now().isoformat(),
                ),
            )
            obs_id = cursor.lastrowid

            for idx, reg in enumerate(regions):
                clf = classifications[idx] if idx < len(classifications) else None
                t_id = tracking_ids[idx] if (tracking_ids and idx < len(tracking_ids)) else f"AR-{getattr(reg, 'id', idx + 1):03d}"
                r_id = getattr(reg, "region_id", f"AR-{getattr(reg, 'id', idx + 1):03d}")

                bx, by, bw, bh = getattr(reg, "bbox", (0, 0, 0, 0))
                cx, cy = getattr(reg, "centroid_pixel", getattr(reg, "centroid", (0.0, 0.0)))

                # Risk breakdown extraction
                risk = getattr(clf, "demonstration_risk", None)
                score = float(risk.score) if risk else 0.0
                attention = getattr(clf, "attention_level", "Low Attention")
                factors = risk.factors.to_dict() if (risk and hasattr(risk, "factors")) else {}
                weights = risk.weights if (risk and hasattr(risk, "weights")) else {}

                cursor.execute(
                    """
                    INSERT INTO active_regions (
                        observation_id, tracking_id, region_id,
                        bbox_x, bbox_y, bbox_w, bbox_h,
                        centroid_x, centroid_y, area_pixels, perimeter_pixels,
                        circularity, equivalent_diameter_px, mean_intensity, min_intensity,
                        contrast, umbra_area_pixels, penumbra_area_pixels, has_penumbra,
                        spot_count, scientific_status, detection_confidence,
                        lat_deg, lon_cmd_deg, heliocentric_angle_deg, cos_theta,
                        area_uhem, umbra_area_uhem, penumbra_area_uhem, umbra_penumbra_ratio,
                        longitudinal_extent_deg, latitudinal_extent_deg,
                        mcintosh_class, class_name, attention_level, flare_potential,
                        demonstration_risk_score, risk_area_factor, risk_complexity_factor,
                        risk_penumbra_factor, risk_contrast_factor, risk_compactness_factor,
                        risk_factors_json, risk_weights_json, rule_trace_json, confidence
                    ) VALUES (
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?
                    )
                    """,
                    (
                        obs_id,
                        t_id,
                        r_id,
                        int(bx), int(by), int(bw), int(bh),
                        float(cx), float(cy),
                        int(getattr(reg, "area_pixels", getattr(reg, "projected_area_px", 0))),
                        float(getattr(reg, "perimeter_pixels", 0.0)),
                        float(getattr(reg, "circularity", 0.0)),
                        float(getattr(reg, "equivalent_diameter_px", 0.0)),
                        float(getattr(reg, "mean_intensity", 0.0)),
                        float(getattr(reg, "min_intensity", 0.0)),
                        float(getattr(reg, "contrast", getattr(reg, "mean_contrast", 0.0))),
                        int(getattr(reg, "umbra_area_pixels", 0)),
                        int(getattr(reg, "penumbra_area_pixels", 0)),
                        1 if getattr(reg, "has_penumbra", False) else 0,
                        int(getattr(reg, "spot_count", 1)),
                        str(getattr(reg, "scientific_status", "Candidate Dark Region")),
                        float(getattr(reg, "confidence", 1.0)),
                        float(getattr(reg, "heliographic_lat", 0.0)),
                        float(getattr(reg, "heliographic_lon_cmd", 0.0)),
                        float(getattr(reg, "heliocentric_angle_deg", 0.0)),
                        float(getattr(reg, "cos_theta", 1.0)),
                        float(getattr(reg, "area_uhem", 0.0)),
                        float(getattr(reg, "umbra_area_uhem", 0.0)),
                        float(getattr(reg, "penumbra_area_uhem", 0.0)),
                        float(getattr(reg, "umbra_penumbra_ratio", 0.0)),
                        float(getattr(reg, "longitudinal_extent_deg", 0.0)),
                        float(getattr(reg, "latitudinal_extent_deg", 0.0)),
                        getattr(clf, "class_code", "A") if clf else "A",
                        getattr(clf, "class_name", "Unclassified") if clf else "Unclassified",
                        attention,
                        getattr(clf, "flare_potential", attention) if clf else "Low",
                        score,
                        float(factors.get("area_factor", 0.0)),
                        float(factors.get("complexity_factor", 0.0)),
                        float(factors.get("penumbra_factor", 0.0)),
                        float(factors.get("contrast_factor", 0.0)),
                        float(factors.get("compactness_factor", 0.0)),
                        json.dumps(factors),
                        json.dumps(weights),
                        json.dumps(getattr(clf, "rule_trace", [])) if clf else "[]",
                        float(getattr(clf, "confidence", 1.0)) if clf else 1.0,
                    ),
                )

            conn.commit()
            return obs_id

    def get_all_observations(self) -> List[Dict[str, Any]]:
        """Fetch all recorded observation sessions ordered chronologically."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM observations ORDER BY timestamp DESC, id DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_observation_by_id(self, obs_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single observation record by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM observations WHERE id = ?", (obs_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_observation_by_image_id(self, image_id: int) -> Optional[Dict[str, Any]]:
        """Fetch observation record associated with a given image metadata ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM observations WHERE image_id = ? ORDER BY timestamp DESC, id DESC LIMIT 1", (image_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_regions_for_observation(self, obs_id: int) -> List[Dict[str, Any]]:
        """Fetch all active regions for a specific observation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM active_regions WHERE observation_id = ? ORDER BY area_uhem DESC", (obs_id,))
            return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Multi-Day Tracking & Trajectory Operations
    # -------------------------------------------------------------------------

    def save_track_history(
        self,
        track_history: TrackHistory,
        observation_id: Optional[int] = None,
    ):
        """
        Upsert a persistent TrackHistory record and record trajectory points.
        """
        now_iso = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tracks (
                    track_id, birth_time, last_seen_time, lifecycle_status,
                    observation_count, initial_area_uhem, latest_area_uhem,
                    max_area_uhem, growth_rate_uhem_per_day, mean_latitude,
                    net_longitude_drift_deg, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(track_id) DO UPDATE SET
                    last_seen_time = excluded.last_seen_time,
                    lifecycle_status = excluded.lifecycle_status,
                    observation_count = excluded.observation_count,
                    latest_area_uhem = excluded.latest_area_uhem,
                    max_area_uhem = excluded.max_area_uhem,
                    growth_rate_uhem_per_day = excluded.growth_rate_uhem_per_day,
                    mean_latitude = excluded.mean_latitude,
                    net_longitude_drift_deg = excluded.net_longitude_drift_deg,
                    updated_at = excluded.updated_at
                """,
                (
                    track_history.track_id,
                    track_history.birth_time.isoformat(),
                    track_history.last_seen_time.isoformat(),
                    track_history.status,
                    track_history.observation_count,
                    float(track_history.initial_area_uhem),
                    float(track_history.latest_area_uhem),
                    float(track_history.max_area_uhem),
                    float(track_history.growth_rate_uhem_per_day),
                    float(track_history.mean_latitude),
                    float(track_history.net_longitude_drift_deg),
                    now_iso,
                    now_iso,
                ),
            )

            # Insert trajectory points for observations
            for obs in track_history.observations:
                # Check if this trajectory point is already recorded for this track and timestamp
                cursor.execute(
                    """
                    SELECT id FROM trajectory_points
                    WHERE track_id = ? AND timestamp = ?
                    """,
                    (track_history.track_id, obs.observation_time.isoformat()),
                )
                if not cursor.fetchone():
                    cursor.execute(
                        """
                        INSERT INTO trajectory_points (
                            track_id, observation_id, timestamp,
                            predicted_lon_cmd, actual_lon_cmd, residual_deg,
                            predicted_lat, actual_lat, area_uhem,
                            area_change_rate_uhem_per_day, is_new, event_note
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            track_history.track_id,
                            observation_id,
                            obs.observation_time.isoformat(),
                            float(obs.predicted_lon_cmd),
                            float(obs.actual_lon_cmd),
                            float(obs.residual_deg),
                            float(obs.predicted_lat),
                            float(obs.actual_lat),
                            float(obs.area_uhem),
                            float(obs.area_change_rate_uhem_per_day),
                            1 if obs.is_new else 0,
                            str(obs.event_note),
                        ),
                    )

            conn.commit()

    def get_track_history(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Fetch persistent track metadata and all trajectory points."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE track_id = ?", (track_id,))
            row = cursor.fetchone()
            if not row:
                return None

            track_data = dict(row)
            cursor.execute(
                "SELECT * FROM trajectory_points WHERE track_id = ? ORDER BY timestamp ASC",
                (track_id,),
            )
            track_data["trajectories"] = [dict(r) for r in cursor.fetchall()]
            return track_data

    def get_all_tracks(self) -> List[Dict[str, Any]]:
        """Fetch summary of all tracked active regions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks ORDER BY birth_time DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_trajectory_points(self, track_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch trajectory observation records, optionally filtered by track ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if track_id:
                cursor.execute(
                    "SELECT * FROM trajectory_points WHERE track_id = ? ORDER BY timestamp ASC",
                    (track_id,),
                )
            else:
                cursor.execute("SELECT * FROM trajectory_points ORDER BY timestamp ASC, track_id ASC")
            return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Catalog & Analytical Queries
    # -------------------------------------------------------------------------

    def get_catalog(self) -> List[Dict[str, Any]]:
        """
        Fetch flat joined catalog of all active regions with observation timestamps.
        Maintains backward compatibility with Streamlit dashboard.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    ar.*,
                    obs.timestamp as observation_time,
                    obs.filename as source_image,
                    obs.disk_radius,
                    obs.quiet_sun_intensity
                FROM active_regions ar
                JOIN observations obs ON ar.observation_id = obs.id
                ORDER BY obs.timestamp DESC, ar.area_uhem DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_full_catalog(self) -> List[Dict[str, Any]]:
        """
        Fetch rich joined analytical catalog linking active regions, observations,
        telemetry image metadata, and tracking lifecycles.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    ar.id as region_db_id,
                    ar.tracking_id,
                    ar.region_id,
                    ar.scientific_status,
                    ar.mcintosh_class,
                    ar.class_name,
                    ar.attention_level,
                    ar.flare_potential,
                    ar.demonstration_risk_score,
                    ar.lat_deg,
                    ar.lon_cmd_deg,
                    ar.area_uhem,
                    ar.umbra_area_uhem,
                    ar.penumbra_area_uhem,
                    ar.area_pixels,
                    ar.contrast,
                    ar.circularity,
                    ar.spot_count,
                    obs.id as observation_id,
                    obs.timestamp as observation_time,
                    obs.filename as source_image,
                    img.sha256 as image_sha256,
                    img.observatory,
                    img.instrument,
                    img.wavelength_channel,
                    trk.lifecycle_status as track_status,
                    trk.growth_rate_uhem_per_day as track_growth_rate
                FROM active_regions ar
                JOIN observations obs ON ar.observation_id = obs.id
                LEFT JOIN image_metadata img ON obs.image_id = img.id
                LEFT JOIN tracks trk ON ar.tracking_id = trk.track_id
                ORDER BY obs.timestamp DESC, ar.area_uhem DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_full_catalog_dataframe(self) -> Any:
        """
        Fetch all catalog records directly as a pandas DataFrame.
        Safely returns an empty DataFrame if no records exist.
        """
        import pandas as pd
        records = self.get_full_catalog()
        return pd.DataFrame(records) if records else pd.DataFrame()

    # -------------------------------------------------------------------------
    # Deletion & Maintenance
    # -------------------------------------------------------------------------

    def delete_observation(self, obs_id: int) -> bool:
        """Delete an observation session and cascade to its active region records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM observations WHERE id = ?", (obs_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_database(self):
        """Purge all database records (for clean test isolation or reset)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM trajectory_points;")
            cursor.execute("DELETE FROM tracks;")
            cursor.execute("DELETE FROM active_regions;")
            cursor.execute("DELETE FROM observations;")
            cursor.execute("DELETE FROM image_metadata;")
            conn.commit()
