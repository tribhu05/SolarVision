"""
SQLite database persistence module for SolarVision.
Stores observation metadata, solar disk geometry, and detected active region records.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.feature_extractor import CalibratedActiveRegion
from src.classifier import ClassificationResult


class SolarDatabase:
    """
    Manages SQLite storage for solar observations and active region catalogs.
    """

    def __init__(self, db_path: str = "data/solarvision.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    disk_center_x REAL NOT NULL,
                    disk_center_y REAL NOT NULL,
                    disk_radius REAL NOT NULL,
                    quiet_sun_intensity REAL NOT NULL,
                    active_region_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS active_regions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observation_id INTEGER NOT NULL,
                    tracking_id TEXT,
                    bbox_x INTEGER NOT NULL,
                    bbox_y INTEGER NOT NULL,
                    bbox_w INTEGER NOT NULL,
                    bbox_h INTEGER NOT NULL,
                    centroid_x REAL NOT NULL,
                    centroid_y REAL NOT NULL,
                    lat_deg REAL NOT NULL,
                    lon_cmd_deg REAL NOT NULL,
                    heliocentric_angle_deg REAL NOT NULL,
                    projected_area_px INTEGER NOT NULL,
                    area_uhem REAL NOT NULL,
                    umbra_area_uhem REAL NOT NULL,
                    penumbra_area_uhem REAL NOT NULL,
                    mcintosh_class TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    flare_potential TEXT NOT NULL,
                    rule_trace_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    FOREIGN KEY(observation_id) REFERENCES observations(id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def save_observation(
        self,
        filename: str,
        timestamp: datetime,
        center_x: float,
        center_y: float,
        radius: float,
        quiet_sun_intensity: float,
        regions: List[CalibratedActiveRegion],
        classifications: List[ClassificationResult],
        tracking_ids: Optional[List[str]] = None,
    ) -> int:
        """
        Save an observation session and all detected active regions.

        Returns:
            The created observation_id
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO observations (
                    timestamp, filename, disk_center_x, disk_center_y, disk_radius,
                    quiet_sun_intensity, active_region_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp.isoformat(),
                    filename,
                    center_x,
                    center_y,
                    radius,
                    quiet_sun_intensity,
                    len(regions),
                    datetime.now().isoformat(),
                ),
            )
            obs_id = cursor.lastrowid

            for idx, reg in enumerate(regions):
                clf = classifications[idx]
                t_id = tracking_ids[idx] if (tracking_ids and idx < len(tracking_ids)) else f"AR-{reg.id}"
                bx, by, bw, bh = reg.bbox

                cursor.execute(
                    """
                    INSERT INTO active_regions (
                        observation_id, tracking_id, bbox_x, bbox_y, bbox_w, bbox_h,
                        centroid_x, centroid_y, lat_deg, lon_cmd_deg, heliocentric_angle_deg,
                        projected_area_px, area_uhem, umbra_area_uhem, penumbra_area_uhem,
                        mcintosh_class, class_name, flare_potential, rule_trace_json, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        obs_id,
                        t_id,
                        bx, by, bw, bh,
                        reg.centroid_pixel[0], reg.centroid_pixel[1],
                        reg.heliographic_lat,
                        reg.heliographic_lon_cmd,
                        reg.heliocentric_angle_deg,
                        reg.projected_area_px,
                        reg.area_uhem,
                        reg.umbra_area_uhem,
                        reg.penumbra_area_uhem,
                        clf.class_code,
                        clf.class_name,
                        clf.flare_potential,
                        json.dumps(clf.rule_trace),
                        clf.confidence,
                    ),
                )
            conn.commit()
            return obs_id

    def get_all_observations(self) -> List[Dict[str, Any]]:
        """Fetch all recorded observation sessions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM observations ORDER BY timestamp DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_regions_for_observation(self, obs_id: int) -> List[Dict[str, Any]]:
        """Fetch all active regions for a specific observation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM active_regions WHERE observation_id = ?", (obs_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_catalog(self) -> List[Dict[str, Any]]:
        """Fetch flat joined catalog of all active regions with observation timestamps."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    ar.*,
                    obs.timestamp as observation_time,
                    obs.filename as source_image
                FROM active_regions ar
                JOIN observations obs ON ar.observation_id = obs.id
                ORDER BY obs.timestamp DESC, ar.area_uhem DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def clear_database(self):
        """Purge all records (for testing or reset)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM active_regions")
            cursor.execute("DELETE FROM observations")
            conn.commit()
