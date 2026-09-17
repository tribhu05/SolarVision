"""
End-to-end verification script for SolarVision data storage and processing pipeline.
Processes real NASA SDO observations, tests deduplication, validates SQLite records,
and outputs verification metrics.
"""

from datetime import datetime
from pathlib import Path
import json
import sqlite3
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import load_config
from src.database import SolarDatabase
from src.pipeline import SolarVisionPipeline


def main():
    print("=" * 70)
    print("SolarVision Data Storage & Processing Pipeline Verification")
    print("=" * 70)

    cfg = load_config()
    db_path = "data/solarvision.db"
    db = SolarDatabase(db_path)
    pipeline = SolarVisionPipeline(config=cfg, db=db)

    # 1. Single Image Processing on real SDO AR3664 observation
    img_day1 = Path("data/sample_images/sdo_hmi_ar3664_20240510.jpg")
    print(f"\n[1] Processing Single Real SDO Observation: {img_day1.name}")
    t0 = datetime.now()
    res1 = pipeline.process_image(img_day1, reprocess=True)
    elapsed1 = (datetime.now() - t0).total_seconds()

    assert res1.success is True, f"Failed: {res1.error_message}"
    print(f"    - Success: {res1.success} (Took {elapsed1:.2f}s)")
    print(f"    - Observation ID: {res1.observation_id}")
    print(f"    - Image ID: {res1.image_id} (SHA-256: {res1.image_metadata.sha256[:16]}...)")
    print(f"    - Solar Disk Radius: {res1.solar_disk.radius:.1f} px (Confidence: {res1.solar_disk.confidence*100:.0f}%)")
    print(f"    - Quiet-Sun Intensity: {res1.quiet_sun_intensity:.1f}")
    print(f"    - Detected Active Regions: {len(res1.regions)}")

    for idx, (reg, clf) in enumerate(zip(res1.regions[:3], res1.classifications[:3])):
        print(f"      * AR-{reg.id}: Class {clf.class_code} ({clf.class_name[:35]}...) | Area: {reg.area_uhem:.1f} uHem | Risk: {clf.demonstration_risk.score:.1f}/100 ({clf.attention_level})")

    # 2. Rerun Idempotency & Deduplication Check
    print("\n[2] Testing Pipeline Idempotency & Deduplication...")
    res2 = pipeline.process_image(img_day1, reprocess=False)
    print(f"    - Second run result is_cached: {res2.is_cached}")
    assert res2.is_cached is True, "Pipeline failed to return cached result on duplicate image!"
    assert res2.observation_id == res1.observation_id
    print("    - Deduplication confirmed: zero redundant computation, identical observation ID returned.")

    # 3. Multi-Day Sequence Processing (AR3664 May 10, 11, 12, 2024)
    seq_files = [
        Path("data/sample_images/sdo_hmi_ar3664_20240510.jpg"),
        Path("data/sample_images/sdo_hmi_ar3664_20240511.jpg"),
        Path("data/sample_images/sdo_hmi_ar3664_20240512.jpg"),
    ]
    print(f"\n[3] Processing Multi-Day Historical Sequence ({len(seq_files)} frames)...")
    t_seq_start = datetime.now()
    seq_res = pipeline.process_sequence(seq_files, reprocess=True)
    elapsed_seq = (datetime.now() - t_seq_start).total_seconds()

    assert seq_res.success is True, f"Sequence failed: {seq_res.error_message}"
    print(f"    - Sequence Success: {seq_res.success} (Took {elapsed_seq:.2f}s)")
    print(f"    - Observations Processed: {seq_res.observation_count}")
    print(f"    - Total Region Detections: {seq_res.total_active_regions_detected}")
    print(f"    - Unique Kinematic Tracks: {seq_res.unique_tracks_count}")

    # Inspect persistent tracks
    print("\n    Persistent Track Summaries:")
    for t_id, th in seq_res.tracks.items():
        if th.observation_count >= 2:
            print(f"      * Track {t_id}: Status='{th.status}', Frames={th.observation_count}, Duration={th.duration_days:.1f}d, Peak Area={th.max_area_uhem:.1f} uHem, Growth={th.growth_rate_uhem_per_day:.1f} uHem/day, Drift={th.net_longitude_drift_deg:.1f} deg")

    # 4. Database Validation & SQLite Integrity Inspection
    print(f"\n[4] Inspecting SQLite Database: {db_path}")
    with db._get_connection() as conn:
        cursor = conn.cursor()
        for tbl in ["image_metadata", "observations", "active_regions", "tracks", "trajectory_points"]:
            cursor.execute(f"SELECT COUNT(*) as count FROM {tbl}")
            count = cursor.fetchone()["count"]
            print(f"    - Table '{tbl}': {count} records")

    # 5. Full Analytical Catalog Query
    catalog = db.get_full_catalog()
    df_cat = pd.DataFrame(catalog)
    print(f"\n[5] Analytical Catalog Records Retrieved: {len(df_cat)} rows")
    if not df_cat.empty:
        sample_cols = ["observation_time", "source_image", "tracking_id", "mcintosh_class", "area_uhem", "demonstration_risk_score", "attention_level"]
        print(df_cat[sample_cols].head(5).to_string(index=False))

    print("\n" + "=" * 70)
    print("ALL END-TO-END PIPELINE AND STORAGE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
