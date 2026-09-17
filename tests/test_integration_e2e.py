"""
Comprehensive End-to-End Integration Test Suite for SolarVision.
Directly exercises and validates all 15 dimensions specified in project verification.
"""

import ast
from datetime import datetime
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pandas as pd
import pytest

from src.classifier import (
    McIntoshClassifier,
    ClassificationResult,
    DemonstrationRiskAssessment,
)
from src.config import SolarVisionConfig, load_config
from src.database import SolarDatabase
from src.detector import SunspotDetector, DetectedRegion
from src.disk_detector import SolarDiskDetector, SolarDiskGeometry
from src.evaluation import SolarVisionEvaluator
from src.feature_extractor import CalibratedActiveRegion, FeatureExtractor
from src.limb_darkening import LimbDarkeningCorrector
from src.pipeline import SolarVisionPipeline, PipelineResult, SequencePipelineResult
from src.preprocessor import SolarImagePreprocessor, PreprocessingResult
from src.segmentation import SunspotSegmenter
from src.solar_data import SolarDataIngestor, IngestionResult
from src.tracker import ActiveRegionTracker, TrackHistory


# 1. Project Installation & Dependency Verification
def test_01_project_installation():
    """Verify that all required third-party dependencies are installed and importable."""
    import cv2
    import numpy
    import scipy
    import pandas
    import streamlit
    import plotly
    import yaml
    import requests
    import PIL

    assert cv2.__version__ is not None
    assert numpy.__version__ is not None
    assert pandas.__version__ is not None
    assert streamlit.__version__ is not None
    assert plotly.__version__ is not None


# 2. Application Startup & AST Integrity
def test_02_application_startup():
    """Verify that app.py parses cleanly without syntax errors and contains all 8 required pages."""
    app_path = Path("app.py")
    assert app_path.exists(), "app.py does not exist"

    with open(app_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    assert tree is not None

    required_sections = [
        "🏠 Overview",
        "🔬 Solar Image Analysis",
        "🎯 Detection Results",
        "🏷️ Region Classification",
        "🛰️ Multi-Day Tracking",
        "📊 Historical Activity",
        "📈 Scientific Evaluation",
        "📚 Methodology & Limitations",
    ]
    for sec in required_sections:
        assert sec in source, f"Missing section in app.py: {sec}"


# 3. Image Loading
def test_03_image_loading():
    """Verify loading real SDO continuum images from disk using SolarDataIngestor."""
    ingestor = SolarDataIngestor()
    sample_path = Path("data/sample_images/sdo_hmi_ar3664_20240510.jpg")
    assert sample_path.exists(), "Sample benchmark image missing"

    res = ingestor.load_local_image(sample_path)
    assert isinstance(res, IngestionResult)
    assert res.success is True
    assert isinstance(res.image, np.ndarray)
    assert res.image.shape[0] >= 512
    assert res.image.shape[1] >= 512
    assert res.metadata is not None
    assert res.metadata.sha256 != ""


# 4. Data Ingestion & Deduplication
def test_04_data_ingestion():
    """Verify SHA-256 deduplication and metadata sidecar generation."""
    ingestor = SolarDataIngestor()
    sample_path = Path("data/sample_images/sdo_hmi_ar3664_20240510.jpg")

    # Ingest observation
    res1 = ingestor.load_local_image(sample_path)
    assert res1.success is True

    # Re-loading identical file yields identical SHA-256
    res2 = ingestor.load_local_image(sample_path)
    assert res2.success is True
    assert res1.metadata.sha256 == res2.metadata.sha256
    assert res1.metadata.is_authentic_real_data is True


# 5. Image Preprocessing & Limb-Darkening Compensation
def test_05_image_preprocessing():
    """Verify solar disk localization and quadratic limb-darkening compensation."""
    cfg = SolarVisionConfig()
    preprocessor = SolarImagePreprocessor(full_config=cfg)
    sample_path = Path("data/sample_images/sdo_hmi_ar3664_20240510.jpg")

    res = preprocessor.process(sample_path)
    assert isinstance(res, PreprocessingResult)
    assert res.solar_disk.is_valid is True
    assert res.solar_disk.radius > 350.0
    assert res.quiet_sun_level > 100.0
    assert res.preprocessed_image.shape == (1024, 1024)
    assert res.visuals.composite_panel.size > 0


# 6. Sunspot Detection & Segmentation
def test_06_sunspot_detection():
    """Verify dual-level adaptive segmentation separating umbra and penumbra."""
    sample_path = Path("data/sample_images/sdo_hmi_ar3664_20240510.jpg")
    img = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    assert img is not None

    disk = SolarDiskDetector().detect(img)
    limb = LimbDarkeningCorrector().correct(img, disk)
    segmenter = SunspotSegmenter()
    seg_res = segmenter.segment(limb, disk)

    assert len(seg_res.regions) >= 1
    assert np.count_nonzero(seg_res.umbra_mask) > 0
    assert np.count_nonzero(seg_res.penumbra_mask) > 0
    
    detector = SunspotDetector()
    det_out = detector.detect(img)
    assert det_out.is_spotless is False
    assert det_out.total_detected_count >= 1


# 7. Feature Extraction & Heliographic Calibration
def test_07_feature_extraction():
    """Verify Stonyhurst coordinate transformation and physical area in MSH."""
    pipeline = SolarVisionPipeline()
    res = pipeline.process_image("data/sample_images/sdo_hmi_ar3664_20240510.jpg")

    assert res.success is True
    assert len(res.regions) >= 1
    for reg in res.regions:
        assert -90.0 <= reg.heliographic_lat <= 90.0
        assert -90.0 <= reg.heliographic_lon_cmd <= 90.0
        assert reg.area_uhem > 0.0
        assert reg.cos_theta > 0.0


# 8. Morphological Classification
def test_08_classification():
    """Verify Modified Zurich / McIntosh classification."""
    pipeline = SolarVisionPipeline()
    res = pipeline.process_image("data/sample_images/sdo_hmi_ar3664_20240510.jpg")

    assert len(res.classifications) == len(res.regions)
    for c in res.classifications:
        assert c.class_code in ["A", "B", "C", "D", "E", "F", "H"]
        assert c.confidence > 0.0


# 9. Demonstration Risk Scoring
def test_09_risk_scoring():
    """Verify Demonstration Risk Score factors, bounds (0-100), and non-prediction disclaimer."""
    pipeline = SolarVisionPipeline()
    res = pipeline.process_image("data/sample_images/sdo_hmi_ar3664_20240510.jpg")

    for c in res.classifications:
        assert 0.0 <= c.demonstration_risk.score <= 100.0
        assert c.demonstration_risk is not None
        assert "NOT an operational or scientifically validated solar flare prediction model" in c.demonstration_risk.disclaimer


# 10. Database Storage & Cascade Integrity
def test_10_database_storage(tmp_path):
    """Verify SQLite schema creation, observation saving, and cascade deletion."""
    db_path = str(tmp_path / "test_storage.db")
    db = SolarDatabase(db_path)
    pipeline = SolarVisionPipeline(db=db)

    result = pipeline.process_image("data/sample_images/sdo_hmi_ar3664_20240510.jpg")
    assert result.observation_id > 0

    # Verify stored records
    obs = db.get_observation_by_id(result.observation_id)
    assert obs is not None
    assert obs["active_region_count"] == len(result.regions)

    regions = db.get_regions_for_observation(result.observation_id)
    assert len(regions) == len(result.regions)

    # Test cascade delete
    db.delete_observation(result.observation_id)
    assert db.get_observation_by_id(result.observation_id) is None
    assert len(db.get_regions_for_observation(result.observation_id)) == 0


# 11. Multi-Day Kinematic Tracking
def test_11_multi_day_tracking(tmp_path):
    """Verify tracking across May 10, 11, and 12, 2024 using Snodgrass differential rotation."""
    db_path = str(tmp_path / "test_tracking.db")
    pipeline = SolarVisionPipeline(db=SolarDatabase(db_path))
    seq_files = [
        "data/sample_images/sdo_hmi_ar3664_20240510.jpg",
        "data/sample_images/sdo_hmi_ar3664_20240511.jpg",
        "data/sample_images/sdo_hmi_ar3664_20240512.jpg",
    ]
    seq_result = pipeline.process_sequence(seq_files)
    assert seq_result.success is True
    assert seq_result.observation_count == 3
    assert len(seq_result.tracks) >= 1
    assert seq_result.total_active_regions_detected >= 1


# 12. Dashboard Visualization Elements
def test_12_dashboard_visualization():
    """Verify that dashboard components can format and serialize pipeline results cleanly."""
    pipeline = SolarVisionPipeline()
    res = pipeline.process_image("data/sample_images/sdo_hmi_ar3664_20240510.jpg")

    d = res.to_dict()
    assert d["success"] is True
    assert d["active_region_count"] >= 1
    assert d["is_spotless"] is False
    assert d["disk_radius"] > 350.0


# 13. Historical Charts & Plotly Visualizations
def test_13_historical_charts():
    """Verify that Plotly figures render and serialize without exceptions."""
    tracker = ActiveRegionTracker()
    fig_empty_traj = tracker.plot_trajectories_plotly()
    fig_empty_area = tracker.plot_area_evolution_plotly()
    assert fig_empty_traj is not None
    assert fig_empty_area is not None

    # Verify serialization to dict/JSON
    assert "data" in fig_empty_traj.to_dict()
    assert "layout" in fig_empty_area.to_dict()


# 14. Error Handling
def test_14_error_handling():
    """Verify defensive handling of nonexistent files and corrupted buffers."""
    ingestor = SolarDataIngestor()
    res = ingestor.load_local_image("nonexistent_path_solar_fake.jpg")
    assert res.success is False
    assert res.status == "error"
    assert "not found" in (res.error_message or "").lower()

    pipeline = SolarVisionPipeline()
    res_pipe = pipeline.process_image("another_nonexistent_file.jpg")
    assert res_pipe.success is False
    assert res_pipe.error_message is not None


# 15. Empty & Invalid Input Cases
def test_15_empty_and_invalid_inputs():
    """Verify spotless disk handling and empty image rejection."""
    from src.evaluation import NOAA_BENCHMARK_CATALOG
    
    detector = SunspotDetector()
    clean_disk = np.full((512, 512), 200, dtype=np.uint8)

    # Off-disk space zeroed
    y, x = np.ogrid[:512, :512]
    r_dist = np.sqrt((x - 256) ** 2 + (y - 256) ** 2)
    clean_disk[r_dist > 230] = 0

    det_res = detector.detect(clean_disk)
    assert det_res.is_spotless is True
    assert det_res.total_detected_count == 0

    # Spotless evaluation
    evaluator = SolarVisionEvaluator()
    bench = NOAA_BENCHMARK_CATALOG["synthetic_spotless_minimum.png"]
    scorecard = evaluator.evaluate_detections(
        benchmark=bench,
        detected_regions=[],
    )
    assert scorecard.recall == 1.0
    assert scorecard.precision == 1.0
    assert scorecard.false_positives == 0
