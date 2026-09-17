"""
Tests for SolarVision Scientific Evaluation Module
"""

from pathlib import Path
import pytest
import numpy as np

from src.evaluation import (
    SolarVisionEvaluator,
    ReferenceActiveRegion,
    BenchmarkObservation,
    NOAA_BENCHMARK_CATALOG,
    EvaluationScorecard,
)
from src.pipeline import SolarVisionPipeline
from src.config import load_config
from src.database import SolarDatabase


def test_calculate_bbox_iou():
    evaluator = SolarVisionEvaluator()

    # Identical bounding boxes -> IoU = 1.0
    b1 = (100, 100, 50, 50)
    assert pytest.approx(evaluator.calculate_bbox_iou(b1, b1), 0.01) == 1.0

    # Completely disjoint boxes -> IoU = 0.0
    b2 = (200, 200, 50, 50)
    assert evaluator.calculate_bbox_iou(b1, b2) == 0.0

    # 50% width overlap
    # b1: (0, 0, 10, 10) -> area 100
    # b3: (5, 0, 10, 10) -> area 100, intersection (5, 0, 5, 10) -> area 50, union 150 -> IoU = 50/150 = 1/3
    b_left = (0, 0, 10, 10)
    b_overlap = (5, 0, 10, 10)
    assert pytest.approx(evaluator.calculate_bbox_iou(b_left, b_overlap), 0.01) == 1.0 / 3.0


def test_noaa_benchmark_catalog_integrity():
    assert len(NOAA_BENCHMARK_CATALOG) >= 4
    for key, bench in NOAA_BENCHMARK_CATALOG.items():
        assert bench.benchmark_id.startswith("BENCH-")
        if not bench.is_spotless:
            assert len(bench.reference_regions) > 0
            for ref in bench.reference_regions:
                assert ref.noaa_number.startswith("NOAA AR")
                assert -90.0 <= ref.heliographic_lat <= 90.0
                assert -90.0 <= ref.heliographic_lon_cmd <= 90.0
                assert ref.area_uhem > 0.0
                assert ref.major_class in ["A", "B", "C", "D", "E", "F", "H"]


def test_spotless_solar_disk_evaluation():
    evaluator = SolarVisionEvaluator()
    bench = NOAA_BENCHMARK_CATALOG["synthetic_spotless_minimum.png"]

    # True spotless disk with zero detections -> perfect 1.0 scorecard
    scorecard = evaluator.evaluate_detections(detected_regions=[], benchmark=bench)
    assert scorecard.precision == 1.0
    assert scorecard.recall == 1.0
    assert scorecard.f1_score == 1.0
    assert scorecard.spotless_disk_accuracy == 1.0
    assert scorecard.false_positives == 0
    assert scorecard.false_negatives == 0


def test_spotless_solar_disk_false_alarms():
    evaluator = SolarVisionEvaluator()
    bench = NOAA_BENCHMARK_CATALOG["synthetic_spotless_minimum.png"]

    # Mock detected region on spotless disk -> False Positive
    class MockDet:
        id = 1
        region_id = "AR-001"
        heliographic_lat = 10.0
        heliographic_lon_cmd = 20.0
        area_uhem = 30.0
        bbox = (500, 500, 30, 30)
        mcintosh_class = "A"

    scorecard = evaluator.evaluate_detections(detected_regions=[MockDet()], benchmark=bench)
    assert scorecard.false_positives == 1
    assert scorecard.true_positives == 0
    assert scorecard.precision == 0.0
    assert scorecard.recall == 0.0
    assert scorecard.f1_score == 0.0
    assert scorecard.spotless_disk_accuracy == 0.0


def test_synthetic_detection_matching():
    evaluator = SolarVisionEvaluator(max_matching_dist_deg=8.0)

    # Reference benchmark with 2 regions
    bench = BenchmarkObservation(
        benchmark_id="BENCH-TEST",
        source_filename="test.jpg",
        observation_date="2024-05-10",
        is_spotless=False,
        reference_regions=[
            ReferenceActiveRegion(
                noaa_number="NOAA AR 13664",
                designation="Target 1",
                heliographic_lat=-17.0,
                heliographic_lon_cmd=32.0,
                area_uhem=2000.0,
                mcintosh_class="Fkc",
                major_class="F",
                spot_count=30,
                approx_bbox_px=(700, 580, 160, 110),
            ),
            ReferenceActiveRegion(
                noaa_number="NOAA AR 13663",
                designation="Target 2",
                heliographic_lat=-20.0,
                heliographic_lon_cmd=66.0,
                area_uhem=250.0,
                mcintosh_class="Hax",
                major_class="H",
                spot_count=5,
                approx_bbox_px=(880, 600, 70, 75),
            ),
        ],
        notes="Synthetic test observation",
    )

    class MockDet1:
        id = 1
        region_id = "AR-001"
        heliographic_lat = -16.8
        heliographic_lon_cmd = 32.5
        area_uhem = 1980.0
        bbox = (705, 582, 155, 108)
        mcintosh_class = "F"

    class MockDet2:
        id = 2
        region_id = "AR-002"
        heliographic_lat = -19.8
        heliographic_lon_cmd = 66.2
        area_uhem = 240.0
        bbox = (882, 602, 68, 72)
        mcintosh_class = "H"

    # Both should be True Positives with high IoU and low coordinate error
    scorecard = evaluator.evaluate_detections([MockDet1(), MockDet2()], bench)
    assert scorecard.true_positives == 2
    assert scorecard.false_positives == 0
    assert scorecard.false_negatives == 0
    assert scorecard.precision == 1.0
    assert scorecard.recall == 1.0
    assert scorecard.f1_score == 1.0
    assert scorecard.mean_total_coord_error_deg < 1.0
    assert scorecard.mean_iou > 0.80
    assert scorecard.mcintosh_class_accuracy == 1.0


def test_real_sdo_image_scientific_evaluation(tmp_path):
    config = load_config()
    db_file = tmp_path / "test_eval.db"
    db = SolarDatabase(str(db_file))
    pipeline = SolarVisionPipeline(config=config, db=db)

    sample_path = Path(config.storage.sample_data_dir) / "sdo_hmi_ar3664_20240510.jpg"
    assert sample_path.exists(), "Sample image missing"

    # Execute pipeline on real SDO observation
    result = pipeline.process_image(sample_path)
    assert result.success is True

    # Evaluate against official NOAA benchmark (passing PipelineResult directly)
    bench = NOAA_BENCHMARK_CATALOG["sdo_hmi_ar3664_20240510.jpg"]
    evaluator = SolarVisionEvaluator()

    scorecard = evaluator.evaluate_detections(result, bench)

    # Validate that AR3664 was detected and matched as True Positive
    ar3664_match = next((m for m in scorecard.matches if m.noaa_number == "NOAA AR 13664"), None)
    assert ar3664_match is not None
    assert ar3664_match.match_type == "True Positive"
    assert ar3664_match.coordinate_distance_deg < 2.0
    assert ar3664_match.detected_class.startswith("F")

    # Overall scorecard should demonstrate high performance
    assert scorecard.true_positives >= 3
    assert scorecard.recall >= 0.75
    assert scorecard.precision >= 0.60
    assert scorecard.mean_total_coord_error_deg < 3.0

    df_report = scorecard.to_dataframe()
    assert len(df_report) == len(scorecard.matches)
