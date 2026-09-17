"""
Test suite for McIntoshClassifier.
"""

import pytest

from src.classifier import McIntoshClassifier
from src.feature_extractor import CalibratedActiveRegion
from src.config import ClassificationConfig


def make_dummy_region(
    area_uhem: float = 100.0,
    span_deg: float = 4.0,
    has_penumbra: bool = True,
    is_bipolar: bool = True,
    penumbra_area_uhem: float = 70.0,
    spot_count: int = 2,
) -> CalibratedActiveRegion:
    return CalibratedActiveRegion(
        id=1,
        bbox=(100, 100, 50, 40),
        centroid_pixel=(125.0, 120.0),
        heliographic_lat=15.0,
        heliographic_lon_cmd=10.0,
        heliocentric_angle_deg=18.0,
        cos_theta=0.95,
        projected_area_px=150,
        corrected_area_px=158.0,
        area_uhem=area_uhem,
        umbra_area_uhem=area_uhem - penumbra_area_uhem,
        penumbra_area_uhem=penumbra_area_uhem,
        umbra_penumbra_ratio=(area_uhem - penumbra_area_uhem) / max(penumbra_area_uhem, 1.0),
        longitudinal_extent_deg=span_deg,
        latitudinal_extent_deg=3.0,
        spot_count=spot_count,
        mean_contrast=0.45,
        has_penumbra=has_penumbra,
        is_bipolar=is_bipolar,
    )


def test_classifier_pore_class_a():
    clf = McIntoshClassifier()
    reg = make_dummy_region(has_penumbra=False, is_bipolar=False, spot_count=1, area_uhem=15.0)
    res = clf.classify(reg)
    assert res.class_code == "A"
    assert "Class A" in res.class_name
    assert len(res.rule_trace) > 0


def test_classifier_bipolar_no_penumbra_class_b():
    clf = McIntoshClassifier()
    reg = make_dummy_region(has_penumbra=False, is_bipolar=True, spot_count=3, span_deg=5.0)
    res = clf.classify(reg)
    assert res.class_code == "B"
    assert "Class B" in res.class_name


def test_classifier_unipolar_penumbra_class_h():
    clf = McIntoshClassifier()
    reg = make_dummy_region(has_penumbra=True, is_bipolar=False, spot_count=1, area_uhem=120.0)
    res = clf.classify(reg)
    assert res.class_code == "H"
    assert "Class H" in res.class_name


def test_classifier_bipolar_compact_class_d():
    clf = McIntoshClassifier()
    reg = make_dummy_region(has_penumbra=True, is_bipolar=True, span_deg=7.5, penumbra_area_uhem=80.0, area_uhem=120.0)
    res = clf.classify(reg)
    assert res.class_code == "D"
    assert "Class D" in res.class_name


def test_classifier_bipolar_extended_class_e_and_f():
    clf = McIntoshClassifier()
    reg_e = make_dummy_region(has_penumbra=True, is_bipolar=True, span_deg=12.0)
    res_e = clf.classify(reg_e)
    assert res_e.class_code == "E"

    reg_f = make_dummy_region(has_penumbra=True, is_bipolar=True, span_deg=18.0)
    res_f = clf.classify(reg_f)
    assert res_f.class_code == "F"


def test_classifier_asymmetric_penumbra_class_c():
    clf = McIntoshClassifier()
    # Penumbra is only 15% of total area (< 25% threshold) -> Class C
    reg = make_dummy_region(
        has_penumbra=True,
        is_bipolar=True,
        span_deg=6.0,
        area_uhem=100.0,
        penumbra_area_uhem=15.0,
    )
    res = clf.classify(reg)
    assert res.class_code == "C"
    assert "Class C" in res.class_name


def test_demonstration_risk_scoring_factors():
    clf = McIntoshClassifier()
    # Test small pore
    reg_pore = make_dummy_region(has_penumbra=False, is_bipolar=False, spot_count=1, area_uhem=10.0)
    res_pore = clf.classify(reg_pore)
    risk_pore = res_pore.demonstration_risk

    assert 0.0 <= risk_pore.score <= 100.0
    assert risk_pore.attention_level == "Low Attention"
    assert risk_pore.factors.area_factor <= 15.0
    assert risk_pore.factors.penumbra_factor == 0.0

    # Test major active region (e.g. AR3664 archetype)
    reg_major = make_dummy_region(
        has_penumbra=True,
        is_bipolar=True,
        spot_count=8,
        span_deg=16.0,
        area_uhem=1500.0,
        penumbra_area_uhem=1100.0,
    )
    res_major = clf.classify(reg_major)
    risk_major = res_major.demonstration_risk

    assert risk_major.score >= 70.0
    assert risk_major.attention_level == "High Attention"
    assert risk_major.factors.area_factor == 100.0
    assert risk_major.factors.complexity_factor >= 80.0
    assert risk_major.factors.penumbra_factor >= 80.0


def test_demonstration_risk_disclaimer_and_assumptions():
    clf = McIntoshClassifier()
    reg = make_dummy_region()
    res = clf.classify(reg)
    risk = res.demonstration_risk

    # Verify disclaimer is explicit and non-predictive
    assert "DEMONSTRATION DISCLAIMER" in risk.disclaimer
    assert "NOT an operational or scientifically validated solar flare prediction model" in risk.disclaimer
    assert "MUST NOT be interpreted as a claim that a region will produce a flare" in risk.disclaimer

    # Verify stated assumptions
    assert len(risk.assumptions) >= 4
    assert any("geometric proxy" in a for a in risk.assumptions)
    assert any("educational" in a for a in risk.assumptions)


def test_morphological_class_documentation():
    clf = McIntoshClassifier()
    for code in ["A", "B", "C", "D", "E", "F", "H"]:
        info = clf.get_class_info(code)
        assert info.code == code
        assert len(info.name) > 0
        assert len(info.description) > 20
        assert len(info.typical_lifespan) > 0
        assert len(info.magnetic_topology) > 0


def test_configurable_thresholds_and_weights():
    custom_cfg = ClassificationConfig(
        area_baseline_uhem=500.0,
        low_attention_threshold=20.0,
        moderate_attention_threshold=50.0,
        risk_weight_area=0.60,
        risk_weight_complexity=0.10,
        risk_weight_penumbra=0.10,
        risk_weight_contrast=0.10,
        risk_weight_compactness=0.10,
    )
    clf = McIntoshClassifier(config=custom_cfg)
    reg = make_dummy_region(area_uhem=300.0, spot_count=2, span_deg=5.0)
    res = clf.classify(reg)

    # With area_baseline=500 and area=300, area_factor = 60.0
    assert res.demonstration_risk.factors.area_factor == 60.0
    assert res.demonstration_risk.weights["area"] == 0.60


def test_classification_with_detected_region():
    from src.detector import DetectedRegion

    det_reg = DetectedRegion(
        id=42,
        region_id="AR-042",
        bbox=(200, 200, 60, 40),
        centroid=(230.0, 220.0),
        area_pixels=500,
        perimeter_pixels=120.0,
        circularity=0.43,
        equivalent_diameter_px=25.2,
        mean_intensity=120.0,
        min_intensity=40.0,
        contrast=0.38,
        umbra_area_pixels=100,
        penumbra_area_pixels=400,
        has_penumbra=True,
        spot_count=3,
        scientific_status="Confirmed Sunspot Group",
        confidence=0.85,
        heliographic_lat=12.0,
        heliographic_lon_cmd=-5.0,
        area_uhem=350.0,
        umbra_area_uhem=70.0,
        penumbra_area_uhem=280.0,
    )

    clf = McIntoshClassifier()
    res = clf.classify(det_reg, solar_radius_px=470.0)

    assert res.region_id == 42
    assert res.class_code in ["C", "D", "E", "F"]
    assert res.attention_level in ["Low Attention", "Moderate Attention", "High Attention"]
    assert len(res.rule_trace) >= 6


def test_structured_serialization():
    clf = McIntoshClassifier()
    reg = make_dummy_region()
    res = clf.classify(reg)
    d = res.to_dict()

    assert isinstance(d, dict)
    assert d["region_id"] == 1
    assert "class_info" in d
    assert "demonstration_risk" in d
    assert "factors" in d["demonstration_risk"]
    assert "disclaimer" in d["demonstration_risk"]
    assert "rule_trace" in d

