"""
SolarVision Classification and Demonstration Risk Scoring Module.
Classifies detected solar active regions into Modified Zurich / McIntosh classes
using transparent geometric and morphological features, and computes a multi-factor
heuristic demonstration risk indicator with explicit non-prediction disclaimers.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.config import ClassificationConfig
from src.feature_extractor import CalibratedActiveRegion


DEMONSTRATION_RISK_DISCLAIMER = (
    "DEMONSTRATION DISCLAIMER: This risk score is a heuristic educational index "
    "quantifying visible-light structural complexity (area, multiplicity, contrast, "
    "and compactness). It is NOT an operational or scientifically validated solar "
    "flare prediction model and MUST NOT be interpreted as a claim that a region will "
    "produce a flare. Operational space weather forecasting requires vector magnetograms "
    "(HMI/SDO magnetic field vectors), electric current helicity, magnetic shear along "
    "neutral lines, free magnetic energy, and empirical flare history models (e.g. NOAA/SWPC). "
    "White-light continuum imagery alone cannot measure sub-photospheric magnetic reconnection "
    "or free magnetic energy."
)


@dataclass
class MorphologicalClassInfo:
    """Scientific documentation and metadata for a Zurich / McIntosh class."""
    code: str                                         # 'A', 'B', 'C', 'D', 'E', 'F', 'H'
    name: str                                         # e.g. "Class D: Moderate Bipolar Group with Penumbra at Both Ends"
    description: str                                  # Complete scientific definition and evolutionary stage
    typical_lifespan: str                             # Typical photospheric lifetime
    magnetic_topology: str                            # Visible light configuration
    mcintosh_equivalent: str                          # Zurich/McIntosh designation


CLASS_DEFINITIONS: Dict[str, MorphologicalClassInfo] = {
    "A": MorphologicalClassInfo(
        code="A",
        name="Class A: Unipolar Pore or Small Spot without Penumbra",
        description=(
            "Small, nascent unipolar pore or decaying remnant sunspot with no penumbral halo. "
            "Represents the earliest stage of flux emergence or the final decaying phase of a sunspot."
        ),
        typical_lifespan="Hours to 1 day",
        magnetic_topology="Simple unipolar flux tube",
        mcintosh_equivalent="Axx / Zurich A",
    ),
    "B": MorphologicalClassInfo(
        code="B",
        name="Class B: Bipolar Group without Penumbra",
        description=(
            "Bipolar sunspot cluster containing both leader and follower spots, but lacking any penumbral halo. "
            "Indicates active magnetic flux emergence where magnetic loops connect opposite polarities across the photosphere."
        ),
        typical_lifespan="1 to 2 days",
        magnetic_topology="Simple bipolar flux pair without penumbra",
        mcintosh_equivalent="Bxo / Zurich B",
    ),
    "C": MorphologicalClassInfo(
        code="C",
        name="Class C: Bipolar Group with Penumbra on One End",
        description=(
            "Bipolar active region where penumbra has developed on spots at one end of the group (predominantly "
            "the larger leader spot), while follower spots remain penumbra-free pores."
        ),
        typical_lifespan="Several days to 1 week",
        magnetic_topology="Asymmetric bipolar group (penumbra on leader)",
        mcintosh_equivalent="Cso / Csi / Zurich C",
    ),
    "D": MorphologicalClassInfo(
        code="D",
        name="Class D: Moderate Bipolar Group with Penumbra at Both Ends",
        description=(
            "Mature bipolar sunspot group exhibiting well-defined penumbra on spots at both leader and follower extremities. "
            "Longitudinal span is compact (<= 10° heliocentric extent)."
        ),
        typical_lifespan="1 to 2 weeks",
        magnetic_topology="Mature compact bipolar system",
        mcintosh_equivalent="Dai / Dki / Zurich D",
    ),
    "E": MorphologicalClassInfo(
        code="E",
        name="Class E: Large Extended Bipolar Group with Complex Penumbra",
        description=(
            "Extensive bipolar active region spanning between 10° and 15° in heliographic longitude, featuring "
            "large leader and follower spots with intermediate spots and complex penumbral structures."
        ),
        typical_lifespan="2 to 4 weeks (often persists across full solar rotations)",
        magnetic_topology="Extended bipolar complex with intermediate spots",
        mcintosh_equivalent="Eai / Eki / Zurich E",
    ),
    "F": MorphologicalClassInfo(
        code="F",
        name="Class F: Very Large, Highly Complex Bipolar Sunspot Group",
        description=(
            "The largest and most structurally complex sunspot groups, spanning > 15° in longitude or exceeding "
            "1000 μHem in physical area (e.g. historic super active regions like AR3664). Contains extensive penumbral "
            "belts crowded with multiple umbrae."
        ),
        typical_lifespan="Multiple solar rotations (> 1 month)",
        magnetic_topology="Highly complex multipolar cluster with strong potential magnetic shear",
        mcintosh_equivalent="Fki / Fkc / Zurich F",
    ),
    "H": MorphologicalClassInfo(
        code="H",
        name="Class H: Stable Unipolar Spot with Symmetric Penumbra",
        description=(
            "A solitary, mature leader spot that has shed its follower spots during decay. Retains a circular, "
            "nearly symmetric penumbra and represents a magnetically stable, quiescent configuration."
        ),
        typical_lifespan="1 to 3 weeks",
        magnetic_topology="Quiescent unipolar flux tube with symmetric penumbra",
        mcintosh_equivalent="Hax / Hsx / Zurich H",
    ),
}


@dataclass
class DemonstrationRiskFactors:
    """
    Individual contribution factors to the heuristic demonstration risk indicator.
    Each factor is normalized to [0.0 - 100.0] before weighted summation.
    """
    area_factor: float          # Physical footprint (log-linear scaled by area_baseline_uhem)
    complexity_factor: float    # Spot multiplicity and longitudinal span
    penumbra_factor: float      # Umbra-penumbra ratio and mature penumbral coverage
    contrast_factor: float      # Photospheric darkness/intensity contrast
    compactness_factor: float   # Shape irregularity (1.0 - circularity)

    def to_dict(self) -> Dict[str, float]:
        return {
            "area_factor": round(float(self.area_factor), 1),
            "complexity_factor": round(float(self.complexity_factor), 1),
            "penumbra_factor": round(float(self.penumbra_factor), 1),
            "contrast_factor": round(float(self.contrast_factor), 1),
            "compactness_factor": round(float(self.compactness_factor), 1),
        }


@dataclass
class DemonstrationRiskAssessment:
    """
    Heuristic demonstration risk score for educational/demonstration purposes.
    Explicitly NOT an operational or scientifically validated solar flare prediction.
    """
    score: float                                      # Combined score [0.0 - 100.0]
    attention_level: str                              # 'Low Attention', 'Moderate Attention', 'High Attention'
    factors: DemonstrationRiskFactors
    weights: Dict[str, float]                         # Configurable weights applied
    assumptions: List[str]                            # Explicitly stated physical/heuristic assumptions
    disclaimer: str = DEMONSTRATION_RISK_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(float(self.score), 1),
            "attention_level": str(self.attention_level),
            "factors": self.factors.to_dict(),
            "weights": {k: round(float(v), 2) for k, v in self.weights.items()},
            "assumptions": self.assumptions,
            "disclaimer": self.disclaimer,
        }


@dataclass
class ClassificationResult:
    """Detailed classification and demonstration risk assessment output."""
    region_id: int
    class_code: str                                   # 'A', 'B', 'C', 'D', 'E', 'F', or 'H'
    class_name: str                                   # Short name
    class_info: MorphologicalClassInfo                # Full documented definition
    attention_level: str                              # Neutral attention label: 'Low Attention', 'Moderate Attention', 'High Attention'
    flare_potential: str                              # Backward-compatible alias for attention_level
    demonstration_risk: DemonstrationRiskAssessment   # Full transparent risk breakdown
    rule_trace: List[str]                             # Auditable decision deduction steps
    confidence: float                                 # Rule matching confidence [0.0 - 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": int(self.region_id),
            "class_code": str(self.class_code),
            "class_name": str(self.class_name),
            "attention_level": str(self.attention_level),
            "flare_potential": str(self.flare_potential),
            "confidence": round(float(self.confidence), 2),
            "class_info": asdict(self.class_info),
            "demonstration_risk": self.demonstration_risk.to_dict(),
            "rule_trace": self.rule_trace,
        }


class McIntoshClassifier:
    """
    Classifies solar active regions into Modified Zurich / McIntosh classes
    based on measurable physical features (area, multiplicity, penumbra, compactness,
    and longitudinal extent), and computes an auditable demonstration risk score.
    """

    def __init__(self, config: Optional[ClassificationConfig] = None):
        self.config = config or ClassificationConfig()

    @staticmethod
    def get_class_info(class_code: str) -> MorphologicalClassInfo:
        """Retrieve documented scientific metadata for a given class code."""
        code_upper = class_code.strip().upper()
        if code_upper in CLASS_DEFINITIONS:
            return CLASS_DEFINITIONS[code_upper]
        return MorphologicalClassInfo(
            code=code_upper,
            name=f"Class {code_upper}: Solar Active Region",
            description="Active region on the solar photosphere.",
            typical_lifespan="Unknown",
            magnetic_topology="Undefined",
            mcintosh_equivalent=f"{code_upper}xx",
        )

    def calculate_demonstration_risk(
        self,
        area_uhem: float,
        spot_count: int,
        span_deg: float,
        has_penumbra: bool,
        is_bipolar: bool,
        class_code: str,
        contrast: float,
        circularity: float,
        penumbra_area_uhem: float,
    ) -> DemonstrationRiskAssessment:
        """
        Compute a transparent multi-factor demonstration risk indicator.
        
        Note:
            This metric is strictly an educational complexity index and NOT an operational
            solar flare prediction.
        """
        cfg = self.config

        # 1. Area Factor (0 - 100): Scaled by area_baseline_uhem
        baseline_area = max(cfg.area_baseline_uhem, 1.0)
        f_area = float(np.clip((area_uhem / baseline_area) * 100.0, 0.0, 100.0))

        # 2. Structural Complexity Factor (0 - 100): Multiplicity + Longitudinal extent
        spot_contrib = min(50.0, max(0.0, float(spot_count - 1) * 10.0))
        max_span = max(cfg.class_e_max_span_deg, 1.0)
        span_contrib = min(50.0, max(0.0, (span_deg / max_span) * 50.0))
        f_complexity = float(np.clip(spot_contrib + span_contrib, 0.0, 100.0))

        # 3. Penumbra Coverage & Distribution Factor (0 - 100)
        if not has_penumbra:
            f_penumbra = 0.0
        elif class_code == "H":
            f_penumbra = 25.0
        elif class_code == "C":
            f_penumbra = 50.0
        elif class_code == "D":
            f_penumbra = 75.0
        elif class_code in ("E", "F"):
            f_penumbra = 95.0
        else:
            f_penumbra = 40.0

        # Fine-tune with penumbra area fraction if available
        if area_uhem > 0 and penumbra_area_uhem > 0:
            pen_frac = min(1.0, penumbra_area_uhem / area_uhem)
            f_penumbra = float(np.clip(f_penumbra * (0.5 + 0.5 * pen_frac), 0.0, 100.0))

        # 4. Photospheric Contrast / Intensity Factor (0 - 100)
        baseline_contrast = max(cfg.contrast_baseline, 0.1)
        f_contrast = float(np.clip((contrast / baseline_contrast) * 100.0, 0.0, 100.0))

        # 5. Compactness / Shape Irregularity Factor (0 - 100)
        clamped_circ = float(np.clip(circularity, 0.0, 1.0))
        f_compactness = float((1.0 - clamped_circ) * 100.0)

        # Weighted Sum
        w_area = cfg.risk_weight_area
        w_comp = cfg.risk_weight_complexity
        w_pen = cfg.risk_weight_penumbra
        w_con = cfg.risk_weight_contrast
        w_circ = cfg.risk_weight_compactness
        tot_w = w_area + w_comp + w_pen + w_con + w_circ
        if tot_w <= 0:
            tot_w = 1.0

        raw_score = (
            w_area * f_area +
            w_comp * f_complexity +
            w_pen * f_penumbra +
            w_con * f_contrast +
            w_circ * f_compactness
        ) / tot_w

        score = float(np.clip(round(raw_score, 1), 0.0, 100.0))

        # Neutral attention label determination
        if score < cfg.low_attention_threshold:
            attention = "Low Attention"
        elif score < cfg.moderate_attention_threshold:
            attention = "Moderate Attention"
        else:
            attention = "High Attention"

        factors = DemonstrationRiskFactors(
            area_factor=round(f_area, 1),
            complexity_factor=round(f_complexity, 1),
            penumbra_factor=round(f_penumbra, 1),
            contrast_factor=round(f_contrast, 1),
            compactness_factor=round(f_compactness, 1),
        )

        assumptions = [
            "Visible continuum morphology (area, multiplicity, contrast, circularity) serves as a geometric proxy for magnetic complexity.",
            "Higher physical area and spot multiplicity indicate greater total emerging magnetic flux.",
            "Low isoperimetric circularity proxies convoluted boundary geometry and potential polarity interface shearing.",
            "Darker intensity contrast indicates strong vertical magnetic suppression of photospheric convection.",
            "Score is a normalized additive heuristic index [0-100] constructed solely for educational and comparative analysis.",
        ]

        weights_dict = {
            "area": w_area,
            "complexity": w_comp,
            "penumbra": w_pen,
            "contrast": w_con,
            "compactness": w_circ,
        }

        return DemonstrationRiskAssessment(
            score=score,
            attention_level=attention,
            factors=factors,
            weights=weights_dict,
            assumptions=assumptions,
            disclaimer=DEMONSTRATION_RISK_DISCLAIMER,
        )

    def classify(
        self,
        region: Any,
        solar_radius_px: Optional[float] = None,
    ) -> ClassificationResult:
        """
        Classify a single active region using transparent physical rules and
        compute its demonstration risk assessment.

        Accepts CalibratedActiveRegion, DetectedRegion, or objects with equivalent attributes.
        """
        trace: List[str] = []

        # Extract features with robust fallbacks
        region_id = getattr(region, "id", 1)
        area_uhem = float(getattr(region, "area_uhem", 0.0))
        area_px = int(getattr(region, "projected_area_px", getattr(region, "area_pixels", 0)))
        has_pen = bool(getattr(region, "has_penumbra", False))
        spots = int(getattr(region, "spot_count", 1))
        umbra_uhem = float(getattr(region, "umbra_area_uhem", 0.0))
        penumbra_uhem = float(getattr(region, "penumbra_area_uhem", 0.0))
        contrast = float(getattr(region, "mean_contrast", getattr(region, "contrast", 0.35)))
        circularity = float(getattr(region, "circularity", 0.85))

        # Extract or derive longitudinal extent
        if hasattr(region, "longitudinal_extent_deg"):
            span = float(region.longitudinal_extent_deg)
        elif hasattr(region, "bbox") and solar_radius_px and solar_radius_px > 0:
            bw = region.bbox[2]
            span = float((bw / solar_radius_px) * (180.0 / np.pi))
        else:
            span = 0.0

        # Determine bipolar flag
        if hasattr(region, "is_bipolar"):
            bipolar = bool(region.is_bipolar)
        else:
            bipolar = (spots >= 2) and (span >= self.config.bipolar_min_sep_deg)

        trace.append(f"1. Total physical area: {area_uhem:.1f} uHem (Projected footprint: {area_px} px)")
        trace.append(f"2. Spot count: {spots} distinct sub-element(s) detected")
        trace.append(f"3. Longitudinal extent: {span:.1f}°")
        trace.append(f"4. Penumbra present: {'Yes' if has_pen else 'No'} (Umbra: {umbra_uhem:.1f} uHem, Penumbra: {penumbra_uhem:.1f} uHem)")
        trace.append(f"5. Bipolar configuration: {'Yes' if bipolar else 'No'}")
        trace.append(f"6. Photospheric contrast: {contrast:.3f} | Circularity: {circularity:.3f}")

        # Rule 1: No penumbra present
        if not has_pen:
            if bipolar:
                code = "B"
                name = "Class B: Bipolar Group without Penumbra"
                conf = 0.88
                trace.append("Decision: Multiple spots spanning >= 2.0° with no penumbra -> Class B (Emerging flux group)")
            else:
                code = "A"
                name = "Class A: Unipolar Spot / Pore without Penumbra"
                conf = 0.92
                trace.append("Decision: Single spot or tight pore cluster with no penumbra -> Class A (Pore or decaying remnant)")

        # Rule 2: Penumbra present
        else:
            if not bipolar:
                # Single mature unipolar spot with penumbra
                code = "H"
                name = "Class H: Unipolar Spot with Symmetric Penumbra"
                conf = 0.90
                trace.append("Decision: Single mature spot with well-developed penumbra -> Class H (Stable unipolar spot)")
            else:
                # Bipolar group with penumbra
                if span <= self.config.class_d_max_span_deg:
                    # Check if penumbra is asymmetric (mostly on one end)
                    if area_uhem > 0 and penumbra_uhem < self.config.class_c_penumbra_fraction * area_uhem:
                        code = "C"
                        name = "Class C: Bipolar Group with Penumbra on One End"
                        conf = 0.85
                        trace.append(
                            f"Decision: Bipolar group (span {span:.1f}° <= {self.config.class_d_max_span_deg}°) "
                            f"with asymmetric penumbra (< {self.config.class_c_penumbra_fraction*100:.0f}% area) -> Class C"
                        )
                    else:
                        code = "D"
                        name = "Class D: Moderate Bipolar Group with Penumbra at Both Ends"
                        conf = 0.91
                        trace.append(
                            f"Decision: Bipolar group with penumbra on both ends (span {span:.1f}° <= {self.config.class_d_max_span_deg}°) -> Class D"
                        )

                elif span <= self.config.class_e_max_span_deg:
                    code = "E"
                    name = "Class E: Large Extended Bipolar Group with Complex Penumbra"
                    conf = 0.89
                    trace.append(
                        f"Decision: Extended bipolar group (span {span:.1f}° between {self.config.class_d_max_span_deg}° "
                        f"and {self.config.class_e_max_span_deg}°) -> Class E"
                    )

                else:
                    code = "F"
                    name = "Class F: Very Large, Highly Complex Bipolar Sunspot Group"
                    conf = 0.94
                    trace.append(
                        f"Decision: Major extended bipolar group (span {span:.1f}° > {self.config.class_e_max_span_deg}° "
                        f"or physical area >= 1000 uHem) -> Class F"
                    )

        # Retrieve documented class information
        class_info = self.get_class_info(code)

        # Compute heuristic demonstration risk score
        risk = self.calculate_demonstration_risk(
            area_uhem=area_uhem,
            spot_count=spots,
            span_deg=span,
            has_penumbra=has_pen,
            is_bipolar=bipolar,
            class_code=code,
            contrast=contrast,
            circularity=circularity,
            penumbra_area_uhem=penumbra_uhem,
        )

        trace.append(
            f"Demonstration Risk Assessment: Score {risk.score:.1f}/100 -> '{risk.attention_level}' "
            f"(Area: {risk.factors.area_factor:.1f}, Comp: {risk.factors.complexity_factor:.1f}, "
            f"Pen: {risk.factors.penumbra_factor:.1f}, Con: {risk.factors.contrast_factor:.1f}, "
            f"Circ: {risk.factors.compactness_factor:.1f})"
        )

        return ClassificationResult(
            region_id=region_id,
            class_code=code,
            class_name=name,
            class_info=class_info,
            attention_level=risk.attention_level,
            flare_potential=risk.attention_level,
            demonstration_risk=risk,
            rule_trace=trace,
            confidence=conf,
        )

    def classify_all(
        self,
        regions: List[Any],
        solar_radius_px: Optional[float] = None,
    ) -> List[ClassificationResult]:
        """Classify a list of active regions."""
        return [self.classify(r, solar_radius_px=solar_radius_px) for r in regions]

