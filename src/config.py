"""
Configuration loader and validator for SolarVision.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class DataSourceConfig:
    primary_url: str = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMIIC.jpg"
    backup_urls: List[str] = field(default_factory=lambda: [
        "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_512_HMIIC.jpg",
        "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMII.jpg"
    ])
    archive_url_template: str = "https://sdo.gsfc.nasa.gov/assets/img/browse/{year}/{month:02d}/{day:02d}/{year}{month:02d}{day:02d}_{hour:02d}0000_1024_HMIIC.jpg"
    timeout_seconds: int = 15
    max_retries: int = 3
    retry_backoff_factor: float = 1.5
    user_agent: str = "SolarVision-Research-Client/1.1"
    raw_dir: str = "data/raw"
    sample_dir: str = "data/sample_images"
    avoid_duplicates: bool = True
    min_image_dimension: int = 256
    min_file_size_bytes: int = 8192


@dataclass
class DiskDetectionConfig:
    blur_kernel_size: int = 5
    canny_threshold1: int = 30
    canny_threshold2: int = 100
    disk_threshold_ratio: float = 0.18
    min_radius_ratio: float = 0.25
    max_radius_ratio: float = 0.52
    edge_margin_fraction: float = 0.02


@dataclass
class PreprocessingConfig:
    denoise_method: str = "bilateral"
    bilateral_d: int = 5
    bilateral_sigma_color: float = 25.0
    bilateral_sigma_space: float = 3.0
    gaussian_kernel_size: int = 3
    gaussian_sigma: float = 0.8
    contrast_method: str = "clahe"
    clahe_clip_limit: float = 1.8
    clahe_tile_grid_size: int = 8
    enable_blackhat: bool = True
    blackhat_kernel_size: int = 15
    morph_cleanup_kernel_size: int = 3


@dataclass
class LimbDarkeningConfig:
    model: str = "linear_u"
    u_coefficient: float = 0.60
    quiet_sun_percentile: float = 90.0
    inner_disk_sample_radius: float = 0.60


@dataclass
class SegmentationConfig:
    umbra_threshold_factor: float = 0.58
    penumbra_threshold_factor: float = 0.88
    min_sunspot_area_pixels: int = 12
    max_sunspot_area_pixels: int = 150000
    min_contrast: float = 0.08
    morphology_kernel_size: int = 3
    clustering_distance_deg: float = 5.0


@dataclass
class SolarPhysicsConfig:
    solar_radius_km: float = 696340.0
    diff_rot_a: float = 14.713
    diff_rot_b: float = -2.396
    diff_rot_c: float = -1.787
    b0_angle_deg: float = 0.0
    p_angle_deg: float = 0.0


@dataclass
class ClassificationConfig:
    pore_max_area_uhem: float = 25.0
    mature_spot_min_area_uhem: float = 50.0
    bipolar_min_sep_deg: float = 2.0
    class_d_max_span_deg: float = 10.0
    class_e_max_span_deg: float = 15.0
    class_c_penumbra_fraction: float = 0.25

    # Heuristic Demonstration Risk Scoring (NOT an operational flare prediction model)
    risk_weight_area: float = 0.30
    risk_weight_complexity: float = 0.25
    risk_weight_penumbra: float = 0.20
    risk_weight_contrast: float = 0.15
    risk_weight_compactness: float = 0.10

    area_baseline_uhem: float = 1000.0
    contrast_baseline: float = 0.60
    low_attention_threshold: float = 35.0
    moderate_attention_threshold: float = 70.0


@dataclass
class TrackingConfig:
    max_matching_dist_deg: float = 7.0       # Max angular residual for valid kinematic match
    max_lat_diff_deg: float = 4.0            # Max latitude drift
    min_area_ratio: float = 0.15             # Min area ratio between consecutive observations
    max_area_ratio: float = 6.0              # Max area ratio between consecutive observations
    max_prediction_gap_days: float = 3.5     # Max gap before terminating track continuity
    limb_cutoff_deg: float = 75.0            # Longitude threshold for western limb rotation
    east_limb_threshold_deg: float = -65.0   # Longitude threshold for eastern limb appearance
    track_id_prefix: str = "TRK"             # Prefix for assigned persistent tracks


@dataclass
class StorageConfig:
    database_path: str = "data/solarvision.db"
    sample_data_dir: str = "data/sample_images"
    raw_data_dir: str = "data/raw"
    catalog_index_path: str = "data/ingestion_catalog.json"


@dataclass
class SolarVisionConfig:
    pipeline_name: str = "SolarVision"
    version: str = "1.1.0"
    log_level: str = "INFO"
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    disk_detection: DiskDetectionConfig = field(default_factory=DiskDetectionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    limb_darkening: LimbDarkeningConfig = field(default_factory=LimbDarkeningConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    solar_physics: SolarPhysicsConfig = field(default_factory=SolarPhysicsConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


def load_config(config_path: Optional[str] = None) -> SolarVisionConfig:
    """
    Load configuration from YAML file or return default configuration.
    """
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return SolarVisionConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    pipe = data.get("pipeline", {})
    ds = data.get("data_source", {})
    dd = data.get("disk_detection", {})
    pp = data.get("preprocessing", {})
    ld = data.get("limb_darkening", {})
    seg = data.get("segmentation", {})
    sp = data.get("solar_physics", {})
    clf = data.get("classification", {})
    trk = data.get("tracking", {})
    st = data.get("storage", {})

    return SolarVisionConfig(
        pipeline_name=pipe.get("name", "SolarVision"),
        version=pipe.get("version", "1.1.0"),
        log_level=pipe.get("log_level", "INFO"),
        data_source=DataSourceConfig(**ds),
        disk_detection=DiskDetectionConfig(**dd),
        preprocessing=PreprocessingConfig(**pp),
        limb_darkening=LimbDarkeningConfig(**ld),
        segmentation=SegmentationConfig(**seg),
        solar_physics=SolarPhysicsConfig(**sp),
        classification=ClassificationConfig(**clf),
        tracking=TrackingConfig(**trk),
        storage=StorageConfig(**st),
    )
