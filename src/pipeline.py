"""
SolarVision Unified Data Processing & Ingestion Pipeline.
Coordinates ingestion, preprocessing, disk localization, limb darkening compensation,
sunspot segmentation, feature extraction, McIntosh classification, demonstration risk scoring,
multi-frame differential rotation tracking, and SQLite persistence.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from src.classifier import ClassificationResult, McIntoshClassifier
from src.config import SolarVisionConfig, load_config
from src.database import SolarDatabase
from src.detector import DetectionOutput, DetectedRegion, SunspotDetector
from src.disk_detector import SolarDiskDetector, SolarDiskGeometry
from src.feature_extractor import CalibratedActiveRegion, FeatureExtractor
from src.limb_darkening import LimbCorrectionResult, LimbDarkeningCorrector
from src.preprocessor import PreprocessingResult, SolarImagePreprocessor
from src.segmentation import SegmentationResult, SunspotSegmenter
from src.solar_data import ImageMetadata, IngestionResult, SolarDataIngestor
from src.tracker import ActiveRegionTracker, TrackHistory, TrackedObservation

logger = logging.getLogger("SolarVision.Pipeline")


def make_empty_disk_geometry() -> SolarDiskGeometry:
    """Create a placeholder invalid SolarDiskGeometry."""
    empty_m = np.zeros((1, 1), dtype=np.uint8)
    return SolarDiskGeometry(
        center_x=0.0,
        center_y=0.0,
        radius=0.0,
        mask=empty_m,
        effective_mask=empty_m,
        confidence=0.0,
        is_valid=False,
    )


@dataclass
class PipelineResult:
    """Standardized output of the end-to-end solar computer vision pipeline."""
    success: bool
    is_cached: bool                                   # True if loaded from existing database record (deduplication)
    observation_id: int
    image_id: Optional[int]
    observation_time: datetime
    image_metadata: Optional[ImageMetadata]
    solar_disk: SolarDiskGeometry
    limb_result: Optional[LimbCorrectionResult]
    segmentation: Optional[SegmentationResult]
    regions: List[CalibratedActiveRegion]
    detected_regions: List[DetectedRegion]
    classifications: List[ClassificationResult]
    detection_output: Optional[DetectionOutput]
    preprocessing_result: Optional[PreprocessingResult] = None
    annotated_image: Optional[np.ndarray] = None
    tracked_observations: List[TrackedObservation] = field(default_factory=list)
    error_message: Optional[str] = None

    @property
    def quiet_sun_intensity(self) -> float:
        """Normalized base intensity of the quiet-Sun photosphere."""
        return self.limb_result.quiet_sun_intensity if self.limb_result else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "is_cached": self.is_cached,
            "observation_id": self.observation_id,
            "image_id": self.image_id,
            "observation_time": self.observation_time.isoformat(),
            "active_region_count": len(self.regions),
            "is_spotless": len(self.regions) == 0,
            "disk_radius": self.solar_disk.radius if self.solar_disk else 0.0,
            "quiet_sun_intensity": self.limb_result.quiet_sun_intensity if self.limb_result else 0.0,
            "error_message": self.error_message,
        }


@dataclass
class SequencePipelineResult:
    """Output for batch or multi-day historical solar sequence processing."""
    success: bool
    observation_count: int
    results: List[PipelineResult]
    tracks: Dict[str, TrackHistory]
    total_active_regions_detected: int
    unique_tracks_count: int
    summary_dataframe: pd.DataFrame
    trajectory_dataframe: pd.DataFrame
    error_message: Optional[str] = None


class SolarVisionPipeline:
    """
    Unified end-to-end processing pipeline connecting data ingestion,
    classical Computer Vision algorithms, classification, tracking, and database storage.
    """

    def __init__(
        self,
        config: Optional[SolarVisionConfig] = None,
        db: Optional[SolarDatabase] = None,
        tracker: Optional[ActiveRegionTracker] = None,
    ):
        self.config = config or load_config()
        self.db = db or SolarDatabase(self.config.storage.database_path)
        self.tracker = tracker or ActiveRegionTracker(
            physics_config=self.config.solar_physics,
            tracking_config=self.config.tracking,
        )

        # Initialize modular stages
        self.ingestor = SolarDataIngestor(
            config=self.config.data_source,
            catalog_index_path=self.config.storage.catalog_index_path,
        )
        self.preprocessor = SolarImagePreprocessor(
            config=self.config.preprocessing,
            full_config=self.config,
        )
        self.disk_detector = SolarDiskDetector(self.config.disk_detection)
        self.limb_corrector = LimbDarkeningCorrector(self.config.limb_darkening)
        self.segmenter = SunspotSegmenter(self.config.segmentation)
        self.feature_extractor = FeatureExtractor(self.config.solar_physics)
        self.classifier = McIntoshClassifier(self.config.classification)
        self.sunspot_detector = SunspotDetector(
            config=self.config.segmentation,
            physics_config=self.config.solar_physics,
            full_config=self.config,
        )

    def _parse_timestamp_from_path(self, filepath: Path) -> datetime:
        """Attempt to extract an authentic observation timestamp from standard NASA SDO filename."""
        name = filepath.stem
        # Pattern 1: YYYYMMDD_HHMMSS
        m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", name)
        if m:
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)), int(m.group(6))
            )

        # Pattern 2: YYYYMMDD
        m = re.search(r"(\d{4})(\d{2})(\d{2})", name)
        if m:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), 0, 0, 0)

        # Fallback to file modification time
        try:
            mtime = filepath.stat().st_mtime
            return datetime.fromtimestamp(mtime)
        except Exception:
            return datetime.now()

    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        observation_time: Optional[datetime] = None,
        metadata: Optional[ImageMetadata] = None,
        reprocess: bool = False,
    ) -> PipelineResult:
        """
        Process a single solar image through the entire Computer Vision pipeline
        and store all structured results in SQLite.

        Args:
            image_input: File path (str/Path) or raw image array (np.ndarray)
            observation_time: Explicit observation timestamp (UTC)
            metadata: Optional pre-existing ImageMetadata sidecar
            reprocess: If True, bypasses database cache and recalculates

        Returns:
            PipelineResult containing calibrated features, classification, tracking, and DB IDs
        """
        try:
            # 1. Resolve image array and metadata
            img_bgr: Optional[np.ndarray] = None
            filename = "solar_image_input.jpg"
            current_metadata: Optional[ImageMetadata] = metadata

            if isinstance(image_input, (str, Path)):
                filepath = Path(image_input)
                if not filepath.exists():
                    return PipelineResult(
                        success=False,
                        is_cached=False,
                        observation_id=-1,
                        image_id=None,
                        observation_time=observation_time or datetime.now(),
                        image_metadata=None,
                        solar_disk=make_empty_disk_geometry(),
                        limb_result=None,
                        segmentation=None,
                        regions=[],
                        detected_regions=[],
                        classifications=[],
                        detection_output=None,
                        error_message=f"Image file does not exist: {filepath}",
                    )

                filename = filepath.name
                if observation_time is None:
                    observation_time = self._parse_timestamp_from_path(filepath)

                ingest_res = self.ingestor.load_local_image(filepath)
                if not ingest_res.success or ingest_res.image is None:
                    return PipelineResult(
                        success=False,
                        is_cached=False,
                        observation_id=-1,
                        image_id=None,
                        observation_time=observation_time or datetime.now(),
                        image_metadata=None,
                        solar_disk=make_empty_disk_geometry(),
                        limb_result=None,
                        segmentation=None,
                        regions=[],
                        detected_regions=[],
                        classifications=[],
                        detection_output=None,
                        error_message=f"Failed to load image: {ingest_res.error_message}",
                    )

                img_bgr = ingest_res.image
                current_metadata = ingest_res.metadata

            elif isinstance(image_input, np.ndarray):
                img_bgr = image_input
                if observation_time is None:
                    observation_time = datetime.now()

                if current_metadata is None:
                    # Calculate SHA-256 checksum from image buffer
                    h = hashlib.sha256(img_bgr.tobytes()).hexdigest()
                    h_px, w_px = img_bgr.shape[:2]
                    channels = img_bgr.shape[2] if img_bgr.ndim == 3 else 1
                    current_metadata = ImageMetadata(
                        filename=filename,
                        filepath="",
                        source_url="in_memory_array",
                        source_type="IN_MEMORY_STREAM",
                        observatory="NASA Solar Dynamics Observatory (SDO)",
                        instrument="Helioseismic and Magnetic Imager (HMI)",
                        wavelength_channel="Fe I 6173 Å Visible Continuum",
                        download_timestamp_utc=observation_time.isoformat(),
                        source_last_modified=None,
                        etag=None,
                        sha256=h,
                        file_size_bytes=img_bgr.nbytes,
                        image_width=w_px,
                        image_height=h_px,
                        channels=channels,
                        is_authentic_real_data=True,
                    )
            else:
                raise ValueError(f"Unsupported image_input type: {type(image_input)}")

            # 2. Check Deduplication & Idempotency
            image_id: Optional[int] = None
            if current_metadata:
                existing_img = self.db.get_image_by_hash(current_metadata.sha256)
                if existing_img:
                    image_id = existing_img["id"]
                    if not reprocess:
                        # Check if observation session already exists
                        existing_obs = self.db.get_observation_by_image_id(image_id)
                        if existing_obs:
                            logger.info(f"Returning cached observation #{existing_obs['id']} for SHA-256 {current_metadata.sha256[:12]}")
                            db_regions = self.db.get_regions_for_observation(existing_obs["id"])
                            cached_disk = SolarDiskGeometry(
                                center_x=existing_obs["disk_center_x"],
                                center_y=existing_obs["disk_center_y"],
                                radius=existing_obs["disk_radius"],
                                mask=np.zeros((1, 1), dtype=np.uint8),
                                effective_mask=np.zeros((1, 1), dtype=np.uint8),
                                confidence=existing_obs["disk_confidence"],
                                is_valid=True,
                            )
                            return PipelineResult(
                                success=True,
                                is_cached=True,
                                observation_id=existing_obs["id"],
                                image_id=image_id,
                                observation_time=datetime.fromisoformat(existing_obs["timestamp"]),
                                image_metadata=current_metadata,
                                solar_disk=cached_disk,
                                limb_result=None,
                                segmentation=None,
                                regions=[],
                                detected_regions=[],
                                classifications=[],
                                detection_output=None,
                                error_message=None,
                            )

            # 3. Execute Core Computer Vision Pipeline Stages
            prep_res = self.preprocessor.process(img_bgr)
            disk = prep_res.solar_disk
            limb = prep_res.limb_result

            # Segmentation & Feature Calibration
            seg = self.segmenter.segment(limb, disk)
            calibrated_regions = self.feature_extractor.extract_features(seg.regions, disk)

            # McIntosh Classification & Demonstration Risk Scoring
            classifications = self.classifier.classify_all(calibrated_regions)

            # Morphological Sunspot Detection Overlay
            det_output = self.sunspot_detector.detect(
                img_bgr,
                disk=disk,
                limb_result=limb,
            )

            # 4. Multi-Day Kinematic Tracking Association
            tracked_obs = self.tracker.track_observation(calibrated_regions, observation_time)
            tracking_ids = [obs.tracking_id for obs in tracked_obs]

            # 5. SQLite Data Persistence
            if current_metadata:
                image_id = self.db.save_image_metadata(current_metadata)

            obs_id = self.db.save_observation(
                filename=filename,
                timestamp=observation_time,
                center_x=disk.center_x,
                center_y=disk.center_y,
                radius=disk.radius,
                quiet_sun_intensity=limb.quiet_sun_intensity,
                regions=calibrated_regions,
                classifications=classifications,
                tracking_ids=tracking_ids,
                image_id=image_id,
                disk_confidence=disk.confidence,
                is_spotless=det_output.is_spotless,
            )

            # Persist updated track histories
            for t_id in tracking_ids:
                th = self.tracker.get_track(t_id)
                if th:
                    self.db.save_track_history(th, observation_id=obs_id)

            return PipelineResult(
                success=True,
                is_cached=False,
                observation_id=obs_id,
                image_id=image_id,
                observation_time=observation_time,
                image_metadata=current_metadata,
                solar_disk=disk,
                limb_result=limb,
                segmentation=seg,
                regions=calibrated_regions,
                detected_regions=det_output.regions,
                classifications=classifications,
                detection_output=det_output,
                preprocessing_result=prep_res,
                annotated_image=det_output.annotated_image,
                tracked_observations=tracked_obs,
                error_message=None,
            )

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return PipelineResult(
                success=False,
                is_cached=False,
                observation_id=-1,
                image_id=None,
                observation_time=observation_time or datetime.now(),
                image_metadata=metadata,
                solar_disk=make_empty_disk_geometry(),
                limb_result=None,
                segmentation=None,
                regions=[],
                detected_regions=[],
                classifications=[],
                detection_output=None,
                error_message=str(e),
            )

    def process_sequence(
        self,
        image_inputs: List[Union[str, Path, np.ndarray]],
        timestamps: Optional[List[datetime]] = None,
        reprocess: bool = False,
    ) -> SequencePipelineResult:
        """
        Process a multi-temporal sequence of solar observations chronologically,
        tracking active regions across observations and persisting all histories in SQLite.

        Args:
            image_inputs: Chronological list of image paths or image arrays
            timestamps: Optional explicit timestamps corresponding to each image
            reprocess: If True, bypasses database cache and recalculates

        Returns:
            SequencePipelineResult containing frame results, tracking histories, and dataframes
        """
        if not image_inputs:
            return SequencePipelineResult(
                success=False,
                observation_count=0,
                results=[],
                tracks={},
                total_active_regions_detected=0,
                unique_tracks_count=0,
                summary_dataframe=pd.DataFrame(),
                trajectory_dataframe=pd.DataFrame(),
                error_message="Empty image sequence provided",
            )

        self.tracker.reset()
        results: List[PipelineResult] = []
        total_ars = 0

        for idx, img_in in enumerate(image_inputs):
            obs_t = timestamps[idx] if (timestamps and idx < len(timestamps)) else None
            res = self.process_image(img_in, observation_time=obs_t, reprocess=reprocess)
            results.append(res)
            if res.success:
                total_ars += len(res.regions)

        summary_df = self.tracker.to_dataframe()
        traj_df = self.tracker.get_trajectory_dataframe()

        return SequencePipelineResult(
            success=all(r.success for r in results),
            observation_count=len(results),
            results=results,
            tracks=dict(self.tracker.tracks),
            total_active_regions_detected=total_ars,
            unique_tracks_count=len(self.tracker.tracks),
            summary_dataframe=summary_df,
            trajectory_dataframe=traj_df,
            error_message=None,
        )
