"""
SolarVision Classical Computer Vision Preprocessing Module.
Prepares solar continuum imagery for robust sunspot segmentation.

Key Operations:
1. Safe input loading and grayscale normalization.
2. Solar disk detection and background isolation (zeroing off-limb deep space).
3. Photospheric limb darkening correction (flat-fielding via Eddington model).
4. Edge-preserving noise reduction (Bilateral filtering to smooth granulation).
5. Controlled contrast enhancement (strictly bounded CLAHE or linear quiet-Sun scaling).
6. Morphological dark feature isolation (Black-Hat transformation).
7. Strict preservation of dimensions and coordinates (1:1 spatial mapping).
8. Generation of comprehensive before-and-after comparison panels.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.config import PreprocessingConfig, SolarVisionConfig, load_config
from src.disk_detector import SolarDiskDetector, SolarDiskGeometry
from src.limb_darkening import LimbDarkeningCorrector, LimbCorrectionResult


class PreprocessingError(Exception):
    """Base exception for preprocessing failures."""
    pass


class InvalidImageInputError(PreprocessingError):
    """Raised when an input image is None, empty, or improperly formatted."""
    pass


@dataclass
class PreprocessingStageVisuals:
    """Visual snapshots of intermediate preprocessing stages."""
    raw_input: np.ndarray             # Original raw image (BGR or grayscale)
    grayscale: np.ndarray             # Grayscale representation
    disk_isolated: np.ndarray         # Disk masked with background zeroed
    flat_fielded: np.ndarray          # Photospheric limb darkening flattened
    denoised: np.ndarray              # Bilateral / edge-preserving noise reduced
    enhanced: np.ndarray              # Contrast-enhanced normalized image
    blackhat_feature_map: np.ndarray  # Morphological Black-Hat dark feature map
    composite_panel: np.ndarray       # Multi-panel before-and-after comparison


@dataclass
class PreprocessingResult:
    """Complete preprocessed output ready for active region segmentation."""
    preprocessed_image: np.ndarray    # Final clean preprocessed uint8 image
    solar_disk: SolarDiskGeometry     # Solar disk geometry (center, radius, mask)
    limb_result: LimbCorrectionResult # Limb darkening correction maps and quiet-Sun level
    visuals: PreprocessingStageVisuals
    quiet_sun_level: float            # Normalized quiet-Sun base intensity
    original_shape: Tuple[int, int]   # Original (H, W) preserved exactly
    metadata: Dict[str, Any]          # Summary of operations applied

    @property
    def annotated_disk_image(self) -> np.ndarray:
        """Render solar disk boundary and center crosshair overlay."""
        if hasattr(self, "visuals") and self.visuals and hasattr(self.visuals, "raw_input") and self.visuals.raw_input is not None:
            base = self.visuals.raw_input.copy()
        elif hasattr(self, "preprocessed_image") and self.preprocessed_image is not None:
            base = self.preprocessed_image.copy()
        else:
            base = np.zeros((1024, 1024, 3), dtype=np.uint8)

        if base.ndim == 2:
            vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        else:
            vis = base.copy()

        if hasattr(self, "solar_disk") and self.solar_disk and self.solar_disk.is_valid:
            cx = int(round(self.solar_disk.center_x))
            cy = int(round(self.solar_disk.center_y))
            r = int(round(self.solar_disk.radius))
            # Photospheric limb circle (Green)
            cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
            # Effective processing margin kappa = 0.985 (Cyan)
            cv2.circle(vis, (cx, cy), max(1, int(round(r * 0.985))), (255, 255, 0), 1)
            # Center marker (Amber)
            cv2.drawMarker(vis, (cx, cy), (0, 165, 255), cv2.MARKER_CROSS, 24, 2)
        return vis

    @property
    def gray_image(self) -> np.ndarray:
        """Grayscale image representation."""
        if hasattr(self, "visuals") and self.visuals and hasattr(self.visuals, "grayscale") and self.visuals.grayscale is not None:
            return self.visuals.grayscale
        return self.preprocessed_image

    @property
    def clahe_enhanced(self) -> np.ndarray:
        """Contrast enhanced representation."""
        if hasattr(self, "visuals") and self.visuals and hasattr(self.visuals, "enhanced") and self.visuals.enhanced is not None:
            return self.visuals.enhanced
        return self.preprocessed_image

    @property
    def blackhat_dark_map(self) -> np.ndarray:
        """Morphological dark feature map."""
        if hasattr(self, "visuals") and self.visuals and hasattr(self.visuals, "blackhat_feature_map") and self.visuals.blackhat_feature_map is not None:
            return self.visuals.blackhat_feature_map
        return np.zeros_like(self.preprocessed_image)



class SolarImagePreprocessor:
    """
    Modular, classical Computer Vision preprocessor for solar disk images.
    Preserves exact spatial coordinates while optimizing signal-to-noise ratio.
    """

    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        full_config: Optional[SolarVisionConfig] = None,
    ):
        base_cfg = full_config or load_config()
        self.config = config or base_cfg.preprocessing
        self.disk_detector = SolarDiskDetector(base_cfg.disk_detection)
        self.limb_corrector = LimbDarkeningCorrector(base_cfg.limb_darkening)

    def load_image(self, source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
        """
        Safely load an image from a filepath, raw bytes, or existing numpy array.

        Args:
            source: Path to image, raw byte payload, or numpy array.

        Returns:
            Decoded numpy array (2D grayscale or 3D BGR).

        Raises:
            InvalidImageInputError: If source cannot be loaded or decoded.
        """
        if source is None:
            raise InvalidImageInputError("Input image source cannot be None.")

        if isinstance(source, np.ndarray):
            if source.size == 0 or source.ndim not in (2, 3):
                raise InvalidImageInputError(f"Invalid numpy array input with shape {source.shape}.")
            return source.copy()

        if isinstance(source, bytes):
            if len(source) == 0:
                raise InvalidImageInputError("Input bytes payload is empty.")
            nparr = np.frombuffer(source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                raise InvalidImageInputError("Failed to decode image from byte buffer.")
            return img

        if isinstance(source, (str, Path)):
            p = Path(source)
            if not p.exists() or not p.is_file():
                raise InvalidImageInputError(f"Image file does not exist: {p}")
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                raise InvalidImageInputError(f"Failed to read image file via OpenCV: {p}")
            return img

        raise InvalidImageInputError(f"Unsupported image source type: {type(source)}")

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to single-channel 8-bit grayscale using standard luminance weighting.
        Preserves 2D spatial dimensions (H, W).
        """
        if image.ndim == 2:
            return image.copy()
        elif image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise InvalidImageInputError(f"Unexpected image dimensions: {image.ndim}D")

    def isolate_solar_disk(
        self, gray_image: np.ndarray, disk: SolarDiskGeometry
    ) -> np.ndarray:
        """
        Zero out off-limb sky and telemetry borders outside the effective solar disk.
        Does not clip or translate pixel coordinates, strictly maintaining spatial registry.
        """
        isolated = np.zeros_like(gray_image)
        isolated[disk.effective_mask > 0] = gray_image[disk.effective_mask > 0]
        return isolated

    def apply_noise_reduction(
        self, image: np.ndarray, effective_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply edge-preserving smoothing to reduce photospheric granulation noise
        (~2-4 pixel convective cells) while preserving sharp umbra/penumbra boundaries.

        Method:
        - 'bilateral': cv2.bilateralFilter smooths low-amplitude intensity variations
          (granulation standard deviation ~3-5% of quiet Sun) while maintaining high-gradient
          sunspot borders.
        - 'gaussian': Gentle Gaussian blur (sigma ~0.8) as an alternative.
        - 'median': 3x3 median filter for impulse noise.
        """
        method = self.config.denoise_method.lower()

        if method == "bilateral":
            d = self.config.bilateral_d
            sc = self.config.bilateral_sigma_color
            ss = self.config.bilateral_sigma_space
            denoised = cv2.bilateralFilter(image, d, sc, ss)

        elif method == "gaussian":
            k = self.config.gaussian_kernel_size
            if k % 2 == 0:
                k += 1
            denoised = cv2.GaussianBlur(image, (k, k), self.config.gaussian_sigma)

        elif method == "median":
            denoised = cv2.medianBlur(image, 3)

        else:
            denoised = image.copy()

        # Zero out regions outside effective mask
        denoised[effective_mask == 0] = 0
        return denoised

    def apply_contrast_enhancement(
        self, image: np.ndarray, effective_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply controlled contrast enhancement strictly within the solar disk.
        Avoids over-processing to prevent creating artificial sunspots or
        artificially inflating solar granulation.

        Method:
        - 'clahe': Contrast Limited Adaptive Histogram Equalization with a strict
          conservative clipLimit (e.g. 1.8) and tileGridSize (8, 8).
        - 'linear_stretch': Percentile-based linear normalization.
        """
        method = self.config.contrast_method.lower()

        if method == "clahe":
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=(self.config.clahe_tile_grid_size, self.config.clahe_tile_grid_size),
            )
            enhanced = clahe.apply(image)

        elif method == "linear_stretch":
            valid_pixels = image[effective_mask > 0]
            if len(valid_pixels) > 0:
                p_min = float(np.percentile(valid_pixels, 1.0))
                p_max = float(np.percentile(valid_pixels, 99.0))
                denom = max(p_max - p_min, 1.0)
                scaled = np.clip((image.astype(np.float32) - p_min) / denom * 255.0, 0, 255)
                enhanced = scaled.astype(np.uint8)
            else:
                enhanced = image.copy()

        else:
            enhanced = image.copy()

        enhanced[effective_mask == 0] = 0
        return enhanced

    def apply_blackhat_transformation(
        self, image: np.ndarray, effective_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply morphological Black-Hat transform:
        BlackHat(I) = Closing(I) - I

        Physical & Mathematical Justification:
        Sunspots are localized dark depressions on a brighter quiet-Sun background.
        Closing the image with an elliptical structuring element fills in dark structures
        smaller than the kernel (pores, umbrae, penumbrae). Subtracting the original image
        isolates these dark features as bright peaks against a flat background,
        greatly aiding subsequent boundary localization without altering the original signal.
        """
        k = self.config.blackhat_kernel_size
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

        # Cleanup residual 1-pixel noise using morphological opening
        cleanup_k = self.config.morph_cleanup_kernel_size
        if cleanup_k % 2 == 0:
            cleanup_k += 1
        c_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cleanup_k, cleanup_k))
        blackhat_clean = cv2.morphologyEx(blackhat, cv2.MORPH_OPEN, c_kernel)

        blackhat_clean[effective_mask == 0] = 0
        return blackhat_clean

    def build_comparison_panel(self, visuals: PreprocessingStageVisuals) -> np.ndarray:
        """
        Generate a comprehensive multi-panel (2x3) visual grid showing before-and-after
        preprocessing steps with labeled scientific headers.
        """
        h, w = visuals.grayscale.shape
        target_w = 400
        target_h = int(h * (target_w / w))

        def prep_thumb(img: np.ndarray, title: str) -> np.ndarray:
            if img.ndim == 2:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                bgr = img.copy()
            resized = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            # Add header banner
            banner = np.zeros((32, target_w, 3), dtype=np.uint8)
            cv2.putText(
                banner, title, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
            )
            return np.vstack([banner, resized])

        t1 = prep_thumb(visuals.raw_input, "1. Raw Input Image")
        t2 = prep_thumb(visuals.disk_isolated, "2. Solar Disk & Masked Sky")
        t3 = prep_thumb(visuals.flat_fielded, "3. Photospheric Flat-Field")
        t4 = prep_thumb(visuals.denoised, "4. Bilateral Denoised")
        t5 = prep_thumb(visuals.enhanced, "5. Controlled Contrast CLAHE")
        t6 = prep_thumb(visuals.blackhat_feature_map, "6. Black-Hat Dark Features")

        row1 = np.hstack([t1, t2, t3])
        row2 = np.hstack([t4, t5, t6])
        grid = np.vstack([row1, row2])

        return grid

    def process(self, source: Union[str, Path, bytes, np.ndarray]) -> PreprocessingResult:
        """
        Execute the full classical Computer Vision preprocessing pipeline on a solar image.

        Args:
            source: Image path, bytes, or numpy array.

        Returns:
            PreprocessingResult containing clean preprocessed image, geometry, and visuals.
        """
        # Step 1: Load and validate input
        raw_bgr = self.load_image(source)
        h, w = raw_bgr.shape[:2]

        # Step 2: Grayscale conversion
        gray = self.convert_to_grayscale(raw_bgr)

        # Step 3: Solar Disk Localization
        disk = self.disk_detector.detect(gray)

        # Step 4: Background Isolation (Zero off-limb cosmic sky)
        disk_isolated = self.isolate_solar_disk(gray, disk)

        # Step 5: Photospheric Limb Darkening Correction (Flat-fielding)
        limb_res = self.limb_corrector.correct(disk_isolated, disk)
        flat_uint8 = limb_res.flattened_uint8

        # Step 6: Edge-Preserving Noise Reduction
        denoised = self.apply_noise_reduction(flat_uint8, disk.effective_mask)

        # Step 7: Controlled Contrast Enhancement
        enhanced = self.apply_contrast_enhancement(denoised, disk.effective_mask)

        # Step 8: Morphological Dark Feature Isolation (Black-Hat)
        if self.config.enable_blackhat:
            blackhat = self.apply_blackhat_transformation(enhanced, disk.effective_mask)
        else:
            blackhat = np.zeros_like(enhanced)

        # Step 9: Assemble Visual Stages
        visuals = PreprocessingStageVisuals(
            raw_input=raw_bgr,
            grayscale=gray,
            disk_isolated=disk_isolated,
            flat_fielded=flat_uint8,
            denoised=denoised,
            enhanced=enhanced,
            blackhat_feature_map=blackhat,
            composite_panel=np.array([]),  # Filled below
        )
        visuals.composite_panel = self.build_comparison_panel(visuals)

        # Final preprocessed image for segmentation is the clean, flattened, denoised image
        preprocessed = denoised.copy()

        metadata = {
            "original_width": w,
            "original_height": h,
            "disk_center_x": disk.center_x,
            "disk_center_y": disk.center_y,
            "disk_radius": disk.radius,
            "disk_confidence": disk.confidence,
            "quiet_sun_intensity": limb_res.quiet_sun_intensity,
            "denoise_method": self.config.denoise_method,
            "contrast_method": self.config.contrast_method,
            "blackhat_enabled": self.config.enable_blackhat,
        }

        return PreprocessingResult(
            preprocessed_image=preprocessed,
            solar_disk=disk,
            limb_result=limb_res,
            visuals=visuals,
            quiet_sun_level=limb_res.quiet_sun_intensity,
            original_shape=(h, w),
            metadata=metadata,
        )
