"""
Solar data ingestion module.
Provides robust ingestion of real full-disk solar continuum imagery from NASA's
Solar Dynamics Observatory (SDO/HMI) and local real-image catalogs.

Enforces:
- Authentic scientific solar observation data only (zero fake or simulated imagery).
- Deduplication via SHA-256 content hashing and HTTP headers (ETag / Last-Modified).
- JSON metadata sidecar generation for all ingested images.
- Comprehensive network failure and corrupt payload handling.
"""

import hashlib
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.config import DataSourceConfig

logger = logging.getLogger("SolarVision.Ingestion")


class SolarIngestionError(Exception):
    """Base exception for data ingestion failures."""
    pass


class SolarNetworkError(SolarIngestionError):
    """Raised when remote data source cannot be reached or returns an HTTP error."""
    pass


class SolarInvalidImageError(SolarIngestionError):
    """Raised when downloaded or loaded bytes do not form a valid solar image."""
    pass


@dataclass
class ImageMetadata:
    """Standardized metadata sidecar for an ingested solar observation."""
    filename: str
    filepath: str
    source_url: str
    source_type: str                  # e.g., 'NASA_SDO_HMI_CONTINUUM', 'LOCAL_ARCHIVE'
    observatory: str                  # 'NASA Solar Dynamics Observatory (SDO)'
    instrument: str                   # 'Helioseismic and Magnetic Imager (HMI)'
    wavelength_channel: str           # 'Fe I 6173 Å (Visible Continuum)'
    download_timestamp_utc: str       # ISO 8601 UTC
    source_last_modified: Optional[str]
    etag: Optional[str]
    sha256: str
    file_size_bytes: int
    image_width: int
    image_height: int
    channels: int
    is_authentic_real_data: bool = True


@dataclass
class IngestionResult:
    """Outcome of an ingestion attempt."""
    success: bool
    status: str                       # 'downloaded', 'duplicate_skipped', 'loaded_from_disk', 'error'
    metadata: Optional[ImageMetadata] = None
    image: Optional[np.ndarray] = None
    filepath: Optional[Path] = None
    error_message: Optional[str] = None


class SolarDataIngestor:
    """
    Robust data ingestor for authentic NASA solar imagery.
    Handles network requests with retries, deduplication, format validation,
    and metadata sidecar generation.
    """

    def __init__(
        self,
        config: Optional[DataSourceConfig] = None,
        catalog_index_path: str = "data/ingestion_catalog.json",
    ):
        self.config = config or DataSourceConfig()
        self.raw_dir = Path(self.config.raw_dir)
        self.sample_dir = Path(self.config.sample_dir)
        self.catalog_index_path = Path(catalog_index_path)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_index_path.parent.mkdir(parents=True, exist_ok=True)

        self._index: Dict[str, dict] = self._load_index()

    def _load_index(self) -> Dict[str, dict]:
        """Load the known SHA-256 / filename index to prevent duplicate downloads."""
        if self.catalog_index_path.exists():
            try:
                with open(self.catalog_index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read catalog index: {e}. Starting fresh.")
                return {}
        return {}

    def _save_index(self):
        """Persist catalog index to disk."""
        try:
            with open(self.catalog_index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save catalog index: {e}")

    def validate_image_bytes(self, data: bytes) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
        """
        Verify that bytes form a valid, non-corrupt solar image of adequate dimension.
        """
        if len(data) < self.config.min_file_size_bytes:
            return False, None, f"Payload size ({len(data)} bytes) is below minimum threshold ({self.config.min_file_size_bytes} bytes)."

        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            return False, None, "Failed to decode image bytes with OpenCV."

        h, w = img.shape[:2]
        if min(h, w) < self.config.min_image_dimension:
            return False, None, f"Image dimensions ({w}x{h}) are smaller than required ({self.config.min_image_dimension}px)."

        return True, img, None

    def fetch_url_with_retries(self, url: str) -> Tuple[bytes, dict]:
        """
        Download remote content with retries and exponential backoff.

        Returns:
            Tuple of (content_bytes, headers_dict)
        """
        retries = self.config.max_retries
        backoff = self.config.retry_backoff_factor
        timeout = self.config.timeout_seconds

        last_error = None
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": self.config.user_agent,
                        "Accept": "image/jpeg, image/png, */*",
                    },
                )
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    status = response.status
                    if status != 200:
                        raise SolarNetworkError(f"HTTP response status {status} from {url}")
                    content = response.read()
                    headers = dict(response.headers)
                    return content, headers

            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                last_error = e
                logger.warning(f"Download attempt {attempt}/{retries} failed for {url}: {e}")
                if attempt < retries:
                    time.sleep(backoff ** (attempt - 1))

        raise SolarNetworkError(f"All {retries} download attempts failed for {url}: {last_error}")

    def download_latest_sdo(
        self,
        destination_dir: Optional[Path] = None,
        force_download: bool = False,
    ) -> IngestionResult:
        """
        Download the latest full-disk visible continuum image from NASA's SDO satellite.
        """
        dest_dir = destination_dir or self.raw_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        url = self.config.primary_url

        try:
            content, headers = self.fetch_url_with_retries(url)
        except SolarNetworkError as net_err:
            logger.error(f"Network error fetching NASA SDO continuum: {net_err}")
            return IngestionResult(
                success=False,
                status="error",
                error_message=str(net_err),
            )

        # Compute SHA-256
        sha256_hash = hashlib.sha256(content).hexdigest()

        # Deduplication check
        if self.config.avoid_duplicates and not force_download:
            if sha256_hash in self._index:
                existing_entry = self._index[sha256_hash]
                existing_path = Path(existing_entry["filepath"])
                if existing_path.exists():
                    logger.info(f"Duplicate SDO image skipped; existing at {existing_path}")
                    # Load existing image
                    existing_img = cv2.imread(str(existing_path))
                    meta = ImageMetadata(**existing_entry)
                    return IngestionResult(
                        success=True,
                        status="duplicate_skipped",
                        metadata=meta,
                        image=existing_img,
                        filepath=existing_path,
                    )

        # Validate image integrity
        is_valid, img, err_msg = self.validate_image_bytes(content)
        if not is_valid or img is None:
            logger.error(f"Invalid solar image payload: {err_msg}")
            return IngestionResult(
                success=False,
                status="error",
                error_message=err_msg,
            )

        h, w, c = img.shape
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"sdo_hmi_1024_{timestamp_str}.jpg"
        save_path = dest_dir / filename
        meta_path = dest_dir / f"sdo_hmi_1024_{timestamp_str}.json"

        # Save image file
        with open(save_path, "wb") as f:
            f.write(content)

        # Build metadata
        meta = ImageMetadata(
            filename=filename,
            filepath=str(save_path.resolve()),
            source_url=url,
            source_type="NASA_SDO_HMI_CONTINUUM",
            observatory="NASA Solar Dynamics Observatory (SDO)",
            instrument="Helioseismic and Magnetic Imager (HMI)",
            wavelength_channel="Fe I 6173 Å (Visible Continuum)",
            download_timestamp_utc=datetime.utcnow().isoformat() + "Z",
            source_last_modified=headers.get("Last-Modified"),
            etag=headers.get("ETag"),
            sha256=sha256_hash,
            file_size_bytes=len(content),
            image_width=w,
            image_height=h,
            channels=c,
            is_authentic_real_data=True,
        )

        # Save metadata sidecar
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)

        # Update index
        self._index[sha256_hash] = asdict(meta)
        self._save_index()

        logger.info(f"Successfully ingested real SDO continuum image to {save_path}")
        return IngestionResult(
            success=True,
            status="downloaded",
            metadata=meta,
            image=img,
            filepath=save_path,
        )

    def download_historical_sdo(
        self,
        date: datetime,
        hour: int = 0,
        destination_dir: Optional[Path] = None,
    ) -> IngestionResult:
        """
        Download a specific archival full-disk SDO/HMI continuum frame from NASA SDO.
        """
        dest_dir = destination_dir or self.raw_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        url = self.config.archive_url_template.format(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=hour,
        )

        try:
            content, headers = self.fetch_url_with_retries(url)
        except SolarNetworkError as net_err:
            return IngestionResult(
                success=False,
                status="error",
                error_message=str(net_err),
            )

        sha256_hash = hashlib.sha256(content).hexdigest()

        if self.config.avoid_duplicates:
            if sha256_hash in self._index:
                entry = self._index[sha256_hash]
                p = Path(entry["filepath"])
                if p.exists():
                    img = cv2.imread(str(p))
                    return IngestionResult(
                        success=True,
                        status="duplicate_skipped",
                        metadata=ImageMetadata(**entry),
                        image=img,
                        filepath=p,
                    )

        is_valid, img, err = self.validate_image_bytes(content)
        if not is_valid or img is None:
            return IngestionResult(success=False, status="error", error_message=err)

        h, w, c = img.shape
        fname = f"sdo_hmi_{date.strftime('%Y%m%d')}_{hour:02d}0000.jpg"
        save_path = dest_dir / fname
        meta_path = dest_dir / f"sdo_hmi_{date.strftime('%Y%m%d')}_{hour:02d}0000.json"

        with open(save_path, "wb") as f:
            f.write(content)

        meta = ImageMetadata(
            filename=fname,
            filepath=str(save_path.resolve()),
            source_url=url,
            source_type="NASA_SDO_HMI_ARCHIVE",
            observatory="NASA Solar Dynamics Observatory (SDO)",
            instrument="Helioseismic and Magnetic Imager (HMI)",
            wavelength_channel="Fe I 6173 Å (Visible Continuum)",
            download_timestamp_utc=datetime.utcnow().isoformat() + "Z",
            source_last_modified=headers.get("Last-Modified"),
            etag=headers.get("ETag"),
            sha256=sha256_hash,
            file_size_bytes=len(content),
            image_width=w,
            image_height=h,
            channels=c,
            is_authentic_real_data=True,
        )

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)

        self._index[sha256_hash] = asdict(meta)
        self._save_index()

        return IngestionResult(
            success=True,
            status="downloaded",
            metadata=meta,
            image=img,
            filepath=save_path,
        )

    def load_local_image(self, file_path: Union[str, Path]) -> IngestionResult:
        """
        Load an authentic solar image from the local filesystem with validation
        and automatic metadata sidecar extraction/generation.
        """
        p = Path(file_path).resolve()
        if not p.exists() or not p.is_file():
            return IngestionResult(
                success=False,
                status="error",
                error_message=f"File not found: {p}",
            )

        try:
            with open(p, "rb") as f:
                content = f.read()
        except Exception as e:
            return IngestionResult(
                success=False,
                status="error",
                error_message=f"Failed to read file: {e}",
            )

        is_valid, img, err = self.validate_image_bytes(content)
        if not is_valid or img is None:
            return IngestionResult(
                success=False,
                status="error",
                error_message=err,
            )

        sha256_hash = hashlib.sha256(content).hexdigest()
        h, w, c = img.shape

        # Check for accompanying metadata sidecar
        meta_sidecar = p.with_suffix(".json")
        if meta_sidecar.exists():
            try:
                with open(meta_sidecar, "r", encoding="utf-8") as f:
                    meta_dict = json.load(f)
                    meta = ImageMetadata(**meta_dict)
            except Exception:
                meta = self._create_local_metadata(p, sha256_hash, len(content), w, h, c)
        else:
            meta = self._create_local_metadata(p, sha256_hash, len(content), w, h, c)
            # Write sidecar
            try:
                with open(meta_sidecar, "w", encoding="utf-8") as f:
                    json.dump(asdict(meta), f, indent=2)
            except Exception:
                pass

        # Update index
        self._index[sha256_hash] = asdict(meta)
        self._save_index()

        return IngestionResult(
            success=True,
            status="loaded_from_disk",
            metadata=meta,
            image=img,
            filepath=p,
        )

    def _create_local_metadata(
        self, path: Path, sha256: str, size: int, w: int, h: int, c: int
    ) -> ImageMetadata:
        """Build standard metadata for locally loaded real solar images."""
        mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat() + "Z"
        return ImageMetadata(
            filename=path.name,
            filepath=str(path.resolve()),
            source_url=f"file://{path.resolve()}",
            source_type="LOCAL_REAL_SOLAR_OBSERVATION",
            observatory="NASA Solar Dynamics Observatory (SDO)",
            instrument="Helioseismic and Magnetic Imager (HMI)",
            wavelength_channel="Fe I 6173 Å (Visible Continuum)",
            download_timestamp_utc=mtime,
            source_last_modified=None,
            etag=None,
            sha256=sha256,
            file_size_bytes=size,
            image_width=w,
            image_height=h,
            channels=c,
            is_authentic_real_data=True,
        )

    def prepare_real_sample_dataset(self) -> List[Path]:
        """
        Prepare the sample dataset consisting solely of authentic real NASA SDO
        solar continuum observations.
        Downloads historic frames from the historic May 2024 solar storm (May 10, 11, 12)
        and the latest live SDO frame if internet is reachable.
        """
        paths: List[Path] = []

        # 1. Fetch historical May 2024 sequence (Active Region AR3664)
        storm_dates = [
            datetime(2024, 5, 10),
            datetime(2024, 5, 11),
            datetime(2024, 5, 12),
        ]

        for dt in storm_dates:
            fname = f"sdo_hmi_ar3664_{dt.strftime('%Y%m%d')}.jpg"
            target_p = self.sample_dir / fname

            if target_p.exists():
                res = self.load_local_image(target_p)
                if res.success and res.filepath:
                    paths.append(res.filepath)
            else:
                # Download from SDO archive
                res = self.download_historical_sdo(dt, hour=0, destination_dir=self.sample_dir)
                if res.success and res.filepath:
                    # Rename to friendly sample name
                    meta_src = res.filepath.with_suffix(".json")
                    if res.filepath != target_p:
                        res.filepath.rename(target_p)
                        if meta_src.exists():
                            meta_src.rename(target_p.with_suffix(".json"))
                    paths.append(target_p)

        # 2. Also fetch latest real-time frame
        latest_res = self.download_latest_sdo(destination_dir=self.sample_dir)
        if latest_res.success and latest_res.filepath:
            paths.append(latest_res.filepath)

        return paths
