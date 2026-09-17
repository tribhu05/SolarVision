"""
Test suite for SolarDataIngestor.
Validates real image loading, network error handling, corrupt image rejection,
and deduplication logic.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.solar_data import (
    SolarDataIngestor,
    SolarNetworkError,
    SolarInvalidImageError,
    ImageMetadata,
    IngestionResult,
)
from src.config import DataSourceConfig


@pytest.fixture
def temp_ingestor(tmp_path):
    cfg = DataSourceConfig(
        raw_dir=str(tmp_path / "raw"),
        sample_dir=str(tmp_path / "sample"),
        min_image_dimension=256,
        min_file_size_bytes=1024,
        timeout_seconds=2,
        max_retries=1,
    )
    return SolarDataIngestor(
        config=cfg,
        catalog_index_path=str(tmp_path / "catalog.json"),
    )


def test_load_local_valid_solar_image(temp_ingestor):
    sample_images = list(Path("data/sample_images").glob("sdo_hmi_*.jpg"))
    if not sample_images:
        pytest.skip("No real sample images found in data/sample_images")

    test_image = sample_images[0]
    res = temp_ingestor.load_local_image(test_image)

    assert res.success is True
    assert res.status == "loaded_from_disk"
    assert res.image is not None
    assert res.image.shape[0] >= 512
    assert res.image.shape[1] >= 512
    assert res.metadata is not None
    assert res.metadata.sha256 != ""
    assert res.metadata.is_authentic_real_data is True


def test_load_nonexistent_file(temp_ingestor):
    res = temp_ingestor.load_local_image("nonexistent_solar_file_9999.jpg")
    assert res.success is False
    assert res.status == "error"
    assert "File not found" in (res.error_message or "")


def test_corrupt_image_payload(temp_ingestor, tmp_path):
    corrupt_file = tmp_path / "corrupt_solar.jpg"
    # Write garbage bytes that cannot be decoded as an image
    corrupt_file.write_bytes(b"CORRUPT_HEADER_NOT_A_JPEG_FILE" * 100)

    res = temp_ingestor.load_local_image(corrupt_file)
    assert res.success is False
    assert res.status == "error"
    assert "decode" in (res.error_message or "").lower()


def test_image_below_minimum_dimension(temp_ingestor, tmp_path):
    small_file = tmp_path / "small_image.png"
    # Create an image smaller than min_image_dimension (256) but large enough in byte size
    np.random.seed(42)
    small_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    # Save as uncompressed BMP or high-quality PNG to exceed min_file_size_bytes (1024 bytes)
    cv2.imwrite(str(small_file), small_img)

    res = temp_ingestor.load_local_image(small_file)
    assert res.success is False
    assert res.status == "error"
    assert "smaller than required" in (res.error_message or "")


def test_network_failure_handling(temp_ingestor):
    invalid_url = "http://invalid-nonexistent-solar-domain-999.gov/solar.jpg"
    with pytest.raises(SolarNetworkError):
        temp_ingestor.fetch_url_with_retries(invalid_url)


def test_deduplication_logic(temp_ingestor):
    sample_images = list(Path("data/sample_images").glob("sdo_hmi_*.jpg"))
    if not sample_images:
        pytest.skip("No real sample images found in data/sample_images")

    img_path = sample_images[0]
    res1 = temp_ingestor.load_local_image(img_path)
    assert res1.success is True

    # Attempt to load or process identical image again
    res2 = temp_ingestor.load_local_image(img_path)
    assert res2.success is True
    assert res2.metadata.sha256 == res1.metadata.sha256


def test_metadata_sidecar_generation(temp_ingestor, tmp_path):
    sample_images = list(Path("data/sample_images").glob("sdo_hmi_*.jpg"))
    if not sample_images:
        pytest.skip("No real sample images found in data/sample_images")

    src_img = sample_images[0]
    dest_img = tmp_path / "sample_copy.jpg"
    dest_img.write_bytes(src_img.read_bytes())

    res = temp_ingestor.load_local_image(dest_img)
    assert res.success is True

    sidecar = dest_img.with_suffix(".json")
    assert sidecar.exists()
    with open(sidecar, "r", encoding="utf-8") as f:
        meta_json = json.load(f)

    assert meta_json["instrument"] == "Helioseismic and Magnetic Imager (HMI)"
    assert meta_json["observatory"] == "NASA Solar Dynamics Observatory (SDO)"
    assert meta_json["is_authentic_real_data"] is True
    assert "sha256" in meta_json
