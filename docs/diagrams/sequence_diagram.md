# SolarVision: UML Sequence Diagram

## 1. Overview
This diagram depicts the end-to-end execution sequence of the SolarVision automated pipeline from image ingestion through visual dashboard rendering.

---

## 2. UML Sequence Diagram (Mermaid)

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Streamlit App
    participant Pipe as SolarVisionPipeline
    participant Ingest as SolarDataFetcher
    participant Disk as SolarDiskDetector
    participant Limb as LimbDarkeningCorrector
    participant Prep as Preprocessor
    participant Seg as SunspotSegmenter
    participant Det as SunspotDetector
    participant Cal as HeliographicCalibrator
    participant Class as McIntoshClassifier
    participant Track as ActiveRegionTracker
    participant DB as SolarDatabase

    User->>Pipe: process_image(image_input)
    activate Pipe

    %% Step 1: Ingestion & Deduplication
    Pipe->>Ingest: load_image(image_input)
    activate Ingest
    Ingest-->>Pipe: raw_rgb_image, image_metadata (SHA-256)
    deactivate Ingest

    Pipe->>DB: query_observation_by_hash(sha256)
    activate DB
    alt Hash Exists in DB (Cache Hit)
        DB-->>Pipe: cached_observation
        Pipe-->>User: ObservationResult (Cached, is_cached=True)
    else Hash Not Found (Cache Miss)
        DB-->>Pipe: None
        deactivate DB

        %% Step 2: Disk Detection
        Pipe->>Disk: detect(raw_image)
        activate Disk
        Disk-->>Pipe: DiskDetectionResult (center_x, center_y, radius, mask)
        deactivate Disk

        %% Step 3: Limb Darkening Flattening
        Pipe->>Limb: correct(raw_image, center, radius)
        activate Limb
        Limb-->>Pipe: LimbCorrectionResult (flattened_image, quiet_sun_intensity)
        deactivate Limb

        %% Step 4: Edge-Preserving Denoising & Enhancement
        Pipe->>Prep: process(flattened_image)
        activate Prep
        Prep-->>Pipe: PreprocessingResult (gray, bilateral_filtered, clahe, blackhat)
        deactivate Prep

        %% Step 5: Dual-Level Segmentation
        Pipe->>Seg: segment(flattened_image, quiet_sun_intensity)
        activate Seg
        Seg-->>Pipe: SegmentationResult (umbra_mask, penumbra_mask, total_mask)
        deactivate Seg

        %% Step 6: Contours & Patch Extraction
        Pipe->>Det: detect(total_mask, raw_image)
        activate Det
        Det-->>Pipe: List[RawRegionDetections] (contours, bounding_boxes, patches)
        deactivate Det

        %% Step 7: Heliographic & Physical Calibration
        loop For Each Detected Region
            Pipe->>Cal: calibrate(contour, center, radius)
            activate Cal
            Cal-->>Pipe: PhysicalFeatures (Stonyhurst B, L, Area in MSH, Circularity)
            deactivate Cal

            Pipe->>Class: classify(PhysicalFeatures)
            activate Class
            Class-->>Pipe: ClassificationResult (Zurich Class A-H, Audit Trace, Risk Score)
            deactivate Class
        end

        %% Step 8: Multi-Day Kinematic Association
        Pipe->>Track: update(calibrated_regions, timestamp)
        activate Track
        Track-->>Pipe: List[TrackedActiveRegion] (track_id, delta_b, delta_l, growth_rate)
        deactivate Track

        %% Step 9: Relational Persistence
        Pipe->>DB: save_observation(observation_record, tracked_regions)
        activate DB
        DB-->>Pipe: observation_id
        deactivate DB

        %% Step 10: Final Return
        Pipe-->>User: ObservationResult (annotated_image, metrics, tables, is_cached=False)
    end
    deactivate Pipe

    User->>User: Render Streamlit Multi-Page View & Plotly Visualizations
```
