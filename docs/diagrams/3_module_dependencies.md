# SolarVision: Module Dependency Diagram

## Architectural Module Hierarchy
The SolarVision codebase is structured into modular, low-coupling components under the `src/` directory. High-level orchestrators (`pipeline.py`, `app.py`, `evaluation.py`) consume lower-level computational kernels without circular dependencies.

```mermaid
graph TD
    CFG["config.py<br/>SolarVisionConfig, Paths, Thresholds"]

    subgraph Data_Layer["Data Ingestion & Persistence"]
        DATA["solar_data.py<br/>SolarDataIngestor, ImageMetadata, IngestionResult"]
        DB["database.py<br/>SolarDatabase, SQLite CRUD"]
    end

    subgraph CV_Core["Computer Vision Core"]
        DISK["disk_detector.py<br/>SolarDiskDetector, SolarDiskGeometry"]
        LIMB["limb_darkening.py<br/>LimbDarkeningCorrector, LimbCorrectionResult"]
        PREP["preprocessor.py<br/>SolarImagePreprocessor, PreprocessingResult"]
        SEG["segmentation.py<br/>SunspotSegmenter, SegmentationResult"]
        DET["detector.py<br/>SunspotDetector, DetectedRegion, DetectionOutput"]
        FEAT["feature_extractor.py<br/>FeatureExtractor, CalibratedActiveRegion"]
        CLASS["classifier.py<br/>McIntoshClassifier, ClassificationResult, DemonstrationRiskAssessment"]
        TRACK["tracker.py<br/>ActiveRegionTracker, TrackHistory, TrackedObservation"]
    end

    subgraph Orchestration["Pipeline & Evaluation Engine"]
        PIPE["pipeline.py<br/>SolarVisionPipeline, PipelineResult, SequencePipelineResult"]
        EVAL["evaluation.py<br/>SolarVisionEvaluator, EvaluationScorecard"]
    end

    subgraph Presentation["User Interface Layer"]
        APP["app.py<br/>Streamlit 8-Section Scientific UI"]
    end

    %% Dependency Connections
    CFG --> DATA
    CFG --> DISK
    CFG --> LIMB
    CFG --> PREP
    CFG --> SEG
    CFG --> DET
    CFG --> FEAT
    CFG --> CLASS
    CFG --> TRACK
    CFG --> DB
    CFG --> PIPE

    DATA --> PIPE
    DISK --> PREP
    LIMB --> PREP
    PREP --> DET
    SEG --> DET
    DET --> FEAT
    FEAT --> CLASS
    FEAT --> TRACK
    CLASS --> TRACK

    DET --> PIPE
    FEAT --> PIPE
    CLASS --> PIPE
    TRACK --> PIPE
    DB --> PIPE

    PIPE --> APP
    DB --> APP
    DATA --> APP
    TRACK --> APP
    EVAL --> APP

    FEAT --> EVAL
    CLASS --> EVAL

    classDef config fill:#eceff1,stroke:#607d8b,stroke-width:2px;
    classDef data fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef cv fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef orch fill:#fff8e1,stroke:#ffa000,stroke-width:2px;
    classDef ui fill:#fce4ec,stroke:#d81b60,stroke-width:2px;

    class CFG config;
    class DATA,DB data;
    class DISK,LIMB,PREP,SEG,DET,FEAT,CLASS,TRACK cv;
    class PIPE,EVAL orch;
    class APP ui;
```

## Module Responsibilities and Interfaces
| Module | Primary Class / Functions | Key Dependencies | Primary Output |
| :--- | :--- | :--- | :--- |
| `config.py` | `SolarVisionConfig`, `load_config` | PyYAML, dataclasses | Centralized configuration dataclass |
| `solar_data.py` | `SolarDataIngestor` | Requests, PIL, hashlib | Validated image arrays & metadata sidecars |
| `disk_detector.py` | `SolarDiskDetector` | OpenCV, NumPy | Solar disk geometry $(x_c, y_c, R, \text{mask})$ |
| `limb_darkening.py`| `LimbDarkeningCorrector` | NumPy, SciPy | Flattened photometric continuum array |
| `preprocessor.py` | `SolarImagePreprocessor` | OpenCV, NumPy | Denoised, CLAHE-enhanced contrast arrays |
| `segmentation.py` | `SunspotSegmenter` | OpenCV, NumPy | Binary umbral and penumbral segmentations |
| `detector.py` | `SunspotDetector` | OpenCV, NumPy | Bounding boxes, contours, and detected regions |
| `feature_extractor.py`| `FeatureExtractor` | NumPy | Heliographic coordinates $(B, L)$ and area in $\mu\text{Hem}$ |
| `classifier.py` | `McIntoshClassifier` | NumPy | Modified Zurich class and Demonstration Risk score |
| `tracker.py` | `ActiveRegionTracker` | SciPy, Pandas, Plotly | Multi-day kinematic trajectory linkage |
| `database.py` | `SolarDatabase` | SQLite3 | Relational tables with cascading foreign keys |
| `pipeline.py` | `SolarVisionPipeline` | All CV modules, Database | End-to-end processing & cached retrieval |
| `evaluation.py` | `SolarVisionEvaluator` | Pipeline, NumPy, NOAA catalog | Quantitative precision, recall, IoU, and MAE |
| `app.py` | Streamlit Dashboard | Streamlit, Plotly, Pipeline | 8-section interactive web dashboard |
