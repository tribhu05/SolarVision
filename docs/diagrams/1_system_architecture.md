# SolarVision: System Architecture Diagram

## Architectural Overview
The SolarVision architecture is structured into decoupled, modular layers designed for scientific fidelity, reproducibility, and real-time interactive exploration:
1. **Data Ingestion Layer**: Fetches, validates, and caches real-time and historical continuum imagery from NASA Solar Dynamics Observatory (SDO) Helioseismic and Magnetic Imager (HMI).
2. **Computer Vision & Physical Processing Pipeline**: Performs solar disk localization, quadratic limb-darkening compensation, hierarchical dual-threshold segmentation, heliographic coordinate reprojection, Zurich-McIntosh classification, and differential rotation tracking.
3. **Relational Persistence Layer**: ACID-compliant SQLite storage with cascading foreign keys and deterministic deduplication.
4. **Benchmarking & Validation Engine**: Compares detections against official NOAA Space Weather Prediction Center (SWPC) ground truth.
5. **Presentation Layer**: Multi-page interactive Streamlit dashboard with real-time Plotly charts and export utilities.

## System Architecture Diagram (Mermaid)

```mermaid
graph TB
    subgraph S1["1. External Data Sources"]
        SDO["NASA SDO / HMI Continuum<br/>(1024x1024 or 4096x4096 Near-Realtime)"]
        NOAA["NOAA SWPC Solar Region Summary (SRS)<br/>(Ground Truth Active Region Records)"]
    end

    subgraph S2["2. Data Ingestion & Storage Interface (solar_data.py)"]
        DL["Image Downloader & Request Session"]
        VAL["Image Validator (Dimensions, Depth, Nan Check)"]
        CAT["Ingestion Catalog & Hash Deduplication"]
    end

    subgraph S3["3. Physical Image Preprocessing"]
        DD["Solar Disk Detection (disk_detector.py)<br/>Otsu + Canny + Minimum Enclosing Circle"]
        LDF["Limb Darkening Flattening (limb_darkening.py)<br/>u=0.56, v=0.20 Quadratic Photometric Profile"]
        PRE["Preprocessor & Enhancement (preprocessor.py)<br/>Grayscale + Bilateral Filter + CLAHE + Black-Hat"]
    end

    subgraph S4["4. Segmentation & Detection (segmentation.py & detector.py)"]
        SEG["Dual-Level Adaptive Thresholding<br/>Umbra Core (<0.55 I_quiet) & Penumbra (<0.88 I_quiet)"]
        CON["Contour Hierarchies & Area Gating<br/>(20 <= Area <= 50,000 px)"]
        PATCH["Sub-Pixel ROI Patch Extraction"]
    end

    subgraph S5["5. Coordinate Calibration & Classification (feature_extractor.py & classifier.py)"]
        CAL["Heliographic Stonyhurst Transformation<br/>Latitude (B) & Longitude (L) in Degrees"]
        FEAT["Geometric & Physical Metrics<br/>(Area in MSH, Circularity, Intensity Contrast)"]
        CLASS["Modified Zurich Classification (Classes A-H)"]
        RISK["Educational Demonstration Risk Scoring<br/>(0-100 Score based on Zurich, McIntosh, Area)"]
    end

    subgraph S6["6. Multi-Day Kinematics & Tracking (tracker.py)"]
        ROT["Snodgrass (1984) Differential Rotation Model<br/>omega(B) = 14.71 - 2.39 sin^2(B) - 1.78 sin^4(B)"]
        GATE["Kinematic Latitude & Area Gating Matrix"]
        HIST["Trajectory History & Active Region Lineage"]
    end

    subgraph S7["7. Relational Persistence (database.py)"]
        SQL[("SQLite Database: solarvision.db<br/>- observations<br/>- active_regions<br/>- tracks<br/>- trajectory_points")]
    end

    subgraph S8["8. Scientific Evaluation Engine (evaluation.py)"]
        BENCH["NOAA Benchmark Catalog (May 2024 Geomagnetic Storm)"]
        METRICS["Spatial Matching (IoU) & Bounding Box Overlap"]
        SCORE["Scorecard Generator (Precision, Recall, F1, MAE)"]
    end

    subgraph S9["9. Interactive Presentation Layer (app.py)"]
        DASH["Streamlit Scientific Dashboard (8 Distinct Pages)<br/>Plotly Visualizations + LaTeX Equations + JSON Export"]
    end

    %% Data flow links
    SDO --> DL
    DL --> VAL
    VAL --> CAT
    CAT --> DD
    DD --> LDF
    LDF --> PRE
    PRE --> SEG
    SEG --> CON
    CON --> PATCH
    CON --> CAL
    CAL --> FEAT
    FEAT --> CLASS
    FEAT --> RISK
    CLASS --> ROT
    ROT --> GATE
    GATE --> HIST
    HIST --> SQL
    FEAT --> SQL
    CLASS --> SQL
    SQL --> DASH
    NOAA --> BENCH
    BENCH --> METRICS
    CON --> METRICS
    METRICS --> SCORE
    SCORE --> DASH

    classDef source fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef cv fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef db fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef eval fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef ui fill:#fce4ec,stroke:#c2185b,stroke-width:2px;

    class SDO,NOAA source;
    class DD,LDF,PRE,SEG,CON,PATCH,CAL,FEAT,CLASS,RISK,ROT,GATE,HIST cv;
    class SQL db;
    class BENCH,METRICS,SCORE eval;
    class DASH ui;
```

## Layer Descriptions
- **Data Ingestion**: Fault-tolerant HTTPS retrieval with SHA-256 content deduplication and local disk fallback.
- **Physical Preprocessing**: Solves the limb darkening phenomenon $I(\theta) = I(0)[1 - u(1 - \cos\theta) - v(1 - \cos\theta)^2]$, transforming solar limb pixels to uniform quiet-Sun disk intensity.
- **Segmentation**: Combines adaptive photometric thresholds with morphological opening/closing to separate umbral cores from penumbral filaments.
- **Kinematic Tracking**: Predicts next-frame active region coordinates using the empirical differential rotation law to maintain identity across multi-day observations.
- **Evaluation**: Validates detection bounding boxes against official NOAA SRS bulletins to measure precision, recall, and angular coordinate mean absolute error.
