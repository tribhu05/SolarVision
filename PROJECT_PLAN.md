# SolarVision: Project Implementation Plan & 3-Day Roadmap

**Project**: SolarVision - Automated Solar Active Region Detection and Analysis  
**Target Course**: VIT Bhopal University B.Tech Computer Vision (Course Project)  
**Student Name**: Tribhuwan Singh | **Registration ID**: 24BAI10358  
**Institution**: VIT Bhopal University  

---

## 📅 3-Day Execution Roadmap

### Day 1: Architecture, Core Pipeline & Working Scaffolding ✅ [COMPLETED]
- [x] **Project Scaffolding**: Setup directory hierarchy (`config/`, `data/`, `src/`, `tests/`).
- [x] **Dependencies & Configuration**: Create `requirements.txt` and `config/config.yaml`.
- [x] **Solar Disk Localization**: Implement `SolarDiskDetector` with Otsu thresholding, morphological closing, and circle fitting.
- [x] **Limb Darkening Compensation**: Implement `LimbDarkeningCorrector` using Eddington visible continuum model ($u = 0.60$).
- [x] **Dual-Threshold Segmentation**: Implement `SunspotSegmenter` to isolate umbra cores and penumbra boundaries.
- [x] **Real Data Pipeline**: Ingest real NASA SDO/HMI continuum feeds and generate calibrated multi-day tracking sequences.
- [x] **Interactive UI Foundation**: Launch Streamlit application (`app.py`) with multi-view layout and live parameter sliders.
- [x] **Automated Testing**: Implement unit test suite with 100% pass rate on core algorithms (`pytest tests/ -v`).

---

### Day 2: Physical Calibration, Transparent Classification & Persistence ✅ [COMPLETED]
- [x] **Physical Area & Foreshortening**: Convert 2D pixel areas to Millionths of Solar Hemisphere ($\mu\text{Hem}$) with $\cos\theta$ correction.
- [x] **Stonyhurst Coordinate Mapping**: Calculate physical Heliographic Latitude ($B$) and Central Meridian Distance ($L$).
- [x] **Transparent McIntosh Rule Engine**: Implement deterministic, auditable classification (Classes A, B, C, D, E, F, H) with reasoning chains.
- [x] **SQLite Database Integration**: Implement schema for observations and active regions with querying and CSV export.
- [x] **Multi-Frame Kinematic Tracker**: Implement differential rotation prediction using Snodgrass (1984) relation.
- [x] **Sunspot Detection & Morphological Feature Extraction**: Implement `SunspotDetector` calculating Area ($A_{\text{px}}$ and $A_{\mu\text{Hem}}$), Perimeter, Bounding Box, Centroid, Circularity ($4\pi A/P^2$), and Contrast.
- [x] **Demonstration Risk Scoring**: Implemented 5-factor educational complexity index with neutral attention labels and explicit non-prediction disclaimers.
- [x] **Unified Processing & Storage Pipeline**: Implemented `SolarVisionPipeline` with automatic SQLite schema migration, SHA-256 content deduplication, multi-day historical batch execution, and transaction integrity.
- [x] **Extended Validation**: Tested on real NASA SDO/HMI images from the historic May 2024 solar storms (AR3664 sequence).
- [x] **Edge Case Handling**: Validated behavior when Sun has zero sunspots (solar minimum) without exceptions or false positives.

---

### Day 3: Final Polish, Scientific Evaluation & Submission Package [FINAL PHASE]
- [ ] **Comparative Evaluation**: Compare SolarVision detections against official NOAA Space Weather Prediction Center (SWPC) Solar Region Summaries.
- [ ] **Performance Benchmarking**: Profile runtime per frame (target: $< 250\text{ ms}$ on $1024 \times 1024$ full disk).
- [ ] **Course Project Report (PDF/LaTeX)**: Prepare formal technical report with mathematical derivations, algorithm pseudo-code, and output figures.
- [ ] **Presentation Slide Deck**: Create a 10-slide presentation highlighting the Computer Vision novelty (limb darkening flat-fielding, umbra/penumbra dual-thresholding, foreshortening calibration).
- [ ] **Demo Video Recording**: 3-minute video walkthrough demonstrating:
  1. Live NASA SDO image ingestion and immediate active region detection.
  2. Inspection of the intermediate CV pipeline stages.
  3. Interactive rule trace explaining McIntosh classification.
  4. Multi-frame differential rotation tracking.
  5. Solar Butterfly diagram in the catalog analytics tab.

---

## 🎯 VIT Course Evaluation Rubric Alignment

| Rubric Component | Weight | SolarVision Technical Evidence |
| :--- | :---: | :--- |
| **Problem Formulation & Domain Significance** | 15% | Detection of solar active regions for space weather hazard prediction (solar flares, geomagnetic storms). Grounded in real solar physics. |
| **Pre-Processing & Image Enhancement** | 20% | Physical Eddington limb darkening correction ($I_{\text{flat}} = I / [1 - u(1-\mu)]$), Gaussian denoising, circular solar disk masking. |
| **Segmentation & Morphological Processing** | 25% | Dual dynamic intensity thresholding (umbra $\le 0.58 I_{QS}$, penumbra $\le 0.88 I_{QS}$), morphological opening to reject granulation noise, spatial connected-component clustering into Active Regions. |
| **Feature Extraction & Calibration** | 15% | Spherical foreshortening correction ($\cos\theta$), physical area calibration in $\mu\text{Hem}$, Stonyhurst heliographic coordinates $(B, L)$, contrast ratio. |
| **Pattern Recognition / Classification** | 15% | Transparent Modified Zurich / McIntosh rule engine with auditable, human-readable deduction steps and flare potential risk ratings. |
| **System Integration, UI & Code Quality** | 10% | Fully modular architecture (`src/`), SQLite database, Streamlit dashboard with interactive Plotly analytics, automated test suite with pytest. |

---

## 🔬 Key Scientific Algorithms Summary

### 1. Solar Disk Edge Localization
- Gaussian filter: $G_\sigma * I$
- Dynamic binary thresholding: $T = 0.18 \times \max(I)$
- Minimum enclosing circle: $(x - x_c)^2 + (y - y_c)^2 \le R_\odot^2$

### 2. Photospheric Flat-Fielding (Limb Darkening)
- Radial distance: $r = \sqrt{(x - x_c)^2 + (y - y_c)^2}$
- Heliocentric angle cosine: $\mu = \sqrt{1 - (r/R_\odot)^2}$
- Normalization: $I_{\text{flat}}(x, y) = I(x, y) / [1 - 0.60(1 - \mu)]$

### 3. Dual Thresholds for Umbra and Penumbra
- Quiet-Sun level $I_{QS} = \text{percentile}_{90}(\text{inner disk})$
- Umbra mask: $M_u = \{ (x, y) \mid I_{\text{flat}}(x, y) \le 0.58 \cdot I_{QS} \}$
- Penumbra mask: $M_p = \{ (x, y) \mid 0.58 \cdot I_{QS} < I_{\text{flat}}(x, y) \le 0.88 \cdot I_{QS} \}$

### 4. Differential Rotation Tracker
- Angular velocity: $\omega(B) = 14.713 - 2.396\sin^2(B) - 1.787\sin^4(B)$
- Predicted position: $L_{t+\Delta t} = L_t + \omega(B) \cdot \Delta t$
