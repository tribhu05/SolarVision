# Project Statement: SolarVision

**System Name**: SolarVision: Autonomous Solar Active Region Segmentation, Heliographic Calibration, and Kinematic Tracking System  
**Course**: B.Tech Computer Vision Course Project  
**Student Name**: Tribhuwan Singh  
**Registration ID**: 24BAI10358  
**Institution**: VIT Bhopal University  
**Version**: 1.0.0 (Production / Submission Release)  
**Academic Year**: 2025–2026  

---

## 1. Problem Statement

Solar active regions—predominantly manifested as sunspots with concentrated magnetic flux—are the prime physical drivers of intense space weather, including M-class and X-class solar flares, coronal mass ejections (CMEs), and solar energetic particle (SEP) events. When directed toward Earth, these geomagnetic perturbations disrupt high-frequency telecommunications, induce high voltages in continental power grids, impair satellite avionics, and degrade Global Navigation Satellite Systems (GNSS) positioning accuracy.

Historically, the monitoring and morphological classification of sunspots have relied on human visual observation and manual cataloging (such as daily NOAA Space Weather Prediction Center Solar Region Summaries). This traditional approach suffers from three critical bottlenecks:

1. **Subjective Observer Variability**: Manual delineation of penumbral borders and assignment of boundary-case Modified Zurich/McIntosh classifications vary significantly across human experts and institutions.
2. **Incompatible Cadence with Modern Space Instruments**: Observatories like NASA's Solar Dynamics Observatory (SDO/HMI) stream multi-megapixel solar continuum images every 45 seconds. Human-in-the-loop cataloging operates on a daily 24-hour cycle, failing to capture rapid region emergence and morphological evolution.
3. **Severe Photometric Non-Uniformity**: Optical solar limb darkening causes quiet-Sun photospheric intensity to drop by over 40% near the solar limb, mimicking sunspot absorption contrast and inducing unacceptable false-positive rates in naive computer vision thresholding algorithms.

### Objective
To overcome these limitations, **SolarVision** provides a fully automated, deterministic, physics-informed Computer Vision pipeline that ingests raw full-disk continuum imagery from NASA SDO/HMI, autonomously localizes the solar disk, removes photometric limb darkening, segments umbral cores and penumbral halos, projects coordinates into true physical Stonyhurst heliographic coordinates, performs deterministic Modified Zurich classification, tracks kinematic trajectories across solar differential rotation, and persists all observations in an ACID-compliant relational catalog.

---

## 2. Scope of the Project

The scope of SolarVision is strictly defined to ensure high scientific rigor, reproducibility, and computational efficiency within the framework of Computer Vision:

### In-Scope:
- **Full-Disk Optical Processing**: Ingestion and normalization of full-disk continuum images ($6173\text{ \AA}$ Fe I line) from NASA SDO/HMI and SOHO/MDI instruments in both grayscale and RGB modes.
- **Autonomous Geometry Localization**: Unsupervised localization of the solar disk center $(x_c, y_c)$ and pixel radius $R_\odot$ via Otsu binarization and Canny edge-fitting.
- **Photometric Radiative Transfer Correction**: Implementation of empirical solar limb-darkening compensation ($u = 0.56, v = 0.20$) to flatten the photospheric background across the full disk.
- **Hierarchical Adaptive Segmentation**: Dual-threshold intensity segmentation isolating deep magnetic umbral cores ($I \le 0.55 \cdot I_{\text{quiet}}$) and penumbral filaments ($0.55 < I/I_{\text{quiet}} \le 0.88$).
- **Foreshortening & Heliographic Projection**: Spherical line-of-sight geometric correction ($\cos\theta$) yielding physical areas in Millionths of a Solar Hemisphere ($\text{MSH}$), paired with Stonyhurst latitude ($B$) and Central Meridian Distance ($L$) coordinate transformations.
- **Transparent Morphological Taxonomy**: Implementation of Modified Zurich / McIntosh classes (**A, B, C, D, E, F, H**) with auditable decision logs.
- **Multi-Day Kinematic Association**: Differential rotation modeling via the Snodgrass (1984) relation to track long-lived active regions across multi-day sequences.
- **Educational Complexity Scoring**: A heuristic 0–100 visible complexity indicator for interactive academic demonstration.
- **Production Dashboard & Storage**: An 8-page Streamlit analytics interface backed by an ACID-compliant SQLite relational database (`solarvision.db`).

### Out-of-Scope (Boundaries & Constraints):
- **Operational Flare Forecasting**: White-light continuum intensity alone cannot reconstruct vector magnetic shear or non-potential coronal magnetic energy. SolarVision classifies visible morphology; it explicitly disclaims real-time physical flare prediction.
- **Far-Side Helioseismology**: Tracking is physically constrained to the visible solar disk ($|\text{CMD}| \le 80^\circ$). Far-side active region acoustic imaging is excluded.
- **Coronal Plasma Diagnostics**: Extreme Ultraviolet (EUV) coronal loop modeling and X-ray flux integration are outside the scope of photospheric continuum vision.
- **Black-Box End-to-End Neural Networks**: The system intentionally prioritizes verifiable, deterministic classical computer vision and radiative transfer physics over opaque deep learning models, ensuring complete interpretability for scientific research.

---

## 3. Target Users

SolarVision is designed to serve four primary user cohorts:

1. **Solar Physicists and Heliophysicists**:
   - Researchers studying active region emergence, decay kinetics, and photospheric morphology.
   - Scientists requiring automated, reproducible segmentation of umbral/penumbral boundaries and objective MSH area calculations calibrated against NOAA SWPC standards.

2. **Space Weather Analysts and Duty Forecasters**:
   - Operational analysts at aerospace organizations and national meteorological services monitoring active region evolution.
   - Users who need automated ingestion of NASA SDO data, rapid detection of complex bipolar regions (Classes D, E, F), and multi-day differential rotation tracking.

3. **Computer Vision Students and Academic Evaluators**:
   - Students and faculty evaluating real-world applications of classical computer vision: Otsu binarization, Canny edge detection, bilateral filtering, CLAHE, morphological operators, connected component analysis, and contour hierarchy parsing.
   - Evaluators assessing compliance with academic engineering rigor, testing coverage, modular software architecture, and interactive dashboard usability.

4. **Amateur Astronomers and Science Communicators**:
   - Educators and astronomy enthusiasts operating solar optical telescopes who seek to calibrate raw telescope frames, identify Zurich classes, and explore the solar butterfly distribution interactively.

---

## 4. High-Level Features

| Feature ID | Module / Subsystem | Capability Description | Primary Output |
| :--- | :--- | :--- | :--- |
| **HLF-01** | **Telemetry Ingestion & Caching** | Real-time NASA SDO/HMI image retrieval, format validation, depth normalization, and content-addressable SHA-256 deduplication. | Validated float32/uint8 arrays, image metadata records |
| **HLF-02** | **Autonomous Disk Localization** | Automated Otsu thresholding, Canny contour detection, and minimum enclosing circle fitting to detect $(x_c, y_c, R_\odot)$ with 98.5% mask margin. | Solar disk center, radius, binary disk mask |
| **HLF-03** | **Radiative Transfer Flat-Fielding** | Quadratic empirical limb-darkening compensation ($6173\text{ \AA}$) normalizing radial quiet-Sun drop-off from 40% to <2.5%. | Photometrically flattened solar continuum disk |
| **HLF-04** | **Bilateral & Morphological Filtering** | Edge-preserving bilateral denoising ($d=9, \sigma=75$) suppressing granulation noise by ~60%, paired with Black-Hat morphological core isolation. | Noise-reduced, high-contrast photospheric representations |
| **HLF-05** | **Hierarchical Dual-Threshold Segmentation** | Physics-derived adaptive thresholding extracting umbral cores ($I \le 0.55 I_0$) and penumbrae ($0.55 < I/I_0 \le 0.88$) with morphological opening/closing. | Binary umbra, penumbra, and total active region masks |
| **HLF-06** | **Heliographic Projection & Calibration** | Spherical orthographic transformation into Stonyhurst latitude ($B$) and longitude ($L$), with line-of-sight foreshortening correction ($\cos\theta$) yielding physical areas in MSH. | Calibrated Stonyhurst coordinates, true physical MSH areas |
| **HLF-07** | **Deterministic Modified Zurich Classification** | Rule-based taxonomy classifying regions into Zurich/McIntosh classes (A, B, C, D, E, F, H) with full audit traces and educational 0–100 complexity scores. | Morphological class, decision trace, complexity rating |
| **HLF-08** | **Snodgrass Kinematic Tracking** | Multi-day differential rotation kinematics matching active regions across observations via latitude-dependent angular velocity and bipartite gating. | Persistent track IDs, kinematic drift vectors, growth rates |
| **HLF-09** | **Relational SQLite Persistence** | ACID-compliant storage across 5 relational tables (`image_metadata`, `observations`, `active_regions`, `tracks`, `trajectory_points`) with JSON export. | Structured relational database (`solarvision.db`) |
| **HLF-10** | **Interactive Scientific Dashboard** | 8-section responsive Streamlit dashboard featuring visual inspection grids, interactive Plotly charts, butterfly diagrams, and live image testing. | Interactive web GUI at `http://localhost:8501` |
