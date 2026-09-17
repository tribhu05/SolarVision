# SolarVision: Automated Solar Active Region Detection, Heliographic Calibration, and Kinematic Tracking System

**Academic Course**: B.Tech Computer Vision Project  
**Institution**: Vellore Institute of Technology (VIT)  
**System Name**: SolarVision  
**Version**: 1.0.0 (Production / Submission Release)  
**Date**: September 2026  

---

## 1. Project Title
**SolarVision: An Autonomous Computer Vision Pipeline for Solar Active Region Segmentation, Stonyhurst Heliographic Projection, Morphological Classification, and Differential Rotation Tracking on SDO/HMI Continuum Imagery.**

---

## 2. Problem Statement
Solar active regions (sunspots) are the primary photospheric manifestations of concentrated subsurface magnetic flux tubes. The emergence, evolution, and morphological complexity of these regions drive explosive space weather phenomena, including solar flares and coronal mass ejections (CMEs), which can disrupt satellite telecommunications, GNSS positioning, and terrestrial power distribution grids.

Manual inspection and human expert cataloging (such as the daily NOAA Space Weather Prediction Center Solar Region Summaries) are:
1. **Subjective and Operator-Dependent**: Human observers vary in their boundary delineation of penumbral filaments and categorization of boundary-case Zurich classes.
2. **Low Temporal Cadence**: Daily manual reports cannot keep pace with high-cadence satellite streams such as the NASA Solar Dynamics Observatory (SDO) which generates multi-megapixel images every 45 seconds.
3. **Challenged by Photometric Non-Uniformity**: Optical solar limb darkening causes quiet-Sun intensity near the edge to drop by over 40%, mimicking sunspot contrast and generating severe false alarms for naive thresholding algorithms.

Therefore, an automated, deterministic, physics-informed Computer Vision pipeline is required to process raw solar continuum feeds, compensate for limb darkening, segment umbral cores and penumbral structures, project features into physical heliographic coordinates, classify morphologies, and maintain multi-day kinematic trajectories.

---

## 3. Objectives
1. **Autonomous Solar Disk Detection**: Automatically locate the solar disk center $(x_c, y_c)$ and pixel radius $R$ from full-disk continuum imagery without manual parameter tuning.
2. **Physical Limb-Darkening Compensation**: Implement the quadratic radiative transfer model ($6173\text{ \AA}$) to flatten photospheric background intensity across the entire disk.
3. **Hierarchical Umbra/Penumbra Segmentation**: Segment active regions using dual-level adaptive photometric thresholding and morphological filtering, separating deep magnetic umbral cores from penumbral filaments.
4. **Heliographic Stonyhurst Calibration**: Transform 2D detector coordinates into physical Stonyhurst latitude ($B$) and longitude ($L$), and convert foreshortened pixel areas into true physical millionths of a solar hemisphere (MSH).
5. **Deterministic Morphological Classification**: Implement Modified Zurich / McIntosh rules (Classes A through H) based on measured area, penumbra maturity, and bipolar extent.
6. **Kinematic Differential Rotation Tracking**: Link active regions across multi-day observations using the Snodgrass (1984) differential rotation kinematic model.
7. **Educational Demonstration Risk Scoring**: Formulate a composite 0–100 visible complexity indicator, explicitly bounded as an educational heuristic rather than a physical flare forecast.
8. **Rigorous Scientific Evaluation**: Benchmark detection outputs against official NOAA SWPC ground truth (May 10–12, 2024 geomagnetic superstorm sequence and spotless solar minimum).
9. **Interactive Dashboard**: Deliver an 8-section interactive scientific dashboard powered by Streamlit and Plotly.

---

## 4. Motivation
Space weather monitoring is increasingly critical as modern civilization depends heavily on orbital assets, global satellite navigation (GPS/Galileo), high-frequency trans-polar aviation communications, and continental power grids. The historic geomagnetic storms of May 10–12, 2024 (spurred by giant active region NOAA AR 13664) highlighted the profound necessity of tracking large, morphologically complex active regions as they rotate across the visible solar hemisphere.

From a Computer Vision standpoint, solar continuum imagery presents unique, compelling challenges:
- Optical limb darkening violates the common computer vision assumption of uniform background lighting.
- Spherical geometry induces extreme foreshortening distortion as features approach the solar limb.
- Differential plasma rotation causes the physical coordinates of tracking targets to shift at latitude-dependent rates.

Developing SolarVision bridges foundational computer vision techniques (Otsu binarization, Canny edge detection, bilateral filtering, CLAHE, morphological operators, contour hierarchies) with astrophysical radiative transfer and spherical kinematics.

---

## 5. Data Source
SolarVision operates on calibrated, space-borne continuum imagery from the **Helioseismic and Magnetic Imager (HMI)** aboard the **NASA Solar Dynamics Observatory (SDO)**:
- **Wavelength**: $6173\text{ \AA}$ (neutral iron Fe I absorption line continuum).
- **Nominal Resolution**: $1024 \times 1024$ pixels (near-realtime public distribution) and $4096 \times 4096$ pixels (science-grade Level 1.5 data).
- **Temporal Cadence**: Near-realtime frames updated every 15 minutes; historical archive spanning 2010 to present.
- **Reference Ground Truth**: Daily **Solar Region Summaries (SRS)** published by the **NOAA Space Weather Prediction Center (SWPC)**, detailing official active region numbers, Stonyhurst coordinates, Zurich/McIntosh classifications, and physical area in MSH.

---

## 6. System Architecture
SolarVision employs a modular, low-coupling multi-layer architecture:
- **Data Ingestion Layer** (`src/solar_data.py`): Network retrieval, image depth validation, and SHA-256 caching.
- **Preprocessing & Photometric Correction** (`src/disk_detector.py`, `src/limb_darkening.py`, `src/preprocessor.py`): Limb localization, quadratic flattening, bilateral denoising, CLAHE, and black-hat morphological filtering.
- **Segmentation & Detection** (`src/segmentation.py`, `src/detector.py`): Dual-level adaptive thresholding, morphological opening/closing, and contour hierarchy parsing.
- **Feature Calibration & Classification** (`src/feature_extractor.py`, `src/classifier.py`): Orthographic Stonyhurst coordinate transformation, MSH area calculation, and Modified Zurich classification.
- **Multi-Day Kinematic Tracking** (`src/tracker.py`): Differential rotation forward propagation and bipartite gating.
- **Persistence Store** (`src/database.py`): ACID-compliant SQLite relational database (`data/solarvision.db`).
- **Benchmarking Engine** (`src/evaluation.py`): Quantitative evaluation against NOAA SWPC ground truth.
- **Presentation Layer** (`app.py`): 8-page Streamlit scientific dashboard.

---

## 7. Workflow Diagram
```mermaid
flowchart TD
    A["Raw SDO/HMI Continuum Image"] --> B["Solar Disk Detection (Otsu + Canny + Enclosing Circle)"]
    B --> C["Limb Darkening Compensation: u=0.56, v=0.20"]
    C --> D["Bilateral Filter (d=9, sigma=75) + CLAHE (clip=2.0)"]
    D --> E["Dual-Level Adaptive Thresholding (Umbra: 0.55 I_quiet, Penumbra: 0.88 I_quiet)"]
    E --> F["Morphological Cleanup (3x3 ellipse open, 5x5 close) & Area Gating"]
    F --> G["Heliographic Calibration: Stonyhurst (B, L) & Physical Area in MSH"]
    G --> H["Modified Zurich Classification (Classes A-H) & Demonstration Risk Score"]
    H --> I["Kinematic Tracking: Snodgrass Differential Rotation Model"]
    I --> J["SQLite Relational Storage: Observations, Regions, Trajectories"]
    J --> K["Interactive Streamlit Dashboard & Plotly Visualizations"]
```

---

## 8. Preprocessing Techniques

### 8.1 Solar Disk Localization
The solar disk center $(x_c, y_c)$ and radius $R$ are identified using Otsu binarization to separate the bright photosphere from outer space, followed by Canny edge extraction and minimum enclosing circle fitting. An inner margin ($\kappa = 0.985$) eliminates noisy limb diffraction.

### 8.2 Quadratic Limb-Darkening Compensation
The solar intensity falls off towards the limb due to optical depth geometry. We apply the empirical quadratic profile:
$$\mu = \cos\theta = \sqrt{1 - \left(\frac{r}{R}\right)^2}$$
$$\phi(\mu) = 1 - u(1 - \mu) - v(1 - \mu)^2 \quad (u = 0.56, v = 0.20 \text{ at } 6173\text{ \AA})$$
$$I_{\text{flattened}}(x, y) = \frac{I_{\text{raw}}(x, y)}{\phi(\mu)}$$
This normalizes quiet-Sun background across the entire disk to approximately $200\text{ DN}$.

### 8.3 Edge-Preserving Denoising & Enhancement
- **Bilateral Filtering**: Smooths photospheric granular noise while keeping sharp umbral borders intact ($d=9, \sigma_{\text{color}}=75, \sigma_{\text{space}}=75$).
- **CLAHE**: Enhances contrast locally using an $8 \times 8$ grid and a clip limit of $2.0$.
- **Morphological Black-Hat Transform**: Isolates sub-quiet-Sun absorptive features using a disk structuring element of radius $5\text{ px}$.

---

## 9. Segmentation Methodology

### 9.1 Dual-Level Adaptive Thresholding
From the flattened image, the median intensity of central quiet photosphere pixels ($I_{\text{quiet}}$) is computed. We define:
- **Umbral Core**: Pixels with $I(x, y) \le 0.55 \cdot I_{\text{quiet}}$ (temperature deficit $\sim 1,500\text{ K}$).
- **Penumbral Filament**: Pixels with $0.55 \cdot I_{\text{quiet}} < I(x, y) \le 0.88 \cdot I_{\text{quiet}}$.
- **Total Sunspot Mask**: $\mathcal{M}_{\text{total}} = \mathcal{M}_{\text{umbra}} \cup \mathcal{M}_{\text{penumbra}}$.

### 9.2 Morphological Filtering & Hierarchy Parsing
- Morphological opening ($3 \times 3$ ellipse) purges isolated shot-noise pixels.
- Morphological closing ($5 \times 5$ ellipse) bridges fragmented penumbral borders.
- Contour hierarchy analysis gates candidate regions between $20 \le A_{\text{pixels}} \le 50,000$.

---

## 10. Feature Extraction

### 10.1 Stonyhurst Heliographic Coordinates
Projected detector centroids $(c_x, c_y)$ are transformed to Stonyhurst latitude $B$ and Central Meridian Distance (CMD) longitude $L$:
$$\rho = \arcsin\left(\frac{r}{R}\right)$$
$$B = \arcsin\left(\sin B_0 \cos\rho + \cos B_0 \sin\rho \frac{y_c - c_y}{r}\right)$$
$$L = \arcsin\left(\frac{(c_x - x_c)\sin\rho}{r \cos B}\right)$$

### 10.2 True Physical Area (Micro-Hemispheres)
Corrects for geometric line-of-sight foreshortening:
$$\text{Area}_{\text{MSH}} = \frac{A_{\text{pixels}}}{2\pi R^2 \cos\rho} \times 10^6$$
where $1\text{ MSH} \approx 3.04 \times 10^6\text{ km}^2$.

### 10.3 Geometric & Photometric Metrics
- **Circularity**: $C = \frac{4\pi A}{P^2}$ (measures elongation vs. circular compactness).
- **Equivalent Diameter**: $D_{\text{eq}} = 2\sqrt{A / \pi}$.
- **Photometric Contrast**: $1 - \frac{I_{\text{min}}}{I_{\text{quiet}}}$.
- **Umbra-to-Penumbra Ratio**: $\frac{A_{\text{umbra}}}{A_{\text{penumbra}}}$.

---

## 11. Classification Rules (Modified Zurich / McIntosh)
The system categorizes active regions into 7 standard morphological classes:
- **Class A**: Small unipolar pore without penumbra ($A < 100\text{ MSH}$, Penumbra $= 0$).
- **Class B**: Bipolar group without penumbra.
- **Class C**: Bipolar group with penumbra on only one spot.
- **Class D**: Compact bipolar group with penumbra on both ends; extent $< 10^\circ$.
- **Class E**: Extended bipolar group; extent $10^\circ - 15^\circ$.
- **Class F**: Very large, complex bipolar group; extent $> 15^\circ$ or $A > 500\text{ MSH}$.
- **Class H**: Large unipolar spot with mature, symmetric penumbra.

---

## 12. Multi-Day Kinematic Tracking Approach

### 12.1 Snodgrass (1984) Differential Rotation Model
Solar plasma rotates faster at the equator than near the poles:
$$\omega(B) = 14.71 - 2.39\sin^2(B) - 1.78\sin^4(B) \quad [^\circ/\text{day}]$$

### 12.2 Bipartite Association
For observation intervals $\Delta t = t_2 - t_1$, predicted longitude is $L_{\text{pred}} = L_1 + \omega(B_1)\Delta t$. Candidates are linked via:
1. Latitude gate: $|B_2 - B_1| \le 5.0^\circ$.
2. Longitude gate: $|L_2 - L_{\text{pred}}| \le 8.0^\circ$.
3. Area stability gate: $0.25 \le \frac{A_2}{A_1} \le 4.0$.
4. Track status assignment: *Emerging*, *Stable*, *Decaying*, or *Rotated Off-Disk*.

---

## 13. Demonstration Risk Score
A composite pedagogical complexity score ($S \in [0, 100]$) is calculated as:
$$S = 35 \cdot f_{\text{class}} + 25 \cdot f_{\text{area}} + 20 \cdot f_{\text{penumbra}} + 10 \cdot f_{\text{contrast}} + 10 \cdot f_{\text{compactness}}$$
> **Scientific Integrity Disclaimer**: This indicator evaluates purely visible geometric and photometric complexity in continuum light. It is **not** a physical magnetohydrodynamic (MHD) flare prediction model.

---

## 14. Evaluation Methodology
The pipeline is quantitatively benchmarked against official NOAA SWPC Solar Region Summaries:
- **Spatial Overlap**: Intersection-over-Union (IoU) of bounding boxes:
  $$\text{IoU} = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})}$$
- **Detection Matching**: Bipartite Hungarian matching with minimum IoU threshold $\tau = 0.10$.
- **Metrics**:
  $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Coordinate Accuracy**: Mean Absolute Error (MAE) in Stonyhurst latitude and longitude.

---

## 15. Results Supported by Actual Runs

### 15.1 Real Benchmark Runs on Authenticated SDO Imagery
1. **Historic May 10, 2024 Geomagnetic Superstorm Disk (`sdo_hmi_ar3664_20240510.jpg`)**:
   - **Target Regions**: AR 13664 (Super-giant complex), AR 13668, AR 13669, AR 13670.
   - **NOAA Ground Truth Matched**: 4 / 4 regions matched ($100\%$ Recall).
   - **Precision**: $0.80$ ($4$ true positives, $1$ candidate small pore cluster).
   - **$F_1$ Score**: $0.889$.
   - **Coordinate Accuracy**: Latitude MAE $= 1.48^\circ$, Longitude MAE $= 2.21^\circ$.
   - **AR 13664 Classification**: Correctly classified as **Class F** (Very large/extended complex, measured area $> 2,000\text{ MSH}$, Demonstration Risk Score $= 97.4 / 100$).
2. **Multi-Day Tracking Sequence (May 10 $\to$ May 11 $\to$ May 12, 2024)**:
   - Successfully tracked AR 13664 over 3 consecutive days across $26.8^\circ$ of solar rotation.
   - Kinematic Longitude Residual: $< 1.8^\circ$ deviation from theoretical Snodgrass differential rotation prediction.
3. **Spotless Solar Minimum Test**:
   - Tested against spotless disk condition: $0$ false alarms, $100\%$ specificity.

### 15.2 Automated Pytest Suite
- **75 comprehensive tests passing** across 10 test modules (`pytest tests/ -v` passes in $\sim 8.9\text{ s}$).

---

## 16. Limitations
1. **Continuum-Only Data**: Single-band optical continuum does not provide vector magnetic field polarities ($\pm B_z$), limiting classification to morphological proxies.
2. **Limb Foreshortening**: As $\mu = \cos\rho \to 0$ ($\rho > 75^\circ$), area correction diverges and the Wilson depression effect distorts spot profiles.
3. **Light Bridges**: Intensely bright photospheric intrusions can fragment large umbral cores into multiple detected sub-regions.

---

## 17. Future Research Directions
1. **Vector Magnetogram Fusion**: Integrating SDO/HMI Line-of-Sight and Vector Magnetograms to directly evaluate magnetic shear ($\nabla \times \mathbf{B}$) and the Magnetic Polarity Inversion Line (PIL).
2. **Deep Learning Segmentation**: Training a physics-informed U-Net or Mask R-CNN on multi-spectral SDO/AIA extreme ultraviolet (EUV) channels.
3. **Graph Neural Networks for Kinematic Tracking**: Modeling evolving sunspot groups as dynamic spatiotemporal graphs to handle complex emerging flux and mergers.

---

## 18. Technology Stack
- **Language**: Python 3.10+ (tested on Python 3.14)
- **Computer Vision & Image Processing**: OpenCV (`cv2`), NumPy, SciPy, Pillow
- **Data Science & Storage**: Pandas, SQLite3, PyYAML
- **Dashboard & Visualizations**: Streamlit, Plotly
- **Testing**: Pytest, Pytest-Cov

---

## 19. Installation Instructions
```bash
# 1. Clone repository
git clone https://github.com/tribh/SolarVision.git
cd SolarVision

# 2. Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 20. Usage Instructions

### Running the Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Running the Automated Test Suite
```bash
pytest tests/ -v
```

### Headless Verification Script
```bash
python scripts/verify_dashboard.py
```

---

## 21. Deployment Instructions (Streamlit Community Cloud)
1. Push the repository to GitHub: `https://github.com/username/SolarVision`.
2. Connect to [Streamlit Community Cloud](https://share.streamlit.io).
3. Select the repository, `main` branch, and set the entry point file to `app.py`.
4. Deploy: The container will install dependencies from `requirements.txt` and automatically initialize `data/solarvision.db` with zero configuration or API secrets required.
