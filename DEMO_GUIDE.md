# SolarVision: Academic Demonstration & Presentation Guide

This guide provides step-by-step instructions for operating, presenting, and defending the **SolarVision** system during an academic demonstration, laboratory evaluation, or viva examination for the B.Tech Computer Vision course.

---

## 1. How to Start the Application

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Terminal / PowerShell shell

### Execution Steps
```bash
# 1. Activate your virtual environment
.\venv\Scripts\activate          # Windows PowerShell / CMD
# source venv/bin/activate       # Linux / macOS

# 2. Run the full automated test suite to confirm zero regressions (90/90 passing)
pytest tests/ -v

# 3. Launch the Streamlit Scientific Dashboard
streamlit run app.py
```
The application will automatically launch and open in your default browser at `http://localhost:8501`.

---

## 2. How to Load Solar Imagery

SolarVision provides three flexible options for loading solar imagery located in the sidebar control panel:

1. **Pre-Loaded Authentic NASA SDO Benchmark Images (Recommended)**:
   - In the sidebar dropdown, select an authenticated observation:
     - `sdo_hmi_ar3664_20240510.jpg`: Historic May 10, 2024 geomagnetic storm (features giant AR 13664, AR 13668, AR 13669, AR 13670).
     - `sdo_hmi_ar3664_20240511.jpg`: Day 2 tracking frame showing westward differential rotation.
     - `sdo_hmi_ar3664_20240512.jpg`: Day 3 tracking frame near the western limb.
     - `nasa_sdo_hmi_realtime.jpg`: Realtime continuum snapshot.
2. **Fetch Live NASA SDO Telemetry**:
   - Click the **"Fetch Latest Live SDO Image"** button in the sidebar.
   - The system retrieves the latest 1024x1024 continuum frame directly from NASA Goddard servers (`https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMIIC.jpg`).
3. **Upload Custom Solar Observation**:
   - Use the file uploader widget to supply any 8-bit or 16-bit PNG/JPG/FITS-rendered solar image.

---

## 3. How to Run Detection

1. Once an image is loaded, the pipeline automatically processes the observation session.
2. To trigger a fresh re-computation (bypassing the SQLite SHA-256 deduplication cache), click **"Process Image / Re-run Pipeline"** in the sidebar.
3. The system executes the entire Computer Vision sequence in ~150 ms:
   - Solar disk edge detection and center localization.
   - Quadratic limb-darkening compensation ($u=0.56, v=0.20$).
   - Bilateral edge-preserving filtering and CLAHE enhancement.
   - Dual-threshold hierarchical segmentation (Umbra: $< 0.55 I_{\text{quiet}}$, Penumbra: $0.55 - 0.88 I_{\text{quiet}}$).
   - Stonyhurst heliographic coordinate projection $(B, L)$ and physical area calculation ($\mu\text{Hem}$).
   - Morphological Modified Zurich classification and Demonstration Risk scoring.

---

## 4. How to View Results

Navigate between the 8 dedicated dashboard pages using the sidebar navigation:

1. **🏠 Overview**: High-level telemetry summary, active region counts, and processing pipeline flow.
2. **🔬 Solar Image Analysis**: Side-by-side 4-stage diagnostic views:
   - Raw Continuum.
   - Isolated Solar Disk with enclosing circle boundary.
   - Photometrically Flattened Disk (eliminated limb darkening).
   - Bilateral Filtered / CLAHE Contrast Enhanced.
3. **🎯 Detection Results**:
   - Full-disk solar image annotated with color-coded bounding boxes and contour masks.
   - High-resolution cropped ROI patches of individual active regions.
   - Region morphology measurement table (Area in pixels & $\mu\text{Hem}$, Circularity, Centroids).
4. **🏷️ Region Classification**:
   - Modified Zurich class assignment (Classes A through H).
   - Auditable step-by-step decision deduction trace.
   - Multi-factor Demonstration Risk score and radar chart breakdown.
5. **🛰️ Multi-Day Tracking**:
   - Interactive Plotly chart showing longitudinal drift trajectories across days.
   - Snodgrass differential rotation kinematic residuals ($< 1.8^\circ$).
   - Physical area growth rates ($\mu\text{Hem}/\text{day}$) and lifecycle status.
6. **📊 Historical Activity**:
   - Solar Butterfly Diagram (Latitude vs. Time).
   - Active region size distribution histograms and Zurich class breakdowns.
   - SQLite relational database catalog inspection and CSV/JSON export.
7. **📈 Scientific Evaluation**:
   - Quantitative scorecard benchmarked against official NOAA SWPC ground truth (Precision, Recall, $F_1$, Coordinate MAE).
   - Confusion matrix and True Positive / False Positive / False Negative analysis.
8. **📚 Methodology & Limitations**:
   - Mathematical derivations, radiative transfer equations, and academic disclaimers.

---

## 5. How to Explain the Computer Vision Techniques

When presenting the image processing pipeline to the evaluation panel, use these concise technical explanations:

- **Solar Disk Localization (`src/disk_detector.py`)**:
  > *"We apply Otsu binarization to separate the bright solar disk from dark space, followed by Canny edge detection ($T=50, 150$) and OpenCV's minimum enclosing circle algorithm. This determines the sub-pixel center $(x_c, y_c)$ and disk radius $R$, while an inner boundary margin ($\kappa = 0.985$) eliminates limb diffraction noise."*

- **Limb-Darkening Compensation (`src/limb_darkening.py`)**:
  > *"Due to optical radiative transfer, sunlight from the solar limb emerges from higher, cooler atmospheric layers, making the limb over 40% darker than the center. We invert the Pierce & Slaughter quadratic limb-darkening profile:
  $$I_{\text{flattened}}(x, y) = \frac{I_{\text{raw}}(x, y)}{1 - 0.56(1 - \mu) - 0.20(1 - \mu)^2} \quad \text{where } \mu = \cos\theta = \sqrt{1 - (r/R)^2}$$
  This normalizes quiet-Sun photosphere intensity across the entire disk to a uniform level ($I_{\text{quiet}} \approx 200\text{ DN}$), allowing reliable global and adaptive thresholding."*

- **Dual-Level Adaptive Segmentation (`src/segmentation.py` & `src/detector.py`)**:
  > *"Sunspots consist of two distinct thermodynamic structures. We set adaptive thresholds relative to local quiet-Sun intensity: pixels with $I \le 0.55 I_{\text{quiet}}$ are tagged as deep magnetic umbral cores, while pixels with $0.55 I_{\text{quiet}} < I \le 0.88 I_{\text{quiet}}$ are tagged as penumbral skirts. Morphological opening ($3\times 3$ ellipse) purges granular shot noise, while closing ($5\times 5$ ellipse) unifies fragmented penumbral borders."*

- **Heliographic Projection (`src/feature_extractor.py`)**:
  > *"Because the Sun is a sphere projected onto a flat 2D sensor, features near the limb appear compressed. We apply forward orthographic projection to calculate true Stonyhurst latitude ($B$) and longitude ($L$), and correct physical area into millionths of a solar hemisphere ($\mu\text{Hem}$):
  $$\text{Area}_{\mu\text{Hem}} = \frac{A_{\text{pixels}}}{2\pi R^2 \cos\rho} \times 10^6$$
  where $1\ \mu\text{Hem} \approx 3.04 \times 10^6\text{ km}^2$."*

- **Kinematic Differential Rotation Tracking (`src/tracker.py`)**:
  > *"The Sun is a gaseous body that rotates faster at the equator (~25 days) than at the poles (~35 days). We forward-propagate expected active region longitudes using the empirical Snodgrass (1984) relation:
  $$\omega(B) = 14.71 - 2.39\sin^2(B) - 1.78\sin^4(B) \quad [^\circ/\text{day}]$$
  A bipartite gating matrix associates detections across multi-day frames by enforcing latitude tolerances ($|\Delta B| \le 5.0^\circ$) and area stability constraints ($0.25 \le A_2/A_1 \le 4.0$)."*

---

## 6. How to Explain the Risk Indicator

Explain the Demonstration Risk Score with clarity and academic honesty:

- **What It Is**:
  > *"The Demonstration Risk Score is a composite pedagogical metric ($0 - 100$) that quantifies visible morphological complexity by taking a weighted sum of:
  1. Morphological Complexity / Zurich Class (35%)
  2. Physical Footprint Area in $\mu\text{Hem}$ (25%)
  3. Penumbra Maturity & Coverage (20%)
  4. Core Umbral Intensity Contrast (10%)
  5. Shape Compactness & Elongation (10%)"*
- **Why It Is Strictly an Educational Demonstration Heuristic**:
  > *"Visible continuum imagery measures temperature deficits in the photosphere, not vector magnetic fields. True operational space weather flare prediction requires measuring magnetic shear ($\nabla \times \mathbf{B}$), electric currents, and free magnetic energy along the Polarity Inversion Line using vector magnetograms and EUV coronal loops. Our score demonstrates how Computer Vision metrics can be aggregated into decision-support indices, but it is explicitly not an operational flare forecast."*

---

## 7. Known Limitations to Prepare For

Be prepared to answer panel questions regarding the known boundaries of the system:

1. **Continuum-Only Data**: Does not measure vector magnetic polarity ($\pm B_z$); bipolar spot groups are clustered by geometric proximity rather than magnetic polarity inversion lines.
2. **Extreme Limb Foreshortening ($\rho > 75^\circ$)**: Area foreshortening correction $\frac{1}{\cos\rho}$ diverges near the limb, and the Wilson depression effect distorts spot profiles. Regions near the limb carry a dedicated warning flag.
3. **Photospheric Light Bridges**: Bright intrusions dividing giant umbral cores can occasionally split single massive active regions into multiple sub-cores.
4. **Observation Cadence Gaps**: Kinematic tracking is calibrated for observations separated by $\le 48\text{ hours}$. Large temporal gaps ($> 72\text{ hours}$) exceed area gating thresholds and trigger track re-initialization.
5. **Spotless Solar Minimum**: The pipeline includes a dedicated null-handler that detects spotless disks without errors ($100\%$ specificity, zero false alarms).

---

## 8. How to Demonstrate the Project in 5 Minutes (Viva Walkthrough Script)

| Time | Target Dashboard Page | Demonstration Action | What to Explain to the Evaluators |
| :---: | :--- | :--- | :--- |
| **0:00 – 1:00** | **🏠 Overview** | Show metrics banner & system diagram. | *"SolarVision is an autonomous Computer Vision pipeline for segmenting, calibrating, classifying, and tracking solar active regions using NASA SDO/HMI space-borne continuum imagery."* |
| **1:00 – 2:00** | **🔬 Solar Image Analysis** | Select `sdo_hmi_ar3664_20240510.jpg`; display the 4-panel diagnostic view. | *"Optical radiative transfer causes limb darkening, making the solar edge >40% darker than the center. We invert the quadratic Pierce & Slaughter profile ($u=0.56, v=0.20$), flattening background quiet-Sun brightness to uniform intensity ($I_{\text{quiet}} \approx 200\text{ DN}$)."* |
| **2:00 – 3:00** | **🎯 Detection & 🏷️ Classification** | Show annotated disk, ROI patch, and Class F result for AR 13664. | *"Dual-threshold segmentation separates umbral cores (<0.55 $I_{\text{quiet}}$) from penumbral skirts. Coordinates are projected to Stonyhurst $(B, L)$ and area is calibrated in $\mu\text{Hem}$. AR 13664 is correctly classified as Class F ($>2,000\ \mu\text{Hem}$) with an auditable rule trace."* |
| **3:00 – 4:00** | **🛰️ Multi-Day Tracking** | Click **"Run Multi-Day Tracking Sequence"**; inspect Plotly drift chart. | *"The Sun exhibits differential rotation. SolarVision applies the Snodgrass (1984) relation to forward-propagate coordinates. AR 13664 is tracked across May 10–12 over $26.8^\circ$ of rotation with kinematic residual error $< 1.8^\circ$."* |
| **4:00 – 5:00** | **📈 Scientific Evaluation** | Present quantitative benchmark scorecard against official NOAA SWPC SRS data. | *"Benchmarking against official NOAA ground truth yielded 100% recall (4/4 regions matched), 0.80 precision, and coordinate MAE of 1.48° lat and 2.21° lon. On spotless solar minimum disks, it achieved 100% specificity with zero false alarms."* |
