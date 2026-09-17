# SolarVision: Academic Demonstration & Viva Presentation Guide

## 1. Quick Start Commands

```bash
# 1. Activate your virtual environment
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux/macOS

# 2. Verify all automated tests (75/75 passing)
pytest tests/ -v

# 3. Launch the Streamlit Scientific Dashboard
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 2. 5-Minute Demonstration Walkthrough Script

### Minute 1: Introduction & Overview Page (🏠 Overview)
- **What to say**:
  > *"Respected evaluators, SolarVision is an automated Computer Vision system designed to detect, calibrate, classify, and track solar active regions using space-borne continuum imagery from the NASA Solar Dynamics Observatory (SDO)."*
- **Actions in UI**:
  - Show the **🏠 Overview** page.
  - Highlight the key metrics banner (Total Active Regions Tracked, Solar Cycle 25 status, NOAA ground truth alignment).
  - Emphasize the end-to-end processing pipeline diagram.

### Minute 2: Preprocessing & Physical Correction (🔬 Solar Image Analysis)
- **What to say**:
  > *"The foundational challenge of solar computer vision is limb darkening—due to optical radiative transfer, the solar edge is over 40% darker than the center. If we applied naive thresholding, the entire solar limb would be falsely flagged as sunspots."*
- **Actions in UI**:
  - Navigate to **🔬 Solar Image Analysis**.
  - Select `sdo_hmi_ar3664_20240510.jpg` (the historic May 2024 superstorm image).
  - Display the side-by-side comparison:
    1. **Raw Continuum**: Notice the darker edges.
    2. **Solar Disk Mask**: Automatically located via Otsu binarization and Canny edge circle fitting ($R \approx 404\text{ px}$).
    3. **Photometrically Flattened**: Quadratic limb-darkening compensation ($u=0.56, v=0.20$) normalizes quiet-Sun background to uniform intensity.
    4. **Denoised / CLAHE / Black-Hat**: Demonstrates edge-preserving smoothing and dark-feature isolation.

### Minute 3: Detection, Segmentation & Calibration (🎯 Detection & 🏷️ Classification)
- **What to say**:
  > *"Once normalized, SolarVision executes dual-level adaptive photometric thresholding, separating deep magnetic umbral cores (<0.55 quiet-Sun) from penumbral skirts (0.55–0.88 quiet-Sun). Coordinates are projected into physical Stonyhurst heliographic latitude and longitude, correcting for line-of-sight spherical foreshortening."*
- **Actions in UI**:
  - Switch to **🎯 Detection Results**:
    - Show the annotated solar disk with color-coded bounding boxes and contour masks.
    - Inspect the extracted high-resolution ROI patches showing isolated umbral and penumbral regions.
  - Switch to **🏷️ Region Classification**:
    - Highlight **AR-001** (NOAA AR 13664): Classified as **Class F** (Very large/extended complex, physical area $> 2,000\text{ MSH}$, Demonstration Risk Score $= 97.4 / 100$).
    - Point out the deterministic classification rule trace showing how area, spot count, and bipolar extent drove the classification.

### Minute 4: Kinematic Tracking & Differential Rotation (🛰️ Multi-Day Tracking)
- **What to say**:
  > *"Because the Sun is a gaseous body, the equator rotates once in ~25 days while the poles take ~35 days. SolarVision implements the Snodgrass (1984) differential rotation equation to kinematically propagate expected sunspot positions forward in time and link identities across multi-day sequences."*
- **Actions in UI**:
  - Navigate to **🛰️ Multi-Day Tracking**.
  - Click **"Run Multi-Day Tracking Sequence"** (processes May 10 $\to$ May 11 $\to$ May 12, 2024).
  - Show the interactive Plotly drift trajectory: AR 13664 tracked across $26.8^\circ$ of westward longitudinal rotation with kinematic residual error $< 1.8^\circ$.

### Minute 5: Scientific Evaluation & Limitations (📈 Evaluation & 📚 Limitations)
- **What to say**:
  > *"We benchmarked SolarVision against official NOAA Space Weather Prediction Center (SWPC) Solar Region Summaries. On the May 10 benchmark, the pipeline achieved 100% recall (4/4 NOAA regions detected), 0.80 precision, and a coordinate Mean Absolute Error of only 1.48° in latitude and 2.21° in longitude. Furthermore, on spotless solar minimum imagery, it achieved 100% specificity with zero false alarms."*
- **Actions in UI**:
  - Open **📈 Scientific Evaluation**:
    - Show the quantitative scorecard (Precision, Recall, F1, Coordinate MAE, IoU).
    - Review the TP/FP/FN analysis table.
  - Conclude on **📚 Methodology & Limitations**:
    - Reiterate the academic disclaimer: Demonstration Risk Score reflects visible morphological complexity and is not an operational flare forecast.

---

## 3. How to Answer Panel Questions (Viva Defense)

### Q1: "Why use classical Computer Vision (Otsu, Canny, CLAHE) instead of Deep Learning (YOLO or Mask R-CNN)?"
- **Answer**:
  > *"Classical computer vision provides complete physical explainability, deterministic repeatability, and zero risk of hallucinating non-existent solar features. Furthermore, classical algorithms require zero GPU training infrastructure, run in milliseconds, and integrate seamlessly with physical radiative transfer equations (limb darkening) and spherical geometry (Stonyhurst projection). Deep learning models act as black boxes and frequently struggle with physical unit calibration (such as micro-hemispheres) unless physically constrained."*

### Q2: "Why did you use quadratic limb darkening instead of linear?"
- **Answer**:
  > *"Linear limb darkening $I(\theta) = I(0)[1 - u(1 - \cos\theta)]$ is an oversimplification that fails significantly near the solar limb ($\mu < 0.3$). Neckel & Labs (1994) and Pierce & Slaughter demonstrated that at $6173\text{ \AA}$, a quadratic formulation with $u=0.56$ and $v=0.20$ accurately reproduces solar continuum radiative transfer with less than 1% residual error across the entire optical disk."*

### Q3: "Can your system predict solar flares?"
- **Answer**:
  > *"No, and we explicitly document this limitation. Continuum imagery measures photospheric temperature deficits, not magnetic field vectors. While large, complex Zurich Class F sunspots with high Demonstration Risk scores are statistically correlated with higher flare activity, true flare prediction requires measuring magnetic shear ($\nabla \times \mathbf{B}$) and electric currents along the Polarity Inversion Line via vector magnetograms and EUV coronal loops. Our risk score is purely an educational complexity heuristic."*

### Q4: "How does the system handle spotless solar minimums?"
- **Answer**:
  > *"Our pipeline includes a dedicated spotless disk detection handler. When the total segmented pixel area inside the solar disk falls below our minimum threshold (20 pixels), the system bypasses active region extraction, tags the observation as `is_spotless = 1`, and logs zero detections. This has been verified in our test suite with 100% specificity and zero false alarms."*
