# SolarVision: Scientific Limitations & Scope Boundaries

## 1. Scientific & Academic Boundaries
To maintain rigorous academic integrity, this document delineates the fundamental physical, algorithmic, and computational boundaries of the **SolarVision** system.

---

## 2. Space Weather Non-Prediction Disclaimer
> [!IMPORTANT]
> **NOT AN OPERATIONAL FLARE PREDICTION SYSTEM**
> SolarVision is an educational and research demonstration system in classical Computer Vision. The "Demonstration Risk Score" and "Flare Potential" indicators produced by the pipeline reflect purely morphological visual complexity derived from continuum imagery (photospheric light intensity). They **do not** constitute physical magnetohydrodynamic (MHD) forecasts or operational space weather warnings. Operational space weather forecasting requires vector magnetic field measurements (magnetograms), coronal EUV imagery (e.g. SDO/AIA 131Å, 94Å), coronal mass ejection (CME) coronagraphs (SOHO/LASCO), and multi-variate statistical/deep-learning models.

---

## 3. Computer Vision & Photometric Limitations

### 3.1 Continuum Imagery vs. Magnetic Field Data
- SolarVision operates exclusively on single-band optical continuum imagery ($6173\text{ \AA}$ SDO/HMI or white-light telescopes).
- Continuum data captures thermal absorption contrast (temperature deficit in sunspots caused by magnetic suppression of convective energy transport), but **cannot directly measure magnetic polarity ($\pm B_z$) or magnetic shear ($\nabla \times \mathbf{B}$)**.
- Bipolar classifications (Classes B, C, D, E, F) are inferred from spatial clustering and geometric proximity rather than direct magnetic polarity inversion lines (PILs). In complex active region complexes where multiple flux systems interlock (e.g., $\delta$-configuration sunspots), visual clustering can occasionally misattribute separate small pores to adjacent bipoles.

### 3.2 Severe Geometric Foreshortening Near Solar Limb ($\mu < 0.20$)
- As active regions approach the solar limb ($\rho > 75^\circ$, $\mu = \cos\rho < 0.20$):
  - Area foreshortening correction $\frac{A}{\cos\rho}$ diverges as $\cos\rho \to 0$.
  - The Wilson depression effect causes the 3D geometry of sunspot umbral floors to appear optically occulted by the surrounding photosphere.
  - Penumbral filaments on the disk-center side become severely compressed, distorting morphological symmetry and Zurich classification accuracy.
- **System Safeguard**: Active regions detected with $\rho > 78^\circ$ ($\approx 0.98 R$) are tagged with a limb-proximity warning in both the database and dashboard.

### 3.3 Threshold-Based Segmentation Constraints
- **Granulation Noise**: While bilateral filtering and morphological opening eliminate the vast majority of intergranular dark lanes, solar granulation patterns during periods of low activity can intermittently generate candidate regions near the minimum area threshold ($A \approx 20\text{ px}$).
- **Light Bridges**: Faint photospheric light bridges dividing massive umbral cores can occasionally cause a single giant sunspot (e.g. AR 13664) to be segmented into two tightly adjacent sub-cores or, conversely, merged depending on morphological structuring element size.
- **Fixed Quiet-Sun Ratio**: The empirical threshold ratios ($0.55 I_{\text{quiet}}$ for umbra, $0.88 I_{\text{quiet}}$ for penumbra) are derived from SDO/HMI 6173 Å calibrations. When processing non-SDO imagery (e.g., ground-based amateur white-light telescopes or digitized historical plates), atmospheric seeing variations and different spectral bandpasses can require threshold tuning.

---

## 4. Multi-Day Kinematic Tracking Limitations

### 4.1 Bipartite Association Across Large Temporal Gaps
- The Snodgrass differential rotation model predicts active region positions with high accuracy over standard cadences ($\Delta t \le 24\text{ hours}$, residual MAE $< 1.5^\circ$).
- However, if the observation gap exceeds $48 - 72\text{ hours}$, rapidly evolving active regions (which can double in area or disintegrate within 36 hours) may exceed the area gating constraint ($0.25 \le A_2 / A_1 \le 4.0$), resulting in track termination and re-initialization under a new track ID.

### 4.2 Emerging Flux & Complex Mergers
- When a new active region emerges in close spatial proximity ($< 5^\circ$) to an existing decaying region, the centroid of the clustered contour may experience an apparent discontinuous jump that challenges simple centroid tracking without spatial graph topology.

---

## 5. Deployment & Runtime Constraints
1. **Hosting Environment**: Streamlit Community Cloud runs in ephemeral Linux microcontainers. In-memory session state and temporary SQLite databases are reset upon container restarts unless backed by persistent external object storage.
2. **Network Resilience**: Live near-realtime retrieval from NASA SDO servers is subject to external network latency and occasional SDO calibration roll maneuvers during which live feeds can be temporarily offline or distorted. SolarVision automatically falls back to bundled benchmark images during network timeouts.
