# SolarVision: Mathematical & Algorithmic Methodology

## 1. Introduction
SolarVision applies deterministic, physics-grounded Computer Vision principles to solar photosphere continuum imagery. This document presents the comprehensive mathematical formulations, algorithmic logic, and scientific foundations underpinning each pipeline phase.

---

## 2. Solar Disk Detection & Boundary Localization
The solar disk forms an optical sphere projected as a 2D planar disk upon the detector. Disk center $(x_c, y_c)$ and pixel radius $R$ are required to establish an absolute coordinate frame.

### 2.1 Algorithm
1. **Otsu Global Binarization**:
   The gray-level histogram is partitioned into background (space) and foreground (photosphere) by maximizing inter-class variance $\sigma_B^2(T)$:
   $$\sigma_B^2(T) = \omega_0(T)\omega_1(T)[\mu_0(T) - \mu_1(T)]^2$$
2. **Canny Edge Detection**:
   Computes gradient magnitude $|\nabla I| = \sqrt{G_x^2 + G_y^2}$ and directional angle $\theta = \arctan(G_y / G_x)$, applying non-maximum suppression and hysteresis thresholding ($T_{\text{low}} = 50, T_{\text{high}} = 150$).
3. **Minimum Enclosing Circle Fitting**:
   Extracts the maximum exterior contour $\mathcal{C}_{\text{disk}}$ and computes the circumscribed circle minimizing radius:
   $$\min_{x_c, y_c, R} R \quad \text{subject to} \quad \sqrt{(x_i - x_c)^2 + (y_i - y_c)^2} \le R \quad \forall (x_i, y_i) \in \mathcal{C}_{\text{disk}}$$
4. **Effective Photosphere Mask**:
   To eliminate high-noise edge diffraction at the limb, an inner margin $\kappa = 0.985$ is enforced:
   $$\mathcal{M}_{\text{disk}}(x, y) = \begin{cases} 1 & \text{if } \sqrt{(x - x_c)^2 + (y - y_c)^2} \le \kappa R \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. Quadratic Limb-Darkening Photometric Correction
As optical depth $\tau = 1$ reaches deeper, hotter solar atmospheric layers at normal incidence ($\theta = 0^\circ$) than at glancing limb angles ($\theta \to 90^\circ$), solar intensity naturally drops off towards the edges.

### 3.1 Radiative Transfer Formulation
The standard quadratic solar limb-darkening approximation (Neckel & Labs, 1994; Pierce & Slaughter, 1977) at $6173\text{ \AA}$ (SDO/HMI Fe I absorption continuum) models intensity as:
$$\mu = \cos\theta = \sqrt{1 - \left(\frac{r}{R}\right)^2}$$
$$\frac{I(\mu)}{I(\mu = 1)} = 1 - u(1 - \mu) - v(1 - \mu)^2$$
For the SDO/HMI optical filter bandpass, empirically calibrated coefficients are:
$$u = 0.56, \quad v = 0.20$$

### 3.2 Photometric Flattening
To normalize the disk background to a uniform intensity $I_{\text{quiet}} \approx 200 \text{ DN}$:
$$I_{\text{flattened}}(x, y) = \frac{I_{\text{raw}}(x, y)}{1 - u(1 - \mu) - v(1 - \mu)^2} \cdot \mathcal{M}_{\text{disk}}(x, y)$$
This step eliminates false detections near the limb where natural solar darkening mimics sunspot penumbral intensities.

---

## 4. Denoising, Contrast Enhancement & Morphological Filtering

### 4.1 Bilateral Filter (Edge-Preserving Smoothing)
Removes granulations and optical shot noise while preserving steep umbra/penumbra boundaries:
$$I_{\text{filtered}}(x, y) = \frac{1}{W_p} \sum_{(x_i, y_i) \in \Omega} I(x_i, y_i) \exp\left(-\frac{\|(x, y) - (x_i, y_i)\|^2}{2\sigma_s^2}\right) \exp\left(-\frac{|I(x, y) - I(x_i, y_i)|^2}{2\sigma_r^2}\right)$$
Parameters: spatial diameter $d = 9$, color standard deviation $\sigma_r = 75$, coordinate standard deviation $\sigma_s = 75$.

### 4.2 Contrast Limited Adaptive Histogram Equalization (CLAHE)
Divides the disk into $8 \times 8$ contextual tiles, redistributing the local histogram with a clip limit of $2.0$ to prevent noise amplification while highlighting subtle magnetic pore structures.

### 4.3 Morphological Black-Hat Transform
Extracts features darker than their surrounding quiet photosphere:
$$T_{\text{blackhat}}(I) = \phi_B(I) - I = (I \bullet B) - I$$
where $\bullet$ represents morphological closing using a disk structuring element $B$ of radius $5\text{ px}$.

---

## 5. Hierarchical Photometric Segmentation
Sunspots consist of two distinct thermodynamic components:
1. **Umbra**: Dark central core ($T \approx 3,700\text{ K}$), strong vertical magnetic fields ($B \sim 2,000 - 3,500\text{ G}$).
2. **Penumbra**: Lighter filamentary skirt ($T \approx 5,400\text{ K}$), inclined transverse magnetic fields ($B \sim 1,000 - 1,800\text{ K}$).

### 5.1 Dual-Level Thresholding Formulation
Let $I_{\text{quiet}}$ denote the median intensity of quiet photosphere pixels in the central disk ($\mu > 0.8$):
$$\mathcal{M}_{\text{umbra}}(x, y) = \begin{cases} 1 & \text{if } I_{\text{flattened}}(x, y) \le 0.55 \cdot I_{\text{quiet}} \\ 0 & \text{otherwise} \end{cases}$$
$$\mathcal{M}_{\text{penumbra}}(x, y) = \begin{cases} 1 & \text{if } 0.55 \cdot I_{\text{quiet}} < I_{\text{flattened}}(x, y) \le 0.88 \cdot I_{\text{quiet}} \\ 0 & \text{otherwise} \end{cases}$$
$$\mathcal{M}_{\text{total}}(x, y) = \mathcal{M}_{\text{umbra}}(x, y) \cup \mathcal{M}_{\text{penumbra}}(x, y)$$

### 5.2 Morphological Cleanup and Area Gating
- Applied an opening operation $\mathcal{M} \circ S$ ($3 \times 3$ ellipse) to eliminate isolated noisy pixels.
- Applied a closing operation $\mathcal{M} \bullet S$ ($5 \times 5$ ellipse) to join closely clustered pores into unified active region complexes.
- Regions with projected pixel area $A_{\text{pixels}} < 20$ (optical noise/granule lanes) or $A_{\text{pixels}} > 50,000$ (non-physical anomalies) are rejected.

---

## 6. Calibrated Heliographic Coordinate Transformations
Because the Sun is a 3D sphere viewed in orthographic projection, pixel coordinates $(c_x, c_y)$ must be transformed into physical Stonyhurst heliographic latitude $B$ and Central Meridian Distance (CMD) longitude $L$.

### 6.1 Spherical Forward Projection
1. Normalize coordinates relative to solar disk center:
   $$x_n = \frac{c_x - x_c}{R}, \quad y_n = \frac{y_c - c_y}{R}$$
2. Radial heliocentric distance $\rho$:
   $$\sin\rho = \sqrt{x_n^2 + y_n^2}, \quad \cos\rho = \sqrt{1 - \sin^2\rho}$$
3. Stonyhurst Heliographic Coordinates ($B_0$ is solar sub-Earth heliographic tilt angle, $B_0 \approx 0^\circ$ nominal approximation):
   $$B = \arcsin\left(\sin B_0 \cos\rho + \cos B_0 \sin\rho \frac{y_n}{\sin\rho}\right)$$
   $$L = \arcsin\left(\frac{x_n \sin\rho}{\sin\rho \cos B}\right) = \arcsin\left(\frac{x_n}{\cos B}\right)$$
   Coordinates are converted to decimal degrees: $B \in [-90^\circ, +90^\circ]$ (positive North), $L \in [-90^\circ, +90^\circ]$ (positive West, negative East).

### 6.2 Foreshortening Area Correction (Micro-Hemispheres)
Sunspots appear foreshortened near the solar limb due to line-of-sight perspective. The true physical area $A_{\text{MSH}}$ expressed in millionths of the solar hemisphere (MSH, standard astrophysical unit) is:
$$\cos\theta = \cos\rho = \sqrt{1 - \left(\frac{r}{R}\right)^2}$$
$$A_{\text{MSH}} = \frac{A_{\text{pixels}}}{2\pi R^2 \cos\theta} \times 10^6$$
where $2\pi R^2$ is the total projected pixel area of one solar hemisphere.

---

## 7. Morphological Classification (Modified Zurich / McIntosh)
The Zurich classification categorizes sunspot groups based on magnetic maturity, bipolar structure, penumbral symmetry, and longitudinal extent:

| Class | Scientific Definition | Deterministic Rule in SolarVision |
| :--- | :--- | :--- |
| **A** | Small unipolar pore without penumbra | $A < 100\text{ MSH}$, Penumbra Area $= 0$, Spot Count $= 1$ |
| **B** | Bipolar group without penumbra | Penumbra Area $= 0$, Spot Count $\ge 2$, Bipolar |
| **C** | Bipolar group with penumbra on only one spot | Penumbra Area $> 0$, Asymmetric penumbra, Extent $< 10^\circ$ |
| **D** | Bipolar group with penumbra on both ends, compact | Penumbra on both poles, Extent $< 10^\circ$ |
| **E** | Extended bipolar group | Extent $10^\circ - 15^\circ$, Penumbra on both ends |
| **F** | Very large, complex bipolar group | Extent $> 15^\circ$ or Area $> 500\text{ MSH}$ |
| **H** | Large unipolar spot with mature, symmetric penumbra | Spot Count $= 1$, Penumbra Area $> 0$, Circularity $> 0.6$ |

---

## 8. Multi-Day Kinematic Tracking (Snodgrass Differential Rotation)
Because the Sun is a gaseous body, its rotation rate varies with latitude $B$.

### 8.1 Empirical Differential Rotation Law
SolarVision employs the Snodgrass (1984) surface rotation model:
$$\omega(B) = A + B_{\text{rot}}\sin^2(B) + C_{\text{rot}}\sin^4(B)$$
where:
$$A = 14.71^\circ/\text{day} \quad (\text{equatorial rate})$$
$$B_{\text{rot}} = -2.39^\circ/\text{day}$$
$$C_{\text{rot}} = -1.78^\circ/\text{day}$$

### 8.2 Kinematic Forward Propagation
Given an active region at $(B_1, L_1)$ observed at timestamp $t_1$, its expected longitude $L_{\text{pred}}$ at timestamp $t_2$ (with $\Delta t = t_2 - t_1$ in fractional days) is:
$$L_{\text{pred}} = L_1 + \omega(B_1) \cdot \Delta t$$

### 8.3 Bipartite Association Cost Matrix
Candidate regions in frame 2 are associated with tracks using a multi-parameter gating filter:
1. **Latitude Gate**: $|B_2 - B_1| \le 5.0^\circ$ (active regions exhibit minimal latitudinal drift over days).
2. **Longitude Gate**: $|L_2 - L_{\text{pred}}| \le 8.0^\circ$.
3. **Area Change Gate**: $0.25 \le \frac{A_2}{A_1} \le 4.0$ (disallows physically implausible instantaneous growth/decay).
4. **Optimal Matching**: Solved via minimum Euclidean distance in $(B, L_{\text{pred}})$ parameter space.
5. **Lifecycle Assignment**:
   - $\Delta L_{\text{pred}} > 80^\circ$: *Rotated Off-Disk* (passed the Western limb).
   - Unmatched active track $> 48\text{ hours}$: *Decayed*.
   - Unmatched new detection: *New Track Initialized*.

---

## 9. Educational Demonstration Risk Scoring
The Demonstration Risk Score ($S \in [0, 100]$) provides an intuitive pedagogical indicator aggregating five physical factors:
$$S = \sum_{i=1}^5 w_i \cdot f_i$$
with weights:
1. **Morphological Complexity ($w_1 = 0.35$)**: Scaled by Zurich class (Class A = 0.1, D = 0.6, F = 1.0).
2. **Physical Area ($w_2 = 0.25$)**: Normalized logarithmic scale $\min\left(1.0, \frac{\log_{10}(\text{Area}_{\text{MSH}} + 1)}{3.0}\right)$.
3. **Penumbra Maturity ($w_3 = 0.20$)**: Ratio of penumbral area to total region area.
4. **Intensity Contrast ($w_4 = 0.10$)**: Core umbral contrast depth $1 - \frac{I_{\text{min}}}{I_{\text{quiet}}}$.
5. **Geometric Compactness ($w_5 = 0.10$)**: Normalized longitudinal span $\min\left(1.0, \frac{\Delta L}{20^\circ}\right)$.

> **Academic Disclaimer**: This score is strictly an educational complexity heuristic designed for computer vision demonstration. It does **not** constitute a validated physical solar flare prediction model.
