# SolarVision: Data Processing Workflow Diagram

## Pipeline Workflow Specification
The SolarVision processing pipeline executes sequentially from raw satellite payload ingestion to interactive visualization and persistence. Each stage is strictly typed, deterministic, and idempotent.

```mermaid
flowchart TD
    %% Stages
    DS["1. Data Source<br/>NASA SDO / HMI 6173 Å Continuum Stream"] --> DI["2. Data Ingestion (solar_data.py)<br/>Network retrieval, format validation, SHA-256 caching"]
    DI --> IP["3. Image Preprocessing (preprocessor.py & limb_darkening.py)<br/>Disk localization, Limb-darkening flattening, CLAHE & Black-Hat filtering"]
    IP --> SG["4. Segmentation (segmentation.py & detector.py)<br/>Dual-level adaptive thresholding, Umbra/Penumbra separation, Area filter"]
    SG --> FE["5. Feature Extraction (feature_extractor.py)<br/>Stonyhurst heliographic projection (B, L), Area in MSH, Circularity, Intensity"]
    FE --> CL["6. Classification & Risk (classifier.py)<br/>Modified Zurich-McIntosh classification & Demonstration Risk score"]
    CL --> TR["7. Multi-Day Tracking (tracker.py)<br/>Snodgrass differential rotation kinematic propagation & bipartite matching"]
    TR --> DB["8. Database Persistence (database.py)<br/>SQLite relational storage: observations, regions, and trajectories"]
    DB --> UI["9. Streamlit Dashboard (app.py)<br/>Interactive 8-page scientific analysis, Plotly charts, CSV/JSON export"]

    %% Detailed sub-steps
    subgraph S_IP["Preprocessing Detail"]
        IP1["Otsu Threshold + Canny Edges"] --> IP2["Fit Minimum Enclosing Circle (xc, yc, R)"]
        IP2 --> IP3["Compute mu = cos(theta) = sqrt(1 - (r/R)^2)"]
        IP3 --> IP4["Flattening: I_norm = I_raw / (1 - u(1-mu) - v(1-mu)^2)"]
        IP4 --> IP5["Bilateral Filter (d=9, sigma=75) + CLAHE (clip=2.0)"]
    end

    subgraph S_SG["Segmentation Detail"]
        SG1["Quiet-Sun Median Intensity Estimation (I_quiet)"] --> SG2["Umbra Threshold: I <= 0.55 * I_quiet"]
        SG1 --> SG3["Penumbra Threshold: 0.55 < I <= 0.88 * I_quiet"]
        SG2 --> SG4["Morphological Close (3x3 ellipse)"]
        SG3 --> SG4
        SG4 --> SG5["Hierarchy Parsing & Area Filtering (20 <= A <= 50,000 px)"]
    end

    subgraph S_FE["Feature Extraction Detail"]
        FE1["Centroid (cx, cy) -> Heliocentric Normalized (x/R, y/R)"] --> FE2["Forward Orthographic -> Stonyhurst Latitude B & Central Meridian Distance L"]
        FE2 --> FE3["Area in Micro-Hemispheres: A_MSH = (A_px / (2 * pi * R^2 * cos(rho))) * 10^6"]
        FE3 --> FE4["Circularity: C = 4 * pi * Area / Perimeter^2"]
        FE4 --> FE5["Core-to-Penumbra Intensity Ratio"]
    end

    subgraph S_TR["Tracking Kinematics Detail"]
        TR1["Observation Delta Time: dt = t2 - t1 (days)"] --> TR2["Snodgrass Rotation: dL = (14.71 - 2.39 sin^2(B) - 1.78 sin^4(B)) * dt"]
        TR2 --> TR3["Predicted Heliographic Longitude: L_pred = L1 + dL"]
        TR3 --> TR4["Gating Filter: |B2 - B1| < 5.0 deg & 0.25 <= A2/A1 <= 4.0"]
        TR4 --> TR5["Maintain Persistent Track ID or Assign New Track ID"]
    end

    IP -.-> S_IP
    SG -.-> S_SG
    FE -.-> S_FE
    TR -.-> S_TR

    classDef stage fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef sub fill:#f5f5f5,stroke:#9e9e9e,stroke-width:1px;

    class DS,DI,IP,SG,FE,CL,TR,DB,UI stage;
    class S_IP,S_SG,S_FE,S_TR sub;
```

## Step-by-Step Mathematical & Algorithmic Execution
1. **Data Ingestion**: Accepts local JPG/PNG continuum images or queries NASA SDO JSON feed. Verifies 8-bit or 16-bit depth and generates SHA-256 hash.
2. **Preprocessing**:
   - Isolates the solar disk with radius $R$ and center $(x_c, y_c)$.
   - Applies the limb-darkening compensation function:
     $$\mu = \cos\theta = \sqrt{1 - \left(\frac{r}{R}\right)^2}$$
     $$I_{\text{flattened}}(x, y) = \frac{I_{\text{raw}}(x, y)}{1 - u(1 - \mu) - v(1 - \mu)^2}$$
     where $u = 0.56$ and $v = 0.20$ at $6173\text{ \AA}$.
3. **Dual-Level Segmentation**:
   - Computes quiet-Sun median disk intensity $I_{\text{quiet}}$.
   - Umbral mask: $M_u = \{ (x, y) \mid I(x, y) \le 0.55 \cdot I_{\text{quiet}} \}$.
   - Penumbral mask: $M_p = \{ (x, y) \mid 0.55 \cdot I_{\text{quiet}} < I(x, y) \le 0.88 \cdot I_{\text{quiet}} \}$.
4. **Heliographic Projection**:
   - Corrects foreshortening using orthographic projection onto Stonyhurst coordinates $(\text{B}, \text{L})$.
   - Corrects physical area:
     $$\text{Area}_{\text{MSH}} = \frac{A_{\text{pixels}}}{2\pi R^2 \cos\rho} \times 10^6$$
5. **Modified Zurich Classification**:
   - Classifies regions into classes A through H based on area, umbra/penumbra presence, and bipolar longitudinal span.
6. **Multi-Day Kinematic Tracking**:
   - Propagates heliographic longitude using Snodgrass (1984) differential rotation:
     $$\omega(B) = 14.71 - 2.39\sin^2 B - 1.78\sin^4 B \quad [^\circ/\text{day}]$$
7. **Persistence & Evaluation**:
   - Stores all metadata and regions in SQLite.
   - Evaluates performance against NOAA benchmark observations.
