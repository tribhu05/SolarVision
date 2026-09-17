# SolarVision: UML Use Case Diagram

## 1. Overview
The SolarVision system interacts with three primary actors:
- **Solar Researcher / Space Weather Analyst**: Explores active region morphology, monitors multi-day kinematic trajectories, inspects Stonyhurst coordinates, and exports verified region catalogs.
- **Student / Course Evaluator**: Interactively navigates the 8-page Streamlit dashboard, tests synthetic and real edge-case frames, examines before/after visual diagnostic stages, and inspects algorithmic decision traces.
- **Automated Pipeline / SDO Telemetry Service**: Ingests automated telemetry feeds, deduplicates frames via SHA-256 hashing, orchestrates computer vision transformations, and persists records into the SQLite catalog.

---

## 2. UML Use Case Diagram (Mermaid)

```mermaid
flowchart LR
    %% Actors
    subgraph Actors ["System Actors"]
        Researcher(("Solar Researcher /<br/>Space Weather Analyst"))
        Evaluator(("Student /<br/>Course Evaluator"))
        SDOFeed(("NASA SDO Telemetry<br/>Service"))
    end

    %% Use Cases
    subgraph SolarVisionSystem ["SolarVision System Boundaries"]
        UC1(["UC-01: Ingest & Deduplicate Solar Imagery"])
        UC2(["UC-02: Detect Solar Disk & Radius (Otsu/Canny)"])
        UC3(["UC-03: Compensate Photospheric Limb Darkening"])
        UC4(["UC-04: Segment Umbra & Penumbra (Adaptive Dual-Thresh)"])
        UC5(["UC-05: Extract Heliographic Stonyhurst Coordinates (B, L)"])
        UC6(["UC-06: Calculate True Physical Area in MSH (Foreshortening)"])
        UC7(["UC-07: Classify Morphology (Modified Zurich A-H)"])
        UC8(["UC-08: Track Multi-Day Kinematics (Snodgrass Rotation)"])
        UC9(["UC-09: Persist Observations in Relational SQLite Catalog"])
        UC10(["UC-10: Benchmark against NOAA SWPC Ground Truth"])
        UC11(["UC-11: Interactive Multi-View Dashboard Exploration"])
        UC12(["UC-12: Export Catalog Data (CSV / JSON)"])
    end

    %% Relationships
    SDOFeed --> UC1
    UC1 --> UC2
    UC2 --> UC3
    UC3 --> UC4
    UC4 --> UC5
    UC5 --> UC6
    UC6 --> UC7
    UC7 --> UC8
    UC8 --> UC9

    Evaluator --> UC11
    Evaluator --> UC10
    Evaluator --> UC3
    Evaluator --> UC4

    Researcher --> UC11
    Researcher --> UC7
    Researcher --> UC8
    Researcher --> UC10
    Researcher --> UC12
```

---

## 3. Detailed Use Case Specifications

### UC-01: Ingest & Deduplicate Solar Imagery
- **Primary Actor**: Automated SDO Telemetry Service / User
- **Preconditions**: Network connection available or local SDO/HMI image provided.
- **Flow of Events**:
  1. Fetch image from NASA SDO or load from disk.
  2. Compute SHA-256 hash.
  3. Query database; if hash exists, retrieve cached observation.
  4. If new, validate dimensions ($1024 \times 1024$ minimum), format, and depth.
- **Postconditions**: Ingested image ready for preprocessing pipeline.

### UC-04: Segment Umbra & Penumbra
- **Primary Actor**: Solar Researcher / Evaluator
- **Preconditions**: Limb-darkening flattened image available.
- **Flow of Events**:
  1. Calculate quiet-Sun central photospheric median intensity $I_{\text{quiet}}$.
  2. Apply umbral threshold $I \le 0.55 \cdot I_{\text{quiet}}$.
  3. Apply penumbral threshold $0.55 \cdot I_{\text{quiet}} < I \le 0.88 \cdot I_{\text{quiet}}$.
  4. Perform morphological opening ($3\times 3$) and closing ($5\times 5$).
  5. Extract 8-way connected contours with area gating ($20 \le A_{\text{px}} \le 50,000$).
- **Postconditions**: Distinct umbral and penumbral binary masks and contour lists generated.

### UC-08: Track Multi-Day Kinematics
- **Primary Actor**: Solar Researcher / Pipeline
- **Preconditions**: Multi-day sequence of observations processed.
- **Flow of Events**:
  1. For each active region at Stonyhurst latitude $B$, evaluate Snodgrass angular velocity:
     $$\omega(B) = 14.713 - 2.396\sin^2(B) - 1.787\sin^4(B) \quad [\text{deg/day}]$$
  2. Forward-propagate expected longitude: $L_{\text{pred}} = L_0 + \omega(B) \cdot \Delta t$.
  3. Gate spatial candidates: $|\Delta B| \le 5.0^\circ$, $|\Delta L - \omega\Delta t| \le 6.0^\circ$.
  4. Match regions using bipartite assignment, assign persistent `track_id`, update growth rates.
- **Postconditions**: Persistent trajectories and growth rates stored in SQLite.
