# SolarVision: Deployment Architecture Diagram

## Deployment Topology
SolarVision is engineered to operate seamlessly in two distinct environments:
1. **Local Developer & Academic Workstation**: Direct Python environment, local filesystem, persistent SQLite database (`data/solarvision.db`), and full hardware access.
2. **Streamlit Community Cloud (Hosted Container)**: Ephemeral Linux container, headless Streamlit runtime, zero-secret environment, dynamically initialized SQLite database, and automatic port binding.

```mermaid
graph TB
    subgraph Client["Client Browser Layer"]
        B_DESK["Desktop Browser (Chrome / Edge / Firefox)"]
        B_MOB["Mobile / Tablet Viewport"]
    end

    subgraph Streamlit_Cloud["Streamlit Community Cloud (Production / Academic Demo)"]
        GH["GitHub Repository<br/>(tribh/SolarVision main branch)"] -->|Auto Webhook Deployment| POD["Ephemeral Linux Microcontainer<br/>(Debian Base, Python 3.10-3.14)"]
        
        subgraph Container["Container Runtime"]
            REQ["requirements.txt<br/>(opencv-python, numpy, streamlit, plotly, etc.)"]
            BOOT["Streamlit Server Entry: app.py<br/>(Headless mode: --server.headless=true)"]
            
            subgraph App_Context["In-Memory App Session State"]
                ST_CACHE["@st.cache_resource & Session State<br/>(Active Pipeline, Ingestor, Config)"]
                CORE_PIPE["SolarVision Pipeline Engine<br/>(src/pipeline.py)"]
            end

            subgraph Local_Storage["Ephemeral Storage System"]
                SQL_CLOUD[("SQLite File: data/solarvision.db<br/>Auto-initialized via schema")]
                SAMPLE_IMG["data/sample_images/<br/>Bundled SDO Benchmark Images"]
            end
        end

        POD --> REQ
        REQ --> BOOT
        BOOT --> ST_CACHE
        ST_CACHE --> CORE_PIPE
        CORE_PIPE --> SQL_CLOUD
        CORE_PIPE --> SAMPLE_IMG
    end

    subgraph External_Web["Public Scientific Endpoints (Zero-Key Access)"]
        SDO_EP["NASA SDO / HMI Near-Realtime JPG<br/>sdo.gsfc.nasa.gov"]
        NOAA_EP["NOAA SWPC Solar Region Summaries<br/>services.swpc.noaa.gov"]
    end

    subgraph Local_Env["Local Workstation Environment"]
        VENV["Python Virtual Environment (venv)"]
        LOCAL_DB[("Local Persistent SQLite<br/>data/solarvision.db")]
        LOCAL_TESTS["Pytest Test Suite (75 Tests)"]
        LOCAL_SCRIPTS["Verification Scripts (verify_dashboard.py)"]
    end

    %% Network interactions
    B_DESK -->|HTTPS / WSS (Port 443)| BOOT
    B_MOB -->|HTTPS / WSS (Port 443)| BOOT
    CORE_PIPE -.->|Direct HTTPS GET| SDO_EP
    CORE_PIPE -.->|Direct HTTPS GET| NOAA_EP

    VENV --> LOCAL_DB
    VENV --> LOCAL_TESTS
    VENV --> LOCAL_SCRIPTS

    classDef client fill:#e0f7fa,stroke:#00838f,stroke-width:2px;
    classDef cloud fill:#ede7f6,stroke:#512da8,stroke-width:2px;
    classDef container fill:#ffffff,stroke:#673ab7,stroke-dasharray: 5 5;
    classDef external fill:#e8eaf6,stroke:#283593,stroke-width:2px;
    classDef local fill:#f1f8e9,stroke:#558b2f,stroke-width:2px;

    class B_DESK,B_MOB client;
    class GH,POD cloud;
    class REQ,BOOT,ST_CACHE,CORE_PIPE,SQL_CLOUD,SAMPLE_IMG container;
    class SDO_EP,NOAA_EP external;
    class VENV,LOCAL_DB,LOCAL_TESTS,LOCAL_SCRIPTS local;
```

## Production & Cloud Readiness Safeguards
1. **Zero Secret Footprint**: The application does not require or store any API keys or tokens. All NASA SDO near-realtime feeds and NOAA SWPC products are openly accessible public domain scientific resources.
2. **Dynamic Database Creation**: If `data/solarvision.db` does not exist upon container initialization, `SolarVisionDatabase` automatically provisions the tables, constraints, and initial records on first launch.
3. **Headless & Network Fault Tolerance**:
   - Streamlit launches with `--server.headless true` to avoid graphical shell assumptions.
   - Network requests to NASA SDO employ defensive timeouts (10 seconds) and fallback to bundled benchmark images if the external endpoint is unreachable.
4. **Mobile & Viewport Responsiveness**: Uses CSS flex columns, responsive Plotly configurations (`autosizable: True`), and metric wrappers for high-DPI tablets and mobile devices.
