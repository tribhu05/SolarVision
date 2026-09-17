# SolarVision: Streamlit Community Cloud Deployment Guide

## 1. Overview
This document specifies the deployment architecture, configuration parameters, environment considerations, and platform limitations for deploying **SolarVision** to [Streamlit Community Cloud](https://share.streamlit.io).

---

## 2. Pre-Deployment Verification Checklist

Before initiating deployment, verify that all local checks pass:

- [x] **Python Compatibility**: Runs on Python 3.10 through 3.14 without deprecation warnings.
- [x] **Headless Entry Point**: `app.py` passes AST compilation and headless initialization without requiring display hardware (`scripts/verify_dashboard.py`).
- [x] **Zero-Secret Architecture**: No API keys, passwords, or personal access tokens are required or stored.
- [x] **Defensive Network Fallback**: Live NASA SDO feeds timeout gracefully after 15 seconds, falling back to bundled benchmark frames.
- [x] **Dynamic Schema Provisioning**: `SolarDatabase` creates the `data/` directory and SQLite tables dynamically on first launch.
- [x] **System Libraries**: `packages.txt` provides Debian/Ubuntu system libraries (`libgl1`, `libglib2.0-0`) for headless OpenCV execution.
- [x] **Test Verification**: All 90 automated tests pass (`pytest tests/ -v`).

---

## 3. Step-by-Step Deployment Instructions

### Step 1: Prepare the GitHub Repository
1. Ensure all code changes are committed and pushed to your GitHub repository:
   ```bash
   git add .
   git commit -m "chore: prepare for Streamlit Community Cloud deployment"
   git push origin main
   ```
2. Verify that your repository is either Public or accessible by your Streamlit Community Cloud account.

### Step 2: Configure Streamlit Community Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click the **"New app"** button.
3. Fill in the deployment form:
   - **Repository**: `your-username/SolarVision` (or `charming-darwin`)
   - **Branch**: `main` (or `master`)
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom subdomain (e.g. `solarvision-cv.streamlit.app`) or leave as default.
4. Expand **Advanced settings**:
   - **Python version**: Select `3.11` or `3.12`.
   - **Secrets**: Leave empty (SolarVision operates entirely on open scientific endpoints with zero API keys).
5. Click **"Deploy!"**.

### Step 3: Deployment Monitoring & Container Build
The Streamlit Community Cloud container will automatically:
1. Detect `packages.txt` and execute `apt-get install -y libgl1 libglib2.0-0`.
2. Detect `requirements.txt` and install Python dependencies.
3. Launch `streamlit run app.py --server.headless=true`.
4. Bind port `8501` to standard HTTPS (`port 443`).

---

## 4. Platform-Specific Deployment Limitations

### 4.1 Ephemeral Container Storage
- **Platform Behavior**: Streamlit Community Cloud runs within ephemeral Linux microcontainers. Any files created during runtime (such as newly processed images in `data/raw/` or database writes to `data/solarvision.db`) are cleared when the container restarts or reboots.
- **SolarVision Solution**: 
  - The SQLite database is automatically provisioned in memory/disk upon boot.
  - The repository bundles authentic NASA SDO benchmark imagery (`data/sample_images/`) from the historic May 10–12, 2024 solar storm sequence.
  - The dashboard operates continuously even when deployed on a container with no persistent disk.

### 4.2 Resource & Memory Limits
- **Platform Limits**: Free-tier Streamlit Community Cloud containers provide approximately 1 GB of RAM.
- **SolarVision Optimization**:
  - Full-disk images are processed at $1024 \times 1024$ resolution, consuming less than 25 MB per array.
  - Intermediate contour arrays and CLAHE buffers are garbage-collected proactively.
  - Pyplot backend is not used; all dynamic charts utilize Plotly interactive WebGL/SVG, offloading rendering computation to the user's browser client.

### 4.3 Network Latency & External Telemetry Outages
- **Platform Consideration**: SDO near-realtime image servers (`sdo.gsfc.nasa.gov`) occasionally experience scheduled maintenance or spacecraft momentum maneuvers.
- **SolarVision Solution**: The `SolarDataIngestor` enforces a 15-second timeout with exponential backoff. If NASA servers do not respond, the dashboard displays an informative notification and seamlessly switches to the pre-bundled local benchmark images.

---

## 5. Local Pre-Deployment Testing Commands

Verify that the local environment behaves identically to the headless cloud container:

```bash
# 1. Run complete automated test suite (90 tests)
pytest tests/ -v

# 2. Run the 15-dimension end-to-end integration tests
pytest tests/test_integration_e2e.py -v

# 3. Test headless dashboard initialization
python scripts/verify_dashboard.py

# 4. Test headless Streamlit server execution
python -m streamlit run app.py --server.headless true --server.port 8501
```
