"""
SolarVision: Automated Solar Active Region Detection and Analysis
VIT B.Tech Computer Vision Project
Interactive Streamlit Dashboard
"""

import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.config import load_config, SolarVisionConfig
from src.database import SolarDatabase
from src.disk_detector import SolarDiskDetector, SolarDiskGeometry
from src.limb_darkening import LimbDarkeningCorrector, LimbCorrectionResult
from src.segmentation import SunspotSegmenter, SegmentationResult
from src.feature_extractor import FeatureExtractor, CalibratedActiveRegion
from src.classifier import McIntoshClassifier, ClassificationResult
from src.tracker import ActiveRegionTracker, TrackedObservation, TrackHistory, TRACKING_DISCLAIMER
from src.solar_data import SolarDataIngestor, ImageMetadata, IngestionResult
from src.preprocessor import SolarImagePreprocessor, PreprocessingResult
from src.detector import SunspotDetector, DetectionOutput, DetectedRegion

# Page setup
st.set_page_config(
    page_title="SolarVision: Solar Active Region Analysis",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF8C00;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #FF8C00;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 8px 16px;
    }
    .meta-box {
        background-color: #f1f3f5;
        border-radius: 6px;
        padding: 10px;
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_system_components():
    config = load_config()
    db = SolarDatabase(config.storage.database_path)
    ingestor = SolarDataIngestor(config.data_source, config.storage.catalog_index_path)
    ingestor.prepare_real_sample_dataset()
    return config, db, ingestor


config, db, ingestor = get_system_components()


# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("## ☀️ **SolarVision Engine**")
st.sidebar.caption("Automated Active Region Detection & Analysis")

nav_choice = st.sidebar.radio(
    "Navigation Mode",
    [
        "🔬 Single-Image Active Region Detector",
        "🛰️ Multi-Frame Sequence Tracking",
        "📊 Solar Catalog & Analytics",
        "🌐 Real NASA SDO Feed",
        "📚 Scientific Methodology",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ **Computer Vision Tuning**")

with st.sidebar.expander("Physics & Threshold Parameters", expanded=False):
    u_coeff = st.slider(
        "Limb Darkening Coeff (u)",
        min_value=0.20,
        max_value=0.90,
        value=float(config.limb_darkening.u_coefficient),
        step=0.05,
        help="Eddington limb darkening coefficient for visible continuum (SDO/HMI 6173 Å: ~0.60)",
    )
    t_umbra_factor = st.slider(
        "Umbra Threshold Factor",
        min_value=0.30,
        max_value=0.75,
        value=float(config.segmentation.umbra_threshold_factor),
        step=0.02,
        help="Fraction of quiet-Sun intensity defining umbra core boundary",
    )
    t_penumbra_factor = st.slider(
        "Penumbra Threshold Factor",
        min_value=0.70,
        max_value=0.95,
        value=float(config.segmentation.penumbra_threshold_factor),
        step=0.02,
        help="Fraction of quiet-Sun intensity defining penumbral outer boundary",
    )
    min_area_px = st.slider(
        "Min Sunspot Area (pixels)",
        min_value=5,
        max_value=60,
        value=int(config.segmentation.min_sunspot_area_pixels),
        step=1,
        help="Minimum connected pixel footprint to filter out granulation noise",
    )
    cluster_dist_deg = st.slider(
        "AR Clustering Distance (deg)",
        min_value=1.0,
        max_value=12.0,
        value=float(config.segmentation.clustering_distance_deg),
        step=0.5,
        help="Angular distance threshold to group multiple spots into a single Active Region",
    )

with st.sidebar.expander("Classical CV Preprocessing", expanded=False):
    denoise_method = st.selectbox(
        "Denoise Filter",
        ["Bilateral (Edge-Preserving)", "Gaussian", "Median", "None"],
        index=0,
        help="Smooths solar granulation noise while preserving sharp sunspot edges",
    )
    bilateral_sc = st.slider(
        "Bilateral Sigma Color",
        min_value=10.0,
        max_value=50.0,
        value=25.0,
        step=5.0,
        help="Intensity range for edge-preserving bilateral filtering",
    )
    contrast_method = st.selectbox(
        "Contrast Method",
        ["CLAHE (Adaptive)", "Linear Stretch", "None"],
        index=0,
        help="Careful enhancement to avoid creating artificial sunspots",
    )
    clahe_clip = st.slider(
        "CLAHE Clip Limit",
        min_value=1.0,
        max_value=3.0,
        value=1.8,
        step=0.2,
        help="Strict threshold to prevent over-amplification of noise",
    )
    enable_bhat = st.checkbox(
        "Enable Black-Hat Dark Feature Map",
        value=True,
        help="Morphological Black-Hat transform isolates dark features (pores/sunspots)",
    )

with st.sidebar.expander("Demonstration Risk Scoring", expanded=False):
    st.caption("Configurable factor weights & attention thresholds (Educational Indicator)")
    w_area = st.slider("Area Weight", 0.05, 0.50, float(config.classification.risk_weight_area), 0.05)
    w_comp = st.slider("Complexity Weight", 0.05, 0.50, float(config.classification.risk_weight_complexity), 0.05)
    w_pen = st.slider("Penumbra Weight", 0.05, 0.50, float(config.classification.risk_weight_penumbra), 0.05)
    w_con = st.slider("Contrast Weight", 0.05, 0.50, float(config.classification.risk_weight_contrast), 0.05)
    w_circ = st.slider("Compactness Weight", 0.05, 0.50, float(config.classification.risk_weight_compactness), 0.05)
    th_low = st.slider("Low Attention Cutoff", 15.0, 50.0, float(config.classification.low_attention_threshold), 5.0)
    th_mod = st.slider("Moderate Attention Cutoff", 50.0, 85.0, float(config.classification.moderate_attention_threshold), 5.0)

with st.sidebar.expander("Multi-Day Kinematic Tracking", expanded=False):
    st.caption("Snodgrass kinematic matching & gating parameters")
    max_match_dist = st.slider(
        "Max Match Distance (deg)",
        2.0, 15.0,
        float(config.tracking.max_matching_dist_deg),
        0.5,
        help="Maximum residual angular distance to associate an active region across observations",
    )
    max_lat_tol = st.slider(
        "Max Latitude Drift (deg)",
        1.0, 10.0,
        float(config.tracking.max_lat_diff_deg),
        0.5,
        help="Maximum allowed heliographic latitude drift between observations",
    )
    limb_cutoff = st.slider(
        "Limb Cutoff (deg)",
        60.0, 85.0,
        float(config.tracking.limb_cutoff_deg),
        1.0,
        help="Stonyhurst longitude beyond which regions are considered to rotate over the western limb",
    )

# Instantiate pipeline modules with user-tuned parameters
trk_cfg = config.tracking
trk_cfg.max_matching_dist_deg = max_match_dist
trk_cfg.max_lat_diff_deg = max_lat_tol
trk_cfg.limb_cutoff_deg = limb_cutoff
disk_detector = SolarDiskDetector(config.disk_detection)
limb_corrector = LimbDarkeningCorrector(config.limb_darkening)
limb_corrector.config.u_coefficient = u_coeff

prep_cfg = config.preprocessing
prep_cfg.denoise_method = denoise_method.split()[0].lower() if denoise_method != "None" else "none"
prep_cfg.bilateral_sigma_color = bilateral_sc
prep_cfg.contrast_method = contrast_method.split()[0].lower() if contrast_method != "None" else "none"
prep_cfg.clahe_clip_limit = clahe_clip
prep_cfg.enable_blackhat = enable_bhat
preprocessor = SolarImagePreprocessor(prep_cfg, config)

seg_config = config.segmentation
seg_config.umbra_threshold_factor = t_umbra_factor
seg_config.penumbra_threshold_factor = t_penumbra_factor
seg_config.min_sunspot_area_pixels = min_area_px
seg_config.clustering_distance_deg = cluster_dist_deg

clf_cfg = config.classification
clf_cfg.risk_weight_area = w_area
clf_cfg.risk_weight_complexity = w_comp
clf_cfg.risk_weight_penumbra = w_pen
clf_cfg.risk_weight_contrast = w_con
clf_cfg.risk_weight_compactness = w_circ
clf_cfg.low_attention_threshold = th_low
clf_cfg.moderate_attention_threshold = th_mod

segmenter = SunspotSegmenter(seg_config)
feature_extractor = FeatureExtractor(config.solar_physics)
classifier = McIntoshClassifier(clf_cfg)
sunspot_detector = SunspotDetector(
    config=seg_config,
    physics_config=config.solar_physics,
    full_config=config,
)


def run_pipeline_on_image(image_bgr: np.ndarray) -> Tuple[
    SolarDiskGeometry,
    LimbCorrectionResult,
    SegmentationResult,
    List[CalibratedActiveRegion],
    List[ClassificationResult],
    np.ndarray,
    PreprocessingResult,
    DetectionOutput,
]:
    """Execute complete end-to-end solar computer vision pipeline."""
    # 1. Classical Preprocessing
    prep_res = preprocessor.process(image_bgr)
    disk = prep_res.solar_disk
    limb = prep_res.limb_result

    # 2. Umbra / Penumbra Segmentation
    seg = segmenter.segment(limb, disk)
    # 3. Heliographic & Physical Feature Calibration
    regions = feature_extractor.extract_features(seg.regions, disk)
    # 4. McIntosh Classification
    classes = classifier.classify_all(regions)

    # 5. Sunspot Detection & Morphological Feature Extraction
    det_output = sunspot_detector.detect(image_bgr, solar_disk=disk, quiet_sun_intensity=limb.quiet_sun_intensity)

    # 6. Annotated Visualization
    annotated = det_output.annotated_image

    return disk, limb, seg, regions, classes, annotated, prep_res, det_output


# ==============================================================================
# VIEW 1: SINGLE-IMAGE ACTIVE REGION DETECTOR
# ==============================================================================
if nav_choice == "🔬 Single-Image Active Region Detector":
    st.markdown('<div class="main-header">🔬 Solar Active Region Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated detection, photospheric limb correction, and McIntosh classification</div>', unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns([2, 1])

    with col_ctrl1:
        # Image Source selector
        sample_files = sorted(list(Path(config.storage.sample_data_dir).glob("*.jpg")) + list(Path(config.storage.sample_data_dir).glob("*.png")))
        sample_names = [p.name for p in sample_files]

        src_mode = st.radio("Image Input Source", ["Select Real NASA SDO Observation", "Upload Solar Image (PNG/JPG)"], horizontal=True)

        selected_image_path = None
        uploaded_bytes = None
        current_metadata = None

        if src_mode == "Select Real NASA SDO Observation":
            if sample_names:
                chosen_sample = st.selectbox("Choose Real Solar Image:", sample_names, index=0)
                selected_image_path = Path(config.storage.sample_data_dir) / chosen_sample
            else:
                st.warning("No sample images found in directory.")
        else:
            uploaded_file = st.file_uploader("Upload full-disk solar continuum image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                uploaded_bytes = uploaded_file.read()

    # Load Image
    current_image_bgr = None
    image_label = ""

    if selected_image_path and selected_image_path.exists():
        ingest_res = ingestor.load_local_image(selected_image_path)
        if ingest_res.success and ingest_res.image is not None:
            current_image_bgr = ingest_res.image
            current_metadata = ingest_res.metadata
            image_label = selected_image_path.name
        else:
            st.error(f"Failed to load image: {ingest_res.error_message}")
    elif uploaded_bytes is not None:
        file_bytes = np.asarray(bytearray(uploaded_bytes), dtype=np.uint8)
        current_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_label = "Uploaded_Solar_Disk.png"

    if current_image_bgr is not None:
        with st.spinner("Processing solar disk through Computer Vision pipeline..."):
            disk, limb, seg, regions, classes, annotated, prep_res, det_output = run_pipeline_on_image(current_image_bgr)

        # Top Metric Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Solar Disk Radius", f"{disk.radius:.1f} px", f"Conf: {disk.confidence*100:.0f}%")
        m2.metric("Detected ARs", det_output.total_detected_count, f"{det_output.confirmed_sunspot_count} Confirmed")
        m3.metric("Total Area", f"{sum(r.area_uhem for r in det_output.regions):.1f} μHem", f"{sum(r.area_pixels for r in det_output.regions)} px")
        m4.metric("Quiet-Sun Intensity", f"{det_output.quiet_sun_intensity:.1f}")
        m5.metric("Umbra/Penumbra Factors", f"{t_umbra_factor:.2f} / {t_penumbra_factor:.2f}")

        if det_output.is_spotless:
            st.info("🟡 **Spotless Solar Disk (Solar Minimum)**: No active regions or dark spots detected on the photosphere exceeding area or contrast thresholds.")

        # Scientific Metadata Card
        if current_metadata:
            with st.expander(f"🛰️ Authentic Telemetry Metadata: `{image_label}`", expanded=False):
                st.markdown(f"""
                - **Observatory**: {current_metadata.observatory}
                - **Instrument**: {current_metadata.instrument}
                - **Channel**: {current_metadata.wavelength_channel}
                - **Dimensions**: {current_metadata.image_width} x {current_metadata.image_height} ({current_metadata.file_size_bytes:,} bytes)
                - **SHA-256**: `{current_metadata.sha256}`
                - **Authentic Real Data**: `{'Yes (NASA SDO Verified)' if current_metadata.is_authentic_real_data else 'No'}`
                """)

        st.markdown("---")

        tab_overview, tab_patches, tab_prep, tab_pipeline, tab_table, tab_reasoning = st.tabs([
            "🔍 Detection Overview",
            "🔬 Sunspot ROI Patches",
            "🖼️ Preprocessing (6-Panel Analysis)",
            "🛠️ CV Pipeline Stages",
            "📋 Morphological & Physical Metrics Table",
            "🧠 Transparent Rule Traces",
        ])

        with tab_overview:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 1. Original Input Image (Real Solar Observation)")
                st.image(cv2.cvtColor(current_image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with c2:
                st.markdown("##### 2. Detected Active Regions & Morphological Annotations")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption("🟢 Green: Bounding Box | 🔵 Blue Cross: Centroid | 🟡 Yellow: Penumbra | 🔴 Red: Umbra Core | 🔷 Cyan: Boundary Contour")

        with tab_patches:
            st.markdown("##### Cropped ROI Patches of Detected Solar Active Regions")
            st.markdown("High-resolution cropped cutouts around each detected sunspot region with 15px context margin.")
            if det_output.regions:
                cols_per_row = 3
                for i in range(0, len(det_output.regions), cols_per_row):
                    batch = det_output.regions[i:i+cols_per_row]
                    pcols = st.columns(cols_per_row)
                    for col, r in zip(pcols, batch):
                        with col:
                            st.markdown(f"**{r.region_id}** — `{r.scientific_status}`")
                            if r.patch is not None and r.patch.size > 0:
                                st.image(cv2.cvtColor(r.patch, cv2.COLOR_BGR2RGB), use_container_width=True)
                            st.markdown(f"""
                            - **Area:** {r.area_pixels} px ({r.area_uhem:.1f} μHem)
                            - **Perimeter:** {r.perimeter_pixels:.1f} px
                            - **Circularity:** `{r.circularity:.3f}`
                            - **Centroid:** `({r.centroid[0]:.1f}, {r.centroid[1]:.1f})`
                            - **Bbox $(x,y,w,h)$:** `{r.bbox}`
                            - **Contrast:** `{r.contrast:.3f}`
                            - **Confidence:** `{r.confidence*100:.0f}%`
                            """)
            else:
                st.info("No sunspots or candidate regions detected on this solar disk.")

        with tab_prep:
            st.markdown("##### Classical CV Preprocessing: 6-Stage Analysis")
            st.markdown("Comprehensive before-and-after visual inspection showing noise reduction, limb-darkening compensation, and dark feature isolation.")
            st.image(cv2.cvtColor(prep_res.visuals.composite_panel, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Interactive diametric intensity profile across solar disk
            st.markdown("##### Photospheric Diametric Intensity Profile (Limb-to-Limb)")
            cy_mid = int(disk.center_y)
            cx_mid = int(disk.center_x)
            r_val = int(disk.radius)

            x_start = max(0, cx_mid - r_val)
            x_end = min(current_image_bgr.shape[1], cx_mid + r_val)
            x_axis = np.arange(x_start, x_end) - cx_mid

            raw_profile = prep_res.visuals.grayscale[cy_mid, x_start:x_end]
            flat_profile = prep_res.visuals.flat_fielded[cy_mid, x_start:x_end]
            denoised_profile = prep_res.visuals.denoised[cy_mid, x_start:x_end]

            df_prof = pd.DataFrame({
                "Distance from Disk Center (pixels)": x_axis,
                "Raw Input (with Limb Darkening)": raw_profile,
                "Flat-Field (Eddington Normalized)": flat_profile,
                "Bilateral Denoised (Granulation Smoothed)": denoised_profile,
            })

            fig_prof = px.line(
                df_prof,
                x="Distance from Disk Center (pixels)",
                y=["Raw Input (with Limb Darkening)", "Flat-Field (Eddington Normalized)", "Bilateral Denoised (Granulation Smoothed)"],
                title="Horizontal Diametric Intensity Profile across Photosphere",
                labels={"value": "Pixel Intensity [0-255]", "variable": "Preprocessing Stage"},
            )
            fig_prof.update_layout(height=380, template="plotly_white")
            st.plotly_chart(fig_prof, use_container_width=True)

        with tab_pipeline:
            st.markdown("##### Step-by-Step Computer Vision Pipeline Inspection")
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.markdown("**1. Raw Disk & Center**")
                raw_preview = current_image_bgr.copy()
                cv2.circle(raw_preview, (int(disk.center_x), int(disk.center_y)), int(disk.radius), (0, 255, 0), 2)
                cv2.drawMarker(raw_preview, (int(disk.center_x), int(disk.center_y)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                st.image(cv2.cvtColor(raw_preview, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"Center: ({disk.center_x:.1f}, {disk.center_y:.1f})")

            with p2:
                st.markdown("**2. Limb Darkening Model**")
                norm_surf = (limb.correction_surface / max(np.max(limb.correction_surface), 1.0) * 255).astype(np.uint8)
                norm_surf[disk.mask == 0] = 0
                st.image(norm_surf, use_container_width=True)
                st.caption(f"Eddington u = {u_coeff:.2f}")

            with p3:
                st.markdown("**3. Photosphere Flat-Field**")
                st.image(limb.flattened_uint8, use_container_width=True)
                st.caption("Limb-corrected uniform background")

            with p4:
                st.markdown("**4. Umbra/Penumbra Mask**")
                color_mask = np.zeros((*seg.combined_mask.shape, 3), dtype=np.uint8)
                color_mask[seg.penumbra_mask > 0] = [255, 215, 0]  # Yellow penumbra
                color_mask[seg.umbra_mask > 0] = [255, 0, 0]       # Red umbra
                st.image(color_mask, use_container_width=True)
                st.caption(f"T_u: {seg.umbra_threshold:.1f}, T_p: {seg.penumbra_threshold:.1f}")

        with tab_table:
            st.markdown("##### Calibrated Solar Active Region & Morphological Catalog")
            if det_output.regions:
                rows = []
                for det_r in det_output.regions:
                    matching_cls = next((c for r_orig, c in zip(regions, classes) if r_orig.id == det_r.id), None)
                    cls_code = matching_cls.class_code if matching_cls else "Axx"
                    flare_risk = matching_cls.flare_potential if matching_cls else "Low"

                    rows.append({
                        "Region ID": det_r.region_id,
                        "Scientific Status": det_r.scientific_status,
                        "McIntosh Class": cls_code,
                        "Attention Level": matching_cls.attention_level if matching_cls else "Low Attention",
                        "Demonstration Score": f"{matching_cls.demonstration_risk.score:.1f}/100" if matching_cls else "0.0/100",
                        "Area (px)": det_r.area_pixels,
                        "Area (μHem)": round(det_r.area_uhem, 1),
                        "Perimeter (px)": det_r.perimeter_pixels,
                        "Circularity": det_r.circularity,
                        "Centroid": f"({det_r.centroid[0]:.1f}, {det_r.centroid[1]:.1f})",
                        "Bbox (x,y,w,h)": str(det_r.bbox),
                        "Contrast": det_r.contrast,
                        "Mean Intensity": det_r.mean_intensity,
                        "Min Intensity": det_r.min_intensity,
                        "Umbra (μHem)": round(det_r.umbra_area_uhem, 1),
                        "Penumbra (μHem)": round(det_r.penumbra_area_uhem, 1),
                        "Latitude (deg)": round(det_r.heliographic_lat, 2),
                        "Longitude CMD (deg)": round(det_r.heliographic_lon_cmd, 2),
                        "Confidence": f"{det_r.confidence*100:.0f}%",
                    })
                df_det = pd.DataFrame(rows)
                st.dataframe(df_det, use_container_width=True)

                col_dl1, col_dl2, col_save = st.columns([1, 1, 2])
                with col_dl1:
                    st.download_button(
                        "📥 Download JSON",
                        data=det_output.to_json(),
                        file_name=f"detected_{Path(image_label).stem}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                with col_dl2:
                    st.download_button(
                        "📥 Download CSV",
                        data=df_det.to_csv(index=False),
                        file_name=f"detected_{Path(image_label).stem}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with col_save:
                    if st.button("💾 Save Observation to Solar Catalog Database", use_container_width=True):
                        obs_id = db.save_observation(
                            filename=image_label,
                            timestamp=datetime.utcnow(),
                            center_x=disk.center_x,
                            center_y=disk.center_y,
                            radius=disk.radius,
                            quiet_sun_intensity=limb.quiet_sun_intensity,
                            regions=regions,
                            classifications=classes,
                        )
                        st.success(f"Successfully recorded observation #{obs_id} with {len(regions)} active regions into SQLite database.")
            else:
                st.info("No active regions detected above the minimum area threshold on this disk.")

        with tab_reasoning:
            st.markdown("##### Transparent Morphological Classification & Demonstration Risk Engine")
            st.warning("""
            ⚠️ **Scientific & Operational Non-Prediction Disclaimer**:
            The demonstration risk scores and attention levels shown below are **heuristic educational indicators** derived solely from visible-light continuum morphology (physical area, spot multiplicity, contrast, and boundary compactness).
            They are **NOT operational or scientifically validated solar flare prediction models** and must never be interpreted as claiming that an active region will or will not produce a solar flare.
            Operational space weather forecasting (such as that conducted by NOAA Space Weather Prediction Center) requires 3D vector magnetograms (HMI/SDO), electric current helicity, magnetic shear along polarity inversion lines, free magnetic energy, and historical flare occurrence rates.
            """)

            for r, c in zip(regions, classes):
                risk = c.demonstration_risk
                with st.expander(f"📌 Active Region AR-{r.id}: **{c.class_code}** ({c.class_name}) — Level: `{c.attention_level}` ({risk.score:.1f}/100)"):
                    rc1, rc2 = st.columns([1, 1])
                    with rc1:
                        st.markdown(f"**Morphological Classification:** `{c.class_name}`")
                        st.markdown(f"- **Definition:** {c.class_info.description}")
                        st.markdown(f"- **Typical Lifespan:** `{c.class_info.typical_lifespan}`")
                        st.markdown(f"- **Magnetic Topology:** `{c.class_info.magnetic_topology}`")
                        st.markdown(f"- **Zurich/McIntosh System:** `{c.class_info.mcintosh_equivalent}`")
                        st.markdown(f"- **Rule Match Confidence:** `{c.confidence * 100:.0f}%`")

                    with rc2:
                        st.markdown(f"**Demonstration Risk Indicator:** `{risk.attention_level}` ({risk.score:.1f}/100)")
                        st.caption("Multi-factor weighted geometric complexity index (Educational Demonstration Only)")
                        factor_data = [
                            {"Factor": "Area Factor (A / A_base)", "Score [0-100]": risk.factors.area_factor, "Weight": f"{risk.weights['area']*100:.0f}%"},
                            {"Factor": "Structural Complexity (Multiplicity + Extent)", "Score [0-100]": risk.factors.complexity_factor, "Weight": f"{risk.weights['complexity']*100:.0f}%"},
                            {"Factor": "Penumbra Coverage & Topology", "Score [0-100]": risk.factors.penumbra_factor, "Weight": f"{risk.weights['penumbra']*100:.0f}%"},
                            {"Factor": "Photospheric Contrast (Core Darkness)", "Score [0-100]": risk.factors.contrast_factor, "Weight": f"{risk.weights['contrast']*100:.0f}%"},
                            {"Factor": "Shape Irregularity (1 - Circularity)", "Score [0-100]": risk.factors.compactness_factor, "Weight": f"{risk.weights['compactness']*100:.0f}%"},
                        ]
                        st.dataframe(pd.DataFrame(factor_data), use_container_width=True, hide_index=True)

                    st.markdown("---")
                    st.markdown("**Auditable Rule Deduction Trace:**")
                    for step in c.rule_trace:
                        st.markdown(f"- {step}")

                    st.markdown("**Stated Physical & Heuristic Assumptions:**")
                    for asm in risk.assumptions:
                        st.caption(f"• {asm}")

    else:
        st.info("Please select or upload a solar continuum image to begin analysis.")


# ==============================================================================
# VIEW 2: MULTI-FRAME SEQUENCE TRACKING
# ==============================================================================
elif nav_choice == "🛰️ Multi-Frame Sequence Tracking":
    st.markdown('<div class="main-header">🛰️ Multi-Frame Solar Active Region Tracking</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Photospheric Differential Rotation Kinematics & Active Region Evolution on Real NASA SDO Sequences</div>', unsafe_allow_html=True)

    st.warning(f"⚠️ **Scientific Tracking Guardrail**: {TRACKING_DISCLAIMER}")

    st.markdown("""
    The solar photosphere rotates differentially with latitude: the equator rotates faster than higher latitudes.
    SolarVision tracks active regions across multi-day observations by projecting expected heliographic coordinates using the **Snodgrass (1984)** relation:
    $$\\omega(B) = 14.713 - 2.396\\sin^2(B) - 1.787\\sin^4(B) \\quad [\\text{deg/day}]$$
    Nearest-neighbor associations are validated through angular distance gating ($d \\le d_{\\max}$), latitude drift constraints ($|\\Delta B| \\le 4^\\circ$), and physical area consistency limits.
    """)

    sample_dir = Path(config.storage.sample_data_dir)
    # Search for real SDO time-series frames (e.g. May 2024 AR3664 sequence)
    seq_files = sorted(list(sample_dir.glob("sdo_hmi_ar3664_*.jpg")))
    if not seq_files:
        seq_files = sorted(list(sample_dir.glob("sdo_hmi_*.jpg")))

    if len(seq_files) >= 2:
        st.write(f"Real SDO multi-day observation frames: `{', '.join(f.name for f in seq_files)}`")

        tracker = ActiveRegionTracker(
            physics_config=config.solar_physics,
            tracking_config=config.tracking,
        )
        all_obs_tracking = []

        cols = st.columns(len(seq_files))
        for idx, (fpath, col) in enumerate(zip(seq_files, cols)):
            img = cv2.imread(str(fpath))
            obs_time = datetime(2024, 5, 10, 0, 0) + timedelta(days=idx)
            disk = disk_detector.detect(img)
            limb = limb_corrector.correct(img, disk)
            seg = segmenter.segment(limb, disk)
            regs = feature_extractor.extract_features(seg.regions, disk)
            tracked = tracker.track_observation(regs, obs_time)
            all_obs_tracking.append((idx + 1, fpath.name, obs_time, tracked, img, disk, limb, seg))

            with col:
                st.markdown(f"**Day {idx + 1}: May {10 + idx}, 2024**")
                st.caption(f"Detected ARs: {len(tracked)}")
                st.image(str(fpath), use_container_width=True)

        st.markdown("---")

        # Summary KPIs
        total_tracks = len(tracker.tracks)
        multi_frame_tracks = sum(1 for t in tracker.tracks.values() if t.observation_count >= 2)
        peak_area = max((t.max_area_uhem for t in tracker.tracks.values()), default=0.0)
        peak_growth = max((t.growth_rate_uhem_per_day for t in tracker.tracks.values()), default=0.0)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Formations Tracked", total_tracks)
        k2.metric("Multi-Day Linked Tracks", multi_frame_tracks)
        k3.metric("Peak Sequence Area", f"{peak_area:.1f} μHem")
        k4.metric("Max Area Growth Rate", f"{peak_growth:.1f} μHem/day")

        st.markdown("---")
        st.markdown("### 📈 Trajectory Migration & Area Evolution")

        tab_traj, tab_area = st.tabs(["🌐 Photospheric Migration Trajectories", "📊 Physical Area Evolution Curves"])

        with tab_traj:
            fig_traj = tracker.plot_trajectories_plotly()
            st.plotly_chart(fig_traj, use_container_width=True)

        with tab_area:
            fig_area = tracker.plot_area_evolution_plotly()
            st.plotly_chart(fig_area, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Active Region Lifecycle & Trajectory Tables")

        df_summary = tracker.to_dataframe()
        df_traj = tracker.get_trajectory_dataframe()

        tab_summary_tbl, tab_traj_tbl, tab_inspect = st.tabs([
            "📊 Track Lifecycle Summary",
            "🔍 Frame-by-Frame Residuals",
            "🔬 Detailed Track Inspection"
        ])

        with tab_summary_tbl:
            st.markdown("##### Persistent Active Region Lifecycles")
            st.dataframe(df_summary, use_container_width=True)
            if not df_summary.empty:
                st.download_button(
                    "📥 Export Track Summary CSV",
                    data=df_summary.to_csv(index=False),
                    file_name="solarvision_track_summary.csv",
                    mime="text/csv",
                )

        with tab_traj_tbl:
            st.markdown("##### Multi-Observation Kinematic Trajectory Residuals")
            st.dataframe(df_traj, use_container_width=True)
            if not df_traj.empty:
                st.download_button(
                    "📥 Export Trajectories CSV",
                    data=df_traj.to_csv(index=False),
                    file_name="solarvision_trajectories.csv",
                    mime="text/csv",
                )

        with tab_inspect:
            st.markdown("##### Deep-Dive Track Inspector")
            if tracker.tracks:
                track_ids = list(tracker.tracks.keys())
                selected_trk = st.selectbox("Select Track ID to Inspect:", track_ids, index=0)
                trk_obj = tracker.get_track(selected_trk)
                if trk_obj:
                    t_col1, t_col2, t_col3 = st.columns(3)
                    t_col1.metric("Status", trk_obj.status)
                    t_col2.metric("Duration", f"{trk_obj.duration_days:.1f} days")
                    t_col3.metric("Net CMD Drift", f"{trk_obj.net_longitude_drift_deg:.2f}°")

                    st.markdown("**Observation Log:**")
                    obs_records = []
                    for o in trk_obj.observations:
                        obs_records.append({
                            "Timestamp": o.observation_time.strftime("%Y-%m-%d %H:%M"),
                            "Pred Lon CMD (deg)": o.predicted_lon_cmd,
                            "Actual Lon CMD (deg)": o.actual_lon_cmd,
                            "Pred Lat (deg)": o.predicted_lat,
                            "Actual Lat (deg)": o.actual_lat,
                            "Residual (deg)": o.residual_deg,
                            "Area (μHem)": o.area_uhem,
                            "Growth Rate (μHem/day)": o.area_change_rate_uhem_per_day,
                            "Event Note": o.event_note,
                        })
                    st.dataframe(pd.DataFrame(obs_records), use_container_width=True)

    else:
        st.warning("Insufficient multi-frame sequence data. Fetching real SDO observations...")
        ingestor.prepare_real_sample_dataset()
        st.rerun()


# ==============================================================================
# VIEW 3: SOLAR CATALOG & ANALYTICS
# ==============================================================================
elif nav_choice == "📊 Solar Catalog & Analytics":
    st.markdown('<div class="main-header">📊 Solar Active Region Catalog & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Persistent SQLite observation catalog and solar cycle statistical analytics</div>', unsafe_allow_html=True)

    catalog_data = db.get_catalog()

    if catalog_data:
        df_cat = pd.DataFrame(catalog_data)

        # Summary KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cataloged ARs", len(df_cat))
        c2.metric("Unique Tracking IDs", df_cat["tracking_id"].nunique())
        c3.metric("Max Area Recorded", f"{df_cat['area_uhem'].max():.1f} μHem")
        c4.metric("Most Common Class", df_cat["mcintosh_class"].mode()[0])

        st.markdown("---")

        # Visualizations
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("##### Solar Butterfly Latitudinal Distribution")
            fig_bf = px.scatter(
                df_cat,
                x="lon_cmd_deg",
                y="lat_deg",
                color="mcintosh_class",
                size="area_uhem",
                hover_data=["tracking_id", "area_uhem", "flare_potential", "source_image"],
                labels={"lon_cmd_deg": "Heliographic Longitude CMD (°)", "lat_deg": "Heliographic Latitude B (°)"},
                title="Solar Active Region Spatial Distribution",
            )
            fig_bf.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_bf.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_bf.update_layout(height=420, template="plotly_white")
            st.plotly_chart(fig_bf, use_container_width=True)

        with ch2:
            st.markdown("##### McIntosh Classification Breakdown")
            fig_bar = px.histogram(
                df_cat,
                x="mcintosh_class",
                color="flare_potential",
                title="Active Region Count by McIntosh Class & Flare Risk",
                labels={"mcintosh_class": "McIntosh Class", "count": "Occurrences"},
                category_orders={"mcintosh_class": ["A", "B", "C", "D", "E", "F", "H"]},
            )
            fig_bar.update_layout(height=420, template="plotly_white")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("##### Searchable Catalog Records")
        st.dataframe(
            df_cat[[
                "observation_time", "source_image", "tracking_id", "mcintosh_class",
                "lat_deg", "lon_cmd_deg", "area_uhem", "flare_potential", "confidence"
            ]],
            use_container_width=True,
        )

        csv = df_cat.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Catalog as CSV", data=csv, file_name="solarvision_catalog.csv", mime="text/csv")

    else:
        st.info("The SQLite catalog database is currently empty. Analyze and save images in the Single-Image Detector tab to populate the catalog!")


# ==============================================================================
# VIEW 4: REAL NASA SDO FEED
# ==============================================================================
elif nav_choice == "🌐 Real NASA SDO Feed":
    st.markdown('<div class="main-header">🌐 Live NASA SDO Satellite Ingestion</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Direct real-time telemetry from NASA Solar Dynamics Observatory (SDO/HMI)</div>', unsafe_allow_html=True)

    st.markdown("""
    **Telemetry Source Specifications**:
    - **Observatory**: NASA Solar Dynamics Observatory (SDO)
    - **Instrument**: Helioseismic and Magnetic Imager (HMI)
    - **Channel**: Fe I 6173 Å Visible Photospheric Continuum
    - **Resolution**: 1024x1024 Full-Disk Browse Feed
    - **Integrity Protocol**: SHA-256 Checksum, HTTP ETag, and Header Deduplication
    """)

    col_btn, col_status = st.columns([1, 2])
    with col_btn:
        if st.button("🔄 Fetch Latest SDO/HMI Image"):
            with st.spinner("Connecting to NASA SDO telemetry servers..."):
                res = ingestor.download_latest_sdo(destination_dir=Path(config.storage.sample_data_dir))
                if res.success:
                    if res.status == "duplicate_skipped":
                        st.info("ℹ️ Latest image is already cached locally (identical SHA-256 hash). Skipped duplicate download.")
                    else:
                        st.success("✅ Successfully ingested fresh NASA SDO frame!")
                else:
                    st.error(f"Failed to fetch NASA SDO image: {res.error_message}")

    # Display most recent SDO download
    sdo_files = sorted(list(Path(config.storage.sample_data_dir).glob("*sdo*.*")), reverse=True)
    if sdo_files:
        latest_file = sdo_files[0]
        st.markdown(f"**Current Telemetry Frame:** `{latest_file.name}`")
        ingest_res = ingestor.load_local_image(latest_file)
        if ingest_res.success and ingest_res.image is not None:
            img_sdo = ingest_res.image
            meta_sdo = ingest_res.metadata

            if meta_sdo:
                st.markdown(f"""
                <div class="meta-box">
                <b>Observatory:</b> {meta_sdo.observatory} | <b>Instrument:</b> {meta_sdo.instrument}<br/>
                <b>Channel:</b> {meta_sdo.wavelength_channel} | <b>Resolution:</b> {meta_sdo.image_width}x{meta_sdo.image_height}<br/>
                <b>SHA-256:</b> {meta_sdo.sha256}<br/>
                <b>Downloaded At (UTC):</b> {meta_sdo.download_timestamp_utc}
                </div>
                """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.image(cv2.cvtColor(img_sdo, cv2.COLOR_BGR2RGB), caption="Live NASA SDO HMI Continuum Feed", use_container_width=True)
            with c2:
                with st.spinner("Processing live frame through SolarVision pipeline..."):
                    disk, limb, seg, regions, classes, annotated, _, det_out = run_pipeline_on_image(img_sdo)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="SolarVision Automated Detection & Classification", use_container_width=True)
                st.metric("Live Active Regions Detected", len(regions))


# ==============================================================================
# VIEW 5: SCIENTIFIC METHODOLOGY
# ==============================================================================
elif nav_choice == "📚 Scientific Methodology":
    st.markdown('<div class="main-header">📚 Scientific Methodology & CV Formulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Theoretical background, optical equations, and citation references</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. Photospheric Limb Darkening Correction
    The optical depth $\\tau = 1$ reaches deeper and hotter layers near the solar center (normal to the surface) than near the limb.
    SolarVision applies the Eddington linear limb darkening approximation:
    $$I(\\mu) = I_0 \\left[ 1 - u(1 - \\mu) \\right]$$
    where $\\mu = \\cos\\theta = \\sqrt{1 - (r / R_\\odot)^2}$ and $u \\approx 0.60$ for visible continuum ($6173\\text{ \\AA}$).
    The flat-fielded image is then computed as:
    $$I_{\\text{flat}}(x, y) = \\frac{I(x, y)}{1 - u(1 - \\mu)}$$
    
    ### 2. Dual-Threshold Umbra/Penumbra Segmentation
    Sunspots consist of an intense dark core (umbra) and a filamentary halo (penumbra):
    - **Umbra Boundary**: $I \\le 0.58 \\cdot I_{QS}$
    - **Penumbra Boundary**: $0.58 \\cdot I_{QS} < I \\le 0.88 \\cdot I_{QS}$
    where $I_{QS}$ is the normalized quiet-Sun intensity.

    ### 3. Geometric Foreshortening Correction & Area Calibration
    Because the spherical solar surface is viewed in 2D orthographic projection, an area element is compressed by $\\cos\\theta$:
    $$A_{\\text{corrected}} = \\frac{A_{\\text{projected}}}{\\cos\\theta}$$
    Area is calibrated into **Millionths of Solar Hemisphere** ($\\mu\\text{Hem}$):
    $$\\text{Area}_{\\mu\\text{Hem}} = \\frac{A_{\\text{corrected}}}{2\\pi R_\\odot^2} \\times 10^6$$

    ### 4. Stonyhurst Heliographic Coordinates
    Let $(x', y')$ be normalized Cartesian coordinates relative to disk center where $+X$ is West and $+Y$ is North:
    $$\\sin B = y' \\quad \\text{(Heliographic Latitude)}$$
    $$\\sin L = \\frac{x'}{\\cos B} \\quad \\text{(Central Meridian Distance / CMD)}$$

    ### 5. Differential Solar Rotation (Snodgrass 1984)
    $$\\omega(B) = 14.713 - 2.396\\sin^2(B) - 1.787\\sin^4(B) \\quad [\\text{deg/day}]$$

    ### 6. Academic References
    - McIntosh, P. S. (1990). *The classification of sunspot groups*. Solar Physics, 125(2), 251-267.
    - Snodgrass, H. B. (1984). *Separation of large-scale solar flows from differential rotation*. Solar Physics, 94(1), 13-31.
    - Hathaway, D. H. (2015). *The solar cycle*. Living Reviews in Solar Physics, 12(1), 4.
    """)
