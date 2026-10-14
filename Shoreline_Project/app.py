import os
import sys
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import folium
from streamlit_folium import st_folium
from backend_functions import *

st.set_page_config(page_title="NISAR MISSION CONTROL", layout="wide")

# NASA-ISRO Professional Dark UI
st.markdown("""<style>
.stApp { background-color: #000; color: #fff; }
[data-testid="stSidebar"] { background-color: #0a0d14; border-right: 2px solid #0052ff; }
.header-box { background: #0052ff; color: white; padding: 10px; font-weight: bold; border-radius: 4px; margin-bottom: 20px; }
.meta-card { background: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; text-align: center; height: 110px; }
.meta-val { color: #00d4ff; font-size: 24px; font-weight: 800; }
.filename-box { background: #1a1a1a; padding: 10px; border-left: 4px solid #ff9933; color: #fff; font-size: 11px; word-break: break-all; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# State initialization
if 'files' not in st.session_state:
    st.session_state.files = {'l': None, 's': None, 't2': None, 'india_shp': None}
if 'run_mission' not in st.session_state:
    st.session_state.run_mission = False
# Parsed India boundary GeoDataFrame is cached here so it's only
# parsed once per file, not once per expander/loop iteration.
if 'india_gdf_cache' not in st.session_state:
    st.session_state.india_gdf_cache = {}


def select_file(filetypes=[("HDF5", "*.h5")]):
    """
    FIX for 'RuntimeError: main thread is not in main loop':
    Streamlit runs the script in its own worker thread, but tkinter's
    dialog only works reliably in a process's actual main thread — it
    works once by luck and then breaks on the next click. Running the
    dialog in a separate short-lived Python subprocess sidesteps this
    completely, since that subprocess gets its own real main thread.
    """
    script = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "import json, sys\n"
        "filetypes = json.loads(sys.argv[1])\n"
        "root = tk.Tk()\n"
        "root.withdraw()\n"
        "root.attributes('-topmost', True)\n"
        "path = filedialog.askopenfilename(filetypes=[tuple(ft) for ft in filetypes])\n"
        "root.destroy()\n"
        "print(path)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, json.dumps(filetypes)],
            capture_output=True, text=True, timeout=300
        )
        path = result.stdout.strip()
        return path if path else None
    except Exception as e:
        st.error(f"File dialog failed: {e}")
        return None


with st.sidebar:
    c1, c2 = st.columns(2)
    with c1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=75)
    with c2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organisation_Logo.svg", width=65)

    st.markdown("### 🛰️ **NISAR COCKPIT**")
    mode = st.radio("SELECT MISSION TASK", ["Single Band", "L+S Band Fusion", "12-Day Change Detection"])

    if st.button("📁 Load Baseline L (T1)"):
        st.session_state.files['l'] = select_file()
        st.session_state.run_mission = False

    if st.session_state.files['l']:
        st.markdown(f"<div class='filename-box'>L1: {os.path.basename(st.session_state.files['l'])}</div>", unsafe_allow_html=True)

    if mode == "L+S Band Fusion":
        if st.button("📁 Load Baseline S (T1)"):
            st.session_state.files['s'] = select_file()
            st.session_state.run_mission = False
        if st.session_state.files['s']:
            st.markdown(f"<div class='filename-box'>S1: {os.path.basename(st.session_state.files['s'])}</div>", unsafe_allow_html=True)

    if mode == "12-Day Change Detection":
        if st.button("📁 Load Comparative T2"):
            st.session_state.files['t2'] = select_file()
            st.session_state.run_mission = False
        if st.session_state.files['t2']:
            st.markdown(f"<div class='filename-box'>T2: {os.path.basename(st.session_state.files['t2'])}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("🗺️ **BASE VECTOR BOUNDARY**")
    if st.button("📁 Load India SHP (.shp / .zip)"):
        st.session_state.files['india_shp'] = select_file([("Shapefile/ZIP", "*.shp *.zip")])

    if st.session_state.files.get('india_shp'):
        st.markdown(f"<div class='filename-box'>SHP: {os.path.basename(st.session_state.files['india_shp'])}</div>", unsafe_allow_html=True)

    st.divider()
    st.header("⚙️ OPTIMIZATION")
    morph_val = st.slider("Morphological Smoothness", 11, 41, 21, step=2)
    use_manual_t = st.checkbox("Manual dB Adjustment")
    man_t = st.slider("dB Cut-off", -30.0, -5.0, -15.0) if use_manual_t else None

    st.divider()
    st.markdown("**STAGE-WISE PIPELINE**")
    f_list = st.multiselect("STAGE 2: DENOISING", ["Lee Filter", "Kuan Filter", "Frost Filter", "NL-Means", "Median Filter"], ["Lee Filter"])
    s_list = st.multiselect("STAGE 3: SEGMENTATION", ["Otsu Thresholding", "K-Means Clustering", "Fuzzy C-Mean (FCM)", "Region Growing"], ["Otsu Thresholding"])
    r_list = st.multiselect("STAGE 4: REFINEMENT", ["CCA", "Morphological", "Douglas-Peuker"], ["CCA", "Morphological"])

    if st.button("🚀 INITIATE JOINT MISSION"):
        st.session_state.run_mission = True

st.markdown("<h1 style='color:#0052ff; font-weight:900;'>NISAR <span style='color:white; font-weight:300;'>Science Suite</span></h1>", unsafe_allow_html=True)

if st.session_state.files['l'] and st.session_state.run_mission:
    meta_l = get_nisar_metadata(st.session_state.files['l'])
    st.markdown("<div class='header-box'>STAGE 1 // RADIOMETRIC METADATA</div>", unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"<div class='meta-card'>HEIGHT<br><span class='meta-val'>{meta_l['height']} px</span></div>", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"<div class='meta-card'>WIDTH<br><span class='meta-val'>{meta_l['width']} px</span></div>", unsafe_allow_html=True)
    with mc3:
        st.markdown(f"<div class='meta-card'>RESO<br><span class='meta-val'>{meta_l['spacing']}m</span></div>", unsafe_allow_html=True)
    with mc4:
        st.markdown(f"<div class='meta-card'>BAND INFO<br><span class='meta-val'>{meta_l['band']}</span></div>", unsafe_allow_html=True)

    if mode == "L+S Band Fusion" and st.session_state.files['s']:
        db_a, viz_a, mask_a = fuse_nisar_bands(st.session_state.files['l'], st.session_state.files['s'])
    else:
        db_a, viz_a, mask_a = apply_log_scaling(get_nisar_data(st.session_state.files['l']))

    # India reference boundary: parsed once and cached in session_state,
    # not re-extracted/re-parsed inside every expander below.
    india_gdf = None
    india_path = st.session_state.files.get('india_shp')
    if india_path and os.path.exists(india_path):
        if india_path not in st.session_state.india_gdf_cache:
            st.session_state.india_gdf_cache[india_path] = load_india_boundary(india_path)
        india_gdf = st.session_state.india_gdf_cache[india_path]

    st.markdown("<div class='header-box'>STAGE 5 // PURIFIED ANALYTICS MATRIX</div>", unsafe_allow_html=True)
    for f in f_list:
        den_a, viz_den_a, enl_a = run_speckle_filter(db_a, f)
        for s in s_list:
            bin_a, t_val = run_segmentation(den_a, s, mask_a, manual_t=man_t)
            edge_a, s4_a, pur_a = run_refinement(bin_a, r_list, meta_l['spacing'] * 6, mask_a, den_a, morph_val)

            with st.expander(f"🔬 MISSION REPORT: {f} + {s}", expanded=True):
                col_v1, col_v2 = st.columns([2.5, 1])
                with col_v1:
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='black')
                    ax.imshow(viz_a, cmap='gray', alpha=0.5)
                    y1, x1 = np.where(edge_a > 0)
                    ax.scatter(x1, y1, s=0.15, c='#00ffff', label="Baseline T1")

                    if mode == "12-Day Change Detection" and st.session_state.files['t2']:
                        db_t2, _, mask_t2 = apply_log_scaling(get_nisar_data(st.session_state.files['t2']))
                        den_t2, _, _ = run_speckle_filter(db_t2, f)
                        bin_t2, _ = run_segmentation(den_t2, s, mask_t2, manual_t=man_t)
                        edge_t2, s4_t2, pur_t2 = run_refinement(bin_t2, r_list, meta_l['spacing'] * 6, mask_t2, den_t2, morph_val)
                        y2, x2 = np.where(edge_t2 > 0)
                        ax.scatter(x2, y2, s=0.15, c='#ff00ff', label="Comparative T2")
                        er, ac = calculate_change(pur_a, pur_t2, meta_l['spacing'] * 6)
                        st.error(f"Erosion: {er} km²")
                        st.success(f"Accretion: {ac} km²")

                    ax.axis('off')
                    ax.legend(labelcolor='white', frameon=False)
                    st.pyplot(fig)
                    plt.close(fig)

                with col_v2:
                    st.metric("LENGTH T1", f"{s4_a['KM']} KM")
                    st.metric("AREA T1", f"{s4_a['Area']} km²")

                    shp_data, gdf_preview = export_shapefile_zip(edge_a, meta_l, st.session_state.files['l'])
                    if shp_data is not None:
                        st.download_button(
                            label="📥 Export Shapefile (.zip)",
                            data=shp_data,
                            file_name=f"shoreline_{f}_{s}.zip",
                            mime="application/zip",
                            key=f"dl_{f}_{s}"
                        )

                        try:
                            if gdf_preview.crs is None:
                                gdf_preview = gdf_preview.set_crs("EPSG:4326")

                            pbounds = gdf_preview.total_bounds
                            pcenter = [(pbounds[1] + pbounds[3]) / 2, (pbounds[0] + pbounds[2]) / 2]

                            # Zoom locked lower so it doesn't over-zoom
                            # into the thin coastline bounding box.
                            fmap = folium.Map(
                                location=pcenter,
                                zoom_start=7,
                                min_zoom=5,
                                max_zoom=8,
                                tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                                attr="Google Hybrid",
                            )

                            if india_gdf is not None:
                                folium.GeoJson(
                                    india_gdf,
                                    name="India Reference",
                                    style_function=lambda x: {"color": "#FF8800", "weight": 2.0, "fill": False}
                                ).add_to(fmap)

                            # Uses the in-memory GeoDataFrame directly
                            # instead of writing the zip to disk and
                            # reading it back just to preview it.
                            folium.GeoJson(
                                gdf_preview,
                                name="NISAR Shoreline",
                                style_function=lambda x: {"color": "#00FFFF", "weight": 3, "opacity": 1.0},
                            ).add_to(fmap)

                            fmap.fit_bounds(
                                [[pbounds[1], pbounds[0]], [pbounds[3], pbounds[2]]],
                                max_zoom=7,
                                padding=(40, 40)
                            )

                            st.markdown("**Shoreline Overlay Preview**")
                            st_folium(fmap, width=400, height=300, key=f"map_{f}_{s}", returned_objects=[])
                        except ImportError:
                            st.info("Install `streamlit-folium` to see live map preview.")
                        except Exception as map_err:
                            st.warning(f"Map preview failed: {map_err}")
                    else:
                        st.warning("No shoreline contour layer extracted for export.")
            st.divider()
else:
    st.info("MISSION READY: Load H5 files and launch mission.")