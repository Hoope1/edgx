"""
🎨  Edge-Detection Studio – Streamlit GUI
----------------------------------------

Tabs:
1. 📷 Bildauswahl
2. 🔧 Methoden
3. ⚙️ Einstellungen
4. 🚀 Verarbeitung
5. 👁️ Vorschau
"""
from __future__ import annotations
import os, time, tempfile, base64, zipfile, cv2, json
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import streamlit as st

# interne Module
try:
    from detectors import (
        get_all_methods, get_max_resolution, standardize_output
    )
    from gui_components import (
        folder_picker, image_gallery, method_selector_advanced,
        progress_tracker, batch_processor
    )
    DETECTORS_AVAILABLE = True
except Exception as e:            # noqa
    DETECTORS_AVAILABLE = False
    DETECTOR_IMPORT_ERROR = e

# ------------------------------------------------------------------
#  Streamlit Config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Edge Detection Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

_CUSTOM_CSS = """
<style>
body { font-family: 'Segoe UI', sans-serif; }
div.main-header {
    font-size:2.0rem; font-weight:600; margin:0.3em 0 1.2em 0;
}
.status-processing { color:#ff9800; font-weight:600; }
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Session-State Defaults
# ------------------------------------------------------------------
def _init_state():
    d = st.session_state
    d.setdefault("selected_methods", [])
    d.setdefault("selected_images",  [])
    d.setdefault("output_dir",       "./results")
    d.setdefault("processing_status","idle")
    d.setdefault("progress",         0.0)
    d.setdefault("processing_log",   [])

_init_state()

# ------------------------------------------------------------------
# Sidebar – globale Konfiguration & Start/Stop
# ------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Konfiguration")

    # Eingabe-Modus
    st.subheader("📁 Eingabe")
    input_mode = st.radio(
        "Modus wählen:",
        ["Ordner auswählen", "Einzelne Bilder"],
        key="input_mode_radio"
    )

    # Ausgabe-Ordner
    st.subheader("📂 Ausgabe")
    st.session_state.output_dir = st.text_input(
        "Ausgabeordner:",
        value=st.session_state.output_dir,
        help="Pfad, in dem Ergebnisse abgelegt werden"
    )

    # Options
    st.subheader("⚙️ Optionen")
    st.checkbox("🎨 Invertierte Ausgabe", value=True, key="opt_invert")
    st.checkbox("📐 Einheitliche Größe",  value=True, key="opt_uniform")

    st.markdown("---")

    # Start / Stop
    if st.session_state.processing_status == "idle":
        if st.button("🚀 VERARBEITUNG STARTEN", type="primary", use_container_width=True):
            st.session_state.processing_status = "running"
            st.experimental_rerun()
    else:
        st.button("⏹️ Stoppen", type="secondary", use_container_width=True,
                  on_click=lambda: st.session_state.update(processing_status="idle"))

# ------------------------------------------------------------------
# Tabs definieren
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📷 Bildauswahl", "🔧 Methoden", "⚙️ Einstellungen",
     "🚀 Verarbeitung", "👁️ Vorschau"]
)

# --------------------------------------------------------------
# Helper
# --------------------------------------------------------------
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

@st.cache_data(show_spinner=False)
def _load_image(path: str) -> Optional[np.ndarray]:
    try:
        return cv2.imread(path)
    except Exception:
        return None

def _find_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, f)
             for f in os.listdir(folder)
             if f.lower().endswith(_IMAGE_EXT)]
    return sorted(files)

def _thumb(img: np.ndarray, size: Tuple[int,int]=(140,140)) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

# --------------------------------------------------------------
# TAB 1 – Bildauswahl
# --------------------------------------------------------------
with tab1:
    st.header("📷 Bildauswahl")

    if input_mode == "Ordner auswählen":
        sel_folder = folder_picker("Ordner wählen", "./images")
        if sel_folder:
            imgs = _find_images(sel_folder)
            st.session_state.selected_images = imgs
            if imgs:
                st.success(f"{len(imgs)} Bilder gefunden")
                cols = st.columns(4)
                for i,p in enumerate(imgs[:8]):
                    img = _load_image(p)
                    if img is not None:
                        cols[i%4].image(_thumb(img), caption=Path(p).name)
    else:
        uploads = st.file_uploader(
            "Dateien auswählen", type=list(_IMAGE_EXT),
            accept_multiple_files=True
        )
        if uploads:
            tmpdir = tempfile.mkdtemp()
            paths: list[str] = []
            for up in uploads:
                p = os.path.join(tmpdir, up.name)
                with open(p,"wb") as fh: fh.write(up.getbuffer())
                paths.append(p)
            st.session_state.selected_images = paths
            if paths:
                st.success(f"{len(paths)} Dateien geladen")
                cols = st.columns(4)
                for i,p in enumerate(paths[:8]):
                    img = _load_image(p)
                    if img is not None:
                        cols[i%4].image(_thumb(img), caption=Path(p).name)

# --------------------------------------------------------------
# TAB 2 – Methoden
# --------------------------------------------------------------
with tab2:
    if DETECTORS_AVAILABLE:
        st.header("🔧 Methoden")
        all_methods = get_all_methods()
        chosen = method_selector_advanced(all_methods)
        st.session_state.selected_methods = chosen
    else:
        st.error(f"detectors.py konnte nicht importiert werden: {DETECTOR_IMPORT_ERROR}")

# --------------------------------------------------------------
# TAB 3 – Einstellungen
# --------------------------------------------------------------
with tab3:
    st.header("⚙️ Einstellungen")
    st.write("Aktuell beschränken sich Einstellungen auf Sidebar-Optionen.")
    st.info("Invertierung & Skalierung sind standardmäßig aktiv.")

# --------------------------------------------------------------
# TAB 4 – Verarbeitung
# --------------------------------------------------------------
with tab4:
    st.header("🚀 Verarbeitung")

    if st.session_state.processing_status == "running":
        if not st.session_state.get("processing_started"):
            st.session_state.processing_started = True
            if not DETECTORS_AVAILABLE:
                st.error("Edge-Module nicht geladen – Abbruch."); st.stop()
            if not st.session_state.selected_images:
                st.error("Keine Bilder gewählt"); st.stop()
            if not st.session_state.selected_methods:
                st.error("Keine Methoden gewählt"); st.stop()

            # Starte Batch-Prozessor
            batch_log = batch_processor(
                st.session_state.selected_images,
                st.session_state.selected_methods,
                st.session_state.output_dir,
                settings={"target_size": get_max_resolution(
                              st.session_state.selected_images)}
            )
            st.session_state.processing_log = batch_log
            st.session_state.processing_status  = "completed"
            st.experimental_rerun()

        else:
            st.info("⏳ Verarbeitung läuft …")

    elif st.session_state.processing_status == "completed":
        st.success("🎉 Verarbeitung abgeschlossen")
        res_dir = os.path.join(st.session_state.output_dir,
                               "edge_detection_results")
        if os.path.isdir(res_dir):
            files = [f for f in os.listdir(res_dir) if f.endswith(".png")]
            st.write(f"{len(files)} Ausgabedateien")
            cols = st.columns(4)
            for i,f in enumerate(files[:8]):
                p = os.path.join(res_dir,f)
                im = _load_image(p)
                cols[i%4].image(_thumb(im), caption=f)

            # Download-ZIP
            def _zip_dir(path:str)->bytes:
                buf = BytesIO()
                with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
                    for root,_,fs in os.walk(path):
                        for fn in fs:
                            fp = os.path.join(root,fn)
                            arc = os.path.relpath(fp,path)
                            z.write(fp, arc)
                return buf.getvalue()

            zip_bytes = _zip_dir(res_dir)
            st.download_button("📥 Ergebnisse als ZIP",
                               data=zip_bytes,
                               file_name="edge_detection_results.zip",
                               mime="application/zip")

# --------------------------------------------------------------
# TAB 5 – Live-Vorschau
# --------------------------------------------------------------
with tab5:
    st.header("👁️ Live-Vorschau")
    if (st.session_state.selected_images
        and st.session_state.selected_methods
        and DETECTORS_AVAILABLE):
        colA,colB = st.columns(2)
        img_sel = colA.selectbox(
            "Bild", [Path(p).name for p in st.session_state.selected_images])
        m_sel   = colB.selectbox(
            "Methode", st.session_state.selected_methods)
        if st.button("🔄 Vorschau generieren"):
            img_path = next(p for p in st.session_state.selected_images
                            if Path(p).name==img_sel)
            func = dict(get_all_methods())[m_sel]
            res  = func(img_path, target_size=(512,512))
            ori  = cv2.resize(_load_image(img_path),(512,512))
            col1,col2 = st.columns(2)
            col1.image(cv2.cvtColor(ori,cv2.COLOR_BGR2RGB),
                       caption="Original", use_column_width=True)
            col2.image(res, caption=m_sel, clamp=True, use_column_width=True)
    else:
        st.info("Bitte zunächst Bilder *und* Methoden auswählen")

