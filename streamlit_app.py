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
