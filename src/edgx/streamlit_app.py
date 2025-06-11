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
    from .detectors import (
        get_all_methods, get_max_resolution, standardize_output
    )
    from .gui_components import (
        folder_picker, image_gallery, method_selector_advanced,
        progress_tracker, batch_processor, validate_image_files
    )
    DETECTORS_AVAILABLE = True
    DETECTOR_IMPORT_ERROR = None
except Exception as e:            # noqa
    DETECTORS_AVAILABLE = False
    DETECTOR_IMPORT_ERROR = str(e)

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
.main-header {
    font-size: 2.0rem; 
    font-weight: 600; 
    margin: 0.3em 0 1.2em 0;
    text-align: center;
    color: #1f77b4;
}
.status-processing { 
    color: #ff9800; 
    font-weight: 600; 
}
.status-success { 
    color: #4caf50; 
    font-weight: 600; 
}
.status-error { 
    color: #f44336; 
    font-weight: 600; 
}
.metric-container {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Session-State Defaults
# ------------------------------------------------------------------
def _init_state():
    """Initialisiert alle Session State Variablen."""
    defaults = {
        "selected_methods": [],
        "selected_images": [],
        "output_dir": "./results",
        "processing_status": "idle",  # idle, running, completed, error
        "progress": 0.0,
        "processing_log": [],
        "processing_started": False,
        "last_error": None,
        "debug_mode": False
    }
    
    for key, default_value in defaults.items():
        st.session_state.setdefault(key, default_value)

_init_state()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown('<div class="main-header">🎨 Edge Detection Studio</div>', 
            unsafe_allow_html=True)

if not DETECTORS_AVAILABLE:
    st.error(f"""
    ❌ **Kritischer Fehler**: Edge-Detection-Module konnten nicht geladen werden!
    
    **Fehler:** {DETECTOR_IMPORT_ERROR}
    
    **Lösungen:**
    1. Führen Sie `pip install -e .` im Projektverzeichnis aus
    2. Starten Sie `run.bat` neu
    3. Überprüfen Sie, ob alle Dependencies installiert sind
    """)
    st.stop()

# ------------------------------------------------------------------
# Sidebar – globale Konfiguration & Start/Stop
# ------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Konfiguration")

    # Eingabe-Modus
    st.subheader("📁 Eingabe")
    input_mode = st.radio(
        "Eingabe-Modus:",
        ["📁 Ordner auswählen", "📎 Einzelne Bilder hochladen"],
        key="input_mode_radio",
        help="Wählen Sie, wie Sie Bilder bereitstellen möchten"
    )

    # Ausgabe-Ordner
    st.subheader("📂 Ausgabe")
    st.session_state.output_dir = st.text_input(
        "Ausgabeordner:",
        value=st.session_state.output_dir,
        help="Pfad, in dem Ergebnisse abgelegt werden"
    )

    # Optionen
    st.subheader("⚙️ Optionen")
    invert_output = st.checkbox(
        "🎨 Invertierte Ausgabe", 
        value=True, 
        key="opt_invert",
        help="Weiße Kanten auf schwarzem Hintergrund"
    )
    uniform_size = st.checkbox(
        "📐 Einheitliche Größe", 
        value=True, 
        key="opt_uniform",
        help="Alle Ausgaben auf gleiche Größe skalieren"
    )
    
    # Erweiterte Optionen
    with st.expander("🔧 Erweiterte Optionen"):
        batch_size = st.slider("Batch-Größe", 1, 10, 5, 
                              help="Anzahl parallel verarbeiteter Bilder")
        st.session_state.debug_mode = st.checkbox("🐛 Debug-Modus", 
                                                 value=st.session_state.debug_mode)

    st.markdown("---")

    # System-Status
    st.subheader("💾 System-Status")
    col1, col2 = st.columns(2)
    col1.metric("🖼️ Bilder", len(st.session_state.selected_images))
    col2.metric("🔧 Methoden", len(st.session_state.selected_methods))
    
    if DETECTORS_AVAILABLE:
        available_methods = len(get_all_methods())
        st.metric("📋 Verfügbare Methoden", available_methods)

    st.markdown("---")

    # Start / Stop / Reset
    can_start = (len(st.session_state.selected_images) > 0 and 
                len(st.session_state.selected_methods) > 0 and
                st.session_state.processing_status == "idle")

    if can_start:
        if st.button("🚀 **VERARBEITUNG STARTEN**", 
                     type="primary", 
                     use_container_width=True):
            st.session_state.processing_status = "running"
            st.session_state.processing_started = False
            st.session_state.processing_log = []
            st.rerun()
    
    elif st.session_state.processing_status == "running":
        if st.button("⏹️ **STOPPEN**", 
                     type="secondary", 
                     use_container_width=True):
            st.session_state.processing_status = "idle"
            st.session_state.processing_started = False
            st.rerun()
    
    elif st.session_state.processing_status in ["completed", "error"]:
        if st.button("🔄 **NEUE VERARBEITUNG**", 
                     type="primary", 
                     use_container_width=True):
            st.session_state.processing_status = "idle"
            st.session_state.processing_started = False
            st.session_state.processing_log = []
            st.rerun()
    
    else:
        st.button("⏸️ Nicht bereit", disabled=True, use_container_width=True)
        if len(st.session_state.selected_images) == 0:
            st.caption("❌ Keine Bilder ausgewählt")
        if len(st.session_state.selected_methods) == 0:
            st.caption("❌ Keine Methoden ausgewählt")

# ------------------------------------------------------------------
# Tabs definieren
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 Bildauswahl", 
    "🔧 Methoden", 
    "⚙️ Einstellungen",
    "🚀 Verarbeitung", 
    "👁️ Vorschau"
])

# --------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

@st.cache_data(show_spinner=False)
def _load_image(path: str) -> Optional[np.ndarray]:
    """Lädt ein Bild mit Caching."""
    try:
        return cv2.imread(path)
    except Exception as e:
        st.session_state.last_error = f"Bild-Ladefehler: {e}"
        return None

def _find_images(folder: str) -> List[str]:
    """Findet alle unterstützten Bilder in einem Ordner."""
    if not os.path.isdir(folder):
        return []
    
    try:
        files = []
        for f in os.listdir(folder):
            if f.lower().endswith(_IMAGE_EXT):
                full_path = os.path.join(folder, f)
                if os.path.isfile(full_path):
                    files.append(full_path)
        return sorted(files)
    except PermissionError:
        st.error(f"❌ Zugriff auf Ordner verweigert: {folder}")
        return []

def _create_thumbnail(img: np.ndarray, size: Tuple[int,int]=(150,150)) -> np.ndarray:
    """Erstellt ein Thumbnail mit korrekten Proportionen."""
    height, width = img.shape[:2]
    target_width, target_height = size
    
    # Berechne Skalierungsfaktor unter Beibehaltung der Proportionen
    scale = min(target_width/width, target_height/height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

# --------------------------------------------------------------
# TAB 1 – Bildauswahl
# --------------------------------------------------------------
with tab1:
    st.header("📷 Bildauswahl")

    if "📁" in input_mode:  # Ordner auswählen
        st.subheader("📁 Ordner-basierte Auswahl")
        
        # Folder Picker
        selected_folder = folder_picker("Wählen Sie einen Ordner mit Bildern", "./images")
        
        if selected_folder:
            found_images = _find_images(selected_folder)
            
            if found_images:
                # Validiere Bilder
                valid_images, invalid_images = validate_image_files(found_images)
                
                st.session_state.selected_images = valid_images
                
                # Statistiken
                col1, col2, col3 = st.columns(3)
                col1.metric("✅ Gültige Bilder", len(valid_images))
                col2.metric("❌ Ungültige Bilder", len(invalid_images))
                col3.metric("📁 Ordner", Path(selected_folder).name)
                
                # Zeige Bilder
                if valid_images:
                    st.success(f"✅ {len(valid_images)} Bilder erfolgreich geladen")
                    image_gallery(valid_images, max_display=12)
                
                # Zeige ungültige Bilder (falls vorhanden)
                if invalid_images:
                    with st.expander(f"❌ {len(invalid_images)} ungültige Dateien"):
                        for invalid_path in invalid_images:
                            st.write(f"• {Path(invalid_path).name}")
                
            else:
                st.warning("⚠️ Keine unterstützten Bildformate in diesem Ordner gefunden.")
                st.info("Unterstützte Formate: " + ", ".join(_IMAGE_EXT))
        
        else:
            st.info("👆 Wählen Sie einen Ordner aus, um zu beginnen.")

    else:  # Einzelne Bilder hochladen
        st.subheader("📎 Datei-Upload")
        
        uploaded_files = st.file_uploader(
            "Wählen Sie Bilddateien aus:",
            type=[ext.lstrip('.') for ext in _IMAGE_EXT],
            accept_multiple_files=True,
            help="Sie können mehrere Dateien gleichzeitig auswählen"
        )
        
        if uploaded_files:
            # Temporäres Verzeichnis erstellen
            temp_dir = tempfile.mkdtemp(prefix="edgx_upload_")
            temp_paths = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Datei speichern
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_paths.append(temp_path)
                    
                    # Progress Update
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Verarbeite {uploaded_file.name}...")
                    
                except Exception as e:
                    st.error(f"❌ Fehler beim Upload von {uploaded_file.name}: {e}")
            
            progress_bar.empty()
            status_text.empty()
            
            # Validiere hochgeladene Bilder
            valid_images, invalid_images = validate_image_files(temp_paths)
            st.session_state.selected_images = valid_images
            
            # Statistiken
            col1, col2 = st.columns(2)
            col1.metric("✅ Erfolgreich", len(valid_images))
            col2.metric("❌ Fehlerhaft", len(invalid_images))
            
            # Zeige Bilder
            if valid_images:
                st.success(f"✅ {len(valid_images)} Bilder hochgeladen")
                image_gallery(valid_images, max_display=8)
            
            if invalid_images:
                st.error(f"❌ {len(invalid_images)} Dateien konnten nicht verarbeitet werden")

# --------------------------------------------------------------
# TAB 2 – Methoden
# --------------------------------------------------------------
with tab2:
    st.header("🔧 Methoden-Auswahl")
    
    if DETECTORS_AVAILABLE:
        try:
            all_methods = get_all_methods()
            
            if all_methods:
                chosen_methods = method_selector_advanced(all_methods)
                st.session_state.selected_methods = chosen_methods
                
                # Methoden-Informationen
                if chosen_methods:
                    st.markdown("---")
                    st.subheader("ℹ️ Ausgewählte Methoden-Details")
                    
                    method_info = {
                        "HED_OpenCV": {"typ": "Deep Learning", "gpu": False, "qualität": "Hoch"},
                        "HED_PyTorch": {"typ": "Deep Learning", "gpu": True, "qualität": "Hoch"},
                        "Kornia_Canny": {"typ": "Klassisch", "gpu": True, "qualität": "Mittel"},
                        "MultiScaleCanny": {"typ": "Klassisch", "gpu": False, "qualität": "Hoch"},
                        "Laplacian": {"typ": "Klassisch", "gpu": False, "qualität": "Mittel"},
                    }
                    
                    for method in chosen_methods[:5]:  # Zeige nur erste 5
                        info = method_info.get(method, {"typ": "Klassisch", "gpu": False, "qualität": "Normal"})
                        col1, col2, col3, col4 = st.columns(4)
                        col1.write(f"**{method}**")
                        col2.write(info["typ"])
                        col3.write("✅ GPU" if info["gpu"] else "💻 CPU")
                        col4.write(info["qualität"])
                
            else:
                st.error("❌ Keine Edge-Detection-Methoden verfügbar!")
                
        except Exception as e:
            st.error(f"❌ Fehler beim Laden der Methoden: {e}")
            st.session_state.last_error = str(e)
    
    else:
        st.error("❌ Detectors-Modul nicht verfügbar - bitte Installation prüfen")

# --------------------------------------------------------------
# TAB 3 – Einstellungen
# --------------------------------------------------------------
with tab3:
    st.header("⚙️ Verarbeitungs-Einstellungen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Bildverarbeitung")
        
        target_resolution = st.selectbox(
            "Ziel-Auflösung:",
            ["Auto (Max-Resolution)", "1920x1080", "1280x720", "640x480", "Custom"],
            help="Größe der Ausgabebilder"
        )
        
        if target_resolution == "Custom":
            custom_width = st.number_input("Breite (px):", min_value=64, max_value=4096, value=1024)
            custom_height = st.number_input("Höhe (px):", min_value=64, max_value=4096, value=768)
        
        edge_thickness = st.slider("Kantenstärke:", 1, 5, 2, 
                                  help="Nachbearbeitung der Kantendicke")
        
        noise_reduction = st.checkbox("🔇 Rauschreduktion", value=True,
                                     help="Reduziert Bildrauschen vor Edge Detection")
    
    with col2:
        st.subheader("⚡ Performance")
        
        use_gpu = st.checkbox("🎮 GPU verwenden (wenn verfügbar)", value=True)
        
        max_workers = st.slider("Max. parallele Prozesse:", 1, 8, 4,
                               help="Anzahl CPU-Kerne für Parallelverarbeitung")
        
        memory_limit = st.selectbox("🧠 Speicher-Limit:",
                                   ["Kein Limit", "2GB", "4GB", "8GB"],
                                   index=1)
        
        save_originals = st.checkbox("💾 Original-Bilder mitexportieren", value=False)
    
    # Vorschau der Einstellungen
    st.markdown("---")
    st.subheader("📋 Aktuelle Konfiguration")
    
    config_summary = {
        "Invertierung": "✅" if st.session_state.opt_invert else "❌",
        "Einheitliche Größe": "✅" if st.session_state.opt_uniform else "❌",
        "Ziel-Auflösung": target_resolution,
        "GPU-Nutzung": "✅" if use_gpu else "❌",
        "Parallel-Prozesse": max_workers,
        "Debug-Modus": "✅" if st.session_state.debug_mode else "❌"
    }
    
    for key, value in config_summary.items():
        st.write(f"**{key}:** {value}")

# --------------------------------------------------------------
# TAB 4 – Verarbeitung
# --------------------------------------------------------------
with tab4:
    st.header("🚀 Batch-Verarbeitung")

    if st.session_state.processing_status == "running":
        if not st.session_state.processing_started:
            st.session_state.processing_started = True
            
            # Pre-Processing Validierungen
            if not DETECTORS_AVAILABLE:
                st.error("❌ Edge-Module nicht geladen – Abbruch.")
                st.session_state.processing_status = "error"
                st.stop()
            
            if not st.session_state.selected_images:
                st.error("❌ Keine Bilder ausgewählt")
                st.session_state.processing_status = "error"
                st.stop()
            
            if not st.session_state.selected_methods:
                st.error("❌ Keine Methoden ausgewählt")
                st.session_state.processing_status = "error"
                st.stop()

            # Starte Batch-Verarbeitung
            st.info("🔄 **Verarbeitung gestartet...** Dies kann einige Minuten dauern.")
            
            try:
                # Berechne Ziel-Auflösung
                if st.session_state.selected_images:
                    target_size = get_max_resolution(st.session_state.selected_images)
                else:
                    target_size = (1920, 1080)
                
                # Verarbeitungs-Einstellungen
                processing_settings = {
                    "target_size": target_size,
                    "invert": st.session_state.get("opt_invert", True),
                    "uniform": st.session_state.get("opt_uniform", True),
                    "debug": st.session_state.debug_mode
                }
                
                # Starte Batch-Processor
                with st.spinner("🔄 Edge-Detection läuft..."):
                    batch_log = batch_processor(
                        st.session_state.selected_images,
                        st.session_state.selected_methods,
                        st.session_state.output_dir,
                        processing_settings
                    )
                
                st.session_state.processing_log = batch_log
                st.session_state.processing_status = "completed"
                st.rerun()
                
            except Exception as e:
                error_msg = f"Verarbeitung fehlgeschlagen: {e}"
                st.error(f"❌ {error_msg}")
                st.session_state.processing_status = "error"
                st.session_state.last_error = error_msg
                st.session_state.processing_started = False
                
                if st.session_state.debug_mode:
                    st.exception(e)

        else:
            st.info("⏳ **Verarbeitung läuft...** Seite wird automatisch aktualisiert.")
            
            # Auto-refresh alle 2 Sekunden
            time.sleep(2)
            st.rerun()

    elif st.session_state.processing_status == "completed":
        st.success("🎉 **Verarbeitung erfolgreich abgeschlossen!**")
        
        # Ergebnis-Verzeichnis
        results_dir = os.path.join(st.session_state.output_dir, "edge_detection_results")
        
        if os.path.isdir(results_dir):
            # Statistiken
            result_files = [f for f in os.listdir(results_dir) if f.endswith(".png")]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Ausgabedateien", len(result_files))
            
            if st.session_state.processing_log:
                success_count = sum(1 for log in st.session_state.processing_log 
                                  if log.get("status") == "success")
                error_count = len(st.session_state.processing_log) - success_count
                col2.metric("✅ Erfolgreich", success_count)
                col3.metric("❌ Fehler", error_count)
            
            # Ergebnis-Galerie
            if result_files:
                st.subheader("📸 Ergebnis-Vorschau")
                result_paths = [os.path.join(results_dir, f) for f in result_files[:12]]
                image_gallery(result_paths, max_display=12)
                
                # Download-ZIP erstellen
                st.subheader("📥 Download")
                
                @st.cache_data
                def create_results_zip(directory_path: str) -> bytes:
                    """Erstellt ZIP-Archiv aller Ergebnisse."""
                    buffer = BytesIO()
                    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for root, dirs, files in os.walk(directory_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                archive_path = os.path.relpath(file_path, directory_path)
                                zip_file.write(file_path, archive_path)
                    return buffer.getvalue()
                
                try:
                    zip_data = create_results_zip(results_dir)
                    
                    st.download_button(
                        label="📥 **Alle Ergebnisse als ZIP herunterladen**",
                        data=zip_data,
                        file_name=f"edge_detection_results_{int(time.time())}.zip",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ ZIP-Erstellung fehlgeschlagen: {e}")
                
                # Verarbeitungsprotokoll
                if st.session_state.processing_log:
                    with st.expander("📋 Detailliertes Verarbeitungsprotokoll"):
                        for i, log_entry in enumerate(st.session_state.processing_log):
                            status_icon = "✅" if log_entry.get("status") == "success" else "❌"
                            img_name = Path(log_entry.get("image", "")).name
                            method = log_entry.get("method", "")
                            
                            col_a, col_b, col_c = st.columns([1, 2, 2])
                            col_a.write(status_icon)
                            col_b.write(img_name)
                            col_c.write(method)
            
            else:
                st.warning("⚠️ Keine Ergebnisdateien gefunden")
        
        else:
            st.error("❌ Ergebnisordner nicht gefunden")

    elif st.session_state.processing_status == "error":
        st.error("❌ **Verarbeitung fehlgeschlagen**")
        
        if st.session_state.last_error:
            st.error(f"**Fehler:** {st.session_state.last_error}")
        
        st.info("💡 Prüfen Sie Ihre Eingaben und versuchen Sie es erneut.")

    else:  # idle
        st.info("👆 **Bereit für Verarbeitung**")
        st.write("Klicken Sie auf '🚀 VERARBEITUNG STARTEN' in der Sidebar, um zu beginnen.")
        
        # Voraussetzungen prüfen
        ready_checklist = []
        ready_checklist.append(("📷 Bilder ausgewählt", len(st.session_state.selected_images) > 0))
        ready_checklist.append(("🔧 Methoden ausgewählt", len(st.session_state.selected_methods) > 0))
        ready_checklist.append(("📂 Ausgabeordner gesetzt", bool(st.session_state.output_dir)))
        ready_checklist.append(("🎛️ Detectors verfügbar", DETECTORS_AVAILABLE))
        
        st.subheader("✅ Bereitschafts-Check")
        for check_name, is_ready in ready_checklist:
            icon = "✅" if is_ready else "❌"
            st.write(f"{icon} {check_name}")

# --------------------------------------------------------------
# TAB 5 – Live-Vorschau
# --------------------------------------------------------------
with tab5:
    st.header("👁️ Live-Vorschau")
    
    if (st.session_state.selected_images and 
        st.session_state.selected_methods and 
        DETECTORS_AVAILABLE):
        
        st.subheader("🎛️ Vorschau-Konfiguration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bildauswahl
            image_names = [Path(p).name for p in st.session_state.selected_images]
            selected_image_name = st.selectbox(
                "📷 Bild auswählen:", 
                image_names,
                help="Wählen Sie ein Bild für die Vorschau"
            )
        
        with col2:
            # Methodenauswahl
            selected_method = st.selectbox(
                "🔧 Methode auswählen:", 
                st.session_state.selected_methods,
                help="Wählen Sie eine Edge-Detection-Methode"
            )
        
        # Vorschau-Optionen
        col3, col4 = st.columns(2)
        with col3:
            preview_size = st.selectbox("📐 Vorschau-Größe:", 
                                       ["512x512", "256x256", "1024x1024"],
                                       index=0)
        with col4:
            show_comparison = st.checkbox("🔄 Vergleichsansicht", value=True)
        
        # Vorschau generieren
        if st.button("🔄 **Vorschau generieren**", type="primary", use_container_width=True):
            try:
                # Finde ausgewähltes Bild
                selected_image_path = next(
                    p for p in st.session_state.selected_images
                    if Path(p).name == selected_image_name
                )
                
                # Parse Vorschau-Größe
                width, height = map(int, preview_size.split('x'))
                target_size = (width, height)
                
                with st.spinner(f"🔄 Generiere Vorschau mit {selected_method}..."):
                    # Edge-Detection ausführen
                    method_functions = dict(get_all_methods())
                    edge_function = method_functions[selected_method]
                    
                    edge_result = edge_function(selected_image_path, target_size=target_size)
                    original_image = cv2.resize(_load_image(selected_image_path), target_size)
                    
                    # Ergebnisse anzeigen
                    st.subheader("📊 Vorschau-Ergebnisse")
                    
                    if show_comparison:
                        col_orig, col_edge = st.columns(2)
                        
                        with col_orig:
                            st.write("**🖼️ Original**")
                            st.image(
                                cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
                                use_column_width=True,
                                caption=f"Original: {selected_image_name}"
                            )
                        
                        with col_edge:
                            st.write(f"**🎨 Edge Detection: {selected_method}**")
                            st.image(
                                edge_result,
                                use_column_width=True,
                                caption=f"Methode: {selected_method}",
                                clamp=True
                            )
                    else:
                        st.write(f"**🎨 Edge Detection: {selected_method}**")
                        st.image(
                            edge_result,
                            use_column_width=True,
                            caption=f"{selected_image_name} → {selected_method}",
                            clamp=True
                        )
                    
                    # Technische Details
                    with st.expander("🔍 Technische Details"):
                        col_tech1, col_tech2 = st.columns(2)
                        
                        col_tech1.write("**Original-Eigenschaften:**")
                        orig_height, orig_width = original_image.shape[:2]
                        col_tech1.write(f"• Größe: {orig_width}×{orig_height}")
                        col_tech1.write(f"• Typ: {original_image.dtype}")
                        
                        col_tech2.write("**Edge-Map-Eigenschaften:**")
                        edge_height, edge_width = edge_result.shape[:2]
                        col_tech2.write(f"• Größe: {edge_width}×{edge_height}")
                        col_tech2.write(f"• Typ: {edge_result.dtype}")
                        col_tech2.write(f"• Wertebereich: {edge_result.min()}-{edge_result.max()}")
                
            except Exception as e:
                st.error(f"❌ **Vorschau-Generierung fehlgeschlagen:** {e}")
                
                if st.session_state.debug_mode:
                    st.exception(e)
    
    else:
        # Fehlende Voraussetzungen
        missing_items = []
        if not DETECTORS_AVAILABLE:
            missing_items.append("⚠️ Detectors-Modul nicht verfügbar")
        if not st.session_state.selected_images:
            missing_items.append("📷 Keine Bilder ausgewählt")
        if not st.session_state.selected_methods:
            missing_items.append("🔧 Keine Methoden ausgewählt")
        
        st.info("**Voraussetzungen für Live-Vorschau:**")
        for item in missing_items:
            st.write(f"❌ {item}")
        
        st.write("📋 Bitte vervollständigen Sie Ihre Auswahl in den anderen Tabs.")

# --------------------------------------------------------------
# Footer
# --------------------------------------------------------------
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.caption("🎨 **Edge Detection Studio**")
    st.caption("Powered by Streamlit + OpenCV + PyTorch")

with col_footer2:
    if st.session_state.debug_mode:
        st.caption("🐛 **Debug-Info:**")
        st.caption(f"Status: {st.session_state.processing_status}")
        st.caption(f"Bilder: {len(st.session_state.selected_images)}")
        st.caption(f"Methoden: {len(st.session_state.selected_methods)}")

with col_footer3:
    st.caption("📊 **System:**")
    st.caption(f"Detectors: {'✅' if DETECTORS_AVAILABLE else '❌'}")
    if DETECTORS_AVAILABLE:
        st.caption(f"Methoden: {len(get_all_methods())}")
