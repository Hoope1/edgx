"""
Wiederverwendbare GUI-Bausteine für das Edge-Detection-Studio
"""
from __future__ import annotations
import os
import time
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import streamlit as st

logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# Ordner-Picker
# --------------------------------------------------------------
def folder_picker(label:str, default_path:str="./")->Optional[str]:
    st.subheader(label)
    if "fp_cur" not in st.session_state:
        st.session_state.fp_cur = os.path.abspath(default_path)
    col1,col2,col3 = st.columns([1,4,1])
    with col1:
        if st.button("⬆️", help="Ein Ordner nach oben"):
            st.session_state.fp_cur = os.path.dirname(
                st.session_state.fp_cur)
    with col2:
        st.text_input("Pfad", key="fp_input",
                      value=st.session_state.fp_cur,
                      on_change=lambda:
                         st.session_state.update(
                             fp_cur=st.session_state.fp_input))
    with col3:
        if st.button("🏠", help="Home-Verzeichnis"):
            st.session_state.fp_cur = os.path.expanduser("~")

    cur = st.session_state.fp_cur
    if not os.path.isdir(cur):
        st.error("Pfad existiert nicht"); return None
    
    try:
        folders = [f for f in os.listdir(cur)
                   if os.path.isdir(os.path.join(cur,f))]
        folders.sort()
        cols = st.columns(3)
        for i,f in enumerate(folders):
            if cols[i%3].button("📁 "+f):
                st.session_state.fp_cur = os.path.join(cur,f)
                st.rerun()
    except PermissionError:
        st.error("Zugriff verweigert")
        return None

    if st.button("✅ Diesen Ordner wählen", type="primary"):
        return st.session_state.fp_cur
    return None

# --------------------------------------------------------------
# Bild-Galerie (FEHLTE IN ORIGINAL - HINZUGEFÜGT)
# --------------------------------------------------------------
def image_gallery(image_paths: List[str], max_display: int = 12) -> None:
    """Zeigt eine Galerie von Bildern an."""
    if not image_paths:
        st.info("Keine Bilder zum Anzeigen.")
        return
    
    st.write(f"📷 {len(image_paths)} Bilder gefunden")
    
    # Zeige nur die ersten max_display Bilder
    display_paths = image_paths[:max_display]
    
    cols = st.columns(4)
    for i, path in enumerate(display_paths):
        try:
            img = cv2.imread(path)
            if img is not None:
                # Thumbnail erstellen
                height, width = img.shape[:2]
                max_size = 150
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                
                thumbnail = cv2.resize(img, (new_width, new_height))
                thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                
                cols[i % 4].image(
                    thumbnail_rgb, 
                    caption=Path(path).name,
                    use_column_width=True
                )
            else:
                cols[i % 4].error(f"❌ {Path(path).name}")
        except Exception as e:
            logger.warning(f"Konnte Bild nicht laden: {path} - {e}")
            cols[i % 4].error(f"❌ Fehler: {Path(path).name}")
    
    if len(image_paths) > max_display:
        st.info(f"... und {len(image_paths) - max_display} weitere Bilder")

# --------------------------------------------------------------
# Fortschritts-Tracker
# --------------------------------------------------------------
def progress_tracker(total:int, current:int,
                     current_image:str="", current_method:str="",
                     start_time:float|None=None)->Dict:
    pct = current/total*100 if total else 0
    progress_bar = st.progress(pct/100)
    
    col1,col2,col3 = st.columns(3)
    col1.metric("Fortschritt", f"{pct:.1f}%")
    col2.metric("Operation", f"{current}/{total}")
    
    if start_time and current > 0:
        elapsed = time.time() - start_time
        avg_time_per_op = elapsed / current
        remaining_ops = total - current
        eta_seconds = avg_time_per_op * remaining_ops
        eta_minutes = eta_seconds / 60
        col3.metric("ETA", f"{eta_minutes:.1f} min")
    else:
        col3.metric("ETA", "Berechne...")
    
    if current_image and current_method:
        st.write(f"🔄 **{Path(current_image).name}** → {current_method}")
    
    return {"pct": pct, "progress_bar": progress_bar}

# --------------------------------------------------------------
# Methoden-Selector (erweitert)
# --------------------------------------------------------------
def method_selector_advanced(all_methods:List[Tuple[str,callable]])->List[str]:
    st.markdown("### 🔧 Methoden auswählen")
    
    # Kategorisierung der Methoden
    cats = {
        "🎯 Klassische Verfahren": [
            "Laplacian","Prewitt","Roberts","Scharr",
            "GradientMagnitude","MorphologicalGradient"
        ],
        "✂️ Canny-Varianten": [
            "Kornia_Canny","MultiScaleCanny","AdaptiveCanny"
        ],
        "🧠 Deep Learning": [
            "HED_OpenCV","HED_PyTorch","StructuredForests",
            "BDCN","FixedCNN"
        ],
        "⚡ GPU-Beschleunigt": [
            "Kornia_Canny","Kornia_Sobel"
        ]
    }
    
    # Initialisierung der Session State
    if "msel" not in st.session_state:
        st.session_state.msel = ["HED_OpenCV", "Kornia_Canny", "MultiScaleCanny"]
    
    # Schnellauswahl-Buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✅ Alle auswählen", help="Alle verfügbaren Methoden"):
            st.session_state.msel = [n for n, _ in all_methods]
            st.rerun()
    
    with col2:
        if st.button("❌ Alle abwählen", help="Keine Methoden"):
            st.session_state.msel = []
            st.rerun()
    
    with col3:
        if st.button("🎯 Empfohlene", help="Bewährte Methoden"):
            st.session_state.msel = [
                "HED_OpenCV", "MultiScaleCanny", "AdaptiveCanny", 
                "Laplacian", "Scharr"
            ]
            st.rerun()
    
    with col4:
        if st.button("⚡ Nur GPU", help="GPU-beschleunigte Methoden"):
            st.session_state.msel = ["Kornia_Canny", "Kornia_Sobel"]
            st.rerun()
    
    # Verfügbare Methoden sammeln
    available_methods = {name: func for name, func in all_methods}
    
    # Kategorisierte Auswahl
    for cat_name, method_names in cats.items():
        with st.expander(cat_name, expanded=True):
            # Nur verfügbare Methoden in dieser Kategorie anzeigen
            cat_methods = [(n, f) for n, f in all_methods if n in method_names]
            
            if not cat_methods:
                st.info("Keine Methoden in dieser Kategorie verfügbar.")
                continue
            
            # Checkbox für jede Methode
            for method_name, method_func in cat_methods:
                current_value = method_name in st.session_state.msel
                
                # Beschreibung der Methode
                descriptions = {
                    "HED_OpenCV": "Holistically-Nested Edge Detection (OpenCV)",
                    "HED_PyTorch": "HED mit PyTorch-Backend (mit Fallback)",
                    "StructuredForests": "Structured Edge Detection Forest",
                    "Kornia_Canny": "GPU-beschleunigter Canny (Kornia)",
                    "Kornia_Sobel": "GPU-beschleunigter Sobel (Kornia)",
                    "Laplacian": "Laplacian-Operator mit Gaussian Blur",
                    "Prewitt": "Prewitt-Kantendetektor",
                    "Roberts": "Roberts Cross-Gradient",
                    "Scharr": "Scharr-Operator (verbesserte Sobel)",
                    "GradientMagnitude": "Kombinierte Sobel + Scharr Gradients",
                    "MultiScaleCanny": "Multi-Scale Canny mit verschiedenen Blur-Levels",
                    "AdaptiveCanny": "Adaptive Threshold-Berechnung für Canny",
                    "MorphologicalGradient": "Morphologischer Gradient",
                    "BDCN": "Bi-Directional Cascade Network (mit Fallback)",
                    "FixedCNN": "Feste CNN-Filter mit PyTorch"
                }
                
                help_text = descriptions.get(method_name, "Edge-Detection-Methode")
                
                checkbox_value = st.checkbox(
                    method_name, 
                    value=current_value,
                    key=f"method_checkbox_{method_name}",
                    help=help_text
                )
                
                # Session State aktualisieren
                if checkbox_value and method_name not in st.session_state.msel:
                    st.session_state.msel.append(method_name)
                elif not checkbox_value and method_name in st.session_state.msel:
                    st.session_state.msel.remove(method_name)
    
    # Zusammenfassung der Auswahl
    selected_count = len(st.session_state.msel)
    total_count = len(all_methods)
    
    if selected_count > 0:
        st.success(f"✅ **{selected_count}** von {total_count} Methoden ausgewählt")
        with st.expander("📋 Ausgewählte Methoden", expanded=False):
            for method in st.session_state.msel:
                st.write(f"• {method}")
    else:
        st.warning("⚠️ Keine Methoden ausgewählt")
    
    return st.session_state.msel

# --------------------------------------------------------------
# Batch-Prozessor (erweitert)
# --------------------------------------------------------------
def batch_processor(images: List[str], methods: List[str],
                    output_dir: str, settings: Dict) -> List[Dict]:
    """
    Erweiterte Batch-Verarbeitung mit detailliertem Logging und Fortschritt.
    """
    try:
        from .detectors import get_all_methods
    except ImportError:
        st.error("❌ Detectors-Modul konnte nicht importiert werden!")
        return []
    
    # Funktions-Mapping erstellen
    funcs = dict(get_all_methods())
    
    # Output-Verzeichnis erstellen
    os.makedirs(output_dir, exist_ok=True)
    
    # Berechnungen
    total_operations = len(images) * len(methods)
    log = []
    operation_count = 0
    start_time = time.time()
    
    # Progress Container
    progress_container = st.container()
    status_container = st.container()
    
    with status_container:
        st.write("### 🔄 Verarbeitung läuft...")
    
    # Haupt-Verarbeitungsschleife
    for img_index, img_path in enumerate(images):
        img_name = Path(img_path).stem
        
        with status_container:
            st.write(f"**Bild {img_index + 1}/{len(images)}:** {Path(img_path).name}")
        
        for method_index, method_name in enumerate(methods):
            operation_count += 1
            
            try:
                # Prüfe, ob Methode verfügbar ist
                if method_name not in funcs:
                    raise ValueError(f"Methode {method_name} nicht verfügbar")
                
                # Führe Edge-Detection aus
                with status_container:
                    st.write(f"🔄 Verarbeite mit {method_name}...")
                
                result = funcs[method_name](
                    img_path, 
                    target_size=settings.get("target_size")
                )
                
                # Speichere Ergebnis
                output_subdir = os.path.join(output_dir, "edge_detection_results")
                os.makedirs(output_subdir, exist_ok=True)
                
                output_filename = f"{img_name}_{method_name}.png"
                output_path = os.path.join(output_subdir, output_filename)
                
                success = cv2.imwrite(output_path, result)
                
                if success:
                    status = "success"
                    logger.info(f"✅ {output_filename} erstellt")
                else:
                    status = "write_error"
                    logger.error(f"❌ Konnte {output_filename} nicht speichern")
                
            except Exception as e:
                status = "error"
                error_msg = str(e)
                logger.error(f"❌ Fehler bei {img_path} mit {method_name}: {error_msg}")
            
            # Log-Eintrag erstellen
            log_entry = {
                "image": img_path,
                "image_name": Path(img_path).name,
                "method": method_name,
                "status": status,
                "operation": operation_count,
                "total": total_operations
            }
            
            if status == "error":
                log_entry["error"] = error_msg
            
            log.append(log_entry)
            
            # Progress Update
            with progress_container:
                progress_tracker(
                    total_operations, 
                    operation_count,
                    img_path, 
                    method_name, 
                    start_time
                )
            
            # Kurze Pause für UI-Update
            time.sleep(0.01)
    
    # Abschluss-Status
    with status_container:
        elapsed_time = time.time() - start_time
        success_count = sum(1 for entry in log if entry["status"] == "success")
        error_count = total_operations - success_count
        
        st.write("### ✅ Verarbeitung abgeschlossen!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Erfolgreich", success_count)
        col2.metric("❌ Fehler", error_count)
        col3.metric("⏱️ Dauer", f"{elapsed_time:.1f}s")
    
    return log

# --------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------
def create_download_link(file_path: str, link_text: str) -> str:
    """Erstellt einen Download-Link für eine Datei."""
    import base64
    
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        b64_data = base64.b64encode(file_data).decode()
        filename = Path(file_path).name
        
        href = f'<a href="data:application/octet-stream;base64,{b64_data}" download="{filename}">{link_text}</a>'
        return href
    except Exception as e:
        logger.error(f"Konnte Download-Link nicht erstellen: {e}")
        return f"Fehler: {e}"

def validate_image_files(file_paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validiert eine Liste von Bildpfaden.
    
    Returns:
        Tuple: (gültige_pfade, ungültige_pfade)
    """
    valid_paths = []
    invalid_paths = []
    
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    for path in file_paths:
        try:
            # Prüfe Dateiendung
            if not path.lower().endswith(supported_extensions):
                invalid_paths.append(path)
                continue
            
            # Prüfe, ob Datei existiert
            if not os.path.exists(path):
                invalid_paths.append(path)
                continue
            
            # Prüfe, ob Bild geladen werden kann
            img = cv2.imread(path)
            if img is None:
                invalid_paths.append(path)
                continue
            
            valid_paths.append(path)
            
        except Exception as e:
            logger.warning(f"Validierung fehlgeschlagen für {path}: {e}")
            invalid_paths.append(path)
    
    return valid_paths, invalid_paths
