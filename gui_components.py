"""
Wiederverwendbare GUI-Bausteine für das Edge-Detection-Studio
"""
from __future__ import annotations
import os, time, cv2, numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import streamlit as st

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
    folders = [f for f in os.listdir(cur)
               if os.path.isdir(os.path.join(cur,f))]
    folders.sort()
    cols = st.columns(3)
    for i,f in enumerate(folders):
        if cols[i%3].button("📁 "+f):
            st.session_state.fp_cur = os.path.join(cur,f); st.experimental_rerun()

    if st.button("✅ Diesen Ordner wählen", type="primary"):
        return st.session_state.fp_cur
    return None

# --------------------------------------------------------------
# Fortschritts-Tracker
# --------------------------------------------------------------
def progress_tracker(total:int, current:int,
                     current_image:str="", current_method:str="",
                     start_time:float|None=None)->Dict:
    pct = current/total*100 if total else 0
    st.progress(pct/100)
    col1,col2,col3 = st.columns(3)
    col1.metric("Fortschritt", f"{pct:.1f}%")
    col2.metric("Op", f"{current}/{total}")
    if start_time and current:
        eta = (time.time()-start_time)/current*(total-current)
        col3.metric("ETA", f"{eta/60:.1f} min")
    if current_image:
        st.write(f"🔄 {Path(current_image).name} → {current_method}")
    return {"pct":pct}

# --------------------------------------------------------------
# Methoden-Selector (erweitert)
# --------------------------------------------------------------
def method_selector_advanced(all_methods:List[Tuple[str,callable]])->List[str]:
    st.markdown("### Methoden auswählen")
    cats = {
        "Klassisch": ["Laplacian","Prewitt","Roberts","Scharr",
                      "GradientMagnitude","MorphologicalGradient"],
        "Canny-Varianten": ["Kornia_Canny","MultiScaleCanny","AdaptiveCanny"],
        "Deep Learning": ["HED_OpenCV","HED_PyTorch","StructuredForests",
                          "BDCN","FixedCNN"],
        "GPU": ["Kornia_Canny","Kornia_Sobel"]
    }
    if "msel" not in st.session_state:
        st.session_state.msel=["HED_PyTorch","Kornia_Canny","MultiScaleCanny"]
    col1,col2 = st.columns(2)
    col1.button("Alle",  on_click=lambda:
        st.session_state.update(msel=[n for n,_ in all_methods]))
    col2.button("Keine", on_click=lambda:
        st.session_state.update(msel=[]))
    for cat,names in cats.items():
        with st.expander(cat, expanded=True):
            for n,_ in [m for m in all_methods if m[0] in names]:
                chk = st.checkbox(n, value=n in st.session_state.msel,
                                  key=f"meth_{n}")
                if chk and n not in st.session_state.msel:
                    st.session_state.msel.append(n)
                if not chk and n in st.session_state.msel:
                    st.session_state.msel.remove(n)
    st.info("Ausgewählt: "+", ".join(st.session_state.msel) if
            st.session_state.msel else "Keine Methode gewählt")
    return st.session_state.msel

# --------------------------------------------------------------
# Batch-Prozessor (einfach)
# --------------------------------------------------------------
def batch_processor(images:List[str], methods:List[str],
                    output_dir:str, settings:Dict)->List[Dict]:
    from detectors import get_all_methods
    funcs = dict(get_all_methods())
    os.makedirs(output_dir, exist_ok=True)
    total = len(images)*len(methods)
    log=[]
    op=0; start=time.time()
    for img in images:
        for m in methods:
            op+=1
            try:
                res = funcs[m](img, target_size=settings.get("target_size"))
                out_dir = os.path.join(output_dir,"edge_detection_results")
                os.makedirs(out_dir, exist_ok=True)
                out_name = f"{Path(img).stem}_{m}.png"
                cv2.imwrite(os.path.join(out_dir,out_name), res)
                status="success"
            except Exception as e:
                status="error"; print(e)
            log.append({"image":img,"method":m,"status":status})
            progress_tracker(total,op,img,m,start)
    return log
