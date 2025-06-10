import argparse
import gzip
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import requests
import torch

BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
HED_DIR    = os.path.join(MODEL_DIR, 'hed')
STRUCT_DIR = os.path.join(MODEL_DIR, 'structured')

# Alternative URLs für HED (fallback wenn erste nicht funktioniert)
HED_PROTO_URLS = [
    'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/hed_edge_detection/deploy.prototxt',
    'https://raw.githubusercontent.com/ashukid/hed-edge-detector/master/deploy.prototxt'
]
HED_WEIGHTS_URLS = [
    'https://github.com/ashukid/hed-edge-detector/raw/master/hed_pretrained_bsds.caffemodel',
    'https://drive.google.com/uc?id=1zc-tSjrZ1Q1q6hzYNDaLdgBCcCRFjYLF'  # Backup Google Drive
]
STRUCT_URL = 'https://github.com/opencv/opencv_extra/raw/master/testdata/cv/ximgproc/model.yml.gz'

# ------------------------------------------------------
# Helper‑Download
# ------------------------------------------------------

def _download_with_fallback(urls: list, dst: str) -> bool:
    """Download mit mehreren Fallback-URLs"""
    if os.path.exists(dst):
        return True
    
    for i, url in enumerate(urls):
        try:
            print(f"[download] {os.path.basename(dst)} (URL {i+1}/{len(urls)})")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(dst, 'wb') as fh:
                fh.write(r.content)
            return True
        except Exception as e:
            print(f"[failed] URL {i+1} fehlgeschlagen: {e}")
            if i < len(urls) - 1:
                print(f"[retry] Versuche nächste URL...")
    
    print(f"[error] Alle URLs für {os.path.basename(dst)} fehlgeschlagen")
    return False

def _download(url: str, dst: str) -> None:
    """Einzelner Download (für rückwärtskompatibilität)"""
    _download_with_fallback([url], dst)

# ------------------------------------------------------
# Modelle initialisieren (HED, Structured, optional BDCN)
# ------------------------------------------------------

def init_models() -> None:
    os.makedirs(HED_DIR, exist_ok=True)
    os.makedirs(STRUCT_DIR, exist_ok=True)

    # HED Modell mit Fallbacks
    proto_path = os.path.join(HED_DIR, 'deploy.prototxt')
    weights_path = os.path.join(HED_DIR, 'hed.caffemodel')
    
    proto_success = _download_with_fallback(HED_PROTO_URLS, proto_path)
    weights_success = _download_with_fallback(HED_WEIGHTS_URLS, weights_path)
    
    if not proto_success or not weights_success:
        print("[warning] HED Modell Download fehlgeschlagen - HED wird nicht verfügbar sein")
        # Erstelle leere Dateien als Marker
        if not proto_success:
            with open(proto_path + '.failed', 'w') as f:
                f.write("Download failed")
        if not weights_success:
            with open(weights_path + '.failed', 'w') as f:
                f.write("Download failed")

    # Structured Forests
    gz_path = os.path.join(STRUCT_DIR, 'model.yml.gz')
    yml_path = os.path.join(STRUCT_DIR, 'model.yml')
    _download(STRUCT_URL, gz_path)
    if not os.path.exists(yml_path) and os.path.exists(gz_path):
        try:
            with gzip.open(gz_path, 'rb') as fi, open(yml_path, 'wb') as fo:
                fo.write(fi.read())
            print('[unpack] Structured Forests Modell entpackt')
        except Exception as e:
            print(f'[error] Entpacken fehlgeschlagen: {e}')

    # Optional: BDCN Repo (Klon + Installation) - VERBESSERT
    bdcn_repo = os.path.join(BASE_DIR, 'bdcn_repo')
    if not os.path.isdir(bdcn_repo):
        print('[clone] BDCN‑Repo …')
        try:
            # Klone das Repository
            subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/YacobBY/bdcn.git', bdcn_repo], check=True)
            
            # Erstelle eine modifizierte requirements.txt ohne problematische Versionen
            original_req = os.path.join(bdcn_repo, 'requirements.txt')
            temp_req = os.path.join(bdcn_repo, 'requirements_fixed.txt')
            
            if os.path.exists(original_req):
                with open(original_req, 'r') as f:
                    lines = f.readlines()
                
                # Filtere problematische Pakete und Versionen
                fixed_lines = []
                skip_packages = {'numpy', 'torch', 'torchvision', 'opencv-python', 'opencv-contrib-python'}
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    package_name = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                    
                    # Überspringe Pakete die bereits installiert sind oder Probleme machen
                    if package_name.lower() not in skip_packages:
                        # Entferne spezifische Versionsangaben für problematische Pakete
                        if 'matplotlib' in package_name.lower():
                            fixed_lines.append('matplotlib>=3.1.0')
                        elif 'pillow' in package_name.lower():
                            fixed_lines.append('pillow')
                        else:
                            fixed_lines.append(line)
                
                # Schreibe die gefixte requirements.txt
                with open(temp_req, 'w') as f:
                    f.write('\n'.join(fixed_lines))
                
                print('[fix] Verwende modifizierte requirements.txt für BDCN')
                
                # Installiere die gefixten Requirements
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', temp_req], check=True)
            
            # Versuche das BDCN Paket zu installieren (ohne setup.py dependencies)
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', bdcn_repo, '--no-deps'], check=True)
                print('[success] BDCN erfolgreich installiert')
            except subprocess.CalledProcessError:
                print('[warning] BDCN Paket-Installation fehlgeschlagen, aber Repository verfügbar')
                
        except subprocess.CalledProcessError as e:
            print(f"[warning] BDCN Installation fehlgeschlagen: {e}")
        except FileNotFoundError:
            print("[warning] Git nicht gefunden - BDCN wird übersprungen")

# ------------------------------------------------------
# Edge‑Methoden
# ------------------------------------------------------

def run_hed(path: str) -> np.ndarray:
    proto  = os.path.join(HED_DIR, 'deploy.prototxt')
    weight = os.path.join(HED_DIR, 'hed.caffemodel')
    
    # Prüfe ob Modell-Dateien existieren
    if not os.path.exists(proto) or not os.path.exists(weight):
        raise RuntimeError("HED Modell nicht verfügbar - Download fehlgeschlagen")
    
    net = cv2.dnn.readNetFromCaffe(proto, weight)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    H, W = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (W, H), (104.00699, 116.66877, 122.67891), False, False)
    net.setInput(blob)
    out = net.forward()[0, 0]
    out = cv2.resize(out, (W, H))
    return (out * 255).astype('uint8')

def run_structured(path: str) -> np.ndarray:
    mdl = os.path.join(STRUCT_DIR, 'model.yml')
    if not os.path.exists(mdl):
        raise RuntimeError("Structured Forests Modell nicht verfügbar")
    
    ed  = cv2.ximgproc.createStructuredEdgeDetection(mdl)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    img = img.astype('float32') / 255.0
    edge = ed.detectEdges(img)
    return (edge * 255).astype('uint8')

def run_kornia(path: str) -> np.ndarray:
    import kornia
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    t = torch.tensor(gray / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    edge = kornia.filters.canny(t)[0][0]
    return (edge.numpy() * 255).astype('uint8')

def run_bdcn(path: str) -> np.ndarray:
    """
    BDCN Edge Detection - Fallback Implementation
    Falls das Original-BDCN nicht verfügbar ist, verwende eine vereinfachte Implementierung
    """
    try:
        # Versuche das originale BDCN zu importieren
        from bdcn_edge import BDCNEdgeDetector
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Bild konnte nicht geladen werden: {path}")
        edge = BDCNEdgeDetector().detect(img)
        return (edge * 255).astype('uint8')
    except ImportError:
        # Fallback: Verwende eine Kombination aus Canny und morphologischen Operationen
        print("[fallback] BDCN nicht verfügbar, verwende Canny-basierte Alternative")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Bild konnte nicht geladen werden: {path}")
        
        # Gaussianisches Rauschen reduzieren
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Canny Edge Detection mit optimierten Parametern
        edges = cv2.Canny(img_blur, 50, 150)
        
        # Morphologische Operationen für bessere Kantenkontinuität
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)
        
        return edges

def run_fixed(path: str) -> np.ndarray:
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    k = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
    cx = torch.nn.Conv2d(1,1,3,padding=1,bias=False)
    cx.weight.data = k.unsqueeze(0).unsqueeze(0)
    cy = torch.nn.Conv2d(1,1,3,padding=1,bias=False)
    cy.weight.data = k.t().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        t = torch.tensor(gray / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        e = torch.sqrt(cx(t) ** 2 + cy(t) ** 2).squeeze().numpy()
    return (e * 255).astype('uint8')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--init-models', action='store_true')
    args = p.parse_args()
    if args.init_models:
        init_models()