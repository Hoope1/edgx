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

# ------------------------------------------------------
# Alternative URLs für HED (Fallback-Strategie)
# ------------------------------------------------------
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
# Helper: Downloads mit Fallback
# ------------------------------------------------------
def _download_with_fallback(urls: list, dst: str) -> bool:
    """Download mit mehreren Fallback-URLs (Stoppt, sobald einer klappt)."""
    if os.path.exists(dst):
        return True

    for i, url in enumerate(urls):
        try:
            print(f"[download] {os.path.basename(dst)}  (URL {i+1}/{len(urls)})")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(dst, "wb") as fh:
                fh.write(r.content)
            return True
        except Exception as e:
            print(f"[failed ] URL {i+1} fehlgeschlagen: {e}")
            if i < len(urls) - 1:
                print("[retry  ] Versuche nächste URL …")

    print(f"[error  ] Alle URLs für {os.path.basename(dst)} fehlgeschlagen")
    return False


def _download(url: str, dst: str) -> None:
    """Einfacher Einzel-Download (Kompatibilität)."""
    _download_with_fallback([url], dst)

# ------------------------------------------------------
# Gemeinsame Bild-Normalisierung & Hilfsfunktionen
# ------------------------------------------------------
def standardize_output(edge_map: np.ndarray,
                       target_size: tuple | None = None,
                       invert: bool = True) -> np.ndarray:
    """
    Vereinheitlicht die Ausgaben aller Edge-Methoden

    • Invert: weißer Hintergrund, dunkle Kanten  
    • Resize: skaliert (CUBIC) auf `target_size`, falls angegeben  
    • uint8 garantiert
    """
    # --- Normierung 0-255 -----------------------------------------------
    if edge_map.dtype != np.uint8:
        edge_map = (edge_map * 255 if edge_map.max() <= 1.0
                    else edge_map).astype(np.uint8)

    # --- Invertieren -----------------------------------------------------
    if invert:
        edge_map = 255 - edge_map

    # --- Resize ----------------------------------------------------------
    if target_size is not None:
        edge_map = cv2.resize(edge_map, target_size,
                              interpolation=cv2.INTER_CUBIC)

    return edge_map


def get_max_resolution(image_paths: list[str]) -> tuple[int, int]:
    """Ermittelt die größte Breite/Höhe aus einer Bildliste."""
    max_w = max_h = 0
    for path in image_paths:
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        max_w, max_h = max(max_w, w), max(max_h, h)

    return (max_w, max_h) if max_w and max_h else (1920, 1080)

# ------------------------------------------------------
# Modell-Initialisierung (HED, Structured Forest, BDCN, …)
# ------------------------------------------------------
def init_models() -> None:
    os.makedirs(HED_DIR,     exist_ok=True)
    os.makedirs(STRUCT_DIR,  exist_ok=True)

    # --- HED -------------------------------------------------------------
    proto_path   = os.path.join(HED_DIR, 'deploy.prototxt')
    weights_path = os.path.join(HED_DIR, 'hed.caffemodel')

    proto_ok   = _download_with_fallback(HED_PROTO_URLS,   proto_path)
    weight_ok  = _download_with_fallback(HED_WEIGHTS_URLS, weights_path)

    if not (proto_ok and weight_ok):
        print("[warning] HED Modell konnte nicht vollständig geladen werden")

    # --- Structured Forests ---------------------------------------------
    gz_path  = os.path.join(STRUCT_DIR, 'model.yml.gz')
    yml_path = os.path.join(STRUCT_DIR, 'model.yml')

    _download(STRUCT_URL, gz_path)
    if not os.path.exists(yml_path) and os.path.exists(gz_path):
        try:
            with gzip.open(gz_path, "rb") as fi, open(yml_path, "wb") as fo:
                fo.write(fi.read())
            print("[unpack ] Structured Forest Modell entpackt")
        except Exception as e:
            print(f"[error  ] Entpacken fehlgeschlagen: {e}")

    # --- pytorch-hed (pip Package) --------------------------------------
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pytorch-hed", "--quiet"],
            check=True, capture_output=True
        )
        print("[success] pytorch-hed installiert")
    except subprocess.CalledProcessError:
        print("[warning] pytorch-hed Installation fehlgeschlagen")

    # --- BDCN (als Beispiel – kann ignoriert werden, falls git fehlt) ----
    bdcn_repo = os.path.join(BASE_DIR, "bdcn_repo")
    if not os.path.isdir(bdcn_repo):
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/YacobBY/bdcn.git", bdcn_repo],
                check=True
            )
            # Requirements patchen …
            req_in  = os.path.join(bdcn_repo, "requirements.txt")
            req_out = os.path.join(bdcn_repo, "requirements_fixed.txt")

            if os.path.exists(req_in):
                with open(req_in, "r", encoding="utf-8") as fh:
                    lines = fh.readlines()

                skip = {"numpy", "torch", "torchvision",
                        "opencv-python", "opencv-contrib-python"}
                fixed = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    pkg = re.split(r"[=<>]", line)[0].lower()
                    if pkg in skip:
                        continue
                    if "matplotlib" in pkg:
                        fixed.append("matplotlib>=3.1.0")
                    elif "pillow" in pkg:
                        fixed.append("pillow")
                    else:
                        fixed.append(line)

                with open(req_out, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(fixed))

                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", req_out,
                     "--quiet"],
                    check=True
                )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", bdcn_repo,
                 "--no-deps", "--quiet"],
                check=True
            )
            print("[success] BDCN installiert")
        except Exception as e:
            print(f"[warning] BDCN konnte nicht geklont/gebaut werden: {e}")

# ------------------------------------------------------
# Edge-Detection Routinen
# ------------------------------------------------------
def run_hed(path: str, target_size: tuple | None = None) -> np.ndarray:
    """Original HED (OpenCV-DNN)."""
    proto  = os.path.join(HED_DIR, 'deploy.prototxt')
    weight = os.path.join(HED_DIR, 'hed.caffemodel')
    if not (os.path.exists(proto) and os.path.exists(weight)):
        raise RuntimeError("HED Modell nicht verfügbar – bitte init_models ausführen")

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")

    H, W = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img, 1.0, (W, H),
        (104.00699, 116.66877, 122.67891),
        swapRB=False, crop=False
    )
    net = cv2.dnn.readNetFromCaffe(proto, weight)
    net.setInput(blob)
    out = net.forward()[0, 0]
    out = cv2.resize(out, (W, H))
    return standardize_output(out, target_size)

def run_pytorch_hed(path: str, target_size: tuple | None = None) -> np.ndarray:
    """PyPI-Paket *pytorch-hed* (Holistically-Nested Edge Detector)."""
    try:
        import torchHED
        from PIL import Image
        pil = Image.open(path).convert("RGB")
        edge = torchHED.process_img(pil)
        arr  = np.array(edge)
        if arr.ndim == 3:  # konvertiere zu Grau
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return standardize_output(arr, target_size)
    except ImportError:
        print("[fallback] pytorch-hed nicht vorhanden → OpenCV-HED")
        return run_hed(path, target_size)

def run_structured(path: str, target_size: tuple | None = None) -> np.ndarray:
    mdl = os.path.join(STRUCT_DIR, 'model.yml')
    if not os.path.exists(mdl):
        raise RuntimeError("Structured Forests Modell fehlt")
    det = cv2.ximgproc.createStructuredEdgeDetection(mdl)
    img = cv2.imread(path).astype("float32") / 255.0
    edges = det.detectEdges(img)
    return standardize_output(edges, target_size)

def run_kornia_canny(path: str, target_size: tuple | None = None) -> np.ndarray:
    import kornia
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    t = torch.tensor(g/255.0, dtype=torch.float32)[None,None]
    edges = kornia.filters.canny(t)[0][0].numpy()
    return standardize_output(edges, target_size)

def run_kornia_sobel(path: str, target_size: tuple | None = None) -> np.ndarray:
    import kornia
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    t = torch.tensor(g/255.0, dtype=torch.float32)[None,None]
    sx = kornia.filters.sobel(t, normalized=True)
    sy = kornia.filters.sobel(t, normalized=True, dim=3)
    mag = torch.sqrt(sx**2 + sy**2)[0,0].numpy()
    return standardize_output(mag, target_size)

def run_laplacian(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(g, (5,5), 0)
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    return standardize_output(np.abs(lap), target_size)

def run_prewitt(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32)
    ky = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], np.float32)
    ex = cv2.filter2D(g, cv2.CV_32F, kx)
    ey = cv2.filter2D(g, cv2.CV_32F, ky)
    return standardize_output(np.hypot(ex, ey), target_size)

def run_roberts(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    kx = np.array([[1,0],[0,-1]], np.float32)
    ky = np.array([[0,1],[-1,0]], np.float32)
    ex = cv2.filter2D(g, cv2.CV_32F, kx)
    ey = cv2.filter2D(g, cv2.CV_32F, ky)
    return standardize_output(np.hypot(ex, ey), target_size)

def run_scharr(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    sx = cv2.Scharr(g, cv2.CV_64F, 1, 0)
    sy = cv2.Scharr(g, cv2.CV_64F, 0, 1)
    return standardize_output(np.hypot(sx, sy), target_size)

def run_gradient_magnitude(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    schx = cv2.Scharr(g, cv2.CV_64F, 1, 0)
    schy = cv2.Scharr(g, cv2.CV_64F, 0, 1)
    mag = 0.6*np.hypot(sx, sy) + 0.4*np.hypot(schx, schy)
    return standardize_output(mag, target_size)

def run_multi_scale_canny(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    comb = np.zeros_like(g, np.float32)
    for blur, low, high, w in [
        ((3,3), 50, 150, 0.4),
        ((5,5), 30, 100, 0.4),
        ((7,7), 20,  80, 0.2),
    ]:
        edges = cv2.Canny(cv2.GaussianBlur(g, blur, 0), low, high)
        comb += w * edges.astype(np.float32)
    return standardize_output(comb, target_size)

def run_adaptive_canny(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    med = np.median(g)
    sigma = 0.33
    lower, upper = int(max(0, (1.0-sigma)*med)), int(min(255, (1.0+sigma)*med))
    edges = cv2.Canny(g, lower, upper)
    return standardize_output(edges, target_size)

def run_morphological_gradient(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, k1)*0.7 + \
           cv2.morphologyEx(g, cv2.MORPH_GRADIENT, k2)*0.3
    return standardize_output(grad, target_size)

def run_bdcn(path: str, target_size: tuple | None = None) -> np.ndarray:
    try:
        from bdcn_edge import BDCNEdgeDetector
        img = cv2.imread(path)
        edge = BDCNEdgeDetector().detect(img)
        return standardize_output(edge, target_size)
    except Exception:
        print("[fallback] BDCN nicht verfügbar → Canny Fallback")
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(cv2.GaussianBlur(g,(5,5),0), 50, 150)
        return standardize_output(edges, target_size)

def run_fixed_cnn(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    k = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
    cx = torch.nn.Conv2d(1,1,3,padding=1,bias=False); cx.weight.data = k[None,None]
    cy = torch.nn.Conv2d(1,1,3,padding=1,bias=False); cy.weight.data = k.T[None,None]
    with torch.no_grad():
        t = torch.tensor(g/255.0, dtype=torch.float32)[None,None]
        e = torch.sqrt(cx(t)**2 + cy(t)**2)[0,0].numpy()
    return standardize_output(e, target_size)

# ------------------------------------------------------
# Verfügbare Methoden
# ------------------------------------------------------
def get_all_methods():
    return [
        ("HED_OpenCV",          run_hed),
        ("HED_PyTorch",         run_pytorch_hed),
        ("StructuredForests",   run_structured),
        ("Kornia_Canny",        run_kornia_canny),
        ("Kornia_Sobel",        run_kornia_sobel),
        ("Laplacian",           run_laplacian),
        ("Prewitt",             run_prewitt),
        ("Roberts",             run_roberts),
        ("Scharr",              run_scharr),
        ("GradientMagnitude",   run_gradient_magnitude),
        ("MultiScaleCanny",     run_multi_scale_canny),
        ("AdaptiveCanny",       run_adaptive_canny),
        ("MorphologicalGradient", run_morphological_gradient),
        ("BDCN",                run_bdcn),
        ("FixedCNN",            run_fixed_cnn),
    ]

# ------------------------------------------------------
# CLI Helfer
# ------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--init-models", action="store_true")
    p.add_argument("--list-methods", action="store_true")
    args = p.parse_args()

    if args.init_models:
        init_models()
    elif args.list_methods:
        for i,(n,_) in enumerate(get_all_methods(),1):
            print(f"{i:02d}. {n}")
    else:
        print("Nutze --init-models oder --list-methods")
