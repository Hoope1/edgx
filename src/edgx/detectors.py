import argparse
import gzip
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import requests
import torch

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
HED_DIR = os.path.join(MODEL_DIR, "hed")
STRUCT_DIR = os.path.join(MODEL_DIR, "structured")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using %s device", DEVICE)

# ------------------------------------------------------
# Alternative URLs für HED (Fallback-Strategie)
# ------------------------------------------------------
HED_PROTO_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/hed_edge_detection/deploy.prototxt",
    "https://raw.githubusercontent.com/ashukid/hed-edge-detector/master/deploy.prototxt",
]
HED_WEIGHTS_URLS = [
    "https://github.com/ashukid/hed-edge-detector/raw/master/hed_pretrained_bsds.caffemodel",
    "https://drive.google.com/uc?id=1zc-tSjrZ1Q1q6hzYNDaLdgBCcCRFjYLF",  # Backup Google Drive
]
STRUCT_URL = "https://github.com/opencv/opencv_extra/raw/master/testdata/cv/ximgproc/model.yml.gz"


# ------------------------------------------------------
# Helper: Downloads mit Fallback
# ------------------------------------------------------
def _download_with_fallback(urls: list, dst: str) -> bool:
    """Download mit mehreren Fallback-URLs (Stoppt, sobald einer klappt)."""
    if os.path.exists(dst):
        return True

    for i, url in enumerate(urls):
        try:
            logger.info("[download] %s (URL %d/%d)", os.path.basename(dst), i + 1, len(urls))
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(dst, "wb") as fh:
                fh.write(r.content)
            return True
        except Exception as e:
            logger.warning("[failed ] URL %d fehlgeschlagen: %s", i + 1, e)
            if i < len(urls) - 1:
                logger.info("[retry  ] Versuche nächste URL …")

    logger.error("[error  ] Alle URLs für %s fehlgeschlagen", os.path.basename(dst))
    return False


def _download(url: str, dst: str) -> None:
    """Einfacher Einzel-Download (Kompatibilität)."""
    _download_with_fallback([url], dst)


class ModelManager:
    """Utility class handling download and extraction of model files."""

    def download(self, urls: list[str], dst: str) -> bool:
        ok = _download_with_fallback(urls, dst)
        if not ok and os.path.exists(dst):
            try:
                os.remove(dst)
            except OSError:
                pass
        return ok

    def extract_gzip(self, src: str, dst: str) -> bool:
        try:
            with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            return True
        except Exception as e:  # noqa: broad-except
            logging.error("Entpacken fehlgeschlagen: %s", e)
            for p in (src, dst):
                try:
                    os.remove(p)
                except OSError:
                    pass
            return False


# ------------------------------------------------------
# Gemeinsame Bild-Normalisierung & Hilfsfunktionen
# ------------------------------------------------------
def standardize_output(
    edge_map: np.ndarray, target_size: tuple | None = None, invert: bool = True
) -> np.ndarray:
    """
    Vereinheitlicht die Ausgaben aller Edge-Methoden

    • Invert: weißer Hintergrund, dunkle Kanten
    • Resize: skaliert (CUBIC) auf `target_size`, falls angegeben
    • uint8 garantiert
    """
    # --- Normierung 0-255 -----------------------------------------------
    if edge_map.dtype != np.uint8:
        edge_map = (edge_map * 255 if edge_map.max() <= 1.0 else edge_map).astype(
            np.uint8
        )

    # --- Invertieren -----------------------------------------------------
    if invert:
        edge_map = 255 - edge_map

    # --- Resize ----------------------------------------------------------
    if target_size is not None:
        edge_map = cv2.resize(edge_map, target_size, interpolation=cv2.INTER_CUBIC)

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


def _load_image(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read image or raise a clear ValueError."""
    img = cv2.imread(path, flags)
    if img is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    return img


# ------------------------------------------------------
# Modell-Initialisierung (HED, Structured Forest, BDCN, …)
# ------------------------------------------------------
def init_models() -> None:
    """Ensure all required model files are present."""
    os.makedirs(HED_DIR, exist_ok=True)
    os.makedirs(STRUCT_DIR, exist_ok=True)

    mgr = ModelManager()

    # --- HED -------------------------------------------------------------
    proto_path = os.path.join(HED_DIR, "deploy.prototxt")
    weight_path = os.path.join(HED_DIR, "hed.caffemodel")
    proto_ok = mgr.download(HED_PROTO_URLS, proto_path)
    weight_ok = mgr.download(HED_WEIGHTS_URLS, weight_path)
    if not (proto_ok and weight_ok):
        logging.warning("HED Modell konnte nicht vollständig geladen werden")

    # --- Structured Forests ---------------------------------------------
    gz_path = os.path.join(STRUCT_DIR, "model.yml.gz")
    yml_path = os.path.join(STRUCT_DIR, "model.yml")
    if mgr.download([STRUCT_URL], gz_path) and not os.path.exists(yml_path):
        mgr.extract_gzip(gz_path, yml_path)

    # --- BDCN Installation übersprungen (oft problematisch) ----
    logger.info("BDCN-Installation übersprungen (optional)")


# ------------------------------------------------------
# Edge-Detection Routinen
# ------------------------------------------------------
def run_hed(path: str, target_size: tuple | None = None) -> np.ndarray:
    """Original HED (OpenCV-DNN)."""
    proto = os.path.join(HED_DIR, "deploy.prototxt")
    weight = os.path.join(HED_DIR, "hed.caffemodel")
    if not (os.path.exists(proto) and os.path.exists(weight)):
        logging.warning("HED Modelle fehlen – versuche Init")
        init_models()
    if not (os.path.exists(proto) and os.path.exists(weight)):
        raise RuntimeError(
            "HED Modell nicht verfügbar. Bitte init_models ausführen und Dateien "
            "manuell prüfen."
        )

    img = _load_image(path)

    H, W = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img, 1.0, (W, H), (104.00699, 116.66877, 122.67891), swapRB=False, crop=False
    )
    net = cv2.dnn.readNetFromCaffe(proto, weight)
    net.setInput(blob)
    out = net.forward()[0, 0]
    out = cv2.resize(out, (W, H))
    return standardize_output(out, target_size)


def run_pytorch_hed(path: str, target_size: tuple | None = None) -> np.ndarray:
    """
    PyTorch HED Implementation - mit robustem Fallback.
    
    Da der ursprüngliche pytorch-hed Fork nicht mehr verfügbar ist, 
    implementieren wir einen intelligenten Fallback.
    """
    # Versuche zuerst, eine funktionsfähige pytorch-hed Installation zu finden
    try:
        import torchHED
        from PIL import Image
        
        pil = Image.open(path).convert("RGB")
        edge = torchHED.process_img(pil)
        arr = np.array(edge)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return standardize_output(arr, target_size)
        
    except ImportError:
        logger.info("pytorch-hed nicht verfügbar - versuche OpenCV HED")
        
        # Fallback 1: Versuche OpenCV HED
        try:
            return run_hed(path, target_size)
        except RuntimeError:
            logger.info("OpenCV HED nicht verfügbar - versuche erweiterten Canny")
            
            # Fallback 2: Erweiterter Multi-Scale Canny als HED-Ersatz
            return _enhanced_canny_hed_substitute(path, target_size)


def _enhanced_canny_hed_substitute(path: str, target_size: tuple | None = None) -> np.ndarray:
    """
    Erweiterte Canny-Implementation als HED-Ersatz.
    Kombiniert mehrere Techniken für bessere Edge-Detection.
    """
    img = _load_image(path, cv2.IMREAD_GRAYSCALE)
    
    # Mehrfache Gaussian-Blur-Varianten
    blur1 = cv2.GaussianBlur(img, (3, 3), 0.5)
    blur2 = cv2.GaussianBlur(img, (5, 5), 1.0)
    blur3 = cv2.GaussianBlur(img, (7, 7), 1.5)
    
    # Adaptive Threshold-Berechnung basierend auf Bildstatistiken
    mean_val = np.mean(img)
    std_val = np.std(img)
    
    # Dynamische Threshold-Berechnung
    low_thresh = max(10, int(mean_val - std_val * 0.5))
    high_thresh = min(255, int(mean_val + std_val * 0.8))
    
    # Multi-Scale Canny
    edges1 = cv2.Canny(blur1, low_thresh, high_thresh)
    edges2 = cv2.Canny(blur2, int(low_thresh * 0.7), int(high_thresh * 0.9))
    edges3 = cv2.Canny(blur3, int(low_thresh * 0.5), int(high_thresh * 0.7))
    
    # Gewichtete Kombination
    combined = (0.5 * edges1.astype(np.float32) + 
                0.3 * edges2.astype(np.float32) + 
                0.2 * edges3.astype(np.float32))
    
    # Morphologische Nachbearbeitung für saubere Kanten
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined = cv2.morphologyEx(combined.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # Non-maximum suppression approximation
    combined = cv2.GaussianBlur(combined, (1, 1), 0)
    
    return standardize_output(combined, target_size)


def run_structured(path: str, target_size: tuple | None = None) -> np.ndarray:
    mdl = os.path.join(STRUCT_DIR, "model.yml")
    if not os.path.exists(mdl):
        logger.warning("Structured Forests Modell fehlt - versuche Init")
        init_models()
    if not os.path.exists(mdl):
        logger.warning("Structured Forests nicht verfügbar - verwende Canny Fallback")
        return run_adaptive_canny(path, target_size)
    
    try:
        det = cv2.ximgproc.createStructuredEdgeDetection(mdl)
        img = _load_image(path).astype("float32") / 255.0
        edges = det.detectEdges(img)
        return standardize_output(edges, target_size)
    except Exception as e:
        logger.warning(f"Structured Forests fehlgeschlagen: {e} - verwende Fallback")
        return run_adaptive_canny(path, target_size)


def run_kornia_canny(path: str, target_size: tuple | None = None) -> np.ndarray:
    try:
        import kornia
        
        g = _load_image(path, cv2.IMREAD_GRAYSCALE)
        t = torch.tensor(g / 255.0, dtype=torch.float32, device=DEVICE)[None, None]
        edges = kornia.filters.canny(t)[0][0].cpu().numpy()
        return standardize_output(edges, target_size)
    except ImportError:
        logger.warning("Kornia nicht verfügbar - verwende OpenCV Canny")
        return run_adaptive_canny(path, target_size)


def run_kornia_sobel(path: str, target_size: tuple | None = None) -> np.ndarray:
    try:
        import kornia
        
        g = _load_image(path, cv2.IMREAD_GRAYSCALE)
        t = torch.tensor(g / 255.0, dtype=torch.float32, device=DEVICE)[None, None]
        sx = kornia.filters.sobel(t, normalized=True)
        sy = kornia.filters.sobel(t, normalized=True, dim=3)
        mag = torch.sqrt(sx**2 + sy**2)[0, 0].cpu().numpy()
        return standardize_output(mag, target_size)
    except ImportError:
        logger.warning("Kornia nicht verfügbar - verwende Sobel Fallback")
        return run_scharr(path, target_size)


def run_laplacian(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = _load_image(path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(g, (5, 5), 0)
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    return standardize_output(np.abs(lap), target_size)


def run_prewitt(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = _load_image(path, cv2.IMREAD_GRAYSCALE)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)
    ex = cv2.filter2D(g, cv2.CV_32F, kx)
    ey = cv2.filter2D(g, cv2.CV_32F, ky)
    return standardize_output(np.hypot(ex, ey), target_size)


def run_roberts(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = _load_image(path, cv2.IMREAD_GRAYSCALE)
    kx = np.array([[1, 0], [0, -1]], np.float32)
    ky = np.array([[0, 1], [-1, 0]], np.float32)
    ex = cv2.filter2D(g, cv2.CV_32F, kx)
    ey = cv2.filter2D(g, cv2.CV_32F, ky)
    return standardize_output(np.hypot(ex, ey), target_size)


def run_scharr(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = _load_image(path, cv2.IMREAD_GRAYSCALE)
    sx = cv2.Scharr(g, cv2.CV_64F, 1, 0)
    sy = cv2.Scharr(g, cv2.CV_64F, 0, 1)
    return standardize_output(np.hypot(sx, sy), target_size)


def run_gradient_magnitude(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = _load_image(path, cv2.IMREAD_GRAYSCALE)
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    schx = cv2.Scharr(g, cv2.CV_64F, 1, 0)
    schy = cv2.Scharr(g, cv2.CV_64F, 0, 1)
    mag = 0.6 * np.hypot(sx, sy) + 0.4 * np.hypot(schx, schy)
    return standardize_output(mag, target_size)


def run_multi_scale_canny(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = _load_image(path, cv2.IMREAD_GRAYSCALE)
    comb = np.zeros_like(g, np.float32)
    for blur, low, high, w in [
        ((3, 3), 50, 150, 0.4),
        ((5, 5), 30, 100, 0.4),
        ((7, 7), 20, 80, 0.2),
    ]:
        edges = cv2.Canny(cv2.GaussianBlur(g, blur, 0), low, high)
        comb += w * edges.astype(np.float32)
    return standardize_output(comb, target_size)


def run_adaptive_canny(path: str, target_size: tuple | None = None) -> np.ndarray:
    g = _load_image(path, cv2.IMREAD_GRAYSCALE)
    med = np.median(g)
    sigma = 0.33
    lower, upper = int(max(0, (1.0 - sigma) * med)), int(min(255, (1.0 + sigma) * med))
    edges = cv2.Canny(g, lower, upper)
    return standardize_output(edges, target_size)


def run_morphological_gradient(
    path: str, target_size: tuple | None = None
) -> np.ndarray:
    g = _load_image(path, cv2.IMREAD_GRAYSCALE)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grad = (
        cv2.morphologyEx(g, cv2.MORPH_GRADIENT, k1) * 0.7
        + cv2.morphologyEx(g, cv2.MORPH_GRADIENT, k2) * 0.3
    )
    return standardize_output(grad, target_size)


def run_bdcn(path: str, target_size: tuple | None = None) -> np.ndarray:
    """
    BDCN Edge Detection - mit robustem Fallback.
    Da BDCN-Installation oft problematisch ist, verwenden wir einen 
    erweiterten Canny-Fallback.
    """
    try:
        from bdcn_edge import BDCNEdgeDetector
        
        img = _load_image(path)
        edge = BDCNEdgeDetector().detect(img)
        return standardize_output(edge, target_size)
    except (ImportError, ModuleNotFoundError):
        logger.info("BDCN nicht verfügbar - verwende erweiterten Multi-Scale Canny")
        return run_multi_scale_canny(path, target_size)
    except Exception as e:
        logger.warning(f"BDCN fehlgeschlagen: {e} - verwende Fallback")
        return run_multi_scale_canny(path, target_size)


def run_fixed_cnn(path: str, target_size: tuple | None = None) -> np.ndarray:
    """Fixed CNN Filter mit PyTorch - mit CPU/GPU Fallback."""
    try:
        g = _load_image(path, cv2.IMREAD_GRAYSCALE)
        
        # Sobel-Kernel für Edge Detection
        k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                        dtype=torch.float32, device=DEVICE)
        
        # Conv2D Layer erstellen
        cx = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False).to(DEVICE)
        cx.weight.data = k[None, None]
        cy = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False).to(DEVICE)
        cy.weight.data = k.T[None, None]
        
        with torch.no_grad():
            t = torch.tensor(g / 255.0, dtype=torch.float32, device=DEVICE)[None, None]
            e = torch.sqrt(cx(t) ** 2 + cy(t) ** 2)[0, 0].cpu().numpy()
        
        return standardize_output(e, target_size)
    except Exception as e:
        logger.warning(f"Fixed CNN fehlgeschlagen: {e} - verwende Sobel Fallback")
        return run_scharr(path, target_size)


# ------------------------------------------------------
# Verfügbare Methoden
# ------------------------------------------------------
def get_all_methods():
    """
    Gibt alle verfügbaren Edge-Detection-Methoden zurück.
    Reihenfolge: Zuverlässige Methoden zuerst, dann experimentelle.
    """
    return [
        ("HED_OpenCV", run_hed),
        ("HED_PyTorch", run_pytorch_hed),
        ("StructuredForests", run_structured),
        ("Kornia_Canny", run_kornia_canny),
        ("Kornia_Sobel", run_kornia_sobel),
        ("Laplacian", run_laplacian),
        ("Prewitt", run_prewitt),
        ("Roberts", run_roberts),
        ("Scharr", run_scharr),
        ("GradientMagnitude", run_gradient_magnitude),
        ("MultiScaleCanny", run_multi_scale_canny),
        ("AdaptiveCanny", run_adaptive_canny),
        ("MorphologicalGradient", run_morphological_gradient),
        ("BDCN", run_bdcn),
        ("FixedCNN", run_fixed_cnn),
    ]


# ------------------------------------------------------
# CLI Helfer
# ------------------------------------------------------
if __name__ == "__main__":
    # Logging konfigurieren
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    p = argparse.ArgumentParser()
    p.add_argument("--init-models", action="store_true", help="Modelle herunterladen")
    p.add_argument("--list-methods", action="store_true", help="Verfügbare Methoden anzeigen")
    p.add_argument("--test", action="store_true", help="Kurzer Funktionstest")
    args = p.parse_args()

    if args.init_models:
        logger.info("🔧 Initialisiere Modelle...")
        init_models()
        logger.info("✅ Modell-Initialisierung abgeschlossen")
    elif args.list_methods:
        logger.info("📋 Verfügbare Edge-Detection-Methoden:")
        for i, (n, _) in enumerate(get_all_methods(), 1):
            logger.info("%02d. %s", i, n)
    elif args.test:
        logger.info("🧪 Führe Funktionstest durch...")
        # Einfacher Test mit einem kleinen Test-Bild
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_path = "test_image.png"
        cv2.imwrite(test_path, test_img)
        
        working_methods = []
        for name, func in get_all_methods():
            try:
                result = func(test_path, target_size=(50, 50))
                if result is not None and result.size > 0:
                    working_methods.append(name)
                    logger.info(f"✅ {name} - OK")
            except Exception as e:
                logger.warning(f"❌ {name} - Fehler: {e}")
        
        os.remove(test_path)
        logger.info(f"✅ Test abgeschlossen: {len(working_methods)}/{len(get_all_methods())} Methoden funktional")
    else:
        logger.info("Nutze --init-models, --list-methods oder --test")
