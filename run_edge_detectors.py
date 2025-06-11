#!/usr/bin/env python3
"""
Erweiterte Edge-Detection-Pipeline

• Skalierung aller Ausgaben auf die höchste Eingabe-Auflösung  
• Invertierte Ergebnisse (weißer BG, dunkle Kanten)  
• Unterstützung von 15 verschiedenen Algorithmen  
• Ergebnis-Zusammenfassung im Zielordner
"""
from __future__ import annotations
import argparse, os, glob, cv2, time
from pathlib import Path
from typing import List, Tuple

from detectors import (
    get_all_methods,
    get_max_resolution,
    standardize_output  # für eventuelles Upscaling in Fallbacks
)

# ---------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------
def create_output_structure(output_dir: str) -> str:
    """legt …/edge_detection_results an"""
    main = os.path.join(output_dir, "edge_detection_results")
    os.makedirs(main, exist_ok=True)
    return main


def get_image_files(input_dir: str) -> List[str]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    imgs: list[str] = []
    for e in exts:
        imgs += glob.glob(os.path.join(input_dir, e))
        imgs += glob.glob(os.path.join(input_dir, e.upper()))
    return sorted(list(set(imgs)))


def create_summary_file(
    output_dir: str,
    image_files: list[str],
    methods: list[Tuple[str, callable]],
    resolution: Tuple[int, int]
) -> None:
    p = os.path.join(output_dir, "processing_summary.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("EDGE DETECTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Ziel-Auflösung: {resolution[0]}×{resolution[1]}\n")
        f.write(f"Eingabebilder:  {len(image_files)}\n")
        f.write(f"Methoden:       {len(methods)}\n")
        f.write(f"Ausgaben:       {len(image_files)*len(methods)}\n\n")

        f.write("BILDER:\n")
        for im in image_files:
            g = cv2.imread(im)
            if g is None: continue
            h,w = g.shape[:2]
            f.write(f"• {Path(im).name}  ({w}×{h})\n")

        f.write("\nMETHODEN:\n")
        for i,(n,_) in enumerate(methods,1):
            f.write(f"{i:02d}. {n}\n")

        f.write("\nFormat: PNG  • invertiert  • einheitliche Auflösung\n")
    print(f"[info] Summary: {p}")


# ---------------------------------------------------------------------
# Verarbeitung
# ---------------------------------------------------------------------
def process_images(
    input_dir: str,
    output_dir: str,
    selected_methods: list[str] | None = None
) -> None:
    out_root = create_output_structure(output_dir)
    imgs = get_image_files(input_dir)
    if not imgs:
        print(f"[error] Keine Bilder in {input_dir}")
        return

    max_res = get_max_resolution(imgs)
    print(f"[max  ] Ziel-Auflösung: {max_res[0]}×{max_res[1]}")

    all_methods = get_all_methods()
    if selected_methods:
        methods = [m for m in all_methods if m[0] in selected_methods]
    else:
        methods = all_methods

    if not methods:
        print("[error] Keine gültigen Methoden ausgewählt")
        return

    tot_ops = len(imgs) * len(methods)
    op = 0
    start = time.time()

    for img_path in imgs:
        name = Path(img_path).stem
        print(f"\n[img ] {Path(img_path).name}")
        for mname, mfun in methods:
            op += 1
            prog = op / tot_ops * 100
            try:
                print(f"  {prog:5.1f}% → {mname} … ", end="", flush=True)
                res = mfun(img_path, target_size=max_res)
                # Sollte eine Methode *nicht* über standardize_output skalieren,
                # sichern wir hier die Größengleichheit ab:
                if res.shape[:2][::-1] != max_res:
                    res = cv2.resize(res, max_res, cv2.INTER_CUBIC)
                    res = standardize_output(res, max_res)  # invert / uint8
                out_name = f"{name}_{mname}.png"
                cv2.imwrite(os.path.join(out_root, out_name), res)
                print("✓")
            except Exception as e:
                print(f"✗ {e}")

    dur = time.time() - start
    print(f"\n[done] {tot_ops} Operationen  • {dur:.1f}s gesamt")

    create_summary_file(out_root, imgs, methods, max_res)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def list_available():
    print("\nVERFÜGBARE METHODEN")
    for i,(n,_) in enumerate(get_all_methods(),1):
        print(f"{i:02d}. {n}")

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Edge-Detection-Batch-Tool (einheitliche Auflösung, invertiert)"
    )
    ap.add_argument("--input_dir",  required=False)
    ap.add_argument("--output_dir", required=False)
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--list-methods", action="store_true")
    args = ap.parse_args()

    if args.list_methods:
        list_available()
        return

    if not (args.input_dir and args.output_dir):
        ap.print_help(); return

    process_images(args.input_dir, args.output_dir, args.methods)

if __name__ == "__main__":
    main()
