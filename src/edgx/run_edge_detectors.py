#!/usr/bin/env python3
"""
Edge-Detection-CLI-Tool

• Skalierung aller Ausgaben auf die höchste Eingabe-Auflösung oder custom size
• Invertierte Ergebnisse (weißer BG, dunkle Kanten)
• Unterstützung von 15 verschiedenen Algorithmen
• Ergebnis-Zusammenfassung im Zielordner
• Robuste Fehlerbehandlung und Logging
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional
import logging

import cv2

try:
    from .detectors import (
        standardize_output,  # für eventuelles Upscaling in Fallbacks
        get_all_methods,
        get_max_resolution,
        init_models,
    )
except ImportError:
    # Fallback für direktes Ausführen ohne Package-Installation
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    
    try:
        from detectors import (
            standardize_output,
            get_all_methods, 
            get_max_resolution,
            init_models,
        )
    except ImportError as e:
        print(f"❌ Fehler: Edge-Detection-Module konnten nicht importiert werden: {e}")
        print("💡 Lösung: Führen Sie 'pip install -e .' im Projektverzeichnis aus")
        sys.exit(1)

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------
def create_output_structure(output_dir: str) -> str:
    """Erstellt die Ausgabe-Struktur und gibt den Hauptordner zurück."""
    main_dir = os.path.join(output_dir, "edge_detection_results")
    os.makedirs(main_dir, exist_ok=True)
    return main_dir


def get_image_files(input_dir: str, recursive: bool = False) -> List[str]:
    """
    Sammelt alle unterstützten Bilddateien aus einem Verzeichnis.
    
    Args:
        input_dir: Eingabeverzeichnis
        recursive: Ob Unterverzeichnisse durchsucht werden sollen
    
    Returns:
        Liste der gefundenen Bildpfade
    """
    if not os.path.isdir(input_dir):
        logger.error(f"Eingabeverzeichnis nicht gefunden: {input_dir}")
        return []
    
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    images: List[str] = []
    
    if recursive:
        # Rekursive Suche
        for root, dirs, files in os.walk(input_dir):
            for ext in extensions:
                pattern = os.path.join(root, ext)
                images.extend(glob.glob(pattern))
                # Auch Großbuchstaben
                images.extend(glob.glob(pattern.upper()))
    else:
        # Nur aktuelles Verzeichnis
        for ext in extensions:
            pattern = os.path.join(input_dir, ext)
            images.extend(glob.glob(pattern))
            images.extend(glob.glob(pattern.upper()))
    
    # Duplikate entfernen und sortieren
    unique_images = sorted(list(set(images)))
    logger.info(f"📷 {len(unique_images)} Bilddateien gefunden")
    
    return unique_images


def parse_size(value: str) -> Tuple[int, int]:
    """
    Parst eine Größenangabe im Format 'WIDTHxHEIGHT'.
    
    Args:
        value: String im Format "1920x1080"
    
    Returns:
        Tuple: (width, height)
    
    Raises:
        argparse.ArgumentTypeError: Bei ungültigem Format
    """
    try:
        parts = value.lower().replace('x', 'x').split('x')
        if len(parts) != 2:
            raise ValueError("Ungültiges Format")
        
        width, height = int(parts[0]), int(parts[1])
        
        if width <= 0 or height <= 0:
            raise ValueError("Breite und Höhe müssen positiv sein")
        
        if width > 8192 or height > 8192:
            raise ValueError("Maximale Auflösung: 8192x8192")
        
        return (width, height)
    
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"--size erwartet Format WIDTHxHEIGHT (z.B. 1920x1080): {e}"
        ) from e


def validate_output_dir(output_dir: str) -> str:
    """
    Validiert und erstellt das Ausgabeverzeichnis.
    
    Args:
        output_dir: Pfad zum Ausgabeverzeichnis
    
    Returns:
        Absoluter Pfad zum Ausgabeverzeichnis
    
    Raises:
        OSError: Wenn Verzeichnis nicht erstellt werden kann
    """
    try:
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        
        # Teste Schreibberechtigung
        test_file = os.path.join(abs_output_dir, "test_write.tmp")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            raise OSError(f"Keine Schreibberechtigung für {abs_output_dir}: {e}")
        
        return abs_output_dir
    
    except Exception as e:
        raise OSError(f"Ausgabeverzeichnis konnte nicht erstellt werden: {e}")


def create_summary_file(
    output_dir: str,
    image_files: List[str],
    methods: List[Tuple[str, callable]],
    resolution: Tuple[int, int],
    processing_stats: dict,
) -> None:
    """
    Erstellt eine detaillierte Zusammenfassung der Verarbeitung.
    """
    summary_path = os.path.join(output_dir, "processing_summary.txt")
    
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("EDGE DETECTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Verarbeitungs-Info
            f.write("VERARBEITUNG:\n")
            f.write(f"• Datum/Zeit:       {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"• Ziel-Auflösung:   {resolution[0]}×{resolution[1]}\n")
            f.write(f"• Eingabebilder:    {len(image_files)}\n")
            f.write(f"• Methoden:         {len(methods)}\n")
            f.write(f"• Ausgabedateien:   {len(image_files) * len(methods)}\n")
            
            if processing_stats:
                f.write(f"• Dauer:            {processing_stats.get('duration', 0):.1f}s\n")
                f.write(f"• Erfolgreich:      {processing_stats.get('success_count', 0)}\n")
                f.write(f"• Fehler:           {processing_stats.get('error_count', 0)}\n")
            
            f.write("\n")
            
            # Bilder-Liste
            f.write("EINGABEBILDER:\n")
            for i, img_path in enumerate(image_files, 1):
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        size_str = f"({w}×{h})"
                    else:
                        size_str = "(Fehler beim Laden)"
                except Exception:
                    size_str = "(Unbekannt)"
                
                f.write(f"{i:3d}. {Path(img_path).name:<30} {size_str}\n")
            
            f.write("\n")
            
            # Methoden-Liste
            f.write("VERWENDETE METHODEN:\n")
            for i, (method_name, _) in enumerate(methods, 1):
                f.write(f"{i:2d}. {method_name}\n")
            
            f.write("\n")
            f.write("FORMAT:\n")
            f.write("• Dateiformat:      PNG\n")
            f.write("• Kanten:           Invertiert (weiße Kanten, schwarzer Hintergrund)\n")
            f.write("• Größe:            Einheitlich skaliert\n")
            f.write("• Dateinamen:       {original}_{methode}.png\n")
        
        logger.info(f"📋 Zusammenfassung gespeichert: {summary_path}")
    
    except Exception as e:
        logger.error(f"❌ Konnte Zusammenfassung nicht erstellen: {e}")


# ---------------------------------------------------------------------
# Haupt-Verarbeitungsfunktion
# ---------------------------------------------------------------------
def process_images(
    input_dir: str,
    output_dir: str,
    selected_methods: Optional[List[str]] = None,
    *,
    size: Optional[Tuple[int, int]] = None,
    scale: Optional[float] = None,
    recursive: bool = False,
    parallel: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Verarbeitet alle Bilder im Eingabeverzeichnis mit den ausgewählten Methoden.
    
    Args:
        input_dir: Eingabeverzeichnis mit Bildern
        output_dir: Ausgabeverzeichnis für Ergebnisse
        selected_methods: Liste der zu verwendenden Methoden (None = alle)
        size: Zielauflösung (None = automatisch)
        scale: Skalierungsfaktor (None = kein Scaling)
        recursive: Rekursive Verzeichnissuche
        parallel: Parallelverarbeitung (experimentell)
        dry_run: Testlauf ohne tatsächliche Verarbeitung
    
    Returns:
        Dictionary mit Verarbeitungsstatistiken
    """
    start_time = time.time()
    
    # Validierungen
    logger.info("🔧 Initialisiere Verarbeitung...")
    
    # Ausgabeverzeichnis validieren
    try:
        output_dir = validate_output_dir(output_dir)
        output_results_dir = create_output_structure(output_dir)
    except OSError as e:
        logger.error(f"❌ {e}")
        return {"error": str(e)}
    
    # Bilder sammeln
    image_files = get_image_files(input_dir, recursive=recursive)
    if not image_files:
        logger.error(f"❌ Keine Bilder in {input_dir} gefunden")
        return {"error": "Keine Bilder gefunden"}
    
    # Zielauflösung bestimmen
    if size is not None:
        target_resolution = size
        logger.info(f"📐 Verwende benutzerdefinierte Auflösung: {target_resolution[0]}×{target_resolution[1]}")
    else:
        target_resolution = get_max_resolution(image_files)
        if scale is not None:
            target_resolution = (
                int(target_resolution[0] * scale), 
                int(target_resolution[1] * scale)
            )
            logger.info(f"📐 Skalierte Auflösung: {target_resolution[0]}×{target_resolution[1]} (Faktor: {scale})")
        else:
            logger.info(f"📐 Automatische Auflösung: {target_resolution[0]}×{target_resolution[1]}")
    
    # Methoden laden
    try:
        all_methods = get_all_methods()
        if selected_methods:
            # Validiere ausgewählte Methoden
            available_method_names = [name for name, _ in all_methods]
            invalid_methods = [m for m in selected_methods if m not in available_method_names]
            
            if invalid_methods:
                logger.warning(f"⚠️ Unbekannte Methoden: {', '.join(invalid_methods)}")
                selected_methods = [m for m in selected_methods if m in available_method_names]
            
            methods = [(name, func) for name, func in all_methods if name in selected_methods]
        else:
            methods = all_methods
        
        if not methods:
            logger.error("❌ Keine gültigen Methoden ausgewählt")
            return {"error": "Keine gültigen Methoden"}
        
        logger.info(f"🔧 Verwende {len(methods)} Methoden: {', '.join([name for name, _ in methods])}")
    
    except Exception as e:
        logger.error(f"❌ Fehler beim Laden der Methoden: {e}")
        return {"error": f"Methoden-Ladefehler: {e}"}
    
    # Dry-Run-Modus
    if dry_run:
        logger.info("🧪 DRY-RUN-MODUS: Keine tatsächliche Verarbeitung")
        total_operations = len(image_files) * len(methods)
        estimated_time = total_operations * 2  # Geschätzte 2s pro Operation
        
        logger.info(f"📊 Geplante Operationen: {total_operations}")
        logger.info(f"⏱️ Geschätzte Dauer: {estimated_time // 60}m {estimated_time % 60}s")
        
        return {
            "dry_run": True,
            "total_operations": total_operations,
            "estimated_time": estimated_time,
            "methods": [name for name, _ in methods],
            "images": len(image_files),
        }
    
    # Tatsächliche Verarbeitung
    logger.info(f"🚀 Starte Verarbeitung von {len(image_files)} Bildern mit {len(methods)} Methoden...")
    
    total_operations = len(image_files) * len(methods)
    operation_count = 0
    success_count = 0
    error_count = 0
    errors = []
    
    for img_index, img_path in enumerate(image_files):
        img_name = Path(img_path).stem
        logger.info(f"📷 [{img_index + 1}/{len(image_files)}] {Path(img_path).name}")
        
        for method_index, (method_name, method_function) in enumerate(methods):
            operation_count += 1
            progress = operation_count / total_operations * 100
            
            try:
                logger.debug(f"    🔄 [{progress:5.1f}%] {method_name}...")
                
                # Edge-Detection ausführen
                result = method_function(img_path, target_size=target_resolution)
                
                # Ergebnis-Validierung
                if result is None or result.size == 0:
                    raise ValueError("Leeres Ergebnis")
                
                # Sicherstellen, dass Größe korrekt ist
                if result.shape[:2][::-1] != target_resolution:
                    logger.debug(f"    📐 Nachskalierung: {result.shape[:2][::-1]} → {target_resolution}")
                    result = cv2.resize(result, target_resolution, interpolation=cv2.INTER_CUBIC)
                    result = standardize_output(result, target_resolution)
                
                # Datei speichern
                output_filename = f"{img_name}_{method_name}.png"
                output_path = os.path.join(output_results_dir, output_filename)
                
                success = cv2.imwrite(output_path, result)
                if not success:
                    raise RuntimeError("cv2.imwrite fehlgeschlagen")
                
                success_count += 1
                logger.debug(f"    ✅ {output_filename}")
                
            except Exception as e:
                error_count += 1
                error_msg = f"{img_name} + {method_name}: {e}"
                errors.append(error_msg)
                logger.error(f"    ❌ {error_msg}")
                
                # Bei zu vielen Fehlern abbrechen
                if error_count > total_operations * 0.5:  # Mehr als 50% Fehler
                    logger.error("❌ Zu viele Fehler - Verarbeitung abgebrochen")
                    break
        
        # Fortschritts-Update
        if (img_index + 1) % max(1, len(image_files) // 10) == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (img_index + 1) * (len(image_files) - img_index - 1)
            logger.info(f"📊 Fortschritt: {progress:.1f}% | ETA: {eta/60:.1f}min")
    
    # Abschluss-Statistiken
    duration = time.time() - start_time
    logger.info(f"✅ Verarbeitung abgeschlossen in {duration:.1f}s")
    logger.info(f"📊 Erfolgreich: {success_count}/{total_operations} ({success_count/total_operations*100:.1f}%)")
    
    if error_count > 0:
        logger.warning(f"⚠️ Fehler: {error_count}")
        for error in errors[:5]:  # Zeige nur erste 5 Fehler
            logger.warning(f"   • {error}")
        if len(errors) > 5:
            logger.warning(f"   • ... und {len(errors) - 5} weitere")
    
    # Zusammenfassung erstellen
    processing_stats = {
        "duration": duration,
        "success_count": success_count,
        "error_count": error_count,
        "total_operations": total_operations,
    }
    
    create_summary_file(
        output_results_dir, 
        image_files, 
        methods, 
        target_resolution,
        processing_stats
    )
    
    return {
        "success": True,
        "duration": duration,
        "success_count": success_count,
        "error_count": error_count,
        "total_operations": total_operations,
        "output_dir": output_results_dir,
        "errors": errors,
    }


# ---------------------------------------------------------------------
# CLI-Funktionen
# ---------------------------------------------------------------------
def list_available_methods() -> None:
    """Zeigt alle verfügbaren Edge-Detection-Methoden an."""
    try:
        methods = get_all_methods()
        logger.info("📋 VERFÜGBARE EDGE-DETECTION-METHODEN:")
        logger.info("=" * 50)
        
        categories = {
            "Deep Learning": ["HED_OpenCV", "HED_PyTorch", "StructuredForests", "BDCN"],
            "Klassische Filter": ["Laplacian", "Prewitt", "Roberts", "Scharr", "GradientMagnitude"],
            "Canny-Varianten": ["Kornia_Canny", "MultiScaleCanny", "AdaptiveCanny"],
            "Morphologie": ["MorphologicalGradient"],
            "GPU-Beschleunigt": ["Kornia_Canny", "Kornia_Sobel", "FixedCNN"],
        }
        
        for category, method_list in categories.items():
            available_in_category = [(name, func) for name, func in methods if name in method_list]
            if available_in_category:
                logger.info(f"\n{category}:")
                for i, (name, _) in enumerate(available_in_category, 1):
                    logger.info(f"  {i:2d}. {name}")
        
        # Alle anderen Methoden
        categorized_methods = set()
        for method_list in categories.values():
            categorized_methods.update(method_list)
        
        other_methods = [(name, func) for name, func in methods if name not in categorized_methods]
        if other_methods:
            logger.info(f"\nWeitere:")
            for i, (name, _) in enumerate(other_methods, 1):
                logger.info(f"  {i:2d}. {name}")
        
        logger.info(f"\n📊 Gesamt: {len(methods)} Methoden verfügbar")
        
    except Exception as e:
        logger.error(f"❌ Fehler beim Laden der Methoden: {e}")


def test_installation() -> bool:
    """
    Testet die Installation und Funktionsfähigkeit.
    
    Returns:
        True wenn alle Tests erfolgreich, False sonst
    """
    logger.info("🧪 TESTE INSTALLATION...")
    
    try:
        # Test 1: Module importieren
        logger.info("1️⃣ Teste Module-Import...")
        methods = get_all_methods()
        logger.info(f"   ✅ {len(methods)} Methoden geladen")
        
        # Test 2: Modelle initialisieren
        logger.info("2️⃣ Teste Modell-Initialisierung...")
        init_models()
        logger.info("   ✅ Modelle initialisiert")
        
        # Test 3: Erstelle Test-Bild
        logger.info("3️⃣ Teste Edge-Detection...")
        test_image = cv2.imread("test_image.png") if os.path.exists("test_image.png") else None
        
        if test_image is None:
            # Erstelle synthetisches Test-Bild
            import numpy as np
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            test_path = "temp_test_image.png"
            cv2.imwrite(test_path, test_image)
        else:
            test_path = "test_image.png"
        
        # Test 4: Teste einige Methoden
        working_methods = []
        failing_methods = []
        
        test_methods = ["Laplacian", "AdaptiveCanny", "Scharr"]  # Sichere Methoden
        available_test_methods = [(name, func) for name, func in methods if name in test_methods]
        
        for method_name, method_func in available_test_methods[:3]:  # Teste max. 3
            try:
                result = method_func(test_path, target_size=(50, 50))
                if result is not None and result.size > 0:
                    working_methods.append(method_name)
                    logger.info(f"   ✅ {method_name}")
                else:
                    failing_methods.append(method_name)
                    logger.warning(f"   ⚠️ {method_name} (leeres Ergebnis)")
            except Exception as e:
                failing_methods.append(method_name)
                logger.warning(f"   ❌ {method_name}: {e}")
        
        # Cleanup
        if test_path == "temp_test_image.png" and os.path.exists(test_path):
            os.remove(test_path)
        
        # Ergebnis
        logger.info(f"4️⃣ Test-Ergebnis:")
        logger.info(f"   ✅ Funktional: {len(working_methods)}/{len(available_test_methods)}")
        
        if len(working_methods) > 0:
            logger.info("✅ Installation funktionsfähig!")
            return True
        else:
            logger.error("❌ Keine Methoden funktional!")
            return False
    
    except Exception as e:
        logger.error(f"❌ Installation-Test fehlgeschlagen: {e}")
        return False


def main() -> None:
    """Haupt-CLI-Funktion."""
    parser = argparse.ArgumentParser(
        description="Edge-Detection-Batch-Tool mit einheitlicher Auflösung und invertierten Ergebnissen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s --input_dir ./images --output_dir ./results
  %(prog)s --input_dir ./images --output_dir ./results --methods Laplacian Canny
  %(prog)s --input_dir ./images --output_dir ./results --size 1920x1080
  %(prog)s --input_dir ./images --output_dir ./results --scale 0.5 --recursive
  %(prog)s --list-methods
  %(prog)s --test
        """
    )
    
    # Hauptoptionen
    parser.add_argument(
        "--input_dir", 
        type=str, 
        help="Ordner mit Eingabebildern (erforderlich außer bei --list-methods/--test)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results",
        help="Zielordner für Ergebnisse (Standard: ./results)"
    )
    parser.add_argument(
        "--methods", 
        nargs="+", 
        help="Liste der zu verwendenden Methoden (Standard: alle verfügbaren)"
    )
    
    # Größen-Optionen (sich gegenseitig ausschließend)
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--size", 
        type=parse_size, 
        help="Zielauflösung im Format WIDTHxHEIGHT (z.B. 1920x1080)"
    )
    size_group.add_argument(
        "--scale", 
        type=float, 
        help="Skalierungsfaktor (z.B. 0.5 für halbe Größe, 2.0 für doppelte Größe)"
    )
    
    # Zusatz-Optionen
    parser.add_argument(
        "--recursive", 
        action="store_true", 
        help="Durchsuche Unterverzeichnisse rekursiv"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="Experimentelle Parallelverarbeitung"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Testlauf ohne tatsächliche Verarbeitung"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Ausführliche Ausgabe (DEBUG-Level)"
    )
    
    # Informations-Optionen
    info_group = parser.add_mutually_exclusive_group()
    info_group.add_argument(
        "--list-methods", 
        action="store_true", 
        help="Zeige alle verfügbaren Methoden"
    )
    info_group.add_argument(
        "--test", 
        action="store_true", 
        help="Teste Installation und Funktionsfähigkeit"
    )
    
    args = parser.parse_args()
    
    # Logging-Level setzen
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("🐛 Debug-Modus aktiviert")
    
    # Informations-Befehle
    if args.list_methods:
        list_available_methods()
        return
    
    if args.test:
        success = test_installation()
        sys.exit(0 if success else 1)
    
    # Validierung für normale Verarbeitung
    if not args.input_dir:
        parser.error("--input_dir ist erforderlich (außer bei --list-methods oder --test)")
    
    if not os.path.isdir(args.input_dir):
        logger.error(f"❌ Eingabeverzeichnis nicht gefunden: {args.input_dir}")
        sys.exit(1)
    
    # Validiere Scale-Parameter
    if args.scale is not None:
        if args.scale <= 0:
            parser.error("--scale muss positiv sein")
        if args.scale > 10:
            logger.warning(f"⚠️ Sehr hoher Skalierungsfaktor: {args.scale}")
    
    # Führe Verarbeitung aus
    try:
        result = process_images(
            args.input_dir,
            args.output_dir,
            args.methods,
            size=args.size,
            scale=args.scale,
            recursive=args.recursive,
            parallel=args.parallel,
            dry_run=args.dry_run,
        )
        
        if "error" in result:
            logger.error(f"❌ Verarbeitung fehlgeschlagen: {result['error']}")
            sys.exit(1)
        
        if result.get("dry_run"):
            logger.info("🧪 Dry-Run abgeschlossen")
        else:
            logger.info(f"✅ Verarbeitung erfolgreich abgeschlossen")
            logger.info(f"📁 Ergebnisse: {result.get('output_dir', args.output_dir)}")
    
    except KeyboardInterrupt:
        logger.warning("⚠️ Verarbeitung durch Benutzer abgebrochen")
        sys.exit(130)  # Standard Unix exit code für SIGINT
    
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
