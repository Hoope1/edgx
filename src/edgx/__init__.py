"""
Edge Detection Studio (edgx)
============================

Eine Zero-Config-Suite zur Kantenerkennung mit 15 klassischen und Deep-Learning-Algorithmen.

Hauptmodule:
    detectors: Kern-Edge-Detection-Algorithmen
    streamlit_app: GUI-Interface
    run_edge_detectors: CLI-Interface
    gui_components: Wiederverwendbare UI-Komponenten

Beispiele:
    >>> from edgx.detectors import get_all_methods, run_laplacian
    >>> methods = get_all_methods()
    >>> result = run_laplacian("path/to/image.jpg", target_size=(512, 512))
    
    # CLI verwenden:
    $ edgx-cli --input_dir ./images --output_dir ./results --methods Laplacian Canny
    
    # GUI starten:
    $ edgx-gui
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Version Information
__version__ = "0.1.0"
__author__ = "Edge Detection Studio Team"
__email__ = ""
__license__ = "MIT"
__description__ = "Edge Detection Studio - Zero-Config Suite für Kantenerkennung"

# Python Version Check
if sys.version_info < (3, 10):
    warnings.warn(
        f"Python 3.10+ wird empfohlen. Aktuelle Version: {sys.version_info.major}.{sys.version_info.minor}",
        RuntimeWarning,
        stacklevel=2
    )

# Package Information
__all__ = [
    # Version & Meta
    "__version__",
    "__author__", 
    "__license__",
    "__description__",
    
    # Core Functions (conditionally imported)
    "get_all_methods",
    "get_max_resolution", 
    "standardize_output",
    "init_models",
    
    # Specific Methods (conditionally imported)
    "run_laplacian",
    "run_adaptive_canny",
    "run_scharr",
    
    # GUI Components (conditionally imported)
    "method_selector_advanced",
    "image_gallery",
    "batch_processor",
    
    # Constants
    "SUPPORTED_IMAGE_FORMATS",
    "DEFAULT_TARGET_SIZE",
]

# Constants
SUPPORTED_IMAGE_FORMATS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
DEFAULT_TARGET_SIZE = (1920, 1080)

# Conditional Imports mit Fallback-Handling
def _safe_import():
    """
    Importiert Kernfunktionen mit robustem Fallback-Handling.
    
    Returns:
        Dictionary mit erfolgreich importierten Funktionen
    """
    imported = {}
    
    # Core Detectors
    try:
        from .detectors import (
            get_all_methods,
            get_max_resolution,
            standardize_output,
            init_models,
            run_laplacian,
            run_adaptive_canny,
            run_scharr,
        )
        
        imported.update({
            "get_all_methods": get_all_methods,
            "get_max_resolution": get_max_resolution,
            "standardize_output": standardize_output,
            "init_models": init_models,
            "run_laplacian": run_laplacian,
            "run_adaptive_canny": run_adaptive_canny,
            "run_scharr": run_scharr,
        })
        
    except ImportError as e:
        warnings.warn(
            f"Detectors-Modul konnte nicht importiert werden: {e}",
            ImportWarning,
            stacklevel=3
        )
    
    # GUI Components
    try:
        from .gui_components import (
            method_selector_advanced,
            image_gallery, 
            batch_processor,
        )
        
        imported.update({
            "method_selector_advanced": method_selector_advanced,
            "image_gallery": image_gallery,
            "batch_processor": batch_processor,
        })
        
    except ImportError as e:
        warnings.warn(
            f"GUI-Komponenten konnten nicht importiert werden: {e}",
            ImportWarning,
            stacklevel=3
        )
    
    return imported

# Führe sichere Imports durch
_IMPORTED_FUNCTIONS = _safe_import()

# Dynamisch __all__ erweitern
for func_name in _IMPORTED_FUNCTIONS:
    if func_name not in __all__:
        __all__.append(func_name)

# Funktionen im Package-Namespace verfügbar machen
locals().update(_IMPORTED_FUNCTIONS)

# Environment Validation
def validate_environment() -> Dict[str, Any]:
    """
    Validiert die Laufzeitumgebung und Dependencies.
    
    Returns:
        Dictionary mit Umgebungsinformationen und Status
    """
    import platform
    
    env_info = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.system(),
        "platform_version": platform.release(),
        "architecture": platform.machine(),
        "edgx_version": __version__,
        "package_path": str(Path(__file__).parent),
        "dependencies": {},
        "warnings": [],
        "errors": [],
    }
    
    # Prüfe wichtige Dependencies
    dependencies_to_check = [
        ("cv2", "opencv-python"),
        ("torch", "torch"),
        ("streamlit", "streamlit"),
        ("numpy", "numpy"),
        ("PIL", "pillow"),
        ("requests", "requests"),
    ]
    
    for module_name, package_name in dependencies_to_check:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            env_info["dependencies"][package_name] = {
                "available": True,
                "version": version,
                "path": getattr(module, "__file__", "unknown")
            }
        except ImportError:
            env_info["dependencies"][package_name] = {
                "available": False,
                "version": None,
                "path": None
            }
            env_info["warnings"].append(f"Dependency '{package_name}' nicht verfügbar")
    
    # Optionale Dependencies
    optional_deps = [
        ("kornia", "kornia"),
        ("matplotlib", "matplotlib"),
        ("sklearn", "scikit-learn"),
    ]
    
    for module_name, package_name in optional_deps:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            env_info["dependencies"][package_name] = {
                "available": True,
                "version": version,
                "optional": True
            }
        except ImportError:
            env_info["dependencies"][package_name] = {
                "available": False,
                "version": None,
                "optional": True
            }
    
    # Platform-spezifische Checks
    if platform.system() == "Windows":
        if platform.release() not in ["10", "11"]:
            env_info["warnings"].append("Windows 10/11 wird empfohlen")
    
    # Python Version Check
    if sys.version_info < (3, 10):
        env_info["errors"].append("Python 3.10+ ist erforderlich")
    
    # Importierte Funktionen prüfen
    env_info["available_functions"] = list(_IMPORTED_FUNCTIONS.keys())
    env_info["missing_functions"] = [
        func for func in ["get_all_methods", "image_gallery", "batch_processor"]
        if func not in _IMPORTED_FUNCTIONS
    ]
    
    return env_info

def get_version_info() -> str:
    """
    Gibt detaillierte Versionsinformationen zurück.
    
    Returns:
        Formatierte Versions-String
    """
    env = validate_environment()
    
    info_lines = [
        f"Edge Detection Studio v{__version__}",
        f"Python {env['python_version']} on {env['platform']} {env['platform_version']}",
        f"Package: {env['package_path']}",
        "",
        "Dependencies:",
    ]
    
    for dep_name, dep_info in env["dependencies"].items():
        if dep_info["available"]:
            optional_tag = " (optional)" if dep_info.get("optional") else ""
            info_lines.append(f"  ✅ {dep_name} {dep_info['version']}{optional_tag}")
        else:
            optional_tag = " (optional)" if dep_info.get("optional") else ""
            status = "⚠️" if dep_info.get("optional") else "❌"
            info_lines.append(f"  {status} {dep_name} not available{optional_tag}")
    
    if env["warnings"]:
        info_lines.extend(["", "Warnings:"] + [f"  ⚠️ {w}" for w in env["warnings"]])
    
    if env["errors"]:
        info_lines.extend(["", "Errors:"] + [f"  ❌ {e}" for e in env["errors"]])
    
    info_lines.extend([
        "",
        f"Available functions: {len(env['available_functions'])}",
        f"Missing functions: {len(env['missing_functions'])}",
    ])
    
    return "\n".join(info_lines)

# Quick Start Functions
def quick_start() -> None:
    """
    Zeigt Quick-Start-Informationen an.
    """
    print("🎨 Edge Detection Studio - Quick Start")
    print("=" * 50)
    print()
    print("📋 Verfügbare Kommandos:")
    print("  edgx-gui          # Starte Streamlit GUI")
    print("  edgx-cli --help   # CLI-Hilfe anzeigen")
    print("  edgx-test         # Installation testen")
    print()
    print("🐍 Python API:")
    print("  from edgx import get_all_methods, run_laplacian")
    print("  methods = get_all_methods()")
    print("  result = run_laplacian('image.jpg')")
    print()
    print("📚 Weitere Hilfe:")
    print("  from edgx import get_version_info")
    print("  print(get_version_info())")

def list_methods() -> Optional[List[Tuple[str, Any]]]:
    """
    Zeigt alle verfügbaren Edge-Detection-Methoden an.
    
    Returns:
        Liste der verfügbaren Methoden oder None bei Fehler
    """
    if "get_all_methods" in _IMPORTED_FUNCTIONS:
        try:
            methods = _IMPORTED_FUNCTIONS["get_all_methods"]()
            print("🔧 Verfügbare Edge-Detection-Methoden:")
            print("-" * 40)
            for i, (name, _) in enumerate(methods, 1):
                print(f"{i:2d}. {name}")
            print(f"\n📊 Total: {len(methods)} Methoden")
            return methods
        except Exception as e:
            print(f"❌ Fehler beim Laden der Methoden: {e}")
            return None
    else:
        print("❌ Detectors-Modul nicht verfügbar")
        print("💡 Versuchen Sie: pip install -e .")
        return None

# Convenience Functions for direct usage
def edge_detect(image_path: str, method: str = "Laplacian", **kwargs) -> Optional[Any]:
    """
    Convenience-Funktion für schnelle Edge-Detection.
    
    Args:
        image_path: Pfad zum Eingabebild
        method: Name der Edge-Detection-Methode
        **kwargs: Zusätzliche Parameter
    
    Returns:
        Edge-Map oder None bei Fehler
    """
    if "get_all_methods" not in _IMPORTED_FUNCTIONS:
        print("❌ Detectors-Modul nicht verfügbar")
        return None
    
    try:
        methods = dict(_IMPORTED_FUNCTIONS["get_all_methods"]())
        if method not in methods:
            available = list(methods.keys())
            print(f"❌ Methode '{method}' nicht verfügbar")
            print(f"💡 Verfügbare Methoden: {', '.join(available[:5])}...")
            return None
        
        return methods[method](image_path, **kwargs)
    
    except Exception as e:
        print(f"❌ Edge-Detection fehlgeschlagen: {e}")
        return None

# Module-Level Dokumentation erweitern
__doc__ += f"""

Installierte Version: {__version__}
Verfügbare Funktionen: {len(_IMPORTED_FUNCTIONS)}

Quick Start:
    >>> import edgx
    >>> edgx.quick_start()
    >>> edgx.list_methods()
    >>> print(edgx.get_version_info())
"""

# Cleanup
del _safe_import, _IMPORTED_FUNCTIONS
