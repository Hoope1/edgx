#!/usr/bin/env python3
"""
Setup-Script für Edge Detection Studio

Diese Datei ermöglicht die Installation des edgx-Packages mit:
    pip install -e .
    
Unterstützt sowohl setup.py als auch pyproject.toml (moderne Python-Standards).
"""

from setuptools import setup, find_packages
from pathlib import Path
import sys

# Mindestanforderungen prüfen
if sys.version_info < (3, 10):
    print("❌ Fehler: Python 3.10 oder neuer ist erforderlich")
    print(f"   Aktuelle Version: {sys.version}")
    print("💡 Lösung: Installieren Sie Python 3.10+ von https://python.org")
    sys.exit(1)

# README einlesen für long_description
README_PATH = Path(__file__).parent / "README.md"
try:
    with open(README_PATH, "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Edge Detection Studio - Zero-Config Suite für Kantenerkennung"

# Version aus pyproject.toml oder Fallback
VERSION = "0.1.0"
try:
    import tomllib
    with open("pyproject.toml", "rb") as f:
        pyproject_data = tomllib.load(f)
        VERSION = pyproject_data.get("project", {}).get("version", VERSION)
except (ImportError, FileNotFoundError):
    pass

# Kern-Abhängigkeiten (minimal für Funktionsfähigkeit)
CORE_DEPENDENCIES = [
    "streamlit>=1.28.0",
    "opencv-python>=4.5.0",
    "opencv-contrib-python>=4.5.0", 
    "torch>=1.9.0",
    "torchvision>=0.10.0",
    "kornia>=0.6.0",
    "numpy>=1.21.0",
    "pillow>=8.0.0",
    "requests>=2.25.0",
    "click>=7.0",
    "altair>=4.0.0",
]

# Optionale Abhängigkeiten
OPTIONAL_DEPENDENCIES = [
    "matplotlib>=3.3.0",
    "scikit-image>=0.18.0",
    "plotly>=5.0.0",
    "watchdog>=2.1.0",
    "jupyter>=1.0.0",
]

# Development-Abhängigkeiten
DEV_DEPENDENCIES = [
    "pytest>=6.0.0",
    "pytest-cov>=2.0.0",
    "black>=22.0.0",
    "flake8>=4.0.0",
    "isort>=5.0.0",
    "mypy>=0.900",
    "pre-commit>=2.0.0",
    "opencv-python-stubs",
    "types-requests",
    "types-Pillow",
]

# Kombinierte Dependencies für einfache Installation
ALL_DEPENDENCIES = CORE_DEPENDENCIES + OPTIONAL_DEPENDENCIES

setup(
    # Grundlegende Package-Information
    name="edgx",
    version=VERSION,
    description="Edge Detection Studio - Zero-Config Suite für Kantenerkennung",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Autor und URLs
    author="Edge Detection Studio Team",
    author_email="",
    url="https://github.com/edge-detection-studio/edgx",
    project_urls={
        "Bug Reports": "https://github.com/edge-detection-studio/edgx/issues",
        "Source": "https://github.com/edge-detection-studio/edgx",
        "Documentation": "https://github.com/edge-detection-studio/edgx/blob/main/README.md",
    },
    
    # Package-Struktur
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    
    # Python-Anforderungen
    python_requires=">=3.10",
    
    # Abhängigkeiten
    install_requires=CORE_DEPENDENCIES,
    extras_require={
        "all": ALL_DEPENDENCIES,
        "optional": OPTIONAL_DEPENDENCIES,
        "dev": DEV_DEPENDENCIES,
        "full": ALL_DEPENDENCIES + DEV_DEPENDENCIES,
        
        # Spezifische Use-Cases
        "gui": [
            "streamlit>=1.28.0",
            "plotly>=5.0.0",
            "altair>=4.0.0",
        ],
        "cpu": [
            "torch>=1.9.0+cpu",
            "torchvision>=0.10.0+cpu",
        ],
        "gpu": [
            "torch>=1.9.0+cu118",
            "torchvision>=0.10.0+cu118",
            "kornia>=0.6.0",
        ],
    },
    
    # Console Scripts (CLI-Tools)
    entry_points={
        "console_scripts": [
            "edgx-cli=edgx.run_edge_detectors:main",
            "edgx-gui=edgx.streamlit_app:main",
            "edgx-test=edgx.detectors:main",
        ],
    },
    
    # Package-Daten
    package_data={
        "edgx": [
            "models/**/*",
            "*.yml",
            "*.yaml", 
            "*.txt",
        ],
    },
    
    # Klassifikationen für PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        "License :: OSI Approved :: MIT License",
        
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Microsoft :: Windows :: Windows 11",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        
        "Framework :: Streamlit",
    ],
    
    # Keywords für bessere Auffindbarkeit
    keywords=[
        "edge detection", "computer vision", "image processing", 
        "opencv", "pytorch", "streamlit", "gui", "cli",
        "canny", "sobel", "laplacian", "hed", "machine learning",
        "deep learning", "kornia", "batch processing"
    ],
    
    # Zip-sichere Installation
    zip_safe=False,
    
    # Zusätzliche Metadaten
    platforms=["Windows", "Linux", "macOS"],
    license="MIT",
    
    # Setup-Hooks für Post-Install-Aktionen
    cmdclass={},
)

# Post-Installation-Nachrichten
def print_post_install_message():
    """Zeigt hilfreiche Nachrichten nach der Installation."""
    print("\n" + "="*60)
    print("🎉 Edge Detection Studio erfolgreich installiert!")
    print("="*60)
    print()
    print("📋 Verfügbare Kommandos:")
    print("  edgx-gui          # Starte Streamlit GUI")
    print("  edgx-cli --help   # CLI-Hilfe anzeigen")
    print("  edgx-test         # Installation testen")
    print()
    print("🚀 Schnellstart:")
    print("  1. GUI starten:     edgx-gui")
    print("  2. CLI verwenden:   edgx-cli --input_dir ./images --output_dir ./results")
    print("  3. Methoden listen: edgx-cli --list-methods")
    print("  4. Installation testen: edgx-test")
    print()
    print("📚 Weitere Informationen:")
    print("  README.md         # Vollständige Dokumentation")
    print("  AGENTS.md         # Entwickler-Richtlinien")
    print()
    print("💡 Bei Problemen:")
    print("  1. Prüfen Sie: python --version (sollte >=3.10 sein)")
    print("  2. Führen Sie aus: edgx-test")
    print("  3. Konsultieren Sie die Dokumentation")
    print()


if __name__ == "__main__":
    # Direkte Ausführung des Setup-Scripts
    print("🔧 Installiere Edge Detection Studio...")
    
    # Prüfe, ob alle Pfade existieren
    required_paths = [
        "src/edgx/__init__.py",
        "src/edgx/detectors.py",
        "src/edgx/streamlit_app.py",
        "src/edgx/run_edge_detectors.py",
    ]
    
    missing_files = []
    for path in required_paths:
        if not Path(path).exists():
            missing_files.append(path)
    
    if missing_files:
        print("❌ Fehler: Erforderliche Dateien fehlen:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n💡 Stellen Sie sicher, dass Sie im Projektverzeichnis sind")
        sys.exit(1)
    
    try:
        # Führe Setup aus
        from setuptools import setup
        print("✅ Setup-Konfiguration geladen")
        print("📦 Installiere Package...")
        
        # Bei direkter Ausführung zeige Post-Install-Message
        import atexit
        atexit.register(print_post_install_message)
        
    except Exception as e:
        print(f"❌ Setup-Fehler: {e}")
        print("\n💡 Versuchen Sie: pip install -e .")
        sys.exit(1)
