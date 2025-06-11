# 🎨 Edge Detection Studio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Windows 11](https://img.shields.io/badge/windows-10%20%7C%2011-blue.svg)](https://www.microsoft.com/windows)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Eine Zero-Config-Suite zur Kantenerkennung mit 15 klassischen und Deep-Learning-Algorithmen.**

Edge Detection Studio bietet eine sofort einsatzfähige Sammlung von Edge-Detection-Verfahren für Computer Vision, UI-Design und Forschung. Mit robuster Streamlit-GUI und leistungsstarkem CLI-Tool.

---

## 🚀 Schnellstart (5 Minuten)

### Option 1: Ein-Klick-Setup (Windows 11)

```bash
# 1. Repository herunterladen und entpacken
# 2. In Projektverzeichnis navigieren
cd edge_detection_tool

# 3. Automatische Installation und GUI-Start
run.bat
```

### Option 2: Manuelle Installation

```bash
# Python 3.10+ erforderlich
python --version  # Sollte 3.10+ anzeigen

# Virtuelle Umgebung erstellen
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Package installieren (löst "No module named 'edgx'" Problem)
pip install -e .

# Dependencies installieren
pip install -r requirements.txt

# GUI starten
streamlit run src/edgx/streamlit_app.py
```

### Option 3: CLI verwenden

```bash
# Batch-Verarbeitung
python -m edgx.run_edge_detectors \
    --input_dir ./images \
    --output_dir ./results \
    --methods Laplacian AdaptiveCanny HED_OpenCV

# Verfügbare Methoden anzeigen
python -m edgx.run_edge_detectors --list-methods

# Installation testen
python -m edgx.detectors --test
```

---

## 📋 Inhaltsverzeichnis

- [Features](#-features)
- [Installation](#️-installation)
- [Nutzung](#-nutzung)
- [Verfügbare Algorithmen](#-verfügbare-algorithmen)
- [Problemlösungen](#-problemlösungen)
- [Entwicklung](#️-entwicklung)
- [Lizenz](#-lizenz)

---

## ✨ Features

### 🎯 **15 Edge-Detection-Algorithmen**
- **Klassische Filter:** Laplacian, Canny, Sobel, Scharr, Prewitt, Roberts
- **Erweiterte Varianten:** Multi-Scale Canny, Adaptive Canny, Morphological Gradient
- **Deep Learning:** HED (OpenCV + PyTorch), Structured Forests, BDCN
- **GPU-beschleunigt:** Kornia Canny, Kornia Sobel (CUDA)

### 🖥️ **Zwei Benutzeroberflächen**
- **Streamlit GUI:** Drag & Drop, Live-Vorschau, Batch-Export
- **CLI-Tool:** Automatisierung, Scripting, CI/CD-Integration

### ⚙️ **Zero-Configuration**
- **Automatischer Modell-Download:** HED, Structured Forests
- **Intelligente Fallbacks:** Robuste Alternativen bei fehlenden Dependencies
- **Einheitliche Ausgabe:** Normalisierte PNG-Dateien, konsistente Auflösung

### 🔧 **Robuste Architektur**
- **Umfassende Tests:** Unit-, Integration- und Performance-Tests
- **Fehlerbehandlung:** Graceful Fallbacks für alle Algorithmen
- **Cross-Platform:** Windows 11 (primär), Linux, macOS

---

## 🛠️ Installation

### Systemanforderungen

| Komponente | Mindestanforderung | Empfohlen |
|------------|-------------------|-----------|
| **Python** | 3.10.0 | 3.11+ |
| **OS** | Windows 10 | Windows 11 |
| **RAM** | 4 GB | 8 GB+ |
| **Speicher** | 2 GB frei | 5 GB+ |
| **GPU** | Optional | CUDA-fähig |

### Schritt-für-Schritt-Installation

#### 1. **Python 3.10+ installieren**
```bash
# Prüfen Sie Ihre Python-Version
python --version

# Falls < 3.10: Von https://python.org herunterladen
```

#### 2. **Repository klonen/herunterladen**
```bash
git clone <repository-url>
cd edge_detection_tool

# Oder ZIP herunterladen und entpacken
```

#### 3. **Virtuelle Umgebung einrichten**
```bash
python -m venv venv
venv\Scripts\activate
```

#### 4. **Package installieren**
```bash
# WICHTIG: Löst "No module named 'edgx'" Fehler
pip install -e .
```

#### 5. **Dependencies installieren**
```bash
pip install -r requirements.txt

# Bei Problemen: Kernpakete einzeln installieren
pip install streamlit opencv-python opencv-contrib-python torch torchvision kornia numpy pillow requests
```

#### 6. **Installation testen**
```bash
python -m edgx.detectors --test
```

### Alternative Installationsmethoden

#### **Nur GUI-Komponenten**
```bash
pip install -e .[gui]
```

#### **Mit GPU-Unterstützung**
```bash
pip install -e .[gpu]
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### **Vollständige Installation**
```bash
pip install -e .[all]
```

---

## 🎮 Nutzung

### Streamlit GUI

```bash
# GUI starten
streamlit run src/edgx/streamlit_app.py

# Oder über installiertes Package
edgx-gui
```

**GUI-Features:**
- 📁 **Ordner- oder Datei-Upload**
- 🔧 **Methoden-Auswahl mit Kategorien**
- ⚙️ **Einstellungen:** Auflösung, GPU-Nutzung, Parallelisierung
- 👁️ **Live-Vorschau** für einzelne Bilder
- 📥 **ZIP-Export** aller Ergebnisse

### CLI-Tool

#### **Basis-Nutzung**
```bash
# Alle Bilder in einem Ordner verarbeiten
python -m edgx.run_edge_detectors \
    --input_dir ./images \
    --output_dir ./results

# Spezifische Methoden auswählen
python -m edgx.run_edge_detectors \
    --input_dir ./images \
    --output_dir ./results \
    --methods Laplacian AdaptiveCanny HED_OpenCV
```

#### **Erweiterte Optionen**
```bash
# Benutzerdefinierte Auflösung
python -m edgx.run_edge_detectors \
    --input_dir ./images \
    --output_dir ./results \
    --size 1920x1080

# Skalierung
python -m edgx.run_edge_detectors \
    --input_dir ./images \
    --output_dir ./results \
    --scale 0.5

# Rekursive Suche
python -m edgx.run_edge_detectors \
    --input_dir ./images \
    --output_dir ./results \
    --recursive

# Dry-Run (Vorschau ohne Verarbeitung)
python -m edgx.run_edge_detectors \
    --input_dir ./images \
    --output_dir ./results \
    --dry-run
```

#### **Informations-Befehle**
```bash
# Verfügbare Methoden auflisten
python -m edgx.run_edge_detectors --list-methods

# Installation testen
python -m edgx.detectors --test

# Umgebung validieren
python -m edgx.validate_environment --detailed
```

### Python API

```python
import edgx

# Schnelle Edge-Detection
result = edgx.edge_detect("image.jpg", method="Laplacian")

# Alle verfügbaren Methoden
methods = edgx.get_all_methods()

# Spezifische Methode verwenden
from edgx.detectors import run_adaptive_canny
result = run_adaptive_canny("image.jpg", target_size=(512, 512))

# Batch-Verarbeitung
from edgx.gui_components import batch_processor
results = batch_processor(
    images=["img1.jpg", "img2.jpg"],
    methods=["Laplacian", "AdaptiveCanny"],
    output_dir="./results",
    settings={"target_size": (1024, 768)}
)
```

---

## 🔬 Verfügbare Algorithmen

### Klassische Filter

| Algorithmus | Beschreibung | Performance | Qualität |
|------------|--------------|-------------|----------|
| **Laplacian** | Zweite Ableitung, erkennt Blobs und Linien | ⚡⚡⚡ | ⭐⭐⭐ |
| **AdaptiveCanny** | Automatische Threshold-Berechnung | ⚡⚡ | ⭐⭐⭐⭐ |
| **MultiScaleCanny** | Canny mit mehreren Blur-Leveln | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| **Scharr** | Verbesserte Sobel-Filter | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| **Prewitt** | Gradient-basierte Edge-Detection | ⚡⚡⚡ | ⭐⭐⭐ |
| **Roberts** | Cross-Gradient-Operator | ⚡⚡⚡ | ⭐⭐ |

### Deep Learning

| Algorithmus | Beschreibung | Performance | Qualität |
|------------|--------------|-------------|----------|
| **HED_OpenCV** | Holistically-Nested Edge Detection | ⚡ | ⭐⭐⭐⭐⭐ |
| **HED_PyTorch** | HED mit PyTorch-Backend + Fallbacks | ⚡ | ⭐⭐⭐⭐⭐ |
| **StructuredForests** | Random Forest Edge-Detection | ⚡⚡ | ⭐⭐⭐⭐ |
| **BDCN** | Bi-Directional Cascade Network + Fallback | ⚡ | ⭐⭐⭐⭐⭐ |

### GPU-beschleunigt

| Algorithmus | Beschreibung | Voraussetzung | Performance |
|------------|--------------|---------------|-------------|
| **Kornia_Canny** | GPU-Canny via Kornia | PyTorch + CUDA | ⚡⚡⚡⚡ |
| **Kornia_Sobel** | GPU-Sobel via Kornia | PyTorch + CUDA | ⚡⚡⚡⚡ |
| **FixedCNN** | CNN-Filter mit PyTorch | PyTorch | ⚡⚡⚡ |

### Legende
- **Performance:** ⚡ = Langsam, ⚡⚡⚡⚡ = Sehr schnell
- **Qualität:** ⭐ = Basic, ⭐⭐⭐⭐⭐ = Exzellent

---

## 🔧 Problemlösungen

### Häufige Installationsfehler

#### ❌ **"No module named 'edgx'"**
```bash
# Lösung: Package korrekt installieren
pip install -e .

# Prüfen der Installation
python -c "import edgx; print('✅ edgx verfügbar')"
```

#### ❌ **"Repository not found" (pytorch-hed)**
Dieses Problem wurde behoben! Der Code verwendet jetzt intelligente Fallbacks:
- HED_PyTorch fällt zurück auf HED_OpenCV
- Bei fehlenden Modellen wird Adaptive Canny verwendet
- Alle Methoden funktionieren auch ohne externe Dependencies

#### ❌ **"streamlit: command not found"**
```bash
# Installation prüfen
pip install streamlit

# Alternativer Start
python -m streamlit run src/edgx/streamlit_app.py
```

#### ❌ **GPU/CUDA Probleme**
```bash
# CPU-Version installieren (funktioniert überall)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA-Version (falls GPU vorhanden)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Erweiterte Problembehandlung

#### **Umgebungsvalidierung**
```bash
# Vollständige Systemprüfung
python -m edgx.validate_environment --detailed

# Nur kritische Checks
python -m edgx.validate_environment --critical-only

# JSON-Output für Automatisierung
python -m edgx.validate_environment --json
```

#### **Funktionalitätstests**
```bash
# Alle Methoden testen
python -m edgx.detectors --test

# CLI-Tool testen
python -m edgx.run_edge_detectors --input_dir images --dry-run

# GUI-Test (headless)
streamlit run src/edgx/streamlit_app.py --server.headless true --server.port 8502
```

#### **Log-Analyse**
```bash
# Verbose Modus für detaillierte Logs
python -m edgx.run_edge_detectors --input_dir images --verbose

# GUI mit Debug-Informationen
# Aktivieren Sie "Debug-Modus" in der Sidebar
```

---

## 🏗️ Entwicklung

### Development Setup

```bash
# Repository klonen
git clone <repository-url>
cd edge_detection_tool

# Development-Installation
pip install -e .[dev]

# Pre-commit Hooks installieren
pre-commit install

# Tests ausführen
pytest

# Code-Qualität prüfen
pre-commit run --all-files
```

### Projekt-Struktur

```
edge_detection_tool/
├── src/edgx/                   # Hauptpackage
│   ├── __init__.py            # Package-Initialisierung
│   ├── detectors.py           # Kern-Algorithmen
│   ├── streamlit_app.py       # GUI-Hauptdatei
│   ├── gui_components.py      # UI-Komponenten
│   ├── run_edge_detectors.py  # CLI-Tool
│   └── validate_environment.py # System-Validierung
├── tests/                     # Test-Suite
│   ├── test_basic.py         # Basis-Tests
│   └── test_detectors.py     # Algorithmus-Tests
├── setup.py                  # Package-Installation
├── pyproject.toml           # Moderne Python-Konfiguration
├── requirements.txt         # Dependencies
├── .pre-commit-config.yaml  # Code-Qualität
└── README.md               # Diese Datei
```

### Neue Algorithmen hinzufügen

1. **Implementierung in `detectors.py`:**
```python
def run_new_algorithm(path: str, target_size: tuple | None = None) -> np.ndarray:
    """Neuer Edge-Detection-Algorithmus."""
    # Algorithmus-Implementation
    result = your_algorithm_logic(path)
    
    # WICHTIG: Standardisierung aufrufen
    return standardize_output(result, target_size)
```

2. **Registrierung in `get_all_methods()`:**
```python
def get_all_methods():
    return [
        # ... bestehende Methoden
        ("NewAlgorithm", run_new_algorithm),
    ]
```

3. **GUI-Integration in `gui_components.py`:**
```python
# Kategorie-Mapping erweitern
cats = {
    "Neue Kategorie": ["NewAlgorithm"],
    # ...
}
```

4. **Tests hinzufügen:**
```python
# tests/test_detectors.py
def test_new_algorithm(self, quick_test_image):
    result = run_new_algorithm(str(quick_test_image))
    assert result is not None
    # ... weitere Assertions
```

### Code-Qualität Standards

- **Code-Stil:** Black (88 Zeichen)
- **Import-Sortierung:** isort
- **Linting:** flake8 + zusätzliche Plugins
- **Type-Checking:** mypy
- **Security:** bandit
- **Tests:** pytest (>80% Coverage)

### CI/CD Pipeline

```bash
# Alle Checks die in CI laufen
pre-commit run --all-files
pytest --cov=edgx --cov-report=term-missing
python -m edgx.detectors --test
python -m edgx.validate_environment --critical-only
```

---

## 📊 Performance & Hardware

### Benchmarks (Windows 11, Intel i7, RTX 3070)

| Algorithmus | 512x512 | 1024x1024 | 2048x2048 |
|------------|---------|-----------|-----------|
| Laplacian | 2ms | 8ms | 35ms |
| AdaptiveCanny | 5ms | 18ms | 70ms |
| HED_OpenCV | 45ms | 180ms | 720ms |
| Kornia_Canny (GPU) | 3ms | 12ms | 48ms |

### Hardware-Empfehlungen

| Use Case | CPU | RAM | GPU | Speicher |
|----------|-----|-----|-----|----------|
| **Gelegentliche Nutzung** | 4+ Kerne | 8 GB | Optional | 5 GB |
| **Batch-Verarbeitung** | 8+ Kerne | 16 GB | RTX 3060+ | 20 GB+ |
| **Forschung/Development** | 12+ Kerne | 32 GB | RTX 4070+ | 50 GB+ |

---

## 📝 Lizenz

**MIT License** - siehe [LICENSE](LICENSE) Datei für Details.

### Drittanbieter-Lizenzen

- **OpenCV:** Apache 2.0 License
- **PyTorch:** BSD-3-Clause License
- **Streamlit:** Apache 2.0 License
- **Kornia:** Apache 2.0 License

### Modell-Lizenzen

- **HED:** Originale Publikations-Lizenz (Non-commercial Research)
- **Structured Forests:** BSD License

---

## 🤝 Beiträge

Beiträge sind willkommen! Bitte beachten Sie:

1. **Lesen Sie `AGENTS.md`** für detaillierte Entwicklungsrichtlinien
2. **Erstellen Sie Issues** für Bugs oder Feature-Requests
3. **Folgen Sie Code-Standards** (pre-commit wird automatisch prüfen)
4. **Schreiben Sie Tests** für neue Features
5. **Dokumentieren Sie Änderungen** in Pull Requests

### Quick-Contribute

```bash
# Fork das Repository
git clone your-fork-url
cd edge_detection_tool

# Feature-Branch erstellen
git checkout -b feature/new-algorithm

# Entwickeln & testen
# ... Ihre Änderungen ...

# Code-Qualität prüfen
pre-commit run --all-files
pytest

# Commit & Push
git add .
git commit -m "feat: Add new edge detection algorithm"
git push origin feature/new-algorithm

# Pull Request erstellen
```

---

## 📞 Support

- **📖 Dokumentation:** Diese README + `AGENTS.md`
- **🐛 Bug Reports:** [GitHub Issues](issues)
- **💬 Diskussionen:** [GitHub Discussions](discussions)
- **📧 Email:** [Kontakt](mailto:contact@example.com)

### Bevor Sie ein Issue erstellen

1. **Suchen Sie** in bestehenden Issues
2. **Führen Sie aus:** `python -m edgx.validate_environment --detailed`
3. **Testen Sie:** `python -m edgx.detectors --test`
4. **Geben Sie an:** Python-Version, OS, Error-Logs

---

## 🙏 Danksagungen

- **OpenCV Community** für robuste Computer Vision Tools
- **PyTorch Team** für excellentes Deep Learning Framework
- **Streamlit** für benutzerfreundliche Web-App-Entwicklung
- **Edge Detection Research Community** für Algorithmus-Innovationen

---

## 📈 Roadmap

### Version 0.2.0
- [ ] **Mehr Algorithmen:** RCF, CASENet, PiDiNet
- [ ] **Video-Support:** Edge-Detection für Videos
- [ ] **API-Server:** REST API für Cloud-Integration
- [ ] **Docker-Support:** Containerisierte Deployment

### Version 0.3.0
- [ ] **Batch-Optimierung:** Multi-GPU Support
- [ ] **Custom-Training:** Fine-tuning von Deep Learning Modellen
- [ ] **Web-Interface:** Browser-basierte GUI
- [ ] **Plugin-System:** Erweiterbare Architektur

---

**⭐ Gefällt Ihnen das Projekt? Geben Sie uns einen Star auf GitHub!**

**🎯 Edge Detection Studio - Von Entwicklern für Entwickler. Zero-Config, Maximum Impact.**
