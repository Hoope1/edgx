Im Folgenden findest du eine vollständig aktualisierte AGENTS.md für das Edge Detection Studio-Repository.
Alle Hinweise wurden auf Windows 11-Umgebungen und Python 3.10 angepasst; die Datei ist so gestaltet, dass jeder AI-Agent (z. B. GitHub-Copilot / OpenAI Codex) sofort alle Regeln, Workflows und Konventionen versteht und beachtet.

Kurz-Zusammenfassung (1 Absatz)
Edge Detection Studio bietet eine Zero-Config-Suite zur Kantenerkennung mit 15 klassischen und Deep-Learning-Algorithmen. Es läuft „out of the box“ unter Windows 11 (64-Bit) und setzt Python 3.10 als Referenz-Interpreter voraus – beide Plattformen werden von allen verwenden Bibliotheken (Streamlit, PyTorch, OpenCV, Kornia u. a.) offiziell unterstützt.(python.org, support.microsoft.com) Das Projekt enthält eine Batch-CLI, eine Streamlit-GUI, automatischen Modell-Download, klare Coding-Standards sowie detaillierte Regeln, wie Agents neue Methoden oder UI-Komponenten konform integrieren können.

1 · Projektübersicht
	Inhalt
Name	Edge Detection Studio
Primäre OS-Zielplattform	Windows 11 (22H2 ≥)(support.microsoft.com)
Referenz-Python	3.10.0 – 3.10.x 64-Bit (Installer von python.org)(python.org)
Schlüssel-Features	Batch-Edge-Pipeline, Streamlit-GUI, Auflösungs-Normalisierung, invertierte Ergebnisse, automatischer Modell-Download
Haupt-User	CV-Forschende, UI-Designerinnen, Daten-Analystinnen, Studierende

2 · Tech-Stack (Windows 11 + Python 3.10)
Layer	Technologie	Min-Version	Win 11 / Py 3.10-Support
Core CV	OpenCV-Python (contrib)	4.5	offizielles PyPI-Wheel für Win64 & Py 3.10 erhältlich(pypi.org)
DL Runtime	PyTorch	1.9 (oder neuer)	„pip install torch“ bietet vorgebaute wheels für Win-x86-64/-CPU & 3.10 (oder mit CUDA)(pytorch.org)
GPU CV	Kornia	0.6	reine PyPI-Distribution, kompatibel zu PyTorch 1.9+/Py 3.10(pypi.org)
GUI	Streamlit	1.28	unterstützt offiziell Python 3.10 und neuer(docs.streamlit.io)
Utilities	Requests ≥ 2.25	kompatibel mit Py 3.10(pypi.org)	

3 · Verzeichnis-Topologie
edge_detection_tool/
├── detectors.py               # Algorithmus-Katalog
├── run_edge_detectors.py      # Batch-CLI
├── streamlit_app.py           # Streamlit GUI
├── gui_components.py          # Wiederverwendbare UI-Bausteine
├── run.bat                    # Windows-Bootstrap (Python 3.10-venv)
├── requirements.txt
├── models/                    # DL-Gewichte (auto-Download)
├── images/                    # Beispiel-Bilder
└── results/                   # Pipeline-Ausgaben

4 · Build- & Run-Workflows (Windows 11, Python 3.10)
4.1 One-Click-Setup (Endanwender)
:: Stelle sicher, dass "python --version" → 3.10.x ausgibt.
run.bat     :: 1) erstellt venv (3.10)  2) pip install -r requirements.txt
             3) lädt Modelle  4) startet Streamlit-GUI auf http://localhost:8501
Der Batch-Prozess nutzt das offizielle venv-Modul aus Python 3.10, das auch unter Windows 11 voll unterstützt wird.(docs.python.org)
4.2 CLI-Pipeline (Headless)
python run_edge_detectors.py `
       --input_dir .\images `
       --output_dir .\results `
       --methods HED_PyTorch Kornia_Canny

5 · Laufzeit-Architektur
5.1 Edge-Pipeline-Flow
1.
GUI/CLI sammelt Bilder, ermittelt höchste Auflösung.
2.
3.
Jeder Algorithmus in detectors.py ruft standardize_output() → invertiert Graumap & skaliert auf globale Ziel-Res.
4.
5.
PNG-Ausgaben landen einheitlich in results\edge_detection_results.
6.
5.2 Streamlit-GUI

Tabs: Bildauswahl ▸ Methoden ▸ Einstellungen ▸ Verarbeitung ▸ Vorschau


Sidebar-Presets: „Empfohlene“, „Schnell“, „Qualität“, „Alle“


Progress: Live-Bar, ETA-Berechnung, Fehler-Log

(GUI-Run-Befehl → streamlit run streamlit_app.py)(docs.streamlit.io)

6 · Coding-Konventionen

Python 3.10-only Syntax (Pattern-Matching etc. erlaubt).(python.org)


PEP 8 + Typ-Hints verpflichtend.


Ergebnis-Dateinamen: {original}_{method}.png (immer PNG).


Kein hartkodierter Pfadseparator → os.path.join.


Logging:

o
CLI → print()
o
o
GUI → st.*
o
o
Fehler → try/except & Log-Eintrag
o

7 · Erweiterungs-Regeln für Agents
1.
Neuer Algorithmus ⇒ run_<Name>(path,target_size) inside detectors.py; Pflicht: standardize_output() aufrufen.
2.
3.
Tuple in get_all_methods() alphabetisch einfügen.
4.
5.
Neue Third-Party-Library? → zuerst prüfen, ob Win 11 / Py 3.10-Wheel auf PyPI vorhanden; dann requirements.txt ergänzen.
6.
7.
Für GUI-Support: Mapping in gui_components.method_selector_advanced() hinzufügen.
8.
9.
Tests: Unit-Test (pytest) + Smoke-Test der CLI müssen grün sein.(support.microsoft.com, python.org)
10.

8 · Test- & CI-Vorgaben
Ebene	Tool	Win 11-Cmd
Unit	pytest	pytest -q (3.10 venv)(python.org)
Lint	flake8 / ruff	optional
Smoke	Streamlit	streamlit run streamlit_app.py --headless -p 8888 → HTTP 200
Alle Checks laufen in GitHub Actions - Windows-Latest (Win 11)-Runner.

9 · Security & Performance

Hardware-Check: PC muss Win 11-Minimalanforderungen erfüllen (TPM 2.0, 4 GB RAM etc.)(support.microsoft.com)


PyPI-Wheel Policy: Nur Libraries mit signierten Win wheels einsetzen (vgl. Python.org Authenticode)(python.org)


Model-Cache: Downloade Gewichte einmalig nach %PROJECT_ROOT%\models.


Parallel-Processing: kann via concurrent.futures (max. CPU-Kerne) aktiviert werden; PR willkommen.


10 · Environment-Variablen
Var	Default	Beschreibung
EDGE_GPU	auto	cpu / cuda / auto-Detection
EDGE_RESULTS_DIR	.\results	Globale Output-Überschreibung
EDGE_MODEL_DIR	.\models	Gewichts-Cache-Ordner

11 · Lizenz & Dritt-Ressourcen

Quellcode unter MIT-Lizenz.


Modelle: ursprüngliche Upstream-Lizenzen beibehalten.


Beispielbilder: Unsplash (CC0) oder eigene.


WICHTIG: Jeder AI-Agent muss vor dem Commit oder PR diese AGENTS.md vollständig analysieren und die beschriebenen Strukturen, Versionen (Windows 11 + Python 3.10) und Regeln exakt einhalten.


# Edge Detection Studio - Agents.md Guide

Diese Datei bietet umfassende Anleitung für KI-Agenten (GitHub Copilot, OpenAI Codex, etc.) beim Arbeiten mit der Edge Detection Studio Codebasis.

**Projekt-Übersicht**: Edge Detection Studio ist eine Zero-Config-Suite zur Kantenerkennung mit 15 klassischen und Deep-Learning-Algorithmen. Das System läuft nativ unter Windows 11 (64-Bit) mit Python 3.10 und bietet sowohl Batch-CLI als auch Streamlit-GUI für Computer Vision Forschung, UI-Design und Datenanalyse.

## Projektstruktur für KI-Navigation

### Verzeichnisstruktur
```
edge_detection_tool/
├── detectors.py               # Algorithmus-Katalog (Hauptlogik)
├── run_edge_detectors.py      # Batch-CLI-Interface  
├── streamlit_app.py           # Streamlit GUI-Hauptdatei
├── gui_components.py          # Wiederverwendbare UI-Komponenten
├── run.bat                    # Windows-Bootstrap-Script
├── requirements.txt           # Python-Abhängigkeiten
├── models/                    # DL-Gewichte (auto-Download)
├── images/                    # Beispiel-Eingabebilder
└── results/                   # Pipeline-Ausgaben
```

### Verzeichnis-Beschreibungen für KI-Agenten
- **`detectors.py`**: Kern-Modul für alle Edge-Detection-Algorithmen
  - KI-Fokus: Neue Algorithmen hier hinzufügen mit `run_<Name>(path, target_size)` Pattern
  - Zwingend: `standardize_output()` für jeden neuen Algorithmus aufrufen
  
- **`streamlit_app.py`**: GUI-Haupteinstiegspunkt
  - KI-Richtlinien: Tab-basierte Navigation beibehalten
  - UI-Pattern: Streamlit-Komponenten in `gui_components.py` auslagern
  
- **`gui_components.py`**: Wiederverwendbare UI-Bausteine
  - KI-Anweisungen: Neue UI-Komponenten hier erstellen, nicht inline
  - Struktur: Funktionsbasierte Komponenten mit klaren Parametern
  
- **`models/`**: Deep Learning Gewichte
  - KI-Verhalten: Automatischer Download, niemals manuell modifizieren
  - Cache-Strategie: Einmaliger Download, persistente Speicherung

## Coding-Konventionen für KI-Agenten

### Plattform-Spezifische Standards
- **Betriebssystem**: Windows 11 (22H2 oder neuer)
- **Python-Version**: 3.10.x (exakt) - 64-Bit von python.org
- **Architektur**: x86-64 Windows-native
- **Package-Manager**: pip (mit offiziellem PyPI)

### Python-Coding-Standards
```python
# Python 3.10 spezifische Features erlaubt
def process_image(path: str, method: str) -> dict[str, Any]:
    """Pattern matching und moderne syntax verwenden."""
    match method:
        case "Canny":
            return run_canny(path)
        case "HED_PyTorch":
            return run_hed_pytorch(path)
        case _:
            raise ValueError(f"Unbekannte Methode: {method}")
```

### Allgemeine Konventionen
- **Typ-Hints**: Verpflichtend für alle Funktionen und Klassen
- **Code-Stil**: PEP 8 strikt befolgen
- **Dateinamen**: Konsistente Benennung: `{original}_{method}.png`
- **Pfad-Handling**: Immer `os.path.join()` oder `pathlib.Path`
- **Fehler-Behandlung**: Try/except mit spezifischen Exception-Typen

### Framework-spezifische Guidelines

#### OpenCV-Konventionen
```python
import cv2
import numpy as np

def edge_detection_template(image_path: str, target_size: tuple[int, int]) -> np.ndarray:
    """Standard-Template für OpenCV-basierte Algorithmen."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Algorithmus-spezifische Verarbeitung hier
    result = cv2.Canny(image, 50, 150)  # Beispiel
    return standardize_output(result, target_size)
```

#### Streamlit-Konventionen
```python
import streamlit as st

def create_ui_component():
    """Wiederverwendbare UI-Komponenten in gui_components.py."""
    with st.container():
        col1, col2 = st.columns(2)
        # UI-Logik hier
```

#### PyTorch-Integration
```python
import torch
import kornia

def pytorch_edge_detection(image_path: str) -> torch.Tensor:
    """PyTorch/Kornia-basierte Implementierungen."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Modell-Laden und Inferenz
```

### Logging-Konventionen
- **CLI-Modus**: `print()` für Benutzer-Feedback
- **GUI-Modus**: `st.write()`, `st.error()`, `st.success()`
- **Debug-Modus**: Standard `logging`-Modul mit Level-basierter Ausgabe

## Testing-Anforderungen für KI-Agenten

### Test-Framework-Setup
```bash
# Test-Umgebung (Windows 11, Python 3.10 venv)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install pytest pytest-cov
```

### Test-Kategorien
```markdown
### Unit Tests
- **Framework**: pytest
- **Abdeckung**: Mindestens 80% für `detectors.py`
- **Dateimuster**: `test_*.py` im Root-Verzeichnis
- **Ausführung**: `pytest -v --cov=detectors`

### Integration Tests  
- **CLI-Tests**: Vollständige Pipeline mit Beispieldaten
- **GUI-Tests**: Streamlit-App-Start und Basis-Funktionalität
- **Model-Tests**: Download und Initialisierung

### Smoke Tests
- **Streamlit**: `streamlit run streamlit_app.py --headless --server.port 8888`
- **CLI**: `python run_edge_detectors.py --input_dir .\images --output_dir .\results`
```

### Test-Erstellung für neue Features
```python
import pytest
from detectors import run_new_algorithm

def test_new_algorithm_basic():
    """Template für neue Algorithmus-Tests."""
    result = run_new_algorithm("test_image.jpg", (512, 512))
    assert result is not None
    assert result.shape == (512, 512)
    assert result.dtype == np.uint8

def test_new_algorithm_standardization():
    """Sicherstellen, dass standardize_output aufgerufen wird."""
    # Test-Implementierung hier
```

## Pull Request Guidelines für KI-Agenten

### PR-Titel Format
```
[feat|fix|docs|style|refactor|test]: Kurze Beschreibung

Beispiele:
- feat: Add Sobel edge detection algorithm
- fix: Resolve PyTorch model loading on Windows 11
- docs: Update installation guide for Python 3.10
```

### PR-Beschreibung Template
```markdown
## Beschreibung
Kurze Zusammenfassung der Änderungen und deren Zweck.

## Typ der Änderung
- [ ] Neuer Edge-Detection-Algorithmus
- [ ] GUI-Komponente hinzugefügt/verbessert
- [ ] Bug Fix in bestehender Funktionalität
- [ ] Dokumentation/AGENTS.md Update
- [ ] Performance-Optimierung

## Windows 11 + Python 3.10 Kompatibilität
- [ ] Neue Dependencies haben offizielle Windows-Wheels für Python 3.10
- [ ] Manuelle Tests auf Windows 11 durchgeführt
- [ ] run.bat funktioniert mit Änderungen

## Testing
- [ ] Unit Tests hinzugefügt/aktualisiert
- [ ] CLI Smoke Test erfolgreich
- [ ] Streamlit GUI Test erfolgreich
- [ ] `pytest -v` läuft ohne Fehler

## Checklist für KI-Agenten
- [ ] `detectors.py`: Neuer Algorithmus folgt `run_<Name>()` Pattern
- [ ] `detectors.py`: `standardize_output()` wird aufgerufen
- [ ] `get_all_methods()`: Tuple alphabetisch eingefügt
- [ ] `gui_components.py`: Mapping für GUI-Support hinzugefügt
- [ ] `requirements.txt`: Neue Dependencies hinzugefügt (falls nötig)
- [ ] Code folgt PEP 8 und nutzt Type Hints
- [ ] Keine hardkodierten Pfadseparatoren
```

### Code Review Kriterien
- **Funktionalität**: Algorithmus produziert erwartete Edge-Maps
- **Integration**: Nahtlose Einbindung in CLI und GUI
- **Performance**: Angemessene Ausführungszeit auf Standard-Hardware
- **Wartbarkeit**: Klarer, dokumentierter Code

## Programmatische Checks für KI-Agenten

### Pre-Commit Validierung
```bash
# Vollständige Validierung vor Code-Submission
# In Python 3.10 venv ausführen:

# 1. Code-Stil prüfen
python -m flake8 . --max-line-length=88 --extend-ignore=E203,W503

# 2. Type Checking (optional, aber empfohlen)
python -m mypy detectors.py streamlit_app.py

# 3. Unit Tests ausführen
pytest -v --cov=detectors --cov-report=term-missing

# 4. CLI Smoke Test
python run_edge_detectors.py --input_dir .\images --output_dir .\results --methods Canny

# 5. GUI Smoke Test  
timeout 10 streamlit run streamlit_app.py --headless --server.port 8888

# 6. Requirements-Validierung
pip check
```

### Automatisierte CI/CD Pipeline
```yaml
# GitHub Actions - Windows-Latest (Win 11) Runner
- name: Setup Python 3.10
  uses: actions/setup-python@v4
  with:
    python-version: '3.10'
    
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install pytest flake8
    
- name: Run tests
  run: |
    flake8 .
    pytest -v
```

### Environment-Validierung
```python
# System-Check für KI-Agenten (vor Code-Ausführung)
import sys
import platform

def validate_environment():
    """Sicherstellen, dass Umgebung den Anforderungen entspricht."""
    assert sys.version_info >= (3, 10), "Python 3.10+ erforderlich"
    assert platform.system() == "Windows", "Windows-Umgebung erforderlich"
    assert platform.release() in ["10", "11"], "Windows 10/11 erforderlich"
```

## Erweiterungs-Guidelines für KI-Agenten

### Neue Edge-Detection-Algorithmen hinzufügen
```python
# 1. Funktion in detectors.py erstellen
def run_new_algorithm(image_path: str, target_size: tuple[int, int]) -> np.ndarray:
    """
    Neuer Edge-Detection-Algorithmus.
    
    Args:
        image_path: Pfad zum Eingabebild
        target_size: Ziel-Auflösung (width, height)
        
    Returns:
        Standardisierte Graustufenkarte
    """
    # Algorithmus-Implementierung hier
    result = your_algorithm_logic(image_path)
    
    # ZWINGEND: Standardisierung aufrufen
    return standardize_output(result, target_size)

# 2. In get_all_methods() alphabetisch einfügen
def get_all_methods():
    return (
        "Canny",
        "HED_PyTorch", 
        "New_Algorithm",  # <- Hier hinzufügen
        "Sobel",
        # ... weitere Methoden
    )

# 3. GUI-Support in gui_components.py
def method_selector_advanced():
    method_descriptions = {
        "New_Algorithm": "Beschreibung des neuen Algorithmus",
        # ... weitere Mappings
    }
```

### Neue Dependencies hinzufügen
```bash
# 1. Windows 11 + Python 3.10 Kompatibilität prüfen
pip install --dry-run new_package  # Prüfung ohne Installation

# 2. Offizielle Wheel-Verfügbarkeit validieren
# PyPI muss Windows-wheels für Python 3.10 bereitstellen

# 3. requirements.txt aktualisieren
echo "new_package>=1.0.0" >> requirements.txt

# 4. run.bat testen
run.bat
```

### GUI-Komponenten erweitern
```python
# gui_components.py - Neue wiederverwendbare Komponenten
def create_advanced_settings_panel():
    """Erweiterte Einstellungen für Algorithmen."""
    with st.expander("Erweiterte Einstellungen"):
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
        use_gpu = st.checkbox("GPU verwenden", value=True)
        
    return {"threshold": threshold, "use_gpu": use_gpu}
```

## Environment-Konfiguration

### Windows 11 System-Anforderungen
- **Mindest-Hardware**: TPM 2.0, 4 GB RAM, UEFI-Boot
- **Python-Installation**: Offizieller Installer von python.org (nicht Microsoft Store)
- **Architektur**: x86-64 (64-Bit) zwingend erforderlich

### Environment Variables
```bash
# Optionale Konfiguration
set EDGE_GPU=auto          # cpu|cuda|auto
set EDGE_RESULTS_DIR=.\results
set EDGE_MODEL_DIR=.\models
set STREAMLIT_SERVER_PORT=8501
```

### Development-Setup
```bash
# Empfohlene Entwicklungsumgebung
git clone [repository-url]
cd edge_detection_tool
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install pytest flake8 mypy  # Entwicklungstools
```

## Sicherheit und Performance

### Sicherheits-Guidelines
- **PyPI-Packages**: Nur signierte Wheels von vertrauenswürdigen Quellen
- **Model-Downloads**: HTTPS-URLs mit Checksum-Validierung
- **File-Handling**: Eingabe-Validierung für Bildpfade
- **Memory-Management**: Große Bilder in Chunks verarbeiten

### Performance-Optimierungen
```python
# Beispiel für optimierte Bildverarbeitung
import concurrent.futures
from multiprocessing import cpu_count

def batch_process_images(image_paths: list[str], method: str) -> list[np.ndarray]:
    """Parallelisierte Batch-Verarbeitung."""
    max_workers = min(cpu_count(), len(image_paths))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, path, method) 
                  for path in image_paths]
        return [future.result() for future in futures]
```

### Monitoring und Debugging
```python
# Logging-Setup für Produktionsumgebung
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('edge_detection.log'),
        logging.StreamHandler()
    ]
)
```

## Lizenz und Dependencies

### Open Source Compliance
- **Projekt-Lizenz**: MIT License
- **Deep Learning Models**: Ursprüngliche Upstream-Lizenzen beachten
- **Third-Party Libraries**: Kompatible Lizenzen (MIT, BSD, Apache 2.0)

### Dependency-Management
```txt
# requirements.txt - Pinned versions für Stabilität
opencv-contrib-python==4.8.1.78
torch==2.0.1
kornia==0.7.0
streamlit==1.28.1
requests==2.31.0
numpy==1.24.3
Pillow==10.0.1
```

---

**WICHTIG für KI-Agenten**: Diese AGENTS.md muss vor jeder Code-Änderung vollständig analysiert werden. Alle beschriebenen Strukturen, Konventionen und Windows 11/Python 3.10-Spezifikationen sind zwingend einzuhalten. Bei Unsicherheiten bezüglich Implementation-Details, diese Dokumentation als Referenz verwenden.


Edge Detection Studio ist eine Zero-Config-Suite zur Kantenerkennung mit 15 klassischen und Deep-Learning-Algorithmen. Das Projekt ist gezielt auf Windows 11 (64-Bit) – inklusive TPM-2.0-Sicherheitsvorgabe – und Python 3.10 ausgerichtet, weil alle Kern-Bibliotheken (OpenCV-Contrib, PyTorch, Kornia, Streamlit) offizielle Wheels für diese Plattform bereitstellen. (support.microsoft.com, python.org, pypi.org, discuss.pytorch.org, docs.streamlit.io) Die Codebasis bietet eine Batch-CLI, eine Streamlit-GUI, automatischen Modell-Download und einen klaren Erweiterungspfad. Dieses AGENTS.md vereint die Stärken beider bisherigen Versionen – kompakte Orientierung und tiefgehende Arbeitsrichtlinien – und folgt dabei den Best-Practice-Kapiteln des offiziellen Agents.md-Guides. (agentsmd.net)

1 · Projekt-One-Pager
Attribut	Wert
Name	Edge Detection Studio
Ziel-OS	Windows 11 22H2 + (x86-64) (theverge.com)
Referenz-Python	3.10.x (64-Bit) (python.org)
Kern-Features	Ein-Klick-Setup (run.bat), 15 Edge-Algorithmen, Auflösungs-Normalisierung & Invertierung, Streamlit-GUI
Primäre Nutzer	CV-Forschende · Designer · Analyst*innen · Studierende

2 · Repository-Topologie
edge_detection_tool/
├── detectors.py               # Algorithmen-Katalog
├── run_edge_detectors.py      # Batch-CLI
├── streamlit_app.py           # GUI-Hauptdatei
├── gui_components.py          # Wiederverwendbare Streamlit-Bausteine
├── run.bat                    # Windows-Bootstrap (venv, pip, GUI-Start)
├── requirements.txt           # Abhängigkeiten (Win 11 + Py 3.10 Wheels)
├── models/                    # DL-Gewichte (auto-Download)
├── images/                    # Beispiel-Bilder
└── results/                   # Pipeline-Ausgaben
Navigations-Tipps für Agents
Datei/Ordner	AI-Agent-Pflichtaufgaben
detectors.py	Neue Algorithmen hier implementieren → Funktion run_<Name>(path, target_size) muss standardize_output() aufrufen.
get_all_methods()	Alphabetisch erweitern, damit CLI & GUI die Methode kennen.
gui_components.py	UI-Bausteine hinzufügen / ändern – keine Inline-Widgets in streamlit_app.py.
models/	Download‐Cache, niemals manuell committen.

3 · Tech-Stack (Windows 11 + Python 3.10)
Layer	Lib / Tool	Min-Version	Win 11 · Py 3.10-Support
CV Core	OpenCV-Contrib	4.5	cp310-win_amd64 Wheel auf PyPI (pypi.org)
DL Runtime	PyTorch	1.9+	vorgebaute Win-Wheels (CPU oder CUDA) (discuss.pytorch.org)
GPU-Filter	Kornia	0.6+	100 % PyTorch-kompatibel (pypi.org)
GUI	Streamlit	1.28+	Offizielle Unterstützung bis Py 3.13 → 3.10 safe (docs.streamlit.io)
Security	pip-audit	–	führt CVE-Scan vor Release durch (pypi.org)
CI	GitHub Actions	windows-latest (Win 11 Runner) (docs.github.com)	

4 · Build- & Run-Workflows
4.1 One-Click-Setup (End-User)
:: Voraussetzung: python --version  →  3.10.x 64-Bit
run.bat      :: erstellt venv, installiert Wheels, lädt Modelle, startet GUI (Port 8501)
4.2 CLI-Batch
python run_edge_detectors.py `
       --input_dir .\images `
       --output_dir .\results `
       --methods HED_PyTorch Kornia_Canny

5 · Coding-Konventionen
Regel	Detail
Sprache	Python 3.10-only Syntax (z. B. match/case).
Stil	PEP 8 + Black-kompatible 88 Spalten; Typ-Hints Pflicht.
Dateinamen	Ergebnis-PNG: {original}_{method}.png.
Pfad-Handling	Immer pathlib.Path oder os.path.join, kein Backslash-Hardcoding.
Logging	CLI → print() / logging; GUI → st.write() u. Ä.
Fehler	try/except Exception as e + Log-Eintrag; GUI zeigt st.error.
Python-Snippet-Template
def run_new_algo(path: str, target_size: tuple[int, int]) -> np.ndarray:
    """Neuer Edge-Detector: Aufruf via get_all_methods()."""
    # … Algorithmus …
    return standardize_output(result, target_size)

6 · Testing- & CI-Matrix
Ebene	Tool / Befehl	Mindest-Kriterium
Lint	flake8 .	0 Error
Types	mypy detectors.py	keine error:-Zeilen
Unit	pytest -q --cov=detectors	≥ 80 % Coverage
Smoke CLI	python run_edge_detectors.py --input_dir images --output_dir tmp	Rückgabecode 0
Smoke GUI	streamlit run streamlit_app.py --headless -p 8888	HTTP 200 in 30 s
GitHub Actions nutzt windows-latest Runner, Python 3.10, setzt obige Checks um und lädt den Ordner results/edge_detection_results als Artefakt hoch. (docs.github.com)

7 · Pull-Request-Leitfaden

Titel → feat|fix|docs|refactor|test: ⟨Kurzbeschreibung⟩


PR-Beschreibung: Vorlage aus deiner vorherigen Version (Check-Box-Liste).


One Feature per PR – vermeide Sammel-Commits.


CI muss grün sein (Lint + Type + PyTest + Smoke).


Screenshot/Clip bei GUI-Änderungen beilegen.


8 · Security & Performance

Hardware → TPM 2.0 & aktuelle CPUs erforderlich; Microsoft hält daran fest. (theverge.com)


Dependency Audit → pip-audit in CI Pipeline. (pypi.org)


Model-Cache → %PROJECT_ROOT%\models; nicht versionieren.


Parallelisierung → concurrent.futures.ThreadPoolExecutor ≤ CPU-Kerne nutzen.


9 · Environment-Variablen
Var	Default	Zweck
EDGE_GPU	auto	cpu / cuda / auto-Detect
EDGE_RESULTS_DIR	.\results	Output-Root überschreiben
EDGE_MODEL_DIR	.\models	DL-Gewichte-Cache

10 · Erweiterungs-Workflow (TL;DR für Agents)
1.
Code → Neue Funktion in detectors.py, standardize_output() aufrufen.
2.
3.
Registrieren → Tuple in get_all_methods() (Alphabet!).
4.
5.
GUI → Mapping in method_selector_advanced() ergänzen.
6.
7.
Tests → Mind. ein Unit-Test hinzufügen.
8.
9.
Docs → AGENTS.md und README.md ggf. updaten.
10.

Schlusswort
Jeder KI-Agent muss dieses Dokument vor Änderungen lesen und die Vorgaben strikt einhalten – nur so bleiben Build-Stabilität, Windows-11-Kompatibilität und Code-Qualität gewährleistet.
