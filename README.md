# Edge Detection Studio

Edge Detection Studio bietet eine sofort einsatzfähige Sammlung klassischer und Deep-Learning-basierter Kantendetektionsverfahren. Die Anwendung richtet sich an Forschende, Designerinnen sowie Studierende, die ohne manuelle Konfiguration Edge-Maps aus Bildern erzeugen möchten. Offiziell unterstützt wird Windows 11 mit Python 3.10.

## 1. Projektbeschreibung & Zielgruppe

Das Projekt löst das Problem, verschiedene Edge-Detection-Algorithmen konsistent auf beliebig viele Bilder anzuwenden. Hauptnutzer sind Computer-Vision-Forschende, Designer*innen und alle, die schnell qualitativ hochwertige Edge-Maps benötigen. Neben einer komfortablen Streamlit-Oberfläche steht ein vollautomatischer CLI-Modus zur Verfügung.

## 2. Features

- Unterstützung von **15 Algorithmen** (Ausschnitt aus `detectors.py`):
```python
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
```
- Einheitliche Normalisierung, Invertierung und Skalierung:
```python
# Gemeinsame Bild-Normalisierung & Hilfsfunktionen
def standardize_output(edge_map: np.ndarray,
                       target_size: tuple | None = None,
                       invert: bool = True) -> np.ndarray:
    """
    Vereinheitlicht die Ausgaben aller Edge-Methoden
    • Invert: weißer Hintergrund, dunkle Kanten
    • Resize: skaliert (CUBIC) auf `target_size`, falls angegeben
    • uint8 garantiert
    """
    if edge_map.dtype != np.uint8:
        edge_map = (edge_map * 255 if edge_map.max() <= 1.0
                    else edge_map).astype(np.uint8)
    if invert:
        edge_map = 255 - edge_map
    if target_size is not None:
        edge_map = cv2.resize(edge_map, target_size,
                              interpolation=cv2.INTER_CUBIC)
    return edge_map
```
- Streamlit-GUI mit Tabs *Bildauswahl → Methoden → Einstellungen → Verarbeitung → Vorschau*
- Fortschrittsanzeige, ETA-Berechnung und ZIP-Export der Ergebnisse
- CLI-Tool `run_edge_detectors.py`
- Automatischer Modell-Download für HED, Structured Forests und BDCN

## 3. Installation (Windows 11 empfohlen)

1. **Python 3.10** installieren und in der Eingabeaufforderung verfügbar machen.
2. `run.bat` ausführen (Ausschnitt):
```bat
:: 1) Python-Verfügbarkeit prüfen
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌  Python nicht gefunden. Bitte installieren!
    pause & exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version') do set PYVER=%%v
echo ✅  Python %%PYVER%% gefunden

:: 2) Virtuelle Umgebung
if not exist venv (
    echo 📦  Erstelle venv …
    python -m venv venv || (echo Fehler & pause & exit /b 1)
) else (
    echo ✅  venv vorhanden
)
call venv\Scripts\activate

:: 3) pip updaten & Requirements
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt || goto :req_fallback
:req_fallback
echo ⚠️  Sammel-Installation fehlgeschlagen – installiere Kernpakete …
python -m pip install streamlit opencv-python opencv-contrib-python torch torchvision kornia requests pillow numpy pytorch-hed
:req_ok

:: 4) Modelle
python detectors.py --init-models

:: 5) Verzeichnisstruktur
if not exist images  mkdir images
if not exist results mkdir results
if not exist models  mkdir models

:: 6) Starte Streamlit
streamlit run streamlit_app.py --server.headless false --server.port 8501
```
   Bei Problemen mit `pytorch-hed` gegebenenfalls `setup.py` auf `python_requires='>=3.7'` anpassen. Sollte `opencv-python==4.5.0.52` fehlen, kann eine aktuelle Version verwendet werden.
3. Die GUI ist anschließend unter http://localhost:8501 erreichbar.

## 4. Nutzung

### Streamlit-GUI
1. `run.bat` ausführen oder innerhalb der venv `streamlit run streamlit_app.py` starten.
2. GUI-Tabs:
```
[Bildauswahl] [Methoden] [Einstellungen] [Verarbeitung] [Vorschau]
```
3. Bilder auswählen, Methoden ankreuzen und **VERARBEITUNG STARTEN** klicken.
4. Nach Abschluss kann ein ZIP aller PNGs heruntergeladen werden.

### CLI
```bash
python run_edge_detectors.py --input_dir images --output_dir results --methods Kornia_Canny HED_PyTorch
```
Die Ergebnisse liegen unter `results/edge_detection_results` als `{bildname}_{algorithmus}.png`. Die Datei `processing_summary.txt` fasst Auflösung und Methoden zusammen.

## 5. Architektur & Code-Struktur
- **detectors.py** – Algorithmen, Modell-Downloads und Hilfsfunktionen
- **gui_components.py** – wiederverwendbare Widgets (Folder-Picker, Batch-Prozessor)
- **streamlit_app.py** – fünf Tabs, Interaktion, ZIP-Export
- **run_edge_detectors.py** – Batch-CLI
- **validate_environment.py** prüft die Plattform:
```python
def validate_environment():
    """Sicherstellen, dass Umgebung den Anforderungen entspricht."""
    assert sys.version_info >= (3, 10), "Python 3.10+ erforderlich"
    assert platform.system() == "Windows", "Windows-Umgebung erforderlich"
    assert platform.release() in ["10", "11"], "Windows 10/11 erforderlich"
```
- **tools/check_all.sh** automatisiert Linting, Typprüfung und Tests:
```bash
#!/bin/bash
echo "✅ Linting mit flake8"
flake8 .

echo "✅ Formatieren mit black"
black .

echo "✅ Sortieren mit isort"
isort .

echo "✅ Typprüfung mit mypy"
mypy .

echo "✅ Tests mit pytest"
pytest --cov=detectors
```
Der Datenfluss lautet: GUI/CLI → Bild- & Methodenwahl → Verarbeitung in `detectors.py` → Ausgabe in `results/edge_detection_results` → ZIP-Export und `processing_summary.txt`.

## 6. Entwicklungsrichtlinien & Tests
- Code muss PEP 8 entsprechen und Typ-Hints nutzen.
- `tools/check_all.sh` führt `flake8`, `black`, `isort`, `mypy` und `pytest` aus.
- Momentan sammelt `pytest` keine Tests ("no tests collected").
- Offiziell werden nur Windows 10/11 und Python 3.10 unterstützt.
- Für lokale Checks kann `pre-commit` verwendet werden:
  ```bash
  pip install pre-commit
  pre-commit install
  ```

## 7. Erweiterung
1. Neue Methode als `run_<Name>(path, target_size)` in `detectors.py` implementieren und `standardize_output()` aufrufen.
2. Alphabetisch in `get_all_methods()` registrieren.
3. GUI-Mapping in `method_selector_advanced()` hinzufügen.
4. Tests, Dokumentation und ggf. Screenshots beisteuern.

## 8. Bekannte Probleme
- `pytorch-hed` benötigt eventuell einen Fix des `setup.py` (`python_requires='>=3.7'`).
- `opencv-python==4.5.0.52` ist nicht mehr verfügbar – aktuelle Version verwenden.
- Der `streamlit`-Befehl muss im `PATH` liegen, sonst startet die GUI nicht.

## 9. Lizenz, Autor:innen & Mitwirken
Falls keine andere Lizenzdatei vorhanden ist, wird die **MIT License** empfohlen. Beiträge müssen die Regeln aus `AGENTS.md` einhalten.

## 10. Beispielausgabe
Beispielbilder befinden sich im Ordner `images/`. Nach der Verarbeitung liegen alle Edge-Maps unter `results/edge_detection_results/` und können als ZIP heruntergeladen werden.
