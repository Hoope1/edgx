@echo off
REM 1) Virtuelle Umgebung anlegen (falls nicht vorhanden)
IF NOT EXIST venv (
    python -m venv venv
)

REM 2) Aktivieren
call venv\Scripts\activate

REM 3) Requirements installieren (KEIN pip‑Upgrade, verhindert WinError 5)
python -m pip install -r requirements.txt

REM 4) Modelle herunterladen
python detectors.py --init-models

REM 5) Batch‑Kantenerkennung
python run_edge_detectors.py --input_dir images --output_dir results
pause