@echo off
echo ========================================
echo  🎨  Edge Detection Studio Setup
echo ========================================

:: -------------------------------------------------
:: 1) Python-Verfügbarkeit prüfen
:: -------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌  Python nicht gefunden. Bitte installieren!
    pause & exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version') do set PYVER=%%v
echo ✅  Python %%PYVER%% gefunden

:: -------------------------------------------------
:: 2) Virtuelle Umgebung
:: -------------------------------------------------
if not exist venv (
    echo 📦  Erstelle venv …
    python -m venv venv || (echo Fehler & pause & exit /b 1)
) else (
    echo ✅  venv vorhanden
)

call venv\Scripts\activate

:: -------------------------------------------------
:: 3) pip updaten & Requirements
:: -------------------------------------------------
echo 📚  Installiere Requirements …
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt || goto :req_fallback
goto :req_ok
:req_fallback
echo ⚠️  Sammel-Installation fehlgeschlagen – installiere Kernpakete …
python -m pip install streamlit opencv-python opencv-contrib-python torch torchvision kornia requests pillow numpy git+https://github.com/Hoope1/pytorch-hed@v0.5.1
:req_ok

:: -------------------------------------------------
:: 4) Modelle
:: -------------------------------------------------
python -m edgx.detectors --init-models

:: -------------------------------------------------
:: 5) Verzeichnisstruktur
:: -------------------------------------------------
if not exist images  mkdir images
if not exist results mkdir results
if not exist models  mkdir models

:: -------------------------------------------------
:: 6) Starte Streamlit
:: -------------------------------------------------
echo 🚀  Starte GUI …
streamlit run src\edgx\streamlit_app.py --server.headless false --server.port 8501
echo 👋  beendet – bye
pause
