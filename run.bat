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
:: 3) pip updaten & Package installieren
:: -------------------------------------------------
echo 📚  Installiere lokales edgx Package...
python -m pip install --upgrade pip --quiet
python -m pip install -e . --quiet || goto :package_error

echo 📚  Installiere Requirements …
python -m pip install -r requirements.txt || goto :req_fallback
goto :req_ok

:package_error
echo ❌  Lokales Package konnte nicht installiert werden
echo Versuche alternative Installation...
python -m pip install --editable . || (
    echo ❌  Package-Installation fehlgeschlagen
    pause & exit /b 1
)

:req_fallback
echo ⚠️  Sammel-Installation fehlgeschlagen – installiere Kernpakete …
python -m pip install streamlit opencv-python opencv-contrib-python torch torchvision kornia requests pillow numpy matplotlib scikit-image

:req_ok

:: -------------------------------------------------
:: 4) Modelle (nur wenn edgx verfügbar ist)
:: -------------------------------------------------
echo 🔧  Initialisiere Modelle...
python -c "
try:
    from edgx.detectors import init_models
    init_models()
    print('✅ Modelle initialisiert')
except Exception as e:
    print(f'⚠️  Modell-Initialisierung übersprungen: {e}')
" || echo ⚠️  Modell-Initialisierung fehlgeschlagen

:: -------------------------------------------------
:: 5) Verzeichnisstruktur
:: -------------------------------------------------
if not exist images  mkdir images
if not exist results mkdir results
if not exist models  mkdir models

:: -------------------------------------------------
:: 6) Teste Installation
:: -------------------------------------------------
echo 🧪  Teste Installation...
python -c "
try:
    from edgx.detectors import get_all_methods
    methods = get_all_methods()
    print(f'✅ edgx Package OK - {len(methods)} Methoden verfügbar')
except Exception as e:
    print(f'❌ edgx Import fehlgeschlagen: {e}')
    exit(1)
"

if errorlevel 1 (
    echo ❌  Installation unvollständig - bitte Fehler prüfen
    pause & exit /b 1
)

:: -------------------------------------------------
:: 7) Starte Streamlit
:: -------------------------------------------------
echo 🚀  Starte GUI …
streamlit run src\edgx\streamlit_app.py --server.headless false --server.port 8501 || (
    echo ❌ Streamlit-Start fehlgeschlagen, versuche alternative Methode...
    python -m streamlit run src\edgx\streamlit_app.py --server.headless false --server.port 8501 || (
        echo ❌ GUI konnte nicht gestartet werden
        echo Versuchen Sie manuell: streamlit run src\edgx\streamlit_app.py
        pause & exit /b 1
    )
)

echo 👋  GUI beendet
pause
