#!/bin/bash
# =============================================================================
# Edge Detection Studio - Vollständige Code-Qualitäts-Checks
# =============================================================================
#
# Dieses Skript führt alle Code-Qualitäts-Checks für Edge Detection Studio aus.
# Kompatibel mit Windows (WSL/Git Bash), Linux und macOS.
#
# Nutzung:
#   ./tools/check_all.sh
#   ./tools/check_all.sh --fix    # Automatische Fixes wo möglich
#   ./tools/check_all.sh --fast   # Nur schnelle Checks
#   ./tools/check_all.sh --help   # Hilfe anzeigen
#
# =============================================================================

set -e  # Exit bei ersten Fehler
set -u  # Exit bei undefined variables

# =============================================================================
# Konfiguration und Variablen
# =============================================================================

# Script-Verzeichnis ermitteln (funktioniert auch bei Symlinks)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Farben für Output (falls Terminal unterstützt)
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
    RED=$(tput setaf 1)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    BLUE=$(tput setaf 4)
    MAGENTA=$(tput setaf 5)
    CYAN=$(tput setaf 6)
    WHITE=$(tput setaf 7)
    BOLD=$(tput bold)
    NC=$(tput sgr0)  # No Color
else
    RED="" GREEN="" YELLOW="" BLUE="" MAGENTA="" CYAN="" WHITE="" BOLD="" NC=""
fi

# Konfiguration
DEFAULT_PYTHON="python"
VENV_PATH="${PROJECT_ROOT}/venv"
SRC_DIR="${PROJECT_ROOT}/src"
TESTS_DIR="${PROJECT_ROOT}/tests"
TOOLS_DIR="${PROJECT_ROOT}/tools"

# Check-Flags
RUN_FORMATTING=true
RUN_LINTING=true
RUN_TYPE_CHECKING=true
RUN_TESTING=true
RUN_SECURITY=true
RUN_DOCUMENTATION=true
AUTO_FIX=false
FAST_MODE=false
VERBOSE=false

# Ergebnis-Tracking
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
declare -a FAILED_CHECK_NAMES=()

# =============================================================================
# Hilfsfunktionen
# =============================================================================

print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "==============================================================================="
    echo "  🎨 Edge Detection Studio - Code Quality Checks"
    echo "==============================================================================="
    echo -e "${NC}"
}

print_section() {
    local section_name="$1"
    echo -e "\n${BLUE}${BOLD}📋 ${section_name}${NC}"
    echo -e "${BLUE}$(printf '%.78s' "$(printf '%*s' 78 '' | tr ' ' '-')")${NC}"
}

print_step() {
    local step_name="$1"
    local step_num="$2"
    local total_steps="$3"
    echo -e "\n${MAGENTA}▶ [${step_num}/${total_steps}] ${step_name}${NC}"
}

print_success() {
    local message="$1"
    echo -e "${GREEN}✅ ${message}${NC}"
    ((PASSED_CHECKS++))
}

print_error() {
    local message="$1"
    echo -e "${RED}❌ ${message}${NC}"
    ((FAILED_CHECKS++))
}

print_warning() {
    local message="$1"
    echo -e "${YELLOW}⚠️  ${message}${NC}"
}

print_info() {
    local message="$1"
    echo -e "${CYAN}ℹ️  ${message}${NC}"
}

# Führt einen Check aus und tracked das Ergebnis
run_check() {
    local check_name="$1"
    local check_command="$2"
    local allow_failure="${3:-false}"
    
    ((TOTAL_CHECKS++))
    
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${WHITE}Ausführen: ${check_command}${NC}"
    fi
    
    if eval "$check_command"; then
        print_success "$check_name"
        return 0
    else
        if [[ "$allow_failure" == true ]]; then
            print_warning "$check_name (optional - Fehler ignoriert)"
            ((PASSED_CHECKS++))
            return 0
        else
            print_error "$check_name"
            FAILED_CHECK_NAMES+=("$check_name")
            return 1
        fi
    fi
}

# Prüft ob ein Kommando verfügbar ist
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Findet Python-Executable
find_python() {
    local python_cmd=""
    
    # Prüfe Virtual Environment zuerst
    if [[ -f "${VENV_PATH}/Scripts/python.exe" ]]; then
        python_cmd="${VENV_PATH}/Scripts/python.exe"  # Windows
    elif [[ -f "${VENV_PATH}/bin/python" ]]; then
        python_cmd="${VENV_PATH}/bin/python"  # Unix
    elif command_exists python3; then
        python_cmd="python3"
    elif command_exists python; then
        python_cmd="python"
    else
        echo -e "${RED}❌ Python nicht gefunden!${NC}"
        exit 1
    fi
    
    echo "$python_cmd"
}

# Aktiviert Virtual Environment falls vorhanden
activate_venv() {
    if [[ -f "${VENV_PATH}/Scripts/activate" ]]; then
        # Windows
        source "${VENV_PATH}/Scripts/activate"
        print_info "Virtual Environment aktiviert (Windows)"
    elif [[ -f "${VENV_PATH}/bin/activate" ]]; then
        # Unix
        source "${VENV_PATH}/bin/activate"
        print_info "Virtual Environment aktiviert (Unix)"
    else
        print_warning "Kein Virtual Environment gefunden"
    fi
}

# Prüft Python-Version
check_python_version() {
    local python_cmd="$1"
    local version_output
    version_output="$($python_cmd --version 2>&1)"
    
    if [[ "$version_output" =~ Python\ 3\.([0-9]+) ]]; then
        local minor_version="${BASH_REMATCH[1]}"
        if [[ "$minor_version" -ge 10 ]]; then
            print_success "Python Version: $version_output"
            return 0
        else
            print_error "Python 3.10+ erforderlich, gefunden: $version_output"
            return 1
        fi
    else
        print_error "Ungültige Python-Version: $version_output"
        return 1
    fi
}

# =============================================================================
# Check-Funktionen
# =============================================================================

setup_environment() {
    print_section "Umgebung einrichten"
    
    # Ins Projektverzeichnis wechseln
    cd "$PROJECT_ROOT"
    print_info "Arbeitsverzeichnis: $PROJECT_ROOT"
    
    # Python finden und Version prüfen
    PYTHON_CMD="$(find_python)"
    check_python_version "$PYTHON_CMD"
    
    # Virtual Environment aktivieren
    activate_venv
    
    # Umgebung validieren
    if [[ -f "${SRC_DIR}/edgx/validate_environment.py" ]]; then
        run_check "Umgebungsvalidierung" \
            "$PYTHON_CMD -m edgx.validate_environment --critical-only" \
            true
    fi
}

check_project_structure() {
    print_section "Projekt-Struktur validieren"
    
    # Wichtige Dateien prüfen
    local required_files=(
        "setup.py"
        "pyproject.toml"
        "requirements.txt"
        "src/edgx/__init__.py"
        "src/edgx/detectors.py"
        "src/edgx/streamlit_app.py"
        "src/edgx/run_edge_detectors.py"
    )
    
    local missing_files=()
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -eq 0 ]]; then
        print_success "Alle erforderlichen Dateien vorhanden"
    else
        print_error "Fehlende Dateien: ${missing_files[*]}"
        return 1
    fi
}

check_dependencies() {
    print_section "Dependencies prüfen"
    
    # Package-Installation prüfen
    run_check "edgx Package Import" \
        "$PYTHON_CMD -c 'import edgx; print(f\"edgx v{edgx.__version__} verfügbar\")'"
    
    # Kritische Dependencies
    local core_deps=("cv2" "torch" "streamlit" "numpy")
    for dep in "${core_deps[@]}"; do
        run_check "Dependency: $dep" \
            "$PYTHON_CMD -c 'import $dep; print(f\"$dep OK\")'" \
            true
    done
    
    # Pip check für Dependency-Konflikte
    if command_exists pip; then
        run_check "Dependency-Konflikte prüfen" \
            "pip check" \
            true
    fi
}

run_formatting_checks() {
    if [[ "$RUN_FORMATTING" != true ]]; then
        return 0
    fi
    
    print_section "Code-Formatierung"
    
    # Black
    if command_exists black; then
        if [[ "$AUTO_FIX" == true ]]; then
            run_check "Black (Auto-Fix)" \
                "black src/ tests/ --line-length=88"
        else
            run_check "Black Check" \
                "black src/ tests/ --check --line-length=88"
        fi
    else
        print_warning "Black nicht installiert - überspringe"
    fi
    
    # isort
    if command_exists isort; then
        if [[ "$AUTO_FIX" == true ]]; then
            run_check "isort (Auto-Fix)" \
                "isort src/ tests/ --profile=black"
        else
            run_check "isort Check" \
                "isort src/ tests/ --check-only --profile=black"
        fi
    else
        print_warning "isort nicht installiert - überspringe"
    fi
}

run_linting_checks() {
    if [[ "$RUN_LINTING" != true ]]; then
        return 0
    fi
    
    print_section "Linting"
    
    # flake8
    if command_exists flake8; then
        run_check "flake8" \
            "flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,E266,E501,W503,E402,E731,E722,E741,E265,W504"
    else
        print_warning "flake8 nicht installiert - überspringe"
    fi
    
    # pylint (optional)
    if command_exists pylint && [[ "$FAST_MODE" != true ]]; then
        run_check "pylint" \
            "pylint src/edgx/ --disable=C0114,C0115,C0116,R0903,R0913,W0613" \
            true
    fi
}

run_type_checking() {
    if [[ "$RUN_TYPE_CHECKING" != true ]]; then
        return 0
    fi
    
    print_section "Type-Checking"
    
    if command_exists mypy; then
        run_check "mypy" \
            "mypy src/edgx/ --config-file=setup.cfg --ignore-missing-imports"
    else
        print_warning "mypy nicht installiert - überspringe"
    fi
}

run_security_checks() {
    if [[ "$RUN_SECURITY" != true ]]; then
        return 0
    fi
    
    print_section "Security-Checks"
    
    # bandit
    if command_exists bandit; then
        run_check "bandit Security Scan" \
            "bandit -r src/ -f json -o bandit-report.json" \
            true
    else
        print_warning "bandit nicht installiert - überspringe"
    fi
    
    # safety (dependency vulnerabilities)
    if command_exists safety && [[ "$FAST_MODE" != true ]]; then
        run_check "safety Vulnerability Scan" \
            "safety check --json" \
            true
    fi
}

run_tests() {
    if [[ "$RUN_TESTING" != true ]]; then
        return 0
    fi
    
    print_section "Tests ausführen"
    
    if command_exists pytest; then
        # Basis-Tests
        run_check "pytest Basis-Tests" \
            "pytest tests/test_basic.py -v --tb=short"
        
        # Erweiterte Tests (falls nicht im Fast-Modus)
        if [[ "$FAST_MODE" != true ]]; then
            run_check "pytest Detector-Tests" \
                "pytest tests/test_detectors.py -v --tb=short -m 'not slow and not gpu'" \
                true
            
            # Coverage-Report
            if command_exists coverage; then
                run_check "Test Coverage" \
                    "pytest tests/ --cov=edgx --cov-report=term-missing --cov-fail-under=70" \
                    true
            fi
        fi
    else
        print_warning "pytest nicht installiert - überspringe Tests"
    fi
}

run_functionality_tests() {
    print_section "Funktionalitäts-Tests"
    
    # edgx Module Test
    run_check "edgx Funktions-Test" \
        "$PYTHON_CMD -m edgx.detectors --test"
    
    # CLI Tool Test
    if [[ -d "images" ]] && [[ "$FAST_MODE" != true ]]; then
        run_check "CLI Tool Test" \
            "$PYTHON_CMD -m edgx.run_edge_detectors --input_dir images --dry-run --methods Laplacian" \
            true
    fi
    
    # GUI Import Test
    run_check "GUI Import Test" \
        "$PYTHON_CMD -c 'from edgx.streamlit_app import *; print(\"GUI imports OK\")'"
}

run_documentation_checks() {
    if [[ "$RUN_DOCUMENTATION" != true ]]; then
        return 0
    fi
    
    print_section "Dokumentation"
    
    # README.md Existenz
    if [[ -f "README.md" ]]; then
        print_success "README.md vorhanden"
    else
        print_error "README.md fehlt"
    fi
    
    # AGENTS.md für Entwickler
    if [[ -f "AGENTS.md" ]]; then
        print_success "AGENTS.md vorhanden"
    else
        print_warning "AGENTS.md fehlt (empfohlen für Entwickler)"
    fi
    
    # Docstring-Coverage (optional)
    if command_exists docstr-coverage && [[ "$FAST_MODE" != true ]]; then
        run_check "Docstring Coverage" \
            "docstr-coverage src/edgx/ --fail-under=50" \
            true
    fi
}

run_pre_commit_checks() {
    print_section "Pre-commit Hooks"
    
    if command_exists pre-commit; then
        if [[ "$AUTO_FIX" == true ]]; then
            run_check "pre-commit (alle Dateien)" \
                "pre-commit run --all-files"
        else
            run_check "pre-commit Check" \
                "pre-commit run --all-files --show-diff-on-failure"
        fi
    else
        print_warning "pre-commit nicht installiert - empfohlen für Entwicklung"
    fi
}

# =============================================================================
# Hauptfunktion
# =============================================================================

show_help() {
    cat << EOF
${BOLD}Edge Detection Studio - Code Quality Checker${NC}

${BOLD}NUTZUNG:${NC}
    $0 [OPTIONEN]

${BOLD}OPTIONEN:${NC}
    --fix           Automatische Fixes wo möglich (black, isort)
    --fast          Nur schnelle Checks (überspringt langsame Tests)
    --no-format     Überspringe Formatierungs-Checks
    --no-lint       Überspringe Linting
    --no-type       Überspringe Type-Checking
    --no-test       Überspringe Tests
    --no-security   Überspringe Security-Checks
    --no-docs       Überspringe Dokumentations-Checks
    --verbose       Zeige alle ausgeführten Kommandos
    --help          Zeige diese Hilfe

${BOLD}BEISPIELE:${NC}
    $0                          # Alle Checks ausführen
    $0 --fix                    # Alle Checks mit Auto-Fix
    $0 --fast                   # Nur schnelle Checks
    $0 --no-test --no-security  # Ohne Tests und Security-Checks
    $0 --verbose --fix          # Verbose Modus mit Auto-Fix

${BOLD}ABHÄNGIGKEITEN:${NC}
    Installiere alle Development-Tools mit:
    pip install -e .[dev]
    pre-commit install

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --fix)
                AUTO_FIX=true
                shift
                ;;
            --fast)
                FAST_MODE=true
                shift
                ;;
            --no-format)
                RUN_FORMATTING=false
                shift
                ;;
            --no-lint)
                RUN_LINTING=false
                shift
                ;;
            --no-type)
                RUN_TYPE_CHECKING=false
                shift
                ;;
            --no-test)
                RUN_TESTING=false
                shift
                ;;
            --no-security)
                RUN_SECURITY=false
                shift
                ;;
            --no-docs)
                RUN_DOCUMENTATION=false
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}Unbekannte Option: $1${NC}"
                echo "Nutzen Sie --help für Hilfe."
                exit 1
                ;;
        esac
    done
}

show_summary() {
    echo -e "\n${CYAN}${BOLD}"
    echo "==============================================================================="
    echo "  📊 Zusammenfassung"
    echo "==============================================================================="
    echo -e "${NC}"
    
    echo -e "${BOLD}Gesamt:${NC} $TOTAL_CHECKS Checks"
    echo -e "${GREEN}${BOLD}Erfolgreich:${NC} $PASSED_CHECKS"
    echo -e "${RED}${BOLD}Fehlgeschlagen:${NC} $FAILED_CHECKS"
    
    if [[ $FAILED_CHECKS -gt 0 ]]; then
        echo -e "\n${RED}${BOLD}Fehlgeschlagene Checks:${NC}"
        for check in "${FAILED_CHECK_NAMES[@]}"; do
            echo -e "${RED}  ❌ $check${NC}"
        done
        
        echo -e "\n${YELLOW}${BOLD}Nächste Schritte:${NC}"
        echo -e "${YELLOW}  1. Beheben Sie die oben genannten Probleme${NC}"
        echo -e "${YELLOW}  2. Führen Sie die Checks erneut aus${NC}"
        echo -e "${YELLOW}  3. Verwenden Sie --fix für automatische Korrekturen${NC}"
        
        return 1
    else
        echo -e "\n${GREEN}${BOLD}🎉 Alle Checks erfolgreich!${NC}"
        echo -e "${GREEN}Code ist bereit für Commit/Push.${NC}"
        return 0
    fi
}

main() {
    # Argument-Parsing
    parse_arguments "$@"
    
    # Banner anzeigen
    print_banner
    
    if [[ "$AUTO_FIX" == true ]]; then
        print_info "Auto-Fix Modus aktiviert"
    fi
    
    if [[ "$FAST_MODE" == true ]]; then
        print_info "Fast Modus aktiviert - langsame Checks übersprungen"
    fi
    
    # Umgebung einrichten
    setup_environment
    
    # Projekt-Struktur prüfen
    check_project_structure
    
    # Dependencies prüfen
    check_dependencies
    
    # Code-Qualitäts-Checks
    run_formatting_checks
    run_linting_checks
    run_type_checking
    
    # Security
    run_security_checks
    
    # Tests
    run_tests
    run_functionality_tests
    
    # Dokumentation
    run_documentation_checks
    
    # Pre-commit (falls verfügbar)
    run_pre_commit_checks
    
    # Zusammenfassung anzeigen
    show_summary
}

# =============================================================================
# Script-Ausführung
# =============================================================================

# Trap für Cleanup bei Unterbrechung
cleanup() {
    echo -e "\n${YELLOW}Script unterbrochen.${NC}"
    exit 130
}
trap cleanup INT TERM

# Hauptfunktion ausführen
main "$@"
