"""
Environment validation utilities für Edge Detection Studio.

Dieses Modul bietet umfassende Validierung der Laufzeitumgebung,
Dependencies und Hardware-Anforderungen.
"""

import os
import platform
import sys
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess

logger = logging.getLogger(__name__)

# Mindestanforderungen
MIN_PYTHON_VERSION = (3, 10)
RECOMMENDED_PYTHON_VERSION = (3, 11)

# Unterstützte Plattformen
SUPPORTED_PLATFORMS = {
    "Windows": ["10", "11"],
    "Linux": ["Ubuntu", "Debian", "CentOS", "RHEL", "Fedora"],
    "Darwin": ["macOS"],  # macOS
}

# Kern-Dependencies
CORE_DEPENDENCIES = [
    ("cv2", "opencv-python", ">=4.5.0", True),
    ("torch", "torch", ">=1.9.0", True),
    ("streamlit", "streamlit", ">=1.28.0", True),
    ("numpy", "numpy", ">=1.21.0", True),
    ("PIL", "pillow", ">=8.0.0", True),
    ("requests", "requests", ">=2.25.0", True),
]

# Optionale Dependencies
OPTIONAL_DEPENDENCIES = [
    ("kornia", "kornia", ">=0.6.0", False),
    ("matplotlib", "matplotlib", ">=3.3.0", False),
    ("sklearn", "scikit-learn", ">=0.24.0", False),
    ("plotly", "plotly", ">=5.0.0", False),
    ("jupyter", "jupyter", ">=1.0.0", False),
]


class EnvironmentValidator:
    """
    Umfassende Validierung der Laufzeitumgebung für Edge Detection Studio.
    """
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: Dict[str, Any] = {}
        
    def validate_all(self) -> Dict[str, Any]:
        """
        Führt alle Validierungen durch.
        
        Returns:
            Dictionary mit vollständigen Umgebungsinformationen
        """
        logger.info("🔍 Starte Umgebungsvalidierung...")
        
        # Grundlegende System-Informationen
        self._collect_system_info()
        
        # Python-Version validieren
        self._validate_python_version()
        
        # Plattform validieren
        self._validate_platform()
        
        # Dependencies validieren
        self._validate_dependencies()
        
        # Hardware-Informationen sammeln
        self._collect_hardware_info()
        
        # GPU-Verfügbarkeit prüfen
        self._check_gpu_availability()
        
        # Speicher und Disk-Space prüfen
        self._check_resources()
        
        # PATH-Umgebung prüfen
        self._validate_path_environment()
        
        # Erstelle Zusammenfassung
        return self._create_summary()
    
    def _collect_system_info(self) -> None:
        """Sammelt grundlegende Systeminformationen."""
        try:
            self.info.update({
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "python_executable": sys.executable,
                "platform_system": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "hostname": platform.node(),
                "user": os.getenv("USER", os.getenv("USERNAME", "unknown")),
            })
            
            # Virtual Environment Detection
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                self.info["virtual_env"] = True
                self.info["virtual_env_path"] = sys.prefix
            else:
                self.info["virtual_env"] = False
                self.warnings.append("Keine virtuelle Umgebung erkannt - empfohlen für isolierte Installation")
            
        except Exception as e:
            self.errors.append(f"Systeminformationen konnten nicht gesammelt werden: {e}")
    
    def _validate_python_version(self) -> None:
        """Validiert die Python-Version."""
        current_version = sys.version_info[:2]
        
        if current_version < MIN_PYTHON_VERSION:
            self.errors.append(
                f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ ist erforderlich. "
                f"Aktuelle Version: {current_version[0]}.{current_version[1]}"
            )
        elif current_version < RECOMMENDED_PYTHON_VERSION:
            self.warnings.append(
                f"Python {RECOMMENDED_PYTHON_VERSION[0]}.{RECOMMENDED_PYTHON_VERSION[1]}+ wird empfohlen. "
                f"Aktuelle Version: {current_version[0]}.{current_version[1]}"
            )
        
        self.info["python_version_valid"] = current_version >= MIN_PYTHON_VERSION
        self.info["python_version_recommended"] = current_version >= RECOMMENDED_PYTHON_VERSION
    
    def _validate_platform(self) -> None:
        """Validiert die Plattform."""
        system = platform.system()
        release = platform.release()
        
        self.info["platform_supported"] = False
        
        if system in SUPPORTED_PLATFORMS:
            supported_releases = SUPPORTED_PLATFORMS[system]
            
            if system == "Windows":
                if release in supported_releases:
                    self.info["platform_supported"] = True
                else:
                    self.warnings.append(f"Windows {release} - Windows 10/11 wird empfohlen")
            
            elif system == "Linux":
                # Linux-Distribution erkennen
                try:
                    with open("/etc/os-release", "r") as f:
                        os_info = f.read()
                        for supported in supported_releases:
                            if supported.lower() in os_info.lower():
                                self.info["platform_supported"] = True
                                break
                    
                    if not self.info["platform_supported"]:
                        self.warnings.append(f"Linux-Distribution möglicherweise nicht getestet")
                except FileNotFoundError:
                    self.warnings.append("Linux-Distribution konnte nicht erkannt werden")
            
            elif system == "Darwin":  # macOS
                self.info["platform_supported"] = True
        
        else:
            self.warnings.append(f"Plattform {system} nicht offiziell unterstützt")
    
    def _validate_dependencies(self) -> None:
        """Validiert alle Dependencies."""
        self.info["dependencies"] = {}
        
        # Kern-Dependencies
        for module_name, package_name, min_version, required in CORE_DEPENDENCIES:
            dep_info = self._check_dependency(module_name, package_name, min_version, required)
            self.info["dependencies"][package_name] = dep_info
            
            if required and not dep_info["available"]:
                self.errors.append(f"Erforderliche Dependency fehlt: {package_name}")
        
        # Optionale Dependencies
        for module_name, package_name, min_version, required in OPTIONAL_DEPENDENCIES:
            dep_info = self._check_dependency(module_name, package_name, min_version, required)
            dep_info["optional"] = True
            self.info["dependencies"][package_name] = dep_info
    
    def _check_dependency(self, module_name: str, package_name: str, 
                         min_version: str, required: bool) -> Dict[str, Any]:
        """
        Prüft eine einzelne Dependency.
        
        Args:
            module_name: Import-Name des Moduls
            package_name: PyPI-Package-Name
            min_version: Mindestversion (z.B. ">=1.0.0")
            required: Ob die Dependency erforderlich ist
        
        Returns:
            Dictionary mit Dependency-Informationen
        """
        dep_info = {
            "available": False,
            "version": None,
            "version_valid": False,
            "path": None,
            "required": required,
        }
        
        try:
            module = __import__(module_name)
            dep_info["available"] = True
            dep_info["path"] = getattr(module, "__file__", "unknown")
            
            # Version extrahieren
            version = getattr(module, "__version__", None)
            if version:
                dep_info["version"] = version
                
                # Version validieren (einfache Implementierung)
                try:
                    from packaging import version as pkg_version
                    min_ver = min_version.replace(">=", "").replace(">", "").replace("=", "").strip()
                    dep_info["version_valid"] = pkg_version.parse(version) >= pkg_version.parse(min_ver)
                except ImportError:
                    # Fallback ohne packaging
                    dep_info["version_valid"] = True  # Assume valid if can't check
            
        except ImportError:
            if required:
                logger.error(f"Erforderliche Dependency nicht gefunden: {package_name}")
        
        return dep_info
    
    def _collect_hardware_info(self) -> None:
        """Sammelt Hardware-Informationen."""
        try:
            import psutil
            
            # CPU-Informationen
            self.info["cpu_count"] = psutil.cpu_count(logical=True)
            self.info["cpu_count_physical"] = psutil.cpu_count(logical=False)
            self.info["cpu_freq"] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            
            # Memory-Informationen
            memory = psutil.virtual_memory()
            self.info["memory_total"] = memory.total
            self.info["memory_available"] = memory.available
            self.info["memory_percent"] = memory.percent
            
            # Disk-Informationen
            disk = psutil.disk_usage('.')
            self.info["disk_total"] = disk.total
            self.info["disk_free"] = disk.free
            self.info["disk_percent"] = (disk.used / disk.total) * 100
            
        except ImportError:
            self.warnings.append("psutil nicht verfügbar - Hardware-Informationen begrenzt")
            
            # Fallback für grundlegende CPU-Info
            try:
                self.info["cpu_count"] = os.cpu_count()
            except:
                self.info["cpu_count"] = "unknown"
    
    def _check_gpu_availability(self) -> None:
        """Prüft GPU-Verfügbarkeit."""
        self.info["gpu_available"] = False
        self.info["gpu_info"] = []
        
        # PyTorch CUDA Check
        try:
            import torch
            if torch.cuda.is_available():
                self.info["gpu_available"] = True
                self.info["gpu_count"] = torch.cuda.device_count()
                
                for i in range(torch.cuda.device_count()):
                    gpu_info = {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total": torch.cuda.get_device_properties(i).total_memory,
                        "compute_capability": torch.cuda.get_device_properties(i).major,
                    }
                    self.info["gpu_info"].append(gpu_info)
                    
                logger.info(f"✅ {self.info['gpu_count']} GPU(s) verfügbar")
            else:
                self.warnings.append("Keine CUDA-fähige GPU erkannt")
                
        except ImportError:
            self.warnings.append("PyTorch nicht verfügbar - GPU-Check übersprungen")
    
    def _check_resources(self) -> None:
        """Prüft verfügbare Systemressourcen."""
        # Memory Check
        if "memory_total" in self.info:
            memory_gb = self.info["memory_total"] / (1024**3)
            if memory_gb < 4:
                self.warnings.append(f"Wenig RAM verfügbar: {memory_gb:.1f}GB (empfohlen: 8GB+)")
            elif memory_gb < 8:
                self.warnings.append(f"Begrenzter RAM: {memory_gb:.1f}GB (empfohlen für große Bilder: 8GB+)")
        
        # Disk Space Check
        if "disk_free" in self.info:
            disk_free_gb = self.info["disk_free"] / (1024**3)
            if disk_free_gb < 1:
                self.errors.append(f"Zu wenig Speicherplatz: {disk_free_gb:.1f}GB")
            elif disk_free_gb < 5:
                self.warnings.append(f"Wenig Speicherplatz: {disk_free_gb:.1f}GB")
    
    def _validate_path_environment(self) -> None:
        """Validiert PATH-Umgebungsvariablen."""
        path_env = os.environ.get("PATH", "")
        python_dir = Path(sys.executable).parent
        
        self.info["python_in_path"] = str(python_dir) in path_env
        
        if not self.info["python_in_path"]:
            self.warnings.append("Python-Verzeichnis nicht in PATH - CLI-Tools möglicherweise nicht verfügbar")
        
        # Check für wichtige Executables
        executables_to_check = ["git", "pip", "streamlit"]
        available_executables = {}
        
        for exe in executables_to_check:
            try:
                result = subprocess.run([exe, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                available_executables[exe] = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                available_executables[exe] = False
        
        self.info["available_executables"] = available_executables
        
        if not available_executables.get("git", False):
            self.warnings.append("Git nicht verfügbar - Repository-Updates nicht möglich")
        
        if not available_executables.get("streamlit", False):
            self.warnings.append("Streamlit CLI nicht verfügbar")
    
    def _create_summary(self) -> Dict[str, Any]:
        """Erstellt Zusammenfassung der Validierung."""
        summary = {
            "status": "unknown",
            "environment_info": self.info,
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": [],
        }
        
        # Status bestimmen
        if self.errors:
            summary["status"] = "critical"
        elif self.warnings:
            summary["status"] = "warning"
        else:
            summary["status"] = "ok"
        
        # Empfehlungen generieren
        recommendations = []
        
        if not self.info.get("virtual_env", False):
            recommendations.append("Verwenden Sie eine virtuelle Umgebung (python -m venv venv)")
        
        if not self.info.get("python_version_recommended", False):
            recommendations.append("Aktualisieren Sie auf Python 3.11+ für beste Performance")
        
        if not self.info.get("gpu_available", False):
            recommendations.append("Installieren Sie CUDA-PyTorch für GPU-Beschleunigung")
        
        missing_deps = [
            name for name, info in self.info.get("dependencies", {}).items()
            if info.get("required", False) and not info.get("available", False)
        ]
        
        if missing_deps:
            recommendations.append(f"Installieren Sie fehlende Dependencies: {', '.join(missing_deps)}")
        
        summary["recommendations"] = recommendations
        
        return summary


def validate_environment() -> Dict[str, Any]:
    """
    Führt eine vollständige Umgebungsvalidierung durch.
    
    Returns:
        Dictionary mit Validierungsergebnissen
    """
    validator = EnvironmentValidator()
    return validator.validate_all()


def print_environment_report(detailed: bool = False) -> None:
    """
    Druckt einen formatierten Umgebungsbericht.
    
    Args:
        detailed: Ob detaillierte Informationen angezeigt werden sollen
    """
    result = validate_environment()
    
    # Header
    status_icons = {"ok": "✅", "warning": "⚠️", "critical": "❌"}
    status_icon = status_icons.get(result["status"], "❓")
    
    print(f"\n{status_icon} Edge Detection Studio - Environment Report")
    print("=" * 60)
    
    # System Info
    env = result["environment_info"]
    print(f"🖥️  System: {env.get('platform_system', 'unknown')} {env.get('platform_release', '')}")
    print(f"🐍  Python: {env.get('python_version', 'unknown')} ({'✅' if env.get('python_version_valid') else '❌'})")
    print(f"📁  Virtual Env: {'✅' if env.get('virtual_env') else '❌'}")
    
    if "memory_total" in env:
        memory_gb = env["memory_total"] / (1024**3)
        print(f"🧠  Memory: {memory_gb:.1f}GB")
    
    if "cpu_count" in env:
        print(f"⚙️  CPU Cores: {env['cpu_count']}")
    
    if env.get("gpu_available"):
        gpu_count = len(env.get("gpu_info", []))
        print(f"🎮  GPU: {gpu_count} CUDA device(s)")
    else:
        print("🎮  GPU: None detected")
    
    # Dependencies
    print(f"\n📦 Dependencies:")
    deps = env.get("dependencies", {})
    for name, info in deps.items():
        if info.get("required", True) or info.get("available", False):
            status = "✅" if info.get("available") else "❌"
            version = f" v{info.get('version', 'unknown')}" if info.get("version") else ""
            optional = " (optional)" if info.get("optional") else ""
            print(f"   {status} {name}{version}{optional}")
    
    # Errors
    if result["errors"]:
        print(f"\n❌ Errors ({len(result['errors'])}):")
        for error in result["errors"]:
            print(f"   • {error}")
    
    # Warnings
    if result["warnings"]:
        print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
        for warning in result["warnings"][:5]:  # Zeige max 5
            print(f"   • {warning}")
        if len(result["warnings"]) > 5:
            print(f"   ... und {len(result['warnings']) - 5} weitere")
    
    # Recommendations
    if result["recommendations"]:
        print(f"\n💡 Empfehlungen:")
        for rec in result["recommendations"]:
            print(f"   • {rec}")
    
    # Detailed Info
    if detailed:
        print(f"\n🔍 Detaillierte Informationen:")
        print(f"   Python Executable: {env.get('python_executable', 'unknown')}")
        print(f"   Platform Version: {env.get('platform_version', 'unknown')}")
        print(f"   Architecture: {env.get('architecture', 'unknown')}")
        
        if env.get("virtual_env"):
            print(f"   Virtual Env Path: {env.get('virtual_env_path', 'unknown')}")
        
        if "disk_free" in env:
            disk_free_gb = env["disk_free"] / (1024**3)
            print(f"   Free Disk Space: {disk_free_gb:.1f}GB")
    
    print()


def check_critical_requirements() -> bool:
    """
    Prüft nur kritische Anforderungen für schnelle Validierung.
    
    Returns:
        True wenn alle kritischen Anforderungen erfüllt sind
    """
    # Python Version
    if sys.version_info < MIN_PYTHON_VERSION:
        return False
    
    # Kern-Dependencies
    critical_modules = ["cv2", "torch", "streamlit", "numpy"]
    for module in critical_modules:
        try:
            __import__(module)
        except ImportError:
            return False
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Edge Detection Studio Environment Validator")
    parser.add_argument("--detailed", action="store_true", help="Zeige detaillierte Informationen")
    parser.add_argument("--json", action="store_true", help="Ausgabe als JSON")
    parser.add_argument("--critical-only", action="store_true", help="Prüfe nur kritische Anforderungen")
    
    args = parser.parse_args()
    
    if args.critical_only:
        success = check_critical_requirements()
        print("✅ Kritische Anforderungen erfüllt" if success else "❌ Kritische Anforderungen nicht erfüllt")
        sys.exit(0 if success else 1)
    
    if args.json:
        import json
        result = validate_environment()
        print(json.dumps(result, indent=2, default=str))
    else:
        print_environment_report(detailed=args.detailed)
