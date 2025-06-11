import sys
import platform

def validate_environment():
    """Sicherstellen, dass Umgebung den Anforderungen entspricht."""
    assert sys.version_info >= (3, 10), "Python 3.10+ erforderlich"
    assert platform.system() == "Windows", "Windows-Umgebung erforderlich"
    assert platform.release() in ["10", "11"], "Windows 10/11 erforderlich"

if __name__ == "__main__":
    try:
        validate_environment()
        print("Environment OK")
    except AssertionError as e:
        print(f"Environment check failed: {e}")
