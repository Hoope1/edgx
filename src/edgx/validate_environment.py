"""Environment validation utilities."""

import platform
import sys
import warnings
import logging

logger = logging.getLogger(__name__)


def validate_environment():
    """Sicherstellen, dass Umgebung den Anforderungen entspricht."""
    if sys.version_info < (3, 10):
        warnings.warn(
            "Python 3.10 oder neuer wird empfohlen. Funktionalität ist sonst "
            "nicht garantiert.",
            RuntimeWarning,
        )
    if platform.system() != "Windows" or platform.release() not in {"10", "11"}:
        warnings.warn(
            "Entwickelt für Windows 10/11. Andere Plattformen sind nicht " "getestet.",
            RuntimeWarning,
        )


if __name__ == "__main__":
    try:
        validate_environment()
        logger.info("Environment OK")
    except AssertionError as e:
        logger.error("Environment check failed: %s", e)
