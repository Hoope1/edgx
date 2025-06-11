#!/usr/bin/env python3
"""
Basis-Tests für Edge Detection Studio

Diese Tests prüfen grundlegende Funktionalität ohne komplexe Dependencies.
Geeignet für schnelle Smoke-Tests und CI/CD-Pipelines.
"""

import sys
import os
import tempfile
from pathlib import Path
from typing import Optional, List

import pytest
import numpy as np

# Package-Import mit robustem Path-Handling
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from edgx.detectors import (
        run_laplacian, 
        run_adaptive_canny, 
        run_scharr,
        run_prewitt,
        run_roberts,
        get_all_methods,
        get_max_resolution,
        standardize_output,
    )
    DETECTORS_AVAILABLE = True
except ImportError as e:
    DETECTORS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    pytest.skip(f"Detectors module nicht verfügbar: {e}", allow_module_level=True)


class TestImageUtils:
    """Hilfsfunktionen für Test-Bilder."""
    
    @staticmethod
    def create_test_image(width: int = 100, height: int = 100) -> np.ndarray:
        """
        Erstellt ein synthetisches Test-Bild.
        
        Args:
            width: Bildbreite
            height: Bildhöhe
        
        Returns:
            RGB-Testbild als numpy array
        """
        # Erstelle ein Schachbrett-Muster für gute Edge-Detection
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Schachbrett-Pattern
        square_size = max(10, min(width, height) // 8)
        for y in range(0, height, square_size):
            for x in range(0, width, square_size):
                if ((x // square_size) + (y // square_size)) % 2 == 0:
                    image[y:y+square_size, x:x+square_size] = [255, 255, 255]  # Weiß
                else:
                    image[y:y+square_size, x:x+square_size] = [0, 0, 0]  # Schwarz
        
        # Füge einige geometrische Formen hinzu
        center_x, center_y = width // 2, height // 2
        
        # Kreis
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= (min(width, height) // 6)**2
        image[mask] = [128, 128, 128]  # Grau
        
        return image
    
    @staticmethod
    def save_test_image(image: np.ndarray, path: Path) -> None:
        """Speichert Test-Bild mit OpenCV."""
        try:
            import cv2
            # OpenCV erwartet BGR
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            success = cv2.imwrite(str(path), image_bgr)
            if not success:
                raise RuntimeError(f"cv2.imwrite fehlgeschlagen für {path}")
        except ImportError:
            # Fallback ohne OpenCV
            from PIL import Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image, 'RGB')
            else:
                pil_image = Image.fromarray(image, 'L')
            pil_image.save(path)
    
    @staticmethod
    def find_test_images() -> List[Path]:
        """Findet verfügbare Test-Bilder."""
        image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        test_images = []
        
        # Suche in verschiedenen möglichen Verzeichnissen
        search_dirs = [
            Path("images"),
            Path("test_images"),
            Path("tests") / "images",
            Path(__file__).parent / "images",
            Path(__file__).parent.parent / "images",
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for ext in image_extensions:
                    test_images.extend(search_dir.glob(f"*{ext}"))
        
        return test_images


@pytest.fixture
def test_image_path() -> Path:
    """
    Fixture für Test-Bild-Pfad.
    
    Erstellt ein temporäres Test-Bild falls keine echten Bilder verfügbar sind.
    """
    # Versuche zuerst echte Test-Bilder zu finden
    existing_images = TestImageUtils.find_test_images()
    
    if existing_images:
        return existing_images[0]
    
    # Erstelle synthetisches Test-Bild
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    
    test_image = TestImageUtils.create_test_image(100, 100)
    TestImageUtils.save_test_image(test_image, tmp_path)
    
    yield tmp_path
    
    # Cleanup
    if tmp_path.exists():
        tmp_path.unlink()


@pytest.fixture
def multiple_test_images() -> List[Path]:
    """
    Fixture für mehrere Test-Bilder verschiedener Größen.
    """
    temp_files = []
    
    # Erstelle Test-Bilder in verschiedenen Größen
    sizes = [(50, 50), (100, 100), (150, 75)]
    
    for i, (width, height) in enumerate(sizes):
        with tempfile.NamedTemporaryFile(suffix=f"_test_{i}.png", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        test_image = TestImageUtils.create_test_image(width, height)
        TestImageUtils.save_test_image(test_image, tmp_path)
        temp_files.append(tmp_path)
    
    yield temp_files
    
    # Cleanup
    for tmp_path in temp_files:
        if tmp_path.exists():
            tmp_path.unlink()


class TestBasicFunctionality:
    """Basis-Tests für Kern-Funktionalität."""
    
    @pytest.mark.unit
    def test_imports(self):
        """Test ob alle wichtigen Module importiert werden können."""
        assert DETECTORS_AVAILABLE, f"Detectors module import failed: {IMPORT_ERROR if not DETECTORS_AVAILABLE else ''}"
        
        # Teste wichtige Funktionen
        assert callable(run_laplacian)
        assert callable(run_adaptive_canny)
        assert callable(get_all_methods)
        assert callable(standardize_output)
    
    @pytest.mark.unit
    def test_get_all_methods(self):
        """Test der get_all_methods Funktion."""
        methods = get_all_methods()
        
        # Grundlegende Validierung
        assert isinstance(methods, list)
        assert len(methods) > 0
        
        # Jeder Eintrag sollte (name, function) Tuple sein
        for method_entry in methods:
            assert isinstance(method_entry, tuple)
            assert len(method_entry) == 2
            
            name, func = method_entry
            assert isinstance(name, str)
            assert callable(func)
        
        # Erwartete Methoden sollten vorhanden sein
        method_names = [name for name, _ in methods]
        expected_methods = ["Laplacian", "AdaptiveCanny", "Scharr"]
        
        for expected in expected_methods:
            assert expected in method_names, f"Erwartete Methode {expected} nicht gefunden"
    
    @pytest.mark.unit
    def test_standardize_output(self):
        """Test der standardize_output Funktion."""
        # Test mit verschiedenen Input-Typen
        test_array = np.random.rand(50, 50).astype(np.float32)
        
        # Basis-Test
        result = standardize_output(test_array)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == test_array.shape
        
        # Test mit Target-Size
        target_size = (100, 100)
        result_resized = standardize_output(test_array, target_size=target_size)
        assert result_resized.shape[:2] == target_size[::-1]  # OpenCV verwendet (height, width)
        
        # Test mit Invertierung
        result_no_invert = standardize_output(test_array, invert=False)
        result_invert = standardize_output(test_array, invert=True)
        assert not np.array_equal(result_no_invert, result_invert)


class TestEdgeDetectionMethods:
    """Tests für Edge-Detection-Methoden."""
    
    @pytest.mark.unit
    def test_laplacian_basic(self, test_image_path):
        """Basis-Test für Laplacian Edge Detection."""
        result = run_laplacian(str(test_image_path))
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.size > 0
        assert result.dtype == np.uint8
        assert len(result.shape) == 2  # Grayscale output
    
    @pytest.mark.unit
    def test_laplacian_with_target_size(self, test_image_path):
        """Test Laplacian mit spezifischer Zielgröße."""
        target_size = (64, 64)
        result = run_laplacian(str(test_image_path), target_size=target_size)
        
        assert result.shape[:2] == target_size[::-1]  # (height, width)
    
    @pytest.mark.unit
    def test_adaptive_canny(self, test_image_path):
        """Test für Adaptive Canny Edge Detection."""
        result = run_adaptive_canny(str(test_image_path))
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.size > 0
        assert result.dtype == np.uint8
    
    @pytest.mark.unit
    def test_scharr_filter(self, test_image_path):
        """Test für Scharr Filter."""
        result = run_scharr(str(test_image_path))
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.size > 0
        assert result.dtype == np.uint8
    
    @pytest.mark.unit
    def test_prewitt_filter(self, test_image_path):
        """Test für Prewitt Filter."""
        result = run_prewitt(str(test_image_path))
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.size > 0
    
    @pytest.mark.unit
    def test_roberts_filter(self, test_image_path):
        """Test für Roberts Cross Filter."""
        result = run_roberts(str(test_image_path))
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.size > 0
    
    @pytest.mark.unit
    @pytest.mark.parametrize("method_name", ["Laplacian", "AdaptiveCanny", "Scharr"])
    def test_multiple_methods(self, test_image_path, method_name):
        """Parametrisierter Test für mehrere Methoden."""
        methods = dict(get_all_methods())
        
        if method_name in methods:
            method_func = methods[method_name]
            result = method_func(str(test_image_path))
            
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert result.size > 0
        else:
            pytest.skip(f"Methode {method_name} nicht verfügbar")


class TestErrorHandling:
    """Tests für Fehlerbehandlung."""
    
    @pytest.mark.unit
    def test_invalid_image_path(self):
        """Test mit ungültigem Bildpfad."""
        invalid_path = "nonexistent_image.png"
        
        with pytest.raises((ValueError, Exception)):
            run_laplacian(invalid_path)
    
    @pytest.mark.unit
    def test_empty_image_path(self):
        """Test mit leerem Pfad."""
        with pytest.raises((ValueError, Exception)):
            run_laplacian("")
    
    @pytest.mark.unit
    def test_invalid_target_size(self, test_image_path):
        """Test mit ungültiger Zielgröße."""
        # Negative Größe sollte Fehler verursachen
        with pytest.raises((ValueError, Exception)):
            run_laplacian(str(test_image_path), target_size=(-10, -10))


class TestUtilityFunctions:
    """Tests für Hilfsfunktionen."""
    
    @pytest.mark.unit
    def test_get_max_resolution_single_image(self, test_image_path):
        """Test get_max_resolution mit einem Bild."""
        resolution = get_max_resolution([str(test_image_path)])
        
        assert isinstance(resolution, tuple)
        assert len(resolution) == 2
        assert all(isinstance(dim, int) for dim in resolution)
        assert all(dim > 0 for dim in resolution)
    
    @pytest.mark.unit
    def test_get_max_resolution_multiple_images(self, multiple_test_images):
        """Test get_max_resolution mit mehreren Bildern."""
        image_paths = [str(path) for path in multiple_test_images]
        resolution = get_max_resolution(image_paths)
        
        assert isinstance(resolution, tuple)
        assert len(resolution) == 2
        
        # Resolution sollte die größte gefundene sein
        # Da wir Test-Bilder mit (50,50), (100,100), (150,75) erstellt haben
        # sollte das Maximum (150, 100) sein
        expected_max_width = 150
        expected_max_height = 100
        
        assert resolution[0] == expected_max_width
        assert resolution[1] == expected_max_height
    
    @pytest.mark.unit
    def test_get_max_resolution_empty_list(self):
        """Test get_max_resolution mit leerer Liste."""
        resolution = get_max_resolution([])
        
        # Sollte Default-Resolution zurückgeben
        assert isinstance(resolution, tuple)
        assert len(resolution) == 2
    
    @pytest.mark.unit
    def test_get_max_resolution_invalid_paths(self):
        """Test get_max_resolution mit ungültigen Pfaden."""
        invalid_paths = ["nonexistent1.png", "nonexistent2.jpg"]
        resolution = get_max_resolution(invalid_paths)
        
        # Sollte Default-Resolution zurückgeben wenn keine Bilder geladen werden können
        assert isinstance(resolution, tuple)
        assert len(resolution) == 2


class TestIntegration:
    """Integrations-Tests für vollständige Workflows."""
    
    @pytest.mark.integration
    def test_end_to_end_workflow(self, multiple_test_images):
        """End-to-End Test eines typischen Workflows."""
        image_paths = [str(path) for path in multiple_test_images]
        
        # 1. Hole verfügbare Methoden
        methods = get_all_methods()
        assert len(methods) > 0
        
        # 2. Bestimme Zielauflösung
        target_resolution = get_max_resolution(image_paths)
        assert target_resolution[0] > 0 and target_resolution[1] > 0
        
        # 3. Wende Edge-Detection auf alle Bilder an
        results = []
        test_methods = ["Laplacian", "AdaptiveCanny"]  # Sichere Methoden
        
        for image_path in image_paths:
            for method_name, method_func in methods:
                if method_name in test_methods:
                    try:
                        result = method_func(image_path, target_size=target_resolution)
                        assert result is not None
                        assert result.shape[:2] == target_resolution[::-1]
                        results.append((image_path, method_name, result))
                    except Exception as e:
                        pytest.fail(f"Workflow failed for {image_path} with {method_name}: {e}")
        
        # Sollte mindestens einige Ergebnisse haben
        assert len(results) > 0
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_performance_benchmark(self, test_image_path):
        """Einfacher Performance-Test."""
        import time
        
        # Teste Performance einer schnellen Methode
        start_time = time.time()
        result = run_laplacian(str(test_image_path))
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Sollte schnell sein (unter 1 Sekunde für kleine Bilder)
        assert processing_time < 1.0, f"Verarbeitung zu langsam: {processing_time:.2f}s"
        assert result is not None


# Smoke Test für CI/CD
def test_smoke():
    """
    Minimaler Smoke Test für schnelle CI/CD-Validierung.
    
    Dieser Test läuft auch wenn keine echten Test-Bilder verfügbar sind.
    """
    # Test ob Imports funktionieren
    assert DETECTORS_AVAILABLE
    
    # Test ob Basis-Funktionen verfügbar sind
    methods = get_all_methods()
    assert len(methods) > 0
    
    # Test mit synthetischem Bild
    test_image = TestImageUtils.create_test_image(32, 32)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    
    try:
        TestImageUtils.save_test_image(test_image, tmp_path)
        
        # Teste eine einfache Methode
        result = run_laplacian(str(tmp_path), target_size=(16, 16))
        assert result is not None
        assert result.shape == (16, 16)
        
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    # Ermöglicht direktes Ausführen der Tests
    pytest.main([__file__, "-v"])
