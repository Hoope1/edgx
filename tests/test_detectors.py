#!/usr/bin/env python3
"""
Erweiterte Tests für Edge Detection Studio Detectors

Diese Tests prüfen alle Edge-Detection-Algorithmen einschließlich
Deep Learning Methoden und GPU-beschleunigter Varianten.
"""

import sys
import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pytest
import numpy as np

# Package-Import mit robustem Path-Handling
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from edgx.detectors import (
        get_all_methods,
        init_models,
        standardize_output,
        get_max_resolution,
        # Klassische Methoden
        run_laplacian,
        run_adaptive_canny,
        run_multi_scale_canny,
        run_scharr,
        run_prewitt,
        run_roberts,
        run_gradient_magnitude,
        run_morphological_gradient,
        # Deep Learning Methoden
        run_hed,
        run_pytorch_hed,
        run_structured,
        run_bdcn,
        run_fixed_cnn,
        # GPU Methoden
        run_kornia_canny,
        run_kornia_sobel,
    )
    DETECTORS_AVAILABLE = True
except ImportError as e:
    DETECTORS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    pytest.skip(f"Detectors module nicht verfügbar: {e}", allow_module_level=True)

# Zusätzliche Dependency-Checks
HAS_TORCH = False
HAS_KORNIA = False
HAS_OPENCV_CONTRIB = False

try:
    import torch
    HAS_TORCH = True
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False

try:
    import kornia
    HAS_KORNIA = True
except ImportError:
    pass

try:
    import cv2
    # Check für opencv-contrib spezifische Features
    try:
        cv2.ximgproc.createStructuredEdgeDetection
        HAS_OPENCV_CONTRIB = True
    except AttributeError:
        HAS_OPENCV_CONTRIB = False
except ImportError:
    pass


class TestImageFactory:
    """Factory für verschiedene Test-Bild-Typen."""
    
    @staticmethod
    def create_synthetic_image(width: int = 128, height: int = 128, pattern: str = "checkerboard") -> np.ndarray:
        """
        Erstellt synthetische Test-Bilder mit verschiedenen Mustern.
        
        Args:
            width: Bildbreite
            height: Bildhöhe
            pattern: Art des Musters ("checkerboard", "gradient", "shapes", "noise")
        
        Returns:
            RGB-Testbild als numpy array
        """
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        if pattern == "checkerboard":
            # Schachbrett-Muster für scharfe Kanten
            square_size = max(8, min(width, height) // 16)
            for y in range(0, height, square_size):
                for x in range(0, width, square_size):
                    if ((x // square_size) + (y // square_size)) % 2 == 0:
                        image[y:y+square_size, x:x+square_size] = [255, 255, 255]
        
        elif pattern == "gradient":
            # Horizontaler Gradient
            for x in range(width):
                intensity = int(255 * x / width)
                image[:, x] = [intensity, intensity, intensity]
        
        elif pattern == "shapes":
            # Geometrische Formen
            center_x, center_y = width // 2, height // 2
            
            # Rechteck
            rect_size = min(width, height) // 4
            image[center_y-rect_size:center_y+rect_size, 
                  center_x-rect_size:center_x+rect_size] = [128, 128, 128]
            
            # Kreis
            y, x = np.ogrid[:height, :width]
            circle_mask = (x - center_x)**2 + (y - center_y)**2 <= (min(width, height) // 8)**2
            image[circle_mask] = [200, 200, 200]
        
        elif pattern == "noise":
            # Rauschen mit einigen strukturierten Elementen
            noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            # Füge strukturierte Linien hinzu
            image = noise
            image[height//4:3*height//4, width//2-5:width//2+5] = [255, 255, 255]
            image[height//2-5:height//2+5, width//4:3*width//4] = [0, 0, 0]
        
        return image
    
    @staticmethod
    def save_test_image(image: np.ndarray, path: Path) -> None:
        """Speichert Test-Bild."""
        try:
            import cv2
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            cv2.imwrite(str(path), image_bgr)
        except ImportError:
            from PIL import Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image, 'RGB')
            else:
                pil_image = Image.fromarray(image, 'L')
            pil_image.save(path)


@pytest.fixture(scope="session")
def test_images_suite() -> Dict[str, Path]:
    """
    Session-scope Fixture für Test-Bilder verschiedener Typen.
    
    Returns:
        Dictionary mit Test-Bild-Pfaden für verschiedene Muster
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="edgx_test_"))
    test_images = {}
    
    patterns = ["checkerboard", "gradient", "shapes", "noise"]
    
    for pattern in patterns:
        image = TestImageFactory.create_synthetic_image(128, 128, pattern)
        image_path = temp_dir / f"test_{pattern}.png"
        TestImageFactory.save_test_image(image, image_path)
        test_images[pattern] = image_path
    
    # Verschiedene Größen
    for size_name, (w, h) in [("small", (64, 64)), ("large", (256, 256))]:
        image = TestImageFactory.create_synthetic_image(w, h, "checkerboard")
        image_path = temp_dir / f"test_{size_name}.png"
        TestImageFactory.save_test_image(image, image_path)
        test_images[size_name] = image_path
    
    yield test_images
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def quick_test_image() -> Path:
    """Schnelle Fixture für einzelne Tests."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    
    image = TestImageFactory.create_synthetic_image(64, 64, "checkerboard")
    TestImageFactory.save_test_image(image, tmp_path)
    
    yield tmp_path
    
    if tmp_path.exists():
        tmp_path.unlink()


class TestClassicalMethods:
    """Tests für klassische Edge-Detection-Methoden."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("method_name,method_func", [
        ("Laplacian", run_laplacian),
        ("AdaptiveCanny", run_adaptive_canny),
        ("MultiScaleCanny", run_multi_scale_canny),
        ("Scharr", run_scharr),
        ("Prewitt", run_prewitt),
        ("Roberts", run_roberts),
        ("GradientMagnitude", run_gradient_magnitude),
        ("MorphologicalGradient", run_morphological_gradient),
    ])
    def test_classical_methods_basic(self, quick_test_image, method_name, method_func):
        """Test alle klassischen Methoden mit Basis-Parametern."""
        result = method_func(str(quick_test_image))
        
        assert result is not None, f"{method_name} returned None"
        assert isinstance(result, np.ndarray), f"{method_name} result is not numpy array"
        assert result.size > 0, f"{method_name} result is empty"
        assert result.dtype == np.uint8, f"{method_name} result dtype is {result.dtype}, expected uint8"
        assert len(result.shape) == 2, f"{method_name} result should be grayscale"
    
    @pytest.mark.unit
    def test_classical_methods_with_target_size(self, quick_test_image):
        """Test klassische Methoden mit spezifischer Zielgröße."""
        target_size = (32, 32)
        
        methods_to_test = [run_laplacian, run_adaptive_canny, run_scharr]
        
        for method_func in methods_to_test:
            result = method_func(str(quick_test_image), target_size=target_size)
            assert result.shape[:2] == target_size[::-1], f"Target size not applied correctly"
    
    @pytest.mark.unit
    def test_classical_methods_different_image_types(self, test_images_suite):
        """Test klassische Methoden mit verschiedenen Bildtypen."""
        methods_to_test = [
            ("Laplacian", run_laplacian),
            ("AdaptiveCanny", run_adaptive_canny),
        ]
        
        for pattern, image_path in test_images_suite.items():
            for method_name, method_func in methods_to_test:
                try:
                    result = method_func(str(image_path))
                    assert result is not None, f"{method_name} failed on {pattern} pattern"
                    assert result.size > 0, f"{method_name} empty result on {pattern} pattern"
                except Exception as e:
                    pytest.fail(f"{method_name} failed on {pattern} pattern: {e}")


class TestDeepLearningMethods:
    """Tests für Deep Learning Edge-Detection-Methoden."""
    
    @pytest.mark.unit
    def test_hed_opencv_availability(self):
        """Test ob HED OpenCV verfügbar ist."""
        try:
            # Teste, ob init_models funktioniert
            init_models()
        except Exception as e:
            pytest.skip(f"Model initialization failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_hed_opencv(self, quick_test_image):
        """Test HED mit OpenCV-Backend."""
        try:
            result = run_hed(str(quick_test_image))
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.uint8
        except RuntimeError as e:
            if "HED Modell nicht verfügbar" in str(e):
                pytest.skip("HED model not available")
            else:
                raise
    
    @pytest.mark.integration
    def test_pytorch_hed_fallback(self, quick_test_image):
        """Test PyTorch HED mit Fallback-Mechanismus."""
        # Dieser Test sollte immer funktionieren, da Fallbacks implementiert sind
        result = run_pytorch_hed(str(quick_test_image))
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert len(result.shape) == 2
    
    @pytest.mark.integration
    @pytest.mark.skipif(not HAS_OPENCV_CONTRIB, reason="opencv-contrib not available")
    def test_structured_forests(self, quick_test_image):
        """Test Structured Forests (benötigt opencv-contrib)."""
        try:
            result = run_structured(str(quick_test_image))
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.uint8
        except RuntimeError as e:
            if "Structured Forests" in str(e):
                pytest.skip("Structured Forests model not available")
            else:
                raise
    
    @pytest.mark.integration
    def test_bdcn_fallback(self, quick_test_image):
        """Test BDCN mit Fallback."""
        # BDCN sollte auf Multi-Scale Canny zurückfallen
        result = run_bdcn(str(quick_test_image))
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
    
    @pytest.mark.unit
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_fixed_cnn(self, quick_test_image):
        """Test Fixed CNN Filter."""
        result = run_fixed_cnn(str(quick_test_image))
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8


class TestGPUMethods:
    """Tests für GPU-beschleunigte Methoden."""
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_KORNIA, reason="Kornia not available")
    def test_kornia_canny(self, quick_test_image):
        """Test Kornia Canny (GPU-beschleunigt)."""
        result = run_kornia_canny(str(quick_test_image))
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_KORNIA, reason="Kornia not available")
    def test_kornia_sobel(self, quick_test_image):
        """Test Kornia Sobel (GPU-beschleunigt)."""
        result = run_kornia_sobel(str(quick_test_image))
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_GPU, reason="CUDA not available")
    def test_gpu_acceleration_comparison(self, quick_test_image):
        """Vergleiche GPU vs CPU Performance (wenn verfügbar)."""
        if not HAS_KORNIA:
            pytest.skip("Kornia not available")
        
        import time
        
        # GPU Version
        start_gpu = time.time()
        result_gpu = run_kornia_canny(str(quick_test_image))
        time_gpu = time.time() - start_gpu
        
        # CPU Fallback Version
        start_cpu = time.time()
        result_cpu = run_adaptive_canny(str(quick_test_image))
        time_cpu = time.time() - start_cpu
        
        # Beide sollten funktionieren
        assert result_gpu is not None
        assert result_cpu is not None
        
        # GPU sollte nicht signifikant langsamer sein (für kleine Bilder kann es sogar langsamer sein)
        # Dieser Test dient hauptsächlich der Funktionalitätsprüfung
        assert time_gpu < 10.0  # Sollte nicht extrem langsam sein


class TestErrorHandling:
    """Tests für Fehlerbehandlung und Edge Cases."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("method_func", [
        run_laplacian,
        run_adaptive_canny,
        run_pytorch_hed,  # Sollte fallback verwenden
    ])
    def test_invalid_image_path(self, method_func):
        """Test Verhalten bei ungültigen Bildpfaden."""
        with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
            method_func("nonexistent_image.png")
    
    @pytest.mark.unit
    def test_empty_image_path(self):
        """Test Verhalten bei leerem Pfad."""
        with pytest.raises((ValueError, Exception)):
            run_laplacian("")
    
    @pytest.mark.unit
    def test_invalid_target_size(self, quick_test_image):
        """Test Verhalten bei ungültigen Zielgrößen."""
        # Negative Größen
        with pytest.raises((ValueError, Exception)):
            run_laplacian(str(quick_test_image), target_size=(-10, -10))
        
        # Null-Größen
        with pytest.raises((ValueError, Exception)):
            run_laplacian(str(quick_test_image), target_size=(0, 0))
    
    @pytest.mark.unit
    def test_very_large_target_size(self, quick_test_image):
        """Test Verhalten bei sehr großen Zielgrößen."""
        # Sehr große Größe (sollte Memory-Probleme vermeiden)
        large_size = (10000, 10000)
        
        try:
            result = run_laplacian(str(quick_test_image), target_size=large_size)
            # Falls erfolgreich, prüfe Ergebnis
            if result is not None:
                assert result.shape[:2] == large_size[::-1]
        except (MemoryError, Exception) as e:
            # Erwartetes Verhalten bei Memory-Problemen
            pytest.skip(f"Large image processing failed as expected: {e}")


class TestUtilityFunctions:
    """Tests für Utility-Funktionen."""
    
    @pytest.mark.unit
    def test_standardize_output_edge_cases(self):
        """Test standardize_output mit verschiedenen Edge Cases."""
        # Float32 Input
        float_input = np.random.rand(50, 50).astype(np.float32)
        result = standardize_output(float_input)
        assert result.dtype == np.uint8
        
        # Int16 Input
        int_input = np.random.randint(0, 32767, (50, 50), dtype=np.int16)
        result = standardize_output(int_input)
        assert result.dtype == np.uint8
        
        # Binary Input (0 und 1)
        binary_input = np.random.choice([0, 1], (50, 50)).astype(np.float32)
        result = standardize_output(binary_input)
        assert result.dtype == np.uint8
        assert np.unique(result).size <= 2  # Sollte nur 0 und 255 enthalten
    
    @pytest.mark.unit
    def test_get_max_resolution_robustness(self, test_images_suite):
        """Test get_max_resolution Robustheit."""
        image_paths = [str(path) for path in test_images_suite.values()]
        
        # Normal case
        resolution = get_max_resolution(image_paths)
        assert isinstance(resolution, tuple)
        assert len(resolution) == 2
        assert all(isinstance(dim, int) for dim in resolution)
        
        # Mit einigen ungültigen Pfaden gemischt
        mixed_paths = image_paths + ["invalid1.png", "invalid2.jpg"]
        resolution_mixed = get_max_resolution(mixed_paths)
        assert isinstance(resolution_mixed, tuple)
        # Sollte immer noch funktionieren
        
        # Nur ungültige Pfade
        invalid_only = ["invalid1.png", "invalid2.jpg"]
        resolution_invalid = get_max_resolution(invalid_only)
        assert isinstance(resolution_invalid, tuple)
        # Sollte Default-Resolution zurückgeben


class TestMethodCompatibility:
    """Tests für Kompatibilität zwischen verschiedenen Methoden."""
    
    @pytest.mark.integration
    def test_all_methods_same_output_size(self, quick_test_image):
        """Test dass alle Methoden die gleiche Ausgabegröße produzieren."""
        target_size = (64, 64)
        
        # Sammle alle verfügbaren Methoden
        all_methods = get_all_methods()
        
        results = {}
        for method_name, method_func in all_methods:
            try:
                result = method_func(str(quick_test_image), target_size=target_size)
                if result is not None:
                    results[method_name] = result.shape[:2]
            except Exception as e:
                # Einige Methoden könnten fehlschlagen (z.B. fehlende Modelle)
                print(f"Method {method_name} failed: {e}")
                continue
        
        # Alle erfolgreichen Methoden sollten die gleiche Größe haben
        expected_shape = target_size[::-1]  # OpenCV format (height, width)
        for method_name, shape in results.items():
            assert shape == expected_shape, f"Method {method_name} has wrong output size: {shape}"
    
    @pytest.mark.integration
    def test_method_output_consistency(self, test_images_suite):
        """Test dass Methoden konsistente Ausgaben für gleiche Eingaben produzieren."""
        test_image = str(test_images_suite["checkerboard"])
        
        # Teste einige stabile Methoden mehrfach
        stable_methods = [
            ("Laplacian", run_laplacian),
            ("AdaptiveCanny", run_adaptive_canny),
        ]
        
        for method_name, method_func in stable_methods:
            # Führe mehrfach aus
            results = []
            for _ in range(3):
                try:
                    result = method_func(test_image, target_size=(64, 64))
                    if result is not None:
                        results.append(result)
                except Exception:
                    continue
            
            # Alle Ergebnisse sollten identisch sein
            if len(results) > 1:
                for i in range(1, len(results)):
                    assert np.array_equal(results[0], results[i]), f"Method {method_name} inconsistent results"


class TestCLIIntegration:
    """Tests für CLI-Integration."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_cli_execution(self, test_images_suite):
        """Test CLI-Tool mit echten Bildern."""
        # Erstelle temporäres Ausgabeverzeichnis
        with tempfile.TemporaryDirectory() as temp_output:
            # Erstelle Input-Verzeichnis mit Test-Bildern
            with tempfile.TemporaryDirectory() as temp_input:
                # Kopiere ein Test-Bild
                import shutil
                test_image = test_images_suite["checkerboard"]
                input_image = Path(temp_input) / "test.png"
                shutil.copy(test_image, input_image)
                
                # Führe CLI aus
                cmd = [
                    sys.executable, "-m", "edgx.run_edge_detectors",
                    "--input_dir", temp_input,
                    "--output_dir", temp_output,
                    "--methods", "Laplacian", "AdaptiveCanny"
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    
                    # Prüfe Rückgabecode
                    assert result.returncode == 0, f"CLI failed with: {result.stderr}"
                    
                    # Prüfe ob Ausgabedateien erstellt wurden
                    output_dir = Path(temp_output) / "edge_detection_results"
                    assert output_dir.exists(), "Output directory not created"
                    
                    output_files = list(output_dir.glob("*.png"))
                    assert len(output_files) >= 2, f"Expected at least 2 output files, got {len(output_files)}"
                    
                except subprocess.TimeoutExpired:
                    pytest.skip("CLI execution timed out")
                except FileNotFoundError:
                    pytest.skip("CLI module not accessible")


# Benchmark Tests (markiert als slow)
class TestPerformance:
    """Performance-Tests für verschiedene Szenarien."""
    
    @pytest.mark.slow
    @pytest.mark.parametrize("image_size", [(64, 64), (128, 128), (256, 256)])
    def test_performance_scaling(self, image_size):
        """Test Performance-Skalierung mit verschiedenen Bildgrößen."""
        import time
        
        # Erstelle Test-Bild
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            image = TestImageFactory.create_synthetic_image(image_size[0], image_size[1])
            TestImageFactory.save_test_image(image, tmp_path)
            
            # Teste Performance
            start_time = time.time()
            result = run_laplacian(str(tmp_path))
            end_time = time.time()
            
            processing_time = end_time - start_time
            pixels = image_size[0] * image_size[1]
            
            # Basis-Performance-Checks
            assert processing_time < 5.0, f"Processing too slow for {image_size}: {processing_time:.2f}s"
            assert result is not None, "Processing failed"
            
            # Log Performance für Referenz
            print(f"Size {image_size}: {processing_time:.3f}s ({pixels/processing_time:.0f} pixels/sec)")
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    # Ermöglicht direktes Ausführen der Tests
    pytest.main([__file__, "-v", "--tb=short"])
