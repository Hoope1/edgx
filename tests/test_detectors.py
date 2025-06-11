import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from edgx.detectors import run_hed, run_structured, run_bdcn


@pytest.mark.parametrize("func", [run_hed, run_structured, run_bdcn])
def test_detectors_smoke(func):
    img = next(pathlib.Path("images").glob("*.png"))
    try:
        result = func(str(img))
    except RuntimeError:
        pytest.skip("model not available")
    assert result is not None and result.size > 0


def test_cli_e2e(tmp_path):
    img_dir = pathlib.Path("images")
    out_dir = tmp_path / "out"
    cmd = [sys.executable, "-m", "edgx.run_edge_detectors", "--input_dir", str(img_dir), "--output_dir", str(out_dir), "--methods", "Laplacian"]
    proc = subprocess.run(cmd, capture_output=True)
    assert proc.returncode == 0
    result_dir = out_dir / "edge_detection_results"
    assert any(result_dir.glob("*.png"))
