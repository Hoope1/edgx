import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors import run_laplacian


def test_laplacian_smoke():
    img = next(pathlib.Path("images").glob("*.png"))
    result = run_laplacian(str(img))
    assert result is not None and result.size > 0
