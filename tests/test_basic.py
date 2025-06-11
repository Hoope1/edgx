import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from edgx.detectors import run_laplacian


def test_laplacian_smoke():
    img = next(pathlib.Path("images").glob("*.png"))
    result = run_laplacian(str(img))
    assert result is not None and result.size > 0
