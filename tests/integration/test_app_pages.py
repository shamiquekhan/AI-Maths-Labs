"""AppTest coverage for every dashboard page and the torch-free fallback."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_ROOT = Path(__file__).resolve().parents[2] / "src" / "from_zero" / "app"
PAGES = [APP_ROOT / "Home.py", *sorted((APP_ROOT / "pages").glob("*.py"))]


@pytest.mark.parametrize("page", PAGES, ids=[page.stem for page in PAGES])
def test_every_page_loads_without_exception(page: Path) -> None:
    app = AppTest.from_file(str(page), default_timeout=90)
    app.run()
    assert not app.exception, [str(exception.value) for exception in app.exception]
    assert len(app.markdown) > 0, "page rendered no content"


def test_strict_logistic_button_trains_and_reports_metrics() -> None:
    page = APP_ROOT / "pages" / "9_Strict_Engine.py"
    app = AppTest.from_file(str(page), default_timeout=120)
    app.run()
    app.slider(key="strict_logistic_epochs").set_value(20)
    app.button(key="strict_logistic_button").click()
    app.run()
    assert not app.exception, [str(exception.value) for exception in app.exception]
    labels = {metric.label for metric in app.metric}
    assert {"Training accuracy", "Final loss"} <= labels


def _run_with_torch_blocked(script: str) -> None:
    blocker = (
        "import importlib.abc, sys\n"
        "class Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'torch' or fullname.startswith('torch.'):\n"
        "            raise ImportError('torch blocked by deployment test')\n"
        "sys.meta_path.insert(0, Blocker())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", blocker + script],
        capture_output=True,
        text=True,
        cwd=str(APP_ROOT.parents[2]),
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout, result.stdout


def test_services_degrade_gracefully_without_torch() -> None:
    _run_with_torch_blocked(
        "sys.path.insert(0, 'src')\n"
        "from from_zero.app import services\n"
        "assert not services.TORCH_AVAILABLE\n"
        "available, reason = services.torch_status()\n"
        "assert not available and 'blocked' in reason\n"
        "try:\n"
        "    services.run_gpt_demo(max_steps=1)\n"
        "except RuntimeError as error:\n"
        "    assert 'requires PyTorch' in str(error)\n"
        "else:\n"
        "    raise AssertionError('expected a guarded RuntimeError')\n"
        "print('OK')\n"
    )


def test_every_page_loads_without_torch() -> None:
    _run_with_torch_blocked(
        "from pathlib import Path\n"
        "from streamlit.testing.v1 import AppTest\n"
        "root = Path('src/from_zero/app')\n"
        "pages = [root / 'Home.py', *sorted((root / 'pages').glob('*.py'))]\n"
        "for page in pages:\n"
        "    app = AppTest.from_file(str(page), default_timeout=90)\n"
        "    app.run()\n"
        "    assert not app.exception, (page.name, [str(e.value) for e in app.exception])\n"
        "    if page.name in ('4_Mini_GPT.py', '5_Diffusion.py'):\n"
        "        assert any('needs PyTorch' in item.value for item in app.markdown)\n"
        "print('OK')\n"
    )
