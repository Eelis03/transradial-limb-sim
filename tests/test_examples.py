"""Integration tier: every example script runs to completion under a reduced step count.

The scripts are run as subprocesses so that the test exercises them exactly as a reader
would, including the import of ``_common`` from the examples directory. Figures are
disabled so that the tier measures the computation and not the plotting backend.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

SCRIPTS = (
    "motor_datasheet_check.py",
    "finger_closing.py",
    "grasp_adaptivity.py",
    "efficiency_chain.py",
    "fingertip_force_curve.py",
    "force_control.py",
    "rating_audit.py",
    "sensitivity_study.py",
    "publish_figures.py",
)


def test_every_example_is_listed() -> None:
    """The list above covers every script in the examples directory.

    Without this the integration tier would silently stop covering a new example.
    """
    present = {
        path.name
        for path in EXAMPLES.glob("*.py")
        if not path.name.startswith("_")
    }
    assert present == set(SCRIPTS)


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_runs_to_completion(script: str) -> None:
    """The script exits cleanly and prints something."""
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / script), "--quick", "--no-figures"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()


def test_figures_are_written_when_requested(tmp_path: Path) -> None:
    """The figure path is exercised once, on the cheapest script that produces one."""
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / "finger_closing.py"), "--quick"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "wrote" in completed.stdout
    del tmp_path
