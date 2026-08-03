"""Tests of what the distribution carries, as opposed to what the model computes.

A package that passes ``mypy --strict`` still delivers no types at all to anything that
installs it unless it ships the PEP 561 marker, and the marker is a file whose absence
nothing else in a test suite would notice.
"""

from __future__ import annotations

from pathlib import Path

import transradial_sim

PACKAGE_ROOT = Path(transradial_sim.__file__).resolve().parent


def test_the_typing_marker_is_inside_the_package_directory() -> None:
    """``py.typed`` sits next to ``__init__.py`` so that installers pick it up.

    PEP 561 requires the marker to be a file in the package directory itself. A copy at
    the repository root, or beside ``src``, is not found by a type checker inspecting an
    installed distribution, so the location is asserted rather than just the existence.
    """
    marker = PACKAGE_ROOT / "py.typed"
    assert marker.is_file()
    assert marker.parent == PACKAGE_ROOT
    assert (PACKAGE_ROOT / "__init__.py").is_file()


def test_the_typing_marker_is_empty() -> None:
    """The marker carries no content. Anything in it would be a partial stub directive."""
    assert (PACKAGE_ROOT / "py.typed").read_bytes() == b""
