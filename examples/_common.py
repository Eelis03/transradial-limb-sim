"""Shared wiring for the example scripts: output directory and step count scaling.

Every example accepts a ``--quick`` flag so that the integration tier of the test suite can
run all of them end to end under a reduced step count without duplicating their bodies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.figure import Figure

__all__ = ["Options", "parse_options", "save"]

OUTPUT_DIRECTORY = Path(__file__).resolve().parent.parent / "outputs"


class Options(argparse.Namespace):
    """Command line options common to every example."""

    quick: bool
    no_figures: bool


def parse_options(description: str) -> Options:
    """Parse the common command line options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run with shortened durations and a coarser step, for smoke testing",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="print the numbers but do not write any figure files",
    )
    return parser.parse_args(namespace=Options())


def save(figure: Figure, name: str, options: Options) -> None:
    """Write a figure into ``outputs/`` unless figures are disabled."""
    if options.no_figures:
        return
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIRECTORY / name
    figure.savefig(path, dpi=110)
    print(f"wrote {path}")
