"""Shared matplotlib style so notebook, report and app figures look consistent."""

from __future__ import annotations

import matplotlib as mpl

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"

# Categorical colours in a fixed, colour-blind-checked order. Assign by entity, never by rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
POSITION_COLORS = {"GK": SERIES[0], "DEF": SERIES[1], "MID": SERIES[2], "ATT": SERIES[3]}
LEAGUE_COLORS = {"GB1": SERIES[0], "ES1": SERIES[1], "IT1": SERIES[2], "L1": SERIES[3], "FR1": SERIES[4]}


def apply_style() -> None:
    """Set global matplotlib defaults: light surface, recessive grid, no top/right spines."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": TEXT_MUTED,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "text.color": TEXT,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })
