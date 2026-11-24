"""
Tesselab: a desktop viewer for scientific data whose analyses are small graphs of nodes, supplied by modules.

What module code uses (see AGENTS.md for the protocol):

    import tesselab
    from tesselab import P, Signature, Var, Plot, PlotSpec, Axis, Line, Points, Lines, Result

The rest of the package is Tesselab itself: modules.py (finding and loading modules), analysis.py (the analysis
files), engine.py (computing them), project.py, export.py, check.py, scaffold.py, cli.py, gui/ (the window).
"""
__version__ = "0.1.0a1"

from tesselab import params as P
from tesselab.node import (Action, Crosshair, Cursor, DrawContext, Node, Pick, Plot, RangeSelect, Reader, Result,
                           Toggle, View, diagnostic, warning)
from tesselab.plotspec import (Axis, Band, Heatmap, HLine, Line, Lines, Panel, PlotSpec, Points, Region, Style, Text,
                               VLine, note_only)
from tesselab.signature import Signature, Var

__all__ = ["P", "Node", "Reader", "View", "Toggle", "Signature", "Var", "Plot", "PlotSpec", "Panel", "Axis", "Style", "Line", "Points", "Lines",
           "Band", "VLine", "HLine", "Region", "Heatmap", "Text", "note_only", "Result", "Cursor", "Crosshair",
           "RangeSelect", "Pick", "Action", "DrawContext", "diagnostic", "warning"]
