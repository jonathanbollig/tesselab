"""
tesselab new module <folder>: a minimal module that already works (it passes tesselab check), to start from. A
reader reads a CSV with an x and a y column, a compute node smooths it, and a view plots both; AGENTS.md in it is the
protocol, written for LLMs.
"""
import os
import re
import shutil
import textwrap

import numpy as np

MANIFEST = """\
[module]
id = "{id}"
name = "{name}"
version = "0.1.0"
api = 1
description = "Reads a curve (CSV with x and y) and smooths it. Replace this with what your module does."
authors = []
requires = ["numpy", "xarray"]

[[nodes]]
id = "read"
class = "nodes.py:ReadCurve"
files = ["*.csv"]

[[nodes]]
id = "smooth"
class = "nodes.py:Smooth"

[[nodes]]
id = "curve-view"
class = "nodes.py:CurveView"

[[templates]]
id = "default"
name = "{name}"
file = "templates/default.json"
help = "A folder with one CSV file (columns x, y)."
recognize = {{ any = ["*.csv"] }}
fill = "nodes.py:fill_template"
"""

NODES = '''\
"""
{name}: the nodes of this module (see AGENTS.md, "Good nodes"). A reader reads the file, a compute node computes
(inputs and parameters in, xarray Datasets out; keep heavy analysis code in analysis.py so notebooks can use it too),
a view draws (its plots are descriptions that Tesselab draws; it is a section of the analysis's card).
"""
import os

import numpy as np
import xarray as xr

import tesselab
from tesselab import P, Plot, PlotSpec, Axis, Line, VLine, Signature, Var, Result, Cursor, note_only

from . import analysis


class ReadCurve(tesselab.Reader):
    name = "Curve file"
    help = "Reads a CSV file with two columns, x and y (a header line is allowed)."
    params = [P.Path("file", filter="CSV (*.csv)", help="The CSV file."),
              P.Str("x_unit", "", help="Unit of x (for the axis label)."),
              P.Str("y_unit", "", help="Unit of y.")]
    outputs = ["curve"]
    key_results = [Result("maximum", label="maximum of y")]

    def run(self, inputs, params):
        x, y = analysis.read_curve(params.file)
        curve = xr.Dataset({{"y": ("x", y, {{"units": params.y_unit}}), "maximum": ((), float(np.max(y)))}},
                           coords={{"x": ("x", x, {{"units": params.x_unit}})}})
        return {{"curve": curve}}


class Smooth(tesselab.Node):
    name = "Smooth"
    help = "A moving average over `window` points."
    inputs = {{"curve": Signature(y=Var("x"), x=Var("x"), help="The curve to smooth.")}}
    params = [P.Int("window", 5, min=1, help="Points averaged (odd numbers keep the curve centered).")]
    outputs = ["smoothed_curve"]

    def run(self, inputs, params):
        y = analysis.moving_average(inputs.curve.y.values, params.window)
        if params.window > len(y):
            tesselab.warning("The window is longer than the curve.")
        return {{"smoothed_curve": xr.Dataset({{"y": ("x", y, inputs.curve.y.attrs)}},
                                             coords={{"x": ("x", inputs.curve.x.values, inputs.curve.x.attrs)}})}}


def draw_curve(ctx):
    if "curve" not in ctx.inputs:
        return note_only("No curve")
    curve = ctx.inputs.curve
    marks = [Line(curve.x, curve.y, name="measured", style=tesselab.Style(color="muted"))]
    if "smoothed" in ctx.inputs:
        marks.append(Line(ctx.inputs.smoothed.x, ctx.inputs.smoothed.y, name="smoothed",
                          style=tesselab.Style(color="data", width=2)))
    marks.append(VLine(ctx.params.marker, label="marker"))
    return PlotSpec(x=Axis("x", curve.x.attrs.get("units", "")), y=Axis("y", curve.y.attrs.get("units", "")),
                    marks=marks, legend="top right")


class CurveView(tesselab.View):
    name = "Curve"
    help = "The curve as measured and smoothed."
    inputs = {{"curve": Signature(y=Var("x"), help="The curve as measured."),
              "smoothed": Signature(y=Var("x"), optional=True, help="The smoothed curve.")}}
    params = [P.Float("marker", 0.0, help="A line in the plot (drag it).")]
    plots = [Plot("curve", "Curve", draw=draw_curve, help="Grey: as measured; blue: smoothed. Drag the line.")]
    interactions = [Cursor("marker", plots=["curve"], snap="x")]


def fill_template(folder, project):
    """A data folder -> the template's placeholders: its first CSV file."""
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".csv"))
    path = os.path.join(folder, names[0])
    return {{"name": os.path.basename(folder), "file": os.path.relpath(path, project).replace("\\\\", "/")}}
'''

ANALYSIS = '''\
"""The analysis itself: plain numpy, no Tesselab, so a notebook or a script can use it the same way."""
import numpy as np


def read_curve(path):
    """x and y from a CSV file with two columns (a header line is skipped)."""
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    try:
        float(lines[0].split(",")[0])
    except ValueError:
        lines = lines[1:]
    data = np.array([[float(v) for v in line.split(",")[:2]] for line in lines])
    return data[:, 0], data[:, 1]


def moving_average(y, window):
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")
'''

TEMPLATE = """\
{
  "tesselab": 1,
  "label": "{name}",
  "nodes": {
    "read": {"node": "%(id)s/read", "params": {"file": "{file}"}},
    "smooth": {"node": "%(id)s/smooth", "inputs": {"curve": "read.curve"}},
    "curve": {"node": "%(id)s/curve-view", "inputs": {"curve": "read.curve", "smoothed": "smooth.smoothed_curve"}}
  },
  "view": {"order": ["curve", "read", "smooth"], "open": ["curve"]}
}
"""

README = """\
# {name}

A Tesselab module. Edit nodes.py (the nodes) and analysis.py (the computation), describe your data in sample/, and
check it:

    tesselab check {folder}

An open Tesselab window follows your edits (Modules → Follow changes on disk; without it, Modules → Reload modules). AGENTS.md explains the protocol.
"""


def new_module(folder, module_id=None) -> str:
    folder = os.path.abspath(os.path.expanduser(folder))
    if os.path.exists(folder) and os.listdir(folder):
        raise FileExistsError(f"{folder} exists and is not empty.")
    base = os.path.basename(folder)
    module_id = module_id or "my." + re.sub(r"[^A-Za-z0-9]+", "-", base).strip("-").lower()
    name = base.replace("_", " ").replace("-", " ").strip().capitalize() or "My module"
    os.makedirs(os.path.join(folder, "templates"))
    os.makedirs(os.path.join(folder, "sample", "example"))
    with open(os.path.join(folder, "module.toml"), "w", encoding="utf-8") as f:
        f.write(MANIFEST.format(id=module_id, name=name))
    with open(os.path.join(folder, "nodes.py"), "w", encoding="utf-8") as f:
        f.write(NODES.format(name=name))
    with open(os.path.join(folder, "analysis.py"), "w", encoding="utf-8") as f:
        f.write(ANALYSIS)
    with open(os.path.join(folder, "templates", "default.json"), "w", encoding="utf-8") as f:
        f.write(TEMPLATE % {"id": module_id})
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as f:
        f.write(README.format(name=name, folder=folder))
    x = np.linspace(0, 10, 201)
    y = np.exp(-((x - 4) / 1.2) ** 2) + 0.05 * np.sin(17 * x)
    with open(os.path.join(folder, "sample", "example", "curve.csv"), "w", encoding="utf-8") as f:
        f.write("x,y\n" + "".join(f"{a:.4f},{b:.6f}\n" for a, b in zip(x, y)))
    agents = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AGENTS.md")
    if os.path.isfile(agents):
        shutil.copy(agents, os.path.join(folder, "AGENTS.md"))
    return folder
