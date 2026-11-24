"""
A table of numbers as a dataset: the first column is x, the others are series. The output "table" has the matrix
`values` (series, x): one row per series, as spectra are one row per measurement, so a node that takes a matrix
(SVD, MCR) sees the series as rows and x as the axis of its components.
"""
import os

import numpy as np
import xarray as xr

import tesselab
from tesselab import P, Plot, PlotSpec, Axis, Lines, Signature, Var


def read_table(path, delimiter=None):
    """Header (if the first line is not numbers) and the numbers of a delimited text file."""
    with open(path, encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{os.path.basename(path)} is empty.")
    sep = delimiter or next((d for d in ("\t", ";", ",") if d in lines[0]), None)

    def split(line):
        return [part.strip().strip('"') for part in (line.split(sep) if sep else line.split())]

    first = split(lines[0])
    try:
        [float(x) for x in first]
        header, rows = None, lines
    except ValueError:
        header, rows = first, lines[1:]
    data = np.array([[float(x) if x not in ("", "nan", "NaN") else np.nan for x in split(line)] for line in rows])
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{os.path.basename(path)}: at least two columns of numbers are needed.")
    header = header or ["x"] + [f"column {i}" for i in range(1, data.shape[1])]
    return header, data


def _unit_axis(label, unit):
    try:
        return Axis(label, unit)
    except ValueError:
        return Axis(f"{label} [{unit}]" if unit else label)


def plot_table(ctx):
    table = ctx.inputs.table
    x_name = table.attrs.get("x_name", "x")
    return PlotSpec(x=_unit_axis(x_name, table.x.attrs.get("units", "")),
                    y=_unit_axis("Value", table["values"].attrs.get("units", "")),
                    marks=[Lines(table.x, table["values"], labels=list(table.series.values),
                                 colormap="default")],
                    legend="top right" if table.sizes["series"] <= 10 else None)


class CsvTable(tesselab.Reader):
    name = "CSV table"
    help = "Reads a table: the first column is x, every other column one series."
    params = [P.Path("file", filter="Tables (*.csv *.txt *.tsv *.dat)", help="The table file."),
              P.Str("x_unit", "", help="Unit of the first column, e.g. m or s (for the axis label)."),
              P.Str("y_unit", "", help="Unit of the values.")]
    outputs = ["table"]
    views = ["table-plot"]
    cache = "memory"

    def run(self, inputs, params):
        if not params.file:
            raise ValueError("No file chosen.")
        header, data = read_table(params.file)
        x = data[:, 0]
        table = xr.Dataset({"values": (("series", "x"), data[:, 1:].T, {"units": params.y_unit})},
                           coords={"x": ("x", x, {"units": params.x_unit}), "series": header[1:]},
                           attrs={"x_name": header[0], "source": os.path.basename(params.file)})
        if np.isnan(data).any():
            tesselab.warning(f"{int(np.isnan(data).sum())} cells are empty or not numbers.")
        return {"table": table}


class TablePlot(tesselab.View):
    name = "Table plot"
    help = "Every series of a table against its first column."
    inputs = {"table": Signature(values=Var("series", "x"), help="A table of series.")}
    plots = [Plot("table", "Table", draw=plot_table, help="Every series of the table against its first column.")]


def fill_template(folder, project):
    """A data folder -> the template's placeholders: the first table file in it."""
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith((".csv", ".txt", ".tsv", ".dat")))
    path = os.path.join(folder, names[0]) if names else folder
    return {"name": os.path.basename(folder), "file": os.path.relpath(path, project).replace("\\", "/")}
