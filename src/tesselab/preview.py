"""
A look at any output without a view (the graph window's side panel): Tesselab draws a variable of a Dataset from its
shape alone, so that a reader or compute node can be checked while the graph is built. For inspection only: no
interactions, not numbered, not exported (the views are what the analysis shows).

    single numbers  ->  a table (scalars)
    1-D             ->  a line over its coordinate (points for a few values)
    2-D             ->  one line per row when one dimension is short (≤ 60, e.g. 5 nucleosides, 50 spectra),
                        else a heatmap
"""
import numpy as np

from tesselab.plotspec import Axis, Heatmap, Line, Lines, PlotSpec, Points, Style, note_only

MAX_LINES = 60


def _numeric(array) -> bool:
    return np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)


def variables(dataset) -> list[str]:
    """The data variables worth a plot: numeric, with one or two dimensions."""
    return [str(n) for n, v in dataset.data_vars.items() if v.ndim in (1, 2) and _numeric(v)]


def scalars(dataset) -> list[tuple[str, str]]:
    """[(name, value as text)] of the single numbers of a Dataset."""
    found = []
    for name, variable in dataset.data_vars.items():
        if variable.ndim == 0:
            value = variable.values.item()
            unit = variable.attrs.get("units", "")
            text = f"{value:.6g}" if isinstance(value, (int, float, np.number)) else str(value)
            found.append((str(name), f"{text} {unit}".strip()))
    return found


def _axis(name, array):
    """An axis for a coordinate or variable, in a readable unit where the stored one gives tiny numbers."""
    unit = str(array.attrs.get("units", "")) if hasattr(array, "attrs") else ""
    values = np.asarray(array.values if hasattr(array, "values") else array, dtype=float)
    largest = float(np.nanmax(np.abs(values))) if values.size and np.isfinite(values).any() else 0.0
    display = None
    if unit == "m" and 0 < largest < 1e-5:
        display = "nm"
    elif unit == "s" and 0 < largest < 1e-6:
        display = "ps"
    elif unit == "s" and largest > 600:
        display = "min"
    elif unit == "OD" and 0 < largest < 0.05:
        display = "mOD"
    try:
        return Axis(str(name), unit, display=display)
    except ValueError:  # a unit Tesselab does not convert: shown as it is
        return Axis(f"{name} [{unit}]" if unit else str(name))


def _coordinate(dataset, variable, dim):
    """The coordinate along a dimension (numeric), else the index."""
    if dim in dataset.coords and dataset[dim].ndim == 1 and _numeric(dataset[dim]):
        return dataset[dim]
    return np.arange(variable.sizes[dim], dtype=float)


def preview_spec(dataset, name) -> PlotSpec:
    """A plot of one variable of a Dataset, chosen by its shape."""
    if name not in dataset.variables:
        return note_only(f"No {name}")
    variable = dataset[name]
    if not _numeric(variable):
        return note_only(f"{name} is not numeric")
    y_axis = _axis(name, variable)
    if variable.ndim == 1:
        dim = variable.dims[0]
        x = _coordinate(dataset, variable, dim)
        x_axis = _axis(dim, x)
        if variable.size <= 30:
            mark = Points(x, variable.values, style=Style(color="data"))
        else:
            mark = Line(x, variable.values, style=Style(color="data"))
        return PlotSpec(x=x_axis, y=y_axis, marks=[mark])
    if variable.ndim == 2:
        rows, columns = variable.dims
        if variable.sizes[rows] > variable.sizes[columns]:  # the lines run along the longer dimension
            rows, columns = columns, rows
        values = variable.transpose(rows, columns).values.astype(float)
        x = _coordinate(dataset, variable, columns)
        if variable.sizes[rows] <= MAX_LINES:
            labels = [f"{rows} {v}" for v in (dataset[rows].values if rows in dataset.coords
                                              else range(variable.sizes[rows]))]
            colormap = "default" if variable.sizes[rows] <= 10 else "viridis"
            return PlotSpec(x=_axis(columns, x), y=y_axis, marks=[Lines(x, values, labels=labels, colormap=colormap)],
                            legend="top right" if variable.sizes[rows] <= 6 else None)
        y = _coordinate(dataset, variable, rows)
        return PlotSpec(x=_axis(columns, x), y=_axis(rows, y), zero_line=False,
                        marks=[Heatmap(x, y, values.T, colormap="viridis", colorbar=name)])
    return note_only(f"{name} has {variable.ndim} dimensions: no preview")
