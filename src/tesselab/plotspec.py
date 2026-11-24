"""
The plot description (level 1): what a node's plot function returns. Tesselab draws it in the window (gui/plots.py),
in the PDF export (export.py) and in the check's pictures, and writes it into the exported JSON. A module never draws
itself; that is what makes linking, numbering, export and a later change of the drawing library possible.

    PlotSpec(x=Axis("Wavelength", "m", display="nm"), y=Axis("Absorbance", "OD"),
             marks=[Lines(wl, absorbance_2d, colormap="black-red"), VLine(260e-9, label="λ")])

A plot with several panels (shared axes, e.g. a map with profiles):

    PlotSpec(panels={"map": Panel(x=..., y=..., marks=[Heatmap(...)]), "right": Panel(...), "bottom": Panel(...)},
             mosaic=[["map", "right"], ["bottom", None]], share_x=[["map", "bottom"]], share_y=[["map", "right"]],
             sizes=([3, 1], [3, 1]))

Data are given in the axes' stored units (numpy arrays, xarray DataArrays or lists). Colors are "#rrggbb" or a role
("data", "fit", "accent", "warning", "muted", "series:3") that follows the user's theme. The marks and their fields are
part of the frozen protocol; later versions only add.
"""
from dataclasses import dataclass, field

import numpy as np

from tesselab import units

ROLES = ("data", "fit", "accent", "warning", "error", "good", "muted", "ink", "positive", "negative")
COLORMAPS = ("viridis", "diverging", "black-red", "default", "greys")
DASHES = ("solid", "dash", "dot", "dashdot")
SYMBOLS = ("o", "s", "t", "d", "+", "x", "_")
# what a mark is, for the theme (colors.py: each theme's "marks" says how each looks; a user's theme file can change
# them for every module at once; a field the module sets itself wins)
MARK_ROLES = ("measured", "fit", "result", "reference", "unused", "event")
MAX_EXPORT_POINTS = 4000


def _array(values, name, ndim=1):
    if values is None:
        return None
    array = np.asarray(getattr(values, "values", values), dtype=float)
    if array.ndim == 0 and ndim == 1:
        array = array.reshape(1)
    if array.ndim != ndim:
        raise ValueError(f"{name}: {ndim}-dimensional data expected, got shape {array.shape}.")
    return array


def _check_color(color, name):
    if color is None:
        return
    if isinstance(color, str) and (color in ROLES or (color.startswith("series:") and color[7:].isdigit())
                                   or (color.startswith("#") and len(color) in (4, 7, 9))):
        return
    raise ValueError(f"{name}: color {color!r} is no '#rrggbb' and no role ({', '.join(ROLES)}, series:N).")


@dataclass
class Style:
    """How a mark looks; everything optional (Tesselab's look otherwise). role: what the mark is (MARK_ROLES:
    "measured" data points, a "fit", the "result", an expected or literature "reference", "unused" (excluded) data,
    an "event" in time); the theme gives each role its look, and fields set here override it."""
    role: str | None = None
    color: str | None = None
    width: float | None = None
    dash: str | None = None      # solid, dash, dot, dashdot
    symbol: str | None = None    # o, s, t, d, +, x, _ (a short horizontal bar)
    size: float | None = None    # of symbols, in pixels
    alpha: float | None = None   # 0 … 1

    def __post_init__(self):
        if self.role is not None and self.role not in MARK_ROLES:
            raise ValueError(f"Style: role {self.role!r} is not one of {MARK_ROLES}.")
        _check_color(self.color, "Style")
        if self.dash is not None and self.dash not in DASHES:
            raise ValueError(f"Style: dash {self.dash!r} is not one of {DASHES}.")
        if self.symbol is not None and self.symbol not in SYMBOLS:
            raise ValueError(f"Style: symbol {self.symbol!r} is not one of {SYMBOLS}.")

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


class Mark:
    """Common to all marks: name (legend, tool tips, export), style, axis ("left" or "right")."""
    kind = "mark"

    def _common(self, name, style, axis):
        if axis not in ("left", "right"):
            raise ValueError(f"{self.kind}: axis is 'left' or 'right', not {axis!r}.")
        self.name, self.style, self.axis = name, style or Style(), axis

    def to_dict(self, limit=True):
        data = {"kind": self.kind, "name": self.name, "axis": self.axis, "style": self.style.to_dict()}
        for key, value in self.__dict__.items():
            if key in ("name", "style", "axis"):
                continue
            data[key] = _to_json(value)
        if limit:
            _limit_points(data)
        return data

    def bounds(self):
        """(xmin, xmax, ymin, ymax) of the finite data, None for no data."""
        return None


def _to_json(value):
    if isinstance(value, np.ndarray):
        return [None if not np.isfinite(v) else float(v) for v in value.ravel()] if value.ndim == 1 else \
            [_to_json(row) for row in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    return value


def _limit_points(data):
    """Lines with more than MAX_EXPORT_POINTS points: min and max per bin, so the export stays small but looks
    the same."""
    x, y = data.get("x"), data.get("y")
    if data["kind"] != "line" or not isinstance(x, list) or len(x) <= MAX_EXPORT_POINTS:
        return
    xa = np.array([np.nan if v is None else v for v in x])
    ya = np.array([np.nan if v is None else v for v in y])
    bins = np.array_split(np.arange(len(xa)), MAX_EXPORT_POINTS // 2)
    nx, ny = [], []
    for b in bins:
        segment = ya[b]
        if np.all(np.isnan(segment)):
            continue
        i, j = np.nanargmin(segment), np.nanargmax(segment)
        for k in sorted({i, j}):
            nx.append(float(xa[b][k]))
            ny.append(float(segment[k]))
    data["x"], data["y"], data["binned"] = nx, ny, True


def _finite_bounds(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y) if x.shape == y.shape else None
    xs = x[mask] if mask is not None else x[np.isfinite(x)]
    ys = y[mask] if mask is not None else y[np.isfinite(y)]
    if not len(xs) or not len(ys):
        return None
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


class Line(Mark):
    """A line through (x, y). error_y: a band of ± error around it. step: drawn as steps."""
    kind = "line"

    def __init__(self, x, y, name=None, style=None, axis="left", error_y=None, step=False):
        self._common(name, style, axis)
        self.x, self.y = _array(x, "Line x"), _array(y, "Line y")
        self.error_y, self.step = _array(error_y, "Line error_y"), bool(step)
        if len(self.x) != len(self.y):
            raise ValueError(f"Line {name or ''}: x has {len(self.x)} values, y has {len(self.y)}.")

    def bounds(self):
        return _finite_bounds(self.x, self.y)


class Points(Mark):
    """Symbols at (x, y), optionally with error bars. keys: an int per point for a Toggle interaction (clicking a
    point adds its key to a parameter or removes it)."""
    kind = "points"

    def __init__(self, x, y, name=None, style=None, axis="left", error_x=None, error_y=None, labels=None, keys=None):
        self._common(name, style, axis)
        self.x, self.y = _array(x, "Points x"), _array(y, "Points y")
        self.error_x, self.error_y = _array(error_x, "Points error_x"), _array(error_y, "Points error_y")
        self.labels = list(labels) if labels is not None else None  # tool tip per point
        # keys: an int per point, what a Toggle interaction adds to or removes from its parameter on a click
        self.keys = [int(k) for k in keys] if keys is not None else None
        if len(self.x) != len(self.y):
            raise ValueError(f"Points {name or ''}: x has {len(self.x)} values, y has {len(self.y)}.")
        if self.keys is not None and len(self.keys) != len(self.x):
            raise ValueError(f"Points {name or ''}: one key per point.")

    def bounds(self):
        return _finite_bounds(self.x, self.y)


class Lines(Mark):
    """Many lines on one x: y has one row per line (e.g. all spectra). colors: one per line, or colormap
    ("black-red", "viridis", "default"); labels: one per line (tool tips, and legend entries when there are at most
    12 lines and legend is True)."""
    kind = "lines"

    def __init__(self, x, y, name=None, style=None, axis="left", colors=None, colormap="default", labels=None,
                 legend=True):
        self._common(name, style, axis)
        self.legend = bool(legend)  # False: the labels are tool tips only, never legend entries
        self.x, self.y = _array(x, "Lines x"), _array(y, "Lines y", ndim=2)
        if self.y.shape[1] != len(self.x):
            raise ValueError(f"Lines {name or ''}: y needs one row per line with {len(self.x)} values, "
                             f"got shape {self.y.shape}.")
        if colors is not None:
            for color in colors:
                _check_color(color, "Lines colors")
        if colormap not in COLORMAPS:
            raise ValueError(f"Lines: colormap {colormap!r} is not one of {COLORMAPS}.")
        self.colors, self.colormap = list(colors) if colors is not None else None, colormap
        self.labels = [str(label) for label in labels] if labels is not None else None

    def bounds(self):
        return _finite_bounds(np.broadcast_to(self.x, self.y.shape), self.y)


class Band(Mark):
    """A filled area between low and high."""
    kind = "band"

    def __init__(self, x, low, high, name=None, style=None, axis="left"):
        self._common(name, style, axis)
        self.x, self.low, self.high = _array(x, "Band x"), _array(low, "Band low"), _array(high, "Band high")

    def bounds(self):
        b1, b2 = _finite_bounds(self.x, self.low), _finite_bounds(self.x, self.high)
        if not b1 or not b2:
            return b1 or b2
        return min(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), max(b1[3], b2[3])


class VLine(Mark):
    """A vertical line at x (fixed; a draggable line is a Cursor interaction)."""
    kind = "vline"

    def __init__(self, x, name=None, style=None, axis="left", label=None):
        self._common(name, style, axis)
        self.x, self.label = float(x), label


class HLine(Mark):
    """A horizontal line at y."""
    kind = "hline"

    def __init__(self, y, name=None, style=None, axis="left", label=None):
        self._common(name, style, axis)
        self.y, self.label = float(y), label


class Region(Mark):
    """Shaded spans along x, from starts[i] to stops[i] (e.g. stirring times, a baseline range). labels: a tool tip
    per span."""
    kind = "region"

    def __init__(self, starts, stops, name=None, style=None, axis="left", label=None, labels=None):
        self._common(name, style, axis)
        self.starts, self.stops, self.label = _array(starts, "Region starts"), _array(stops, "Region stops"), label
        if len(self.starts) != len(self.stops):
            raise ValueError("Region: as many starts as stops.")
        self.labels = [str(label) for label in labels] if labels is not None else None
        if self.labels is not None and len(self.labels) != len(self.starts):
            raise ValueError("Region: one label per span.")


class Heatmap(Mark):
    """Colored cells: z has shape (len(x), len(y)); x and y are the cell centers (not necessarily equidistant).
    limits: (low, high) of the colors; None: automatic (symmetric around 0 for the diverging colormap, without the
    most extreme 0.2 %)."""
    kind = "heatmap"

    def __init__(self, x, y, z, name=None, style=None, axis="left", colormap="diverging", limits=None,
                 colorbar=None):
        self._common(name, style, axis)
        self.x, self.y, self.z = _array(x, "Heatmap x"), _array(y, "Heatmap y"), _array(z, "Heatmap z", ndim=2)
        if self.z.shape != (len(self.x), len(self.y)):
            raise ValueError(f"Heatmap: z must have shape (len(x), len(y)) = {(len(self.x), len(self.y))}, "
                             f"got {self.z.shape}.")
        if colormap not in COLORMAPS:
            raise ValueError(f"Heatmap: colormap {colormap!r} is not one of {COLORMAPS}.")
        self.colormap, self.colorbar = colormap, colorbar  # colorbar: its label, None: no color bar
        self.limits = None if limits is None else (float(limits[0]), float(limits[1]))

    def color_limits(self):
        if self.limits is not None:
            return self.limits
        finite = self.z[np.isfinite(self.z)]
        if not len(finite):
            return (-1.0, 1.0)
        if self.colormap == "diverging":
            limit = float(np.percentile(np.abs(finite), 99.8)) or 1.0
            return (-limit, limit)
        return (float(np.percentile(finite, 0.1)), float(np.percentile(finite, 99.9)))

    def bounds(self):
        if not len(self.x) or not len(self.y):
            return None
        return float(np.nanmin(self.x)), float(np.nanmax(self.x)), float(np.nanmin(self.y)), float(np.nanmax(self.y))


class Text(Mark):
    """A text at (x, y): coords="data" (in the axes' units) or "tile" (0 … 1 from the lower left corner)."""
    kind = "text"

    def __init__(self, x, y, text, name=None, style=None, axis="left", coords="data", anchor="left"):
        self._common(name, style, axis)
        if coords not in ("data", "tile"):
            raise ValueError("Text: coords is 'data' or 'tile'.")
        self.x, self.y, self.text, self.coords, self.anchor = float(x), float(y), str(text), coords, anchor


@dataclass
class Axis:
    """An axis: its label, the unit the data are in, and optionally another unit to show (display, or a list of
    units the user can switch between, the first shown at first). scale: linear, log, or linlog (linear up to
    ± linthresh, logarithmic beyond)."""
    label: str = ""
    unit: str = ""
    display: str | list | None = None
    scale: str = "linear"
    linthresh: float | None = None
    invert: bool = False
    limits: tuple | None = None

    def __post_init__(self):
        if self.scale not in ("linear", "log", "linlog"):
            raise ValueError(f"Axis {self.label!r}: scale is linear, log or linlog, not {self.scale!r}.")
        if self.scale == "linlog" and not self.linthresh:
            raise ValueError(f"Axis {self.label!r}: a linlog axis needs linthresh (where the log part starts).")
        for display in self.displays:
            if display and not units.is_known(self.unit, display):
                raise ValueError(f"Axis {self.label!r}: unknown display unit {display!r} for {self.unit!r}.")

    @property
    def displays(self) -> list:
        return list(self.display) if isinstance(self.display, (list, tuple)) else [self.display]

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


@dataclass
class Panel:
    """One area of a plot with its own axes and marks."""
    x: Axis = field(default_factory=Axis)
    y: Axis = field(default_factory=Axis)
    marks: list = field(default_factory=list)
    y_right: Axis | None = None
    legend: str | None = None       # "top right", "top left", "bottom right", "bottom left", "below"; None: none
    zero_line: bool = True
    grid: bool = True
    note: str = ""                  # shown in the middle, e.g. why the panel is empty

    def __post_init__(self):
        for mark in self.marks:
            if not isinstance(mark, Mark):
                raise ValueError(f"Panel: {mark!r} is no mark (Line, Points, Lines, Band, VLine, HLine, Region, "
                                 f"Heatmap, Text).")
            if mark.axis == "right" and self.y_right is None:
                raise ValueError(f"Panel: the mark {mark.name or mark.kind!r} is on the right axis, but there is none "
                                 f"(y_right).")

    def to_dict(self):
        return {"x": self.x.to_dict(), "y": self.y.to_dict(),
                "y_right": self.y_right.to_dict() if self.y_right else None, "legend": self.legend,
                "zero_line": self.zero_line, "grid": self.grid, "note": self.note,
                "marks": [mark.to_dict() for mark in self.marks]}


class PlotSpec:
    """A plot: one panel (x, y, marks, ...) or several (panels, mosaic, share_x, share_y, sizes). note: a text in
    the middle of the plot, e.g. "Set the nucleosides" when there is nothing to show."""

    def __init__(self, x=None, y=None, marks=(), y_right=None, legend=None, zero_line=True, grid=True, note="",
                 panels=None, mosaic=None, share_x=(), share_y=(), sizes=None):
        if panels is None:
            panels = {"main": Panel(x or Axis(), y or Axis(), list(marks), y_right, legend, zero_line, grid, note)}
            mosaic = [["main"]]
        elif mosaic is None:
            mosaic = [[name] for name in panels]
        for name, panel in panels.items():
            if not isinstance(panel, Panel):
                raise ValueError(f"PlotSpec: panel {name!r} is no Panel.")
        names = {name for row in mosaic for name in row if name is not None}
        if names != set(panels):
            raise ValueError(f"PlotSpec: the mosaic names {sorted(names)}, the panels are {sorted(panels)}.")
        width = len(mosaic[0])
        if any(len(row) != width for row in mosaic):
            raise ValueError("PlotSpec: every row of the mosaic needs the same number of cells.")
        for group in list(share_x) + list(share_y):
            for name in group:
                if name not in panels:
                    raise ValueError(f"PlotSpec: share_x/share_y name {name!r}, which is no panel.")
        rows, columns = len(mosaic), width
        if sizes is None:
            sizes = ([1] * columns, [1] * rows)
        if len(sizes[0]) != columns or len(sizes[1]) != rows:
            raise ValueError(f"PlotSpec: sizes needs {columns} column widths and {rows} row heights.")
        self.panels, self.mosaic = panels, [list(row) for row in mosaic]
        self.share_x, self.share_y = [list(g) for g in share_x], [list(g) for g in share_y]
        self.sizes = (list(sizes[0]), list(sizes[1]))
        self.note = note if len(panels) > 1 else ""

    def cells(self):
        """{panel name: (row, column, row span, column span)} from the mosaic (a panel may cover several cells)."""
        found = {}
        for r, row in enumerate(self.mosaic):
            for c, name in enumerate(row):
                if name is None:
                    continue
                r0, c0, r1, c1 = found.get(name, (r, c, r, c))
                found[name] = (min(r0, r), min(c0, c), max(r1, r), max(c1, c))
        return {name: (r0, c0, r1 - r0 + 1, c1 - c0 + 1) for name, (r0, c0, r1, c1) in found.items()}

    def to_dict(self, limit=True):
        return {"panels": {name: panel.to_dict() for name, panel in self.panels.items()},
                "mosaic": self.mosaic, "share_x": self.share_x, "share_y": self.share_y,
                "sizes": [self.sizes[0], self.sizes[1]], "note": self.note}


def note_only(text, x=None, y=None) -> PlotSpec:
    """An empty plot that says why it is empty."""
    return PlotSpec(x=x, y=y, note=text)
