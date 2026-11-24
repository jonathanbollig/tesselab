"""
Nodes: the steps of an analysis, and the only thing a module author writes. There are three kinds, drawn apart in
the graph (2026-11-24, the user: input, computation and output visually separated):

- a Reader reads files (it has a path parameter) and returns the data as measured;
- a Node (kind "compute") computes: inputs and parameters in, Datasets out, no plots;
- a View shows results: it takes the outputs of any stage as inputs and only draws plots. Each view is a section of
  the analysis's card; it has no outputs and no run().

    class QuantumYield(tesselab.Node):
        name = "Quantum yield"
        inputs = {"steps": Signature(delta_spectrum=Var("step", "wavelength"))}
        params = [P.Float("lambda_spectrum", 260e-9, unit="m", display="nm")]
        outputs = ["quantum_yields", "phi0_fit"]
        key_results = [Result("phi0", label="φ0", error="phi0_err")]

        def run(self, inputs, params):
            ...
            return {"quantum_yields": steps, "phi0_fit": fit}

    class QuantumYieldView(tesselab.View):
        name = "Quantum yield"
        inputs = {"quantum_yields": Signature(quantum_yield=Var("step"))}
        plots = [Plot("qy", "Quantum yield", draw=draw_qy)]

run() gets its inputs (bound xarray Datasets, see signature.py) and its parameters (values in their stored units,
paths absolute), and returns a dict of xarray Datasets. It must be a pure function: no globals, no files written, no
window. It may report warnings with diagnostic(); they appear in the window next to the node.

A view's draw functions get a DrawContext (ctx.inputs: its bound inputs, ctx.params, ctx.label, ...) and return a
PlotSpec. What they compute must be quick (an interpolation, a sum): anything slow belongs in a compute node, which
is cached and runs in a worker process. Mouse interactions on a view's plots set the view's own parameters; bound to
analysis variables, they reach the compute nodes bound to the same variables (that is how dragging λ in a plot
changes the quantum yield).
"""
import contextlib


class Bunch(dict):
    """A dict whose keys can also be read as attributes: params.fit_num, inputs.spectra."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No {key!r} (there is: {', '.join(self) or 'nothing'}).") from None


class Reads(Bunch):
    """A Bunch that notes which of its keys were read. A draw function's plot is drawn again only when something it
    read has changed (DrawContext.snapshot, same_as): dragging λ changes the quantum yield, but not the spectra that
    other plots of the same view show, and those are not drawn again."""

    def __init__(self, *args, **kwargs):
        self._read, self._all = set(), False  # (before the contents: dict.__init__ may call keys())
        super().__init__(*args, **kwargs)
        self._read.clear()

    def __getitem__(self, key):
        self._read.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self._read.add(key)
        return super().__contains__(key)

    def get(self, key, default=None):
        self._read.add(key)
        return super().get(key, default)

    def _everything(self):
        self._all = True

    def __iter__(self):
        self._everything()
        return super().__iter__()

    def __len__(self):
        self._everything()
        return super().__len__()

    def keys(self):
        self._everything()
        return super().keys()

    def values(self):
        self._everything()
        return super().values()

    def items(self):
        self._everything()
        return super().items()


class Result:
    """A single number worth showing in the header and saving with the analysis (e.g. φ0). It is read from the
    variable `name` of the node's outputs (output: which one; default: the first that has it); error: the variable of
    its standard error."""

    def __init__(self, name, label=None, unit="", display=None, error=None, output=None, digits=4):
        self.name, self.label, self.unit, self.display = name, label or name, unit, display
        self.error, self.output, self.digits = error, output, digits

    def read(self, outputs) -> tuple[float | None, float | None]:
        """(value, error) from the outputs; None where missing."""
        def find(name):
            if name is None:
                return None
            sources = [outputs[self.output]] if self.output in (outputs or {}) else (outputs or {}).values()
            for dataset in sources:
                if hasattr(dataset, "variables") and name in dataset.variables:
                    value = dataset[name]
                    if value.size == 1:
                        number = float(value.values.reshape(-1)[0])
                        return number if number == number else None
            return None
        return find(self.name), find(self.error)


class Plot:
    """A plot of a node. draw: function(ctx) -> PlotSpec (level 1); widget: a QWidget class with
    update(ctx) and optionally export() -> (QImage, dict) (level 2: anything a description cannot say, e.g. 3D).
    span, rows: 1 or 2 tiles wide / high. link: plots with the same link key zoom together across linked analyses
    (default: the plot's id). needs: output variables the plot needs; without them it shows a note. settings: names
    of parameters (usually redraw=True) shown in a row above the plot. items: "one", or "several" for a plot over all
    linked analyses (ctx.others)."""

    def __init__(self, id, title, draw=None, widget=None, span=1, rows=1, help="", link=None, needs=(),
                 settings=(), items="one"):
        if (draw is None) == (widget is None):
            raise ValueError(f"Plot {id!r}: give either draw (a description) or widget (an own widget).")
        if span not in (1, 2) or rows not in (1, 2):
            raise ValueError(f"Plot {id!r}: span and rows are 1 or 2.")
        if items not in ("one", "several"):
            raise ValueError(f"Plot {id!r}: items is 'one' or 'several'.")
        self.id, self.title, self.draw, self.widget = id, title, draw, widget
        self.span, self.rows, self.help, self.link = span, rows, help, link or id
        self.needs, self.settings, self.items = list(needs), list(settings), items

    @property
    def level(self):
        return 1 if self.draw is not None else 2


class Interaction:
    kind = "interaction"
    params: tuple = ()


class Cursor(Interaction):
    """A draggable line on plots that sets a parameter. axis: "x" (a vertical line) or "y". snap: an output
    variable whose values the line snaps to while dragged (typed values are used exactly). average: a parameter
    with a number of points to average over on each side (drawn as a band). link: cursors with the same key follow
    each other across linked analyses."""
    kind = "cursor"

    def __init__(self, param, axis="x", plots=(), snap=None, average=None, link=None, label=None, panel=None):
        if axis not in ("x", "y"):
            raise ValueError("Cursor: axis is 'x' or 'y'.")
        self.param, self.axis, self.plots, self.snap = param, axis, list(plots), snap
        self.average, self.link, self.label, self.panel = average, link or param, label, panel
        self.params = (param,) + ((average,) if average else ())


class Crosshair(Interaction):
    """A cross of two lines on a plot (e.g. a map): x and y are parameters."""
    kind = "crosshair"

    def __init__(self, x, y, plots=(), panel=None, snap_x=None, snap_y=None):
        self.x, self.y, self.plots, self.panel, self.snap_x, self.snap_y = x, y, list(plots), panel, snap_x, snap_y
        self.params = (x, y)


class RangeSelect(Interaction):
    """A draggable range on plots that sets a Range parameter (or the ranges of a Ranges parameter)."""
    kind = "range"

    def __init__(self, param, plots=(), axis="x", panel=None):
        self.param, self.plots, self.axis, self.panel = param, list(plots), axis, panel
        self.params = (param,)


class Pick(Interaction):
    """Points picked on a plot (Ctrl+click adds one, Ctrl+right click removes the nearest) into a Points parameter;
    'steps.chirp.onsets' reaches into a Steps parameter (the first step of that kind)."""
    kind = "pick"

    def __init__(self, param, plots=(), panel=None):
        self.param, self.plots, self.panel = param, list(plots), panel
        self.params = (param,)


class Toggle(Interaction):
    """A click on a point adds its key to an Ints parameter, or removes it (e.g. ignore a spectrum or use it again).
    The keys are those of the plot's Points marks (Points(..., keys=[...]))."""
    kind = "toggle"

    def __init__(self, param, plots=(), panel=None):
        self.param, self.plots, self.panel = param, list(plots), panel
        self.params = (param,)


class Action:
    """A button. run(ctx) returns nothing, a dict of new parameter values (one undoable change), or
    ("file", suggested_name, bytes) to offer for saving. job=True: runs in a worker process; the window stays
    usable meanwhile and the button shows that it is busy.
    enabled: function(ctx) -> bool."""

    def __init__(self, id, label, run, job=False, enabled=None, icon=None, help=""):
        self.id, self.label, self.run, self.job, self.enabled, self.icon, self.help = \
            id, label, run, job, enabled, icon, help


KINDS = ("reader", "compute", "view")


class Node:
    """Base class of all nodes; see the module docstring. Class attributes describe the node, run() computes.
    views: ids of views (in the same module) added with this node, connected to it (Add node → SVD also adds the
    SVD view)."""
    kind = "compute"
    name = ""
    help = ""
    views: list = []
    inputs: dict = {}
    outputs: list | dict = []
    params: list = []
    key_results: list = []
    plots: list = []
    interactions: list = []
    actions: list = []
    cache = "memory"  # or "disk": outputs also kept in tesselab/cache/ (slow nodes)
    schema = 1        # version of the parameters; raise it with a migrate() step when they change
    run_on = "change"  # or "button": runs only when asked (slow jobs such as MCR)

    def run(self, inputs, params) -> dict:
        raise NotImplementedError

    def migrate(self, params: dict, from_schema: int) -> dict:
        """Parameters saved with an older schema -> the current one."""
        return params

    @classmethod
    def output_names(cls) -> list[str]:
        return list(cls.outputs)


class Reader(Node):
    """A node that reads files (a path parameter) and returns the data as measured, with their units."""
    kind = "reader"


class View(Node):
    """A node that only shows: plots of its inputs (ctx.inputs), no outputs, no run(). Its interactions set its own
    parameters; bind them to analysis variables to reach the compute nodes."""
    kind = "view"

    def run(self, inputs, params):
        return {}


class DrawContext:
    """What a plot's draw function gets: inputs (a view's bound inputs; an input whose source has no result is
    missing: check with "name" in ctx.inputs), params, vars, label (of the analysis), instance (the node's name in the
    analysis), diagnostics, out (the outputs, for a compute node's own uses), and for items="several" plots: others,
    a list of (label, DrawContext) of the linked analyses' same view."""

    def __init__(self, out, inputs, params, label="", instance="", vars=None, diagnostics=(), others=()):
        self._out, self._diagnostics, self._others = Bunch(out or {}), list(diagnostics), list(others)
        self.inputs, self.params, self.vars = Reads(inputs or {}), Reads(params or {}), Reads(vars or {})
        self.label, self.instance = label, instance
        self._cacheable = True

    # out, diagnostics and others are not followed (they are not compared): reading one makes the plot uncacheable
    @property
    def out(self):
        self._cacheable = False
        return self._out

    @property
    def diagnostics(self):
        self._cacheable = False
        return self._diagnostics

    @property
    def others(self):
        self._cacheable = False
        return self._others

    def snapshot(self):
        """What the draw function that has just run read, to compare later contexts with (same_as); None when it
        cannot be told (it read out, diagnostics or others). Inputs are compared by identity (the same bound data
        object; the snapshot keeps them, so that an identity is never reused), parameters and variables by value."""
        if not self._cacheable:
            return None
        return {"label": (self.label, self.instance),
                "inputs": _snap(self.inputs, True), "params": _snap(self.params, False), "vars": _snap(self.vars, False)}

    def same_as(self, snapshot) -> bool:
        """Would the draw function read the same as in the snapshot? (Without noting what is read here.)"""
        return snapshot is not None and (self.label, self.instance) == snapshot["label"] and \
            _same(self.inputs, snapshot["inputs"], True) and _same(self.params, snapshot["params"], False) and \
            _same(self.vars, snapshot["vars"], False)


_MISSING = object()


def _token(value, identity):
    if identity:
        return tuple(id(v) for v in value) if isinstance(value, list) else id(value)
    return repr(value)


def _snap(bunch, identity):
    """(read everything?, {key: token of what was read}, the values read (kept alive for their identities))."""
    keys = list(dict.keys(bunch)) if bunch._all else sorted(bunch._read, key=str)
    values = {key: dict.get(bunch, key, _MISSING) for key in keys}
    return bunch._all, {key: _token(value, identity) for key, value in values.items()}, list(values.values())


def _same(bunch, snap, identity) -> bool:
    every, tokens, _ = snap
    if every and set(dict.keys(bunch)) != set(tokens):
        return False
    return all(_token(dict.get(bunch, key, _MISSING), identity) == token for key, token in tokens.items())


# diagnostics reported while a node runs (one node at a time per process)
_collected = None


def diagnostic(level, message, plots=(), code=None):
    """Report something about the result: level "info", "warning" or "error". plots: ids of the plots it concerns;
    code: a stable key, so that a user can switch off a known warning."""
    if level not in ("info", "warning", "error"):
        raise ValueError("diagnostic: level is 'info', 'warning' or 'error'.")
    entry = {"level": level, "message": str(message), "plots": list(plots), "code": code}
    if _collected is not None:
        _collected.append(entry)
    return entry


def warning(message, plots=(), code=None):
    return diagnostic("warning", message, plots, code)


@contextlib.contextmanager
def collecting():
    """Collect the diagnostics reported inside the block (the runner wraps run() in it)."""
    global _collected
    previous, _collected = _collected, []
    try:
        yield _collected
    finally:
        _collected = previous
