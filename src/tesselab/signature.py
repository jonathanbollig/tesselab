"""
Data signatures: what shape of data a node's input needs, independent of which node produces it.

    Signature(matrix=Var("n", "m"), row_pos=Var("n", optional=True), col_pos=Var("m", units="m"))

The names in Var are placeholders for dimensions; the same placeholder means the same dimension. Tesselab matches a
signature against an output (an xarray Dataset) by its variables' dimensions: `bindings` lists every consistent
assignment (a variable with the signature's own name first, so names decide where they fit), `bind` applies one. The bound Dataset has the signature's variable names and placeholder dimensions (the
matrix transposed into the declared order); variables of the output that the signature does not name are kept, so
a node can still use them. The original names are kept in attrs["tesselab_names"] / attrs["tesselab_dims"].
"""
import itertools


class Var:
    """A variable with these dimension placeholders. units: a required unit string (only where the node computes
    physically with it); optional: the node also works without it; numeric: only numbers fit (default; text
    coordinates such as series names are no positions)."""

    def __init__(self, *dims, units=None, optional=False, numeric=True):
        self.dims, self.units, self.optional, self.numeric = tuple(dims), units, optional, numeric

    def __repr__(self):
        return f"Var({', '.join(map(repr, self.dims))})"


class Signature:
    """The variables an input needs. optional: the whole input may be missing. many: a list of datasets, one per
    analysis (overlays, a joint SVD). help: what the input is for (the graph window's tooltip)."""

    def __init__(self, optional=False, many=False, help="", **variables):
        if not variables:
            raise ValueError("A signature needs at least one variable.")
        self.optional, self.many, self.help, self.variables = optional, many, help, variables

    @property
    def required(self):
        return {name: var for name, var in self.variables.items() if not var.optional}

    def __repr__(self):
        return f"Signature({', '.join(f'{k}={v!r}' for k, v in self.variables.items())})"


def _candidates(var, dataset, wanted=None):
    """Variables (data variables and coordinates) of the dataset that could be `var`: same number of dimensions,
    the required unit, numbers. A variable whose name is the signature's own name (wanted) comes first, so that
    names decide wherever they fit and shapes only where they do not."""
    found = []
    names = list(dataset.data_vars) + list(dataset.coords)
    if wanted in names:
        names.remove(wanted)
        names.insert(0, wanted)
    for name in names:
        array = dataset[name]
        if array.ndim != len(var.dims):
            continue
        if var.units is not None and array.attrs.get("units") != var.units:
            continue
        if var.numeric and array.dtype.kind not in "biufc":
            continue
        found.append(name)
    return found


def bindings(signature, dataset) -> list[dict]:
    """All consistent assignments {signature variable: dataset variable or None}; the first is the default. An empty
    list: the dataset does not fit. Required variables are assigned first; optional ones get every candidate that
    fits the dimensions already fixed (or None if there is none)."""
    names = list(signature.required) + [n for n in signature.variables if n not in signature.required]
    results = []

    def assign(i, chosen, dims):
        if i == len(names):
            results.append(dict(chosen))
            return
        name = names[i]
        var = signature.variables[name]
        options = []
        for candidate in _candidates(var, dataset, name):
            if candidate in chosen.values():
                continue
            # every order of the candidate's dimensions that agrees with the placeholders fixed so far (a matrix
            # can be bound either way round; the order as stored comes first)
            for order in itertools.permutations(dataset[candidate].dims):
                new = dict(dims)
                if all(new.setdefault(p, d) == d for p, d in zip(var.dims, order)) and \
                        len(set(new.values())) == len(new):  # different placeholders, different dimensions
                    options.append((candidate, new))
        if not options and var.optional:
            options = [(None, dims)]
        for candidate, new in options:
            chosen[name] = candidate
            assign(i + 1, chosen, new)
            del chosen[name]

    assign(0, {}, {})
    # identical bindings can come from several orders; keep the first of each
    unique = []
    for binding in results:
        if binding not in unique:
            unique.append(binding)
    return unique


def dims_of(signature, dataset, binding) -> dict:
    """{placeholder: dataset dimension} for a binding: an assignment that fits every bound variable."""
    bound = [(name, source) for name, source in binding.items() if source is not None]

    def search(i, dims):
        if i == len(bound):
            return dims
        name, source = bound[i]
        var = signature.variables[name]
        for order in itertools.permutations(dataset[source].dims):
            trial = dict(dims)
            if all(trial.setdefault(p, d) == d for p, d in zip(var.dims, order)) and \
                    len(set(trial.values())) == len(trial):
                found = search(i + 1, trial)
                if found is not None:
                    return found
        return None

    return search(0, {}) or {}


def bind(signature, dataset, binding=None):
    """The dataset as the node sees it: renamed to the signature's names, the bound variables transposed into the
    declared order. binding: a choice from `bindings` (or a partial one saved with the edge); None: the first."""
    options = bindings(signature, dataset)
    if not options:
        raise ValueError(f"The data do not fit {signature!r} (variables: {', '.join(map(str, dataset.variables))}).")
    chosen = options[0]
    if binding:  # the user's choice for some variables, the first fitting option for the rest
        fitting = [o for o in options if all(o.get(k) == v for k, v in binding.items() if k in o)]
        if not fitting:
            raise ValueError(f"The saved choice {binding} does not fit the data any more.")
        chosen = fitting[0]
    dims = dims_of(signature, dataset, chosen)
    renamed_dims = {d: p for p, d in dims.items() if p != d}
    out = dataset.copy()
    # names of the signature win: drop other variables that are in the way
    rename_vars = {source: name for name, source in chosen.items() if source is not None and source != name}
    in_the_way = [n for n in rename_vars.values() if n in out.variables and n not in rename_vars]
    out = out.drop_vars([n for n in in_the_way if n not in out.dims])
    # coordinates that are bound to a name of the signature become data variables of that name (a dimension
    # coordinate keeps its place too: e.g. col_pos = the wavelength coordinate)
    for source, name in rename_vars.items():
        out[name] = out[source].variable.copy() if source in out.dims else out[source].reset_coords(drop=True)
    if renamed_dims:
        # in two steps, so that a placeholder may be the name of another dimension of the data (e.g. SVD weights
        # (n, component) bound as (n, m) while the data also have an m): first to temporary names, the dimensions in
        # the way get a trailing _, then to the placeholders
        temporary = {d: f"__tesselab_{i}" for i, d in enumerate(renamed_dims)}
        out = out.rename_dims(temporary)
        in_the_way = {p: p + "_" for p in renamed_dims.values() if p in out.dims}
        while any(new in out.dims for new in in_the_way.values()):
            in_the_way = {p: new + "_" for p, new in in_the_way.items()}
        if in_the_way:
            out = out.rename_dims(in_the_way)
        out = out.rename_dims({temporary[d]: p for d, p in renamed_dims.items()})
    for name, source in chosen.items():
        if source is not None:
            var = signature.variables[name]
            out[name] = out[name].transpose(*var.dims) if set(out[name].dims) == set(var.dims) else out[name]
    out.attrs = {**dataset.attrs, "tesselab_names": {k: v for k, v in chosen.items() if v},
                 "tesselab_dims": dims}
    return out


def fits(signature, dataset) -> bool:
    return bool(bindings(signature, dataset))
