"""
Parameter types: what a node declares as its settings. Tesselab builds the form for them, checks values, saves them
as JSON and makes every change undoable. Used as `P` in module code:

    from tesselab import P
    params = [P.Float("lambda_spectrum", 260e-9, unit="m", display="nm"), P.Int("fit_num", 5, min=2)]

Every type turns a value into its JSON form and back (check), and into text for the form's input field and back
(to_text, from_text). Values are stored in SI units (or the unit given); `display` changes only the form and plots.
The JSON forms are part of the frozen protocol: see the reference in AGENTS.md.
"""
import copy
import re

from tesselab import units

NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
RANGE = re.compile(rf"^\s*({NUMBER})\s*(?:-|–|to|,|\s)\s*({NUMBER})\s*$")


def _format_number(value) -> str:
    """260.0 -> '260', 259.98712158 -> '259.987122', 1e-12 -> '1e-12'."""
    if value is None:
        return ""
    if value != 0 and (abs(value) < 1e-4 or abs(value) >= 1e7):
        return f"{value:.6g}"
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


class Param:
    """Common fields of all parameter types.

    name: the key in the analysis file (frozen once published); default: its value when nothing is set;
    label: shown in the form (default: the name); help: hover text; group: a heading in the form;
    advanced: hidden unless Advanced is on; optional: null (None) is allowed and means "not set";
    redraw: changing it only redraws the plots (a plot setting), the node's run is not repeated."""
    type_name = "param"

    def __init__(self, name, default=None, *, label=None, help="", group=None, advanced=False, optional=False,
                 redraw=False):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Parameter name {name!r}: letters, digits and _ only (it is a JSON key).")
        self.name, self.label, self.help, self.group = name, label or name.replace("_", " "), help, group
        self.advanced, self.optional, self.redraw = advanced, optional, redraw
        self.default = None if default is None else self.check(default)
        if self.default is None and not optional:
            self.optional = True  # a parameter without a default can be empty

    def default_value(self):
        return copy.deepcopy(self.default)

    def check(self, value):
        """The value in its normal JSON form; ValueError if it is not valid."""
        if value is None:
            if self.optional or not hasattr(self, "default"):
                return None
            raise ValueError(f"{self.name}: a value is needed.")
        return self._check(value)

    def _check(self, value):
        return value

    def to_text(self, value) -> str:
        return "" if value is None else str(value)

    def from_text(self, text: str):
        return self.check(text.strip() or None)

    def describe(self) -> dict:
        """For the check report and the documentation."""
        return {"name": self.name, "type": self.type_name, "default": self.default}

    def __repr__(self):
        return f"P.{type(self).__name__}({self.name!r})"


class Bool(Param):
    type_name = "bool"

    def __init__(self, name, default=False, **kwargs):
        super().__init__(name, default, **kwargs)

    def _check(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "yes", "1", "false", "no", "0"):
            return value.lower() in ("true", "yes", "1")
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        raise ValueError(f"{self.name}: true or false, not {value!r}.")


class Int(Param):
    type_name = "int"

    def __init__(self, name, default=None, *, min=None, max=None, **kwargs):
        self.min, self.max = min, max
        super().__init__(name, default, **kwargs)

    def _check(self, value):
        if isinstance(value, bool):
            raise ValueError(f"{self.name}: a whole number, not {value!r}.")
        try:
            number = int(value) if not isinstance(value, float) or value.is_integer() else None
        except (TypeError, ValueError):
            number = None
        if number is None:
            raise ValueError(f"{self.name}: a whole number, not {value!r}.")
        if self.min is not None and number < self.min or self.max is not None and number > self.max:
            raise ValueError(f"{self.name}: {number} is outside {self.min} … {self.max}.")
        return number


class Float(Param):
    type_name = "float"

    def __init__(self, name, default=None, *, unit="", display=None, min=None, max=None, **kwargs):
        if display and not units.is_known(unit, display):
            raise ValueError(f"{name}: unknown display unit {display!r} for {unit!r}.")
        self.unit, self.display, self.min, self.max = unit, display, min, max
        super().__init__(name, default, **kwargs)

    def _check(self, value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.name}: a number, not {value!r}.") from None
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError(f"{self.name}: NaN and infinity cannot be saved.")
        if self.min is not None and number < self.min or self.max is not None and number > self.max:
            # in the unit the field shows (260 nm, not 2.6e-07); an open end (min or max None) as ∞
            shown = lambda v, end: self.to_text(v) if v is not None else end
            unit = f" {self.display or self.unit}" if self.display or self.unit else ""
            raise ValueError(f"{self.name}: {shown(number, '')}{unit} is outside "
                             f"{shown(self.min, '-∞')} … {shown(self.max, '∞')}{unit}.")
        return number

    def to_text(self, value):
        return "" if value is None else _format_number(units.to_display(value, self.unit, self.display))

    def from_text(self, text):
        text = text.strip()
        if not text:
            return self.check(None)
        try:
            shown = float(text)
        except ValueError:
            raise ValueError(f"{self.name}: a number, not {text!r}.") from None
        return self.check(units.from_display(shown, self.unit, self.display))


class Str(Param):
    type_name = "str"

    def __init__(self, name, default="", *, multiline=False, **kwargs):
        self.multiline = multiline
        super().__init__(name, default, **kwargs)

    def _check(self, value):
        if not isinstance(value, str):
            raise ValueError(f"{self.name}: text, not {value!r}.")
        return value

    def from_text(self, text):
        return self.check(text if text or not self.optional else None)


class Choice(Param):
    """One of a few options. options: a list of keys, or a dict {key: label}; the file stores the key."""
    type_name = "choice"

    def __init__(self, name, default=None, *, options, **kwargs):
        self.options = dict(options) if isinstance(options, dict) else {key: str(key) for key in options}
        if default is None and not kwargs.get("optional"):
            default = next(iter(self.options))
        super().__init__(name, default, **kwargs)

    def _check(self, value):
        if value not in self.options:
            raise ValueError(f"{self.name}: one of {', '.join(map(str, self.options))}, not {value!r}.")
        return value


class Path(Param):
    """A file or folder, relative to the project folder (written with /) or absolute. run() gets it absolute."""
    type_name = "path"

    def __init__(self, name, default=None, *, filter="", folder=False, **kwargs):
        self.filter, self.folder = filter, folder
        super().__init__(name, default, **kwargs)

    def _check(self, value):
        if not isinstance(value, str):
            raise ValueError(f"{self.name}: a path, not {value!r}.")
        return value.replace("\\", "/") if value else None if self.optional else value


class Paths(Param):
    """Several files: a list of paths, or one glob pattern such as 'powers/*.csv' (kept as written)."""
    type_name = "paths"

    def __init__(self, name, default=None, *, filter="", **kwargs):
        self.filter = filter
        super().__init__(name, default, **kwargs)

    def _check(self, value):
        if isinstance(value, str):
            return value.replace("\\", "/")
        if isinstance(value, (list, tuple)) and all(isinstance(path, str) for path in value):
            return [path.replace("\\", "/") for path in value]
        raise ValueError(f"{self.name}: a list of paths or a pattern, not {value!r}.")

    def to_text(self, value):
        return "" if value is None else value if isinstance(value, str) else ", ".join(value)

    def from_text(self, text):
        text = text.strip()
        if not text:
            return self.check(None)
        parts = [part.strip() for part in text.split(",") if part.strip()]
        return self.check(parts[0] if len(parts) == 1 and any(c in parts[0] for c in "*?[") else parts)


def _parse_range(name, text):
    match = RANGE.match(text)
    if not match:
        raise ValueError(f"{name}: {text!r} is no range (e.g. '385-398').")
    return [float(match.group(1)), float(match.group(2))]


class Range(Param):
    """[start, stop], start <= stop, both ends included."""
    type_name = "range"

    def __init__(self, name, default=None, *, unit="", display=None, **kwargs):
        if display and not units.is_known(unit, display):
            raise ValueError(f"{name}: unknown display unit {display!r} for {unit!r}.")
        self.unit, self.display = unit, display
        super().__init__(name, default, **kwargs)

    def _check(self, value):
        try:
            start, stop = (float(x) for x in value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.name}: [start, stop], not {value!r}.") from None
        return [min(start, stop), max(start, stop)]

    def to_text(self, value):
        if value is None:
            return ""
        shown = sorted(units.to_display(x, self.unit, self.display) for x in value)
        return f"{_format_number(shown[0])} - {_format_number(shown[1])}"

    def from_text(self, text):
        text = text.strip()
        if not text:
            return self.check(None)
        return self.check([units.from_display(x, self.unit, self.display) for x in _parse_range(self.name, text)])


class Ranges(Range):
    """A list of ranges, e.g. [[2.6e-7, 2.7e-7], [3.3e-7, 3.4e-7]]."""
    type_name = "ranges"

    def __init__(self, name, default=(), **kwargs):
        super().__init__(name, list(default), **kwargs)

    def _check(self, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{self.name}: a list of [start, stop], not {value!r}.")
        return [Range._check(self, part) for part in value]

    def to_text(self, value):
        return ", ".join(Range.to_text(self, part) for part in value or [])

    def from_text(self, text):
        parts = [part for part in text.split(",") if part.strip()]
        return self.check([[units.from_display(x, self.unit, self.display) for x in _parse_range(self.name, part)]
                           for part in parts])


class Floats(Param):
    """A list of numbers, e.g. extra wavelengths."""
    type_name = "floats"

    def __init__(self, name, default=(), *, unit="", display=None, **kwargs):
        if display and not units.is_known(unit, display):
            raise ValueError(f"{name}: unknown display unit {display!r} for {unit!r}.")
        self.unit, self.display = unit, display
        super().__init__(name, list(default), **kwargs)

    def _check(self, value):
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            raise ValueError(f"{self.name}: a list of numbers, not {value!r}.") from None

    def to_text(self, value):
        return ", ".join(_format_number(units.to_display(x, self.unit, self.display)) for x in value or [])

    def from_text(self, text):
        try:
            shown = [float(part) for part in text.replace(";", ",").split(",") if part.strip()]
        except ValueError:
            raise ValueError(f"{self.name}: numbers separated by commas, not {text!r}.") from None
        return self.check([units.from_display(x, self.unit, self.display) for x in shown])


class Ints(Param):
    """A list of whole numbers, e.g. spectra to ignore; typed as '0-3, 7'."""
    type_name = "ints"

    def __init__(self, name, default=(), **kwargs):
        super().__init__(name, list(default), **kwargs)

    def _check(self, value):
        if isinstance(value, (list, tuple)) and all(isinstance(x, int) and not isinstance(x, bool) for x in value):
            return sorted(set(value))
        raise ValueError(f"{self.name}: a list of whole numbers, not {value!r}.")

    def to_text(self, value):
        parts, numbers, i = [], sorted(set(value or [])), 0
        while i < len(numbers):
            j = i
            while j + 1 < len(numbers) and numbers[j + 1] == numbers[j] + 1:
                j += 1
            parts.append(str(numbers[i]) if i == j else f"{numbers[i]}-{numbers[j]}")
            i = j + 1
        return ", ".join(parts)

    def from_text(self, text):
        numbers = []
        for part in text.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            match = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", part)
            try:
                numbers.extend(range(int(match.group(1)), int(match.group(2)) + 1) if match else [int(part)])
            except ValueError:
                raise ValueError(f"{self.name}: e.g. '0-3, 7', not {text!r}.") from None
        return self.check(numbers)


class Points(Param):
    """Points picked on a plot, [[x, y], ...] in the plot's data units."""
    type_name = "points"

    def __init__(self, name, default=(), **kwargs):
        super().__init__(name, list(default), **kwargs)

    def _check(self, value):
        try:
            return [[float(x), float(y)] for x, y in value]
        except (TypeError, ValueError):
            raise ValueError(f"{self.name}: a list of [x, y], not {value!r}.") from None

    def to_text(self, value):
        return "; ".join(f"{_format_number(x)} {_format_number(y)}" for x, y in value or [])

    def from_text(self, text):
        points = []
        for part in text.split(";"):
            numbers = part.replace(",", " ").split()
            if numbers:
                if len(numbers) != 2:
                    raise ValueError(f"{self.name}: 'x y; x y', not {text!r}.")
                points.append([float(numbers[0]), float(numbers[1])])
        return self.check(points)


class Seed(Int):
    """The seed of random numbers (random starts): the same seed gives the same result. None: not reproducible."""
    type_name = "seed"

    def __init__(self, name="seed", default=0, **kwargs):
        super().__init__(name, default, min=0, optional=True, **kwargs)


class Step:
    """One kind of processing step in a Steps parameter: its key, its label and its own parameters."""

    def __init__(self, kind, label=None, params=(), help=""):
        self.kind, self.label, self.params, self.help = kind, label or kind, list(params), help

    def defaults(self) -> dict:
        return {"kind": self.kind, "on": True, **{p.name: p.default_value() for p in self.params}}


class Steps(Param):
    """An ordered list of processing steps, each with a kind, an on/off switch and its own parameters:
    [{"kind": "chirp", "on": true, "order": 3}, ...]."""
    type_name = "steps"

    def __init__(self, name, default=(), *, kinds, **kwargs):
        self.kinds = {step.kind: step for step in kinds}
        super().__init__(name, list(default), **kwargs)

    def _check(self, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{self.name}: a list of steps, not {value!r}.")
        checked = []
        for entry in value:
            if not isinstance(entry, dict) or entry.get("kind") not in self.kinds:
                raise ValueError(f"{self.name}: unknown step {entry!r} (known: {', '.join(self.kinds)}).")
            step = self.kinds[entry["kind"]]
            normal = {"kind": step.kind, "on": bool(entry.get("on", True))}
            for p in step.params:
                normal[p.name] = p.check(entry[p.name]) if p.name in entry else p.default_value()
            checked.append(normal)
        return checked

    def to_text(self, value):
        return ", ".join(entry["kind"] + ("" if entry["on"] else " (off)") for entry in value or [])


def find(params, name) -> Param | None:
    """The parameter called `name` in a list of parameters, also inside Steps ('steps.chirp.onsets')."""
    head, _, rest = name.partition(".")
    for p in params:
        if p.name == head:
            if not rest:
                return p
            if isinstance(p, Steps):
                kind, _, sub = rest.partition(".")
                return find(p.kinds[kind].params, sub) if kind in p.kinds else None
    return None
