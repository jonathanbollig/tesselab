"""
Transient absorption as Tesselab nodes: a ΔA matrix is read, corrected step by step (background, chirp picked on the
map, time zero, scatter, crop), explored with a crosshair on a map with its spectrum and kinetic, and fitted globally
with exponentials (decay associated spectra). The SVD of the chemometrics module works on its matrix as it is.

    TA matrix (reader) ──> Corrections ──┬──> Corrected map (view: picks the chirp onsets)
                                         ├──> Explorer (view: crosshair, spectrum and kinetic)
                                         ├──> Global fit ──> Global fit view
                                         └──> SVD (chemometrics) ──> SVD view

The views set what is picked or dragged on their plots through analysis variables: the chirp onsets
(chirp_onsets, used by Corrections), the fit range (fit_range, used by Global fit) and the probe wavelength
(wavelength, shown by the Explorer and the Global fit view).
"""
import os

import numpy as np
import xarray as xr

import tesselab
from tesselab import (P, Plot, PlotSpec, Panel, Axis, Line, Lines, Points, Heatmap, Region, Style, Signature, Var,
                      Result, Crosshair, Pick, Cursor, RangeSelect, note_only)

from . import analysis

DELAY = Axis("Delay", "s", display="ps", scale="linlog", linthresh=1e-12)
WAVELENGTH = Axis("Wavelength", "m", display=["nm", "cm-1", "eV"])
DA = Axis("ΔA", "OD", display="mOD")
DATA = Signature(dA=Var("delay", "wavelength"), delay=Var("delay", units="s"), wavelength=Var("wavelength", units="m"),
                 help="A ΔA matrix over delay and wavelength.")


def _dataset(delays, wavelengths, signal, attrs=None):
    return xr.Dataset({"dA": (("delay", "wavelength"), signal, {"units": "OD"})},
                      coords={"delay": ("delay", delays, {"units": "s"}),
                              "wavelength": ("wavelength", wavelengths, {"units": "m"})}, attrs=attrs or {})


def _map(data, note=""):
    return Heatmap(data.delay, data.wavelength, data.dA, colorbar="ΔA [OD]", name="ΔA")


# --- reading ---------------------------------------------------------------------------------------------------------
class Matrix(tesselab.Reader):
    name = "TA matrix file"
    help = "Reads a ΔA matrix: the first row holds the wavelengths, the first column the delays."
    params = [P.Path("file", filter="Matrices (*.csv *.dat *.txt)", help="The matrix file."),
              P.Choice("delay_unit", "ps", options={"fs": "fs", "ps": "ps", "ns": "ns", "s": "s"},
                       help="The unit of the delays in the file."),
              P.Choice("signal_unit", "OD", options={"OD": "OD", "mOD": "mOD"}, help="The unit of ΔA in the file.")]
    outputs = ["data"]

    def run(self, inputs, params):
        factor = {"fs": 1e-15, "ps": 1e-12, "ns": 1e-9, "s": 1.0}[params.delay_unit]
        delays, wavelengths, signal = analysis.read_matrix(params.file, factor, 1e-9,
                                                           1e-3 if params.signal_unit == "mOD" else 1.0)
        return {"data": _dataset(delays, wavelengths, signal, {"source": os.path.basename(params.file)})}


# --- corrections -------------------------------------------------------------------------------------------------------
STEPS = [
    P.Step("background", "Background subtraction", [P.Range("window", [-3e-12, -0.5e-12], unit="s", display="ps")],
           help="Subtract, per wavelength, the mean ΔA in a window before time zero."),
    P.Step("chirp", "Chirp correction", [P.Int("order", 2, min=1, max=5)],
           help="Time zero at every wavelength: a polynomial through the chirp onsets (picked on the corrected map, "
                "Ctrl+click)."),
    P.Step("t0", "Time-zero shift", [P.Float("shift", 0.0, unit="s", display="ps")],
           help="Shift all delays by a constant."),
    P.Step("scatter", "Scatter removal", [P.Ranges("regions", unit="m", display="nm")],
           help="Wavelength ranges set to missing (pump scatter)."),
    P.Step("crop", "Cropping", [P.Range("delays", optional=True, unit="s", display="ps"),
                                P.Range("wavelengths", optional=True, unit="m", display="nm")],
           help="Keep only these delays and wavelengths."),
]


class Corrections(tesselab.Node):
    name = "Corrections"
    help = "The usual corrections of a TA matrix, as an ordered list of steps that can be switched on and off."
    inputs = {"data": DATA}
    params = [P.Steps("steps", [{"kind": "background"}, {"kind": "chirp"}], kinds=STEPS,
                      help="Applied from the top down."),
              P.Points("chirp_onsets", help="(delay, wavelength) where the signal starts, for the chirp step: picked "
                                            "on the corrected map (bound to the analysis variable chirp_onsets).")]
    outputs = ["data"]
    schema = 2  # 2: the chirp onsets are a parameter of their own (were inside the chirp step), so a view can pick them

    def migrate(self, params, from_schema):
        params = dict(params)
        if from_schema < 2 and isinstance(params.get("steps"), list):
            steps = [dict(step) for step in params["steps"]]
            for step in steps:
                onsets = step.pop("onsets", None)
                if step.get("kind") == "chirp" and onsets and "chirp_onsets" not in params:
                    params["chirp_onsets"] = onsets
            params["steps"] = steps
        return params

    def run(self, inputs, params):
        delays = inputs.data.delay.values.copy()
        wavelengths = inputs.data.wavelength.values.copy()
        signal = inputs.data.dA.values.copy()
        t0 = np.zeros_like(wavelengths)
        for step in params.steps:
            if not step["on"]:
                continue
            kind = step["kind"]
            if kind == "background":
                signal = analysis.subtract_background(delays, signal, step["window"])
            elif kind == "chirp" and params.chirp_onsets:
                t0 = analysis.chirp_curve(params.chirp_onsets, step["order"], wavelengths)
                signal = analysis.correct_chirp(delays, signal, t0)
            elif kind == "t0":
                delays = delays - step["shift"]
            elif kind == "scatter":
                for low, high in step["regions"]:
                    signal[:, (wavelengths >= low) & (wavelengths <= high)] = np.nan
            elif kind == "crop":
                keep_d = np.ones(len(delays), bool) if not step["delays"] else \
                    (delays >= step["delays"][0]) & (delays <= step["delays"][1])
                keep_w = np.ones(len(wavelengths), bool) if not step["wavelengths"] else \
                    (wavelengths >= step["wavelengths"][0]) & (wavelengths <= step["wavelengths"][1])
                delays, signal, wavelengths, t0 = delays[keep_d], signal[np.ix_(keep_d, keep_w)], wavelengths[keep_w], \
                    t0[keep_w]
                if not len(delays) or not len(wavelengths):
                    raise ValueError(f"The crop leaves no {'delays' if not len(delays) else 'wavelengths'}: check "
                                     f"its ranges (low - high, inside the data).")
        out = _dataset(delays, wavelengths, signal, dict(inputs.data.attrs))
        out["t0"] = ("wavelength", t0, {"units": "s"})
        return {"data": out}


# --- views: the corrected map, the explorer -------------------------------------------------------------------------
def draw_corrected(ctx):
    if "data" not in ctx.inputs:
        return note_only("No corrected data", x=DELAY, y=WAVELENGTH)
    data = ctx.inputs.data
    marks = [_map(data)]
    if ctx.params.chirp_onsets:
        onsets = np.asarray(ctx.params.chirp_onsets)
        marks.append(Points(onsets[:, 0], onsets[:, 1], name="onsets", style=Style(color="ink", size=8)))
        if "t0" in data and np.any(data.t0.values):
            marks.append(Line(data.t0, data.wavelength, name="time zero", style=Style(color="ink", dash="dash")))
    return PlotSpec(x=DELAY, y=WAVELENGTH, marks=marks, zero_line=False)


class CorrectedMap(tesselab.View):
    name = "Corrected map"
    help = "The corrected ΔA matrix; the chirp onsets are picked here."
    inputs = {"data": DATA}
    params = [P.Points("chirp_onsets", help="The picked onsets (bound to the analysis variable chirp_onsets, which "
                                            "Corrections uses).")]
    plots = [Plot("map", "Corrected ΔA", draw=draw_corrected, span=2,
                  help="The corrected matrix. Ctrl+click picks a chirp onset (where the signal starts), Ctrl+right "
                       "click removes the nearest; the dashed line is the fitted time zero.")]
    interactions = [Pick("chirp_onsets", plots=["map"])]


def cuts(data, delay, wavelength, average):
    """The spectrum at the delay nearest to `delay` and the kinetic at the wavelength nearest to `wavelength`, each
    averaged over `average` neighbours on each side: (i, j, spectrum, kinetic)."""
    delays, wavelengths, signal = data.delay.values, data.wavelength.values, data.dA.values
    i = int(np.nanargmin(np.abs(delays - delay)))
    j = int(np.nanargmin(np.abs(wavelengths - wavelength)))
    return i, j, analysis.average_around(signal.T, i, average), analysis.average_around(signal, j, average)


def draw_explorer(ctx):
    if "data" not in ctx.inputs:
        return note_only("No data", x=DELAY, y=WAVELENGTH)
    data = ctx.inputs.data
    _, _, spectrum, kinetic = cuts(data, ctx.params.cursor_delay, ctx.params.cursor_wavelength, ctx.params.average)
    return PlotSpec(
        panels={"map": Panel(x=DELAY, y=WAVELENGTH, marks=[_map(data)], zero_line=False),
                "spectrum": Panel(x=DA, y=WAVELENGTH, marks=[Line(spectrum, data.wavelength, name="spectrum")],
                                  zero_line=False),
                "kinetic": Panel(x=DELAY, y=DA, marks=[Line(data.delay, kinetic, name="kinetic")])},
        mosaic=[["map", "spectrum"], ["kinetic", None]], share_x=[["map", "kinetic"]],
        share_y=[["map", "spectrum"]], sizes=([3, 1], [3, 1.4]))


def draw_overlay(ctx):
    marks = []
    for i, (label, other) in enumerate([(ctx.label, ctx)] + [o for o in ctx.others if o[0] != ctx.label]):
        if "data" not in other.inputs:
            continue
        _, _, _, kinetic = cuts(other.inputs.data, other.params.cursor_delay, other.params.cursor_wavelength,
                                other.params.average)
        marks.append(Line(other.inputs.data.delay, kinetic, name=label, style=Style(color=f"series:{i}")))
    return PlotSpec(x=DELAY, y=DA, marks=marks, legend="top right")


class Explorer(tesselab.View):
    name = "Explorer"
    help = "The map with the spectrum at the delay cursor (right) and the kinetic at the wavelength cursor (below)."
    inputs = {"data": DATA}
    params = [P.Float("cursor_delay", 1e-12, unit="s", display="ps", help="The delay of the spectrum."),
              P.Float("cursor_wavelength", 520e-9, unit="m", display="nm", help="The wavelength of the kinetic."),
              P.Int("average", 2, min=0, help="Points averaged on each side of the cursors (noise).")]
    plots = [Plot("explorer", "Map with spectrum and kinetic", draw=draw_explorer, span=2, rows=2,
                  help="Drag the crosshair on the map; the panels share their axes with it."),
             Plot("overlay", "Kinetics of the linked analyses", draw=draw_overlay, items="several",
                  help="The kinetic at the wavelength cursor of every linked analysis.")]
    interactions = [Crosshair(x="cursor_delay", y="cursor_wavelength", plots=["explorer"], panel="map",
                              snap_x="delay", snap_y="wavelength")]


# --- global fit -------------------------------------------------------------------------------------------------------
def draw_das(ctx):
    fit = ctx.inputs.fit
    labels = [f"τ = {tau * 1e12:.3g} ps" for tau in fit.tau.values]
    return PlotSpec(x=WAVELENGTH, y=Axis("Amplitude", "OD", display="mOD"),
                    marks=[Lines(fit.wavelength, fit.das, labels=labels)], legend="top right")


def draw_fit_kinetic(ctx):
    fit = ctx.inputs.fit
    j = int(np.nanargmin(np.abs(fit.wavelength.values - ctx.params.cursor_wavelength)))
    low, high = ctx.params.fit_range
    return PlotSpec(x=DELAY, y=DA, marks=[
        Region([low], [high], name="fit range"),
        Points(fit.delay, fit.dA.values[:, j], name=f"data at {fit.wavelength.values[j] * 1e9:.0f} nm",
               style=Style(size=5)),
        Line(fit.delay, fit.model.values[:, j], name="fit", style=Style(color="fit"))], legend="top right")


class GlobalFit(tesselab.Node):
    name = "Global fit"
    help = "Exponential decays with lifetimes shared by all wavelengths; their amplitudes are the decay associated spectra."
    inputs = {"data": DATA}
    params = [P.Floats("lifetimes", [1e-12, 100e-12], unit="s", display="ps",
                       help="Starting lifetimes, one per component (e.g. '1, 100' ps)."),
              P.Range("fit_range", [0.3e-12, 2e-9], unit="s", display="ps",
                      help="The delays fitted (bound to the analysis variable fit_range: dragged in the fit's view)."),
              P.Int("max_iterations", 200, min=10, advanced=True)]
    outputs = ["fit"]
    views = ["global-fit-view"]
    key_results = [Result(f"tau_{k}", label=f"τ{k}", unit="s", display="ps", error=f"tau_{k}_err", output="fit")
                   for k in (1, 2, 3)]

    def run(self, inputs, params):
        data = inputs.data
        if not params.lifetimes:
            raise ValueError("Give at least one starting lifetime.")
        delays, signal = data.delay.values, data.dA.values
        taus, errors, das, model, converged = analysis.global_fit(delays, signal, params.lifetimes, params.fit_range,
                                                                  params.max_iterations)
        if not converged:
            tesselab.warning("The fit did not converge within max_iterations.", code="not-converged")
        window = params.fit_range[1] - max(params.fit_range[0], 0)
        for k, (tau, error) in enumerate(zip(taus, errors), 1):
            if tau > 3 * window:
                tesselab.warning(f"τ{k} = {tau * 1e12:.3g} ps is much longer than the fitted window: read it as "
                                 f"'longer than the window'.", code="tau-long")
            elif not np.isfinite(error) or error > tau:
                tesselab.warning(f"τ{k} is not determined by the data (its error is larger than itself).",
                                 code="tau-undetermined")
        out = _dataset(delays, data.wavelength.values, signal)
        out["model"] = (("delay", "wavelength"), model, {"units": "OD"})
        out["residual"] = (("delay", "wavelength"), signal - model, {"units": "OD"})
        out["das"] = (("component", "wavelength"), das, {"units": "OD"})
        out["tau"] = ("component", taus, {"units": "s"})
        out["tau_err"] = ("component", errors, {"units": "s"})
        for k, (tau, error) in enumerate(zip(taus, errors), 1):
            out[f"tau_{k}"], out[f"tau_{k}_err"] = tau, error
        return {"fit": out}


class GlobalFitView(tesselab.View):
    name = "Global fit view"
    help = "The decay associated spectra, one kinetic with its fit, and the residual."
    inputs = {"fit": Signature(das=Var("component", "wavelength"), model=Var("delay", "wavelength"),
                               help="The global fit.")}
    params = [P.Float("cursor_wavelength", 520e-9, unit="m", display="nm", label="wavelength",
                      help="The kinetic shown with its fit (bound to the analysis variable wavelength)."),
              P.Range("fit_range", [0.3e-12, 2e-9], unit="s", display="ps",
                      help="The fitted delays, shown and dragged here (bound to the analysis variable fit_range).")]
    plots = [Plot("das", "Decay associated spectra", draw=draw_das, help="The amplitude of each lifetime."),
             Plot("kinetic", "Kinetic and fit", draw=draw_fit_kinetic, settings=["cursor_wavelength"],
                  help="Data and fit at one wavelength; the shaded part is the fit range (drag its edges)."),
             Plot("residual", "Residual", draw=lambda ctx: PlotSpec(
                 x=DELAY, y=WAVELENGTH, zero_line=False,
                 marks=[Heatmap(ctx.inputs.fit.delay, ctx.inputs.fit.wavelength, ctx.inputs.fit.residual,
                                colorbar="ΔA [OD]")]),
                 help="Data minus fit: structure here is something the model misses.")]
    interactions = [Cursor("cursor_wavelength", axis="x", plots=["das"], snap="wavelength", link="wavelength"),
                    RangeSelect("fit_range", plots=["kinetic"])]


# --- template ----------------------------------------------------------------------------------------------------------
def _looks_like_matrix(path):
    try:
        with open(path, encoding="utf-8-sig") as f:
            first = f.readline()
            second = f.readline()
    except OSError:
        return False
    sep = next((d for d in (",", ";", "\t") if d in first), None)
    cells = first.strip().split(sep) if sep else first.split()
    try:
        [float(c) for c in cells[1:]]
        float((second.split(sep) if sep else second.split())[0])
    except (ValueError, IndexError):
        return False
    return len(cells) > 10


def _matrices(folder):
    return [os.path.join(folder, n) for n in sorted(os.listdir(folder))
            if n.lower().endswith((".csv", ".dat", ".txt")) and _looks_like_matrix(os.path.join(folder, n))]


def detect(folder):
    """A folder with a ΔA matrix (a first row of more than ten wavelengths, a first column of delays)."""
    return bool(_matrices(folder))


def fill_template(folder, project):
    matrices = _matrices(folder)
    path = matrices[0] if matrices else folder
    return {"name": os.path.basename(folder), "file": os.path.relpath(path, project).replace("\\", "/")}
