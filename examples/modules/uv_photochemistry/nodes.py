"""
The UV irradiation analysis of the author's package kufner_uv_analysis (copied into uv_analysis/) as Tesselab nodes.

Each compute node does a part of IrradiationExperiment.load_data() / calculate() by calling the same methods, on the
same data, in the same order, so the results are bit-identical to the package's (tests/test_uv_module.py compares
them for the sample; tools/compare_uv.py for all real experiments). Reading files and computing are apart, and the
plots are in one view, the old viewer's "Quantum yield" group:

    readers                          compute                                        view
    Agilent spectra file ──────> Spectrum corrections ──┐
    Agilent spectrum file (buffer) ──┘                  ├──> Irradiation steps ──┬──> Quantum yield ─┐
    Power meter folder ─────────────────────────────────┘                        └──> φ₀ vs. λ ──────┼──> Quantum yield
    Nucleoside spectra file ──────────────────────────────────────────────────────────────────────────┘    (6 plots)

The sample's numbers (nucleosides, λ irradiation, thickness, volume, fit_num), λ spectrum, the ignored spectra, the
baseline and the x axis are analysis variables: the compute nodes and the view are bound to them, so dragging the red
line or clicking a point in the view changes the computation. φ₀(λ) is not bound to λ spectrum: dragging λ does not
compute it again.
"""
import os
from collections import Counter

import numpy as np
import xarray as xr

import tesselab
from tesselab import (P, Plot, PlotSpec, Axis, Line, Lines, Points, Region, VLine, Signature, Var, Result, Cursor,
                      Toggle, Action, note_only, Style, Text)

from .uv_analysis.irradiation import IrradiationExperiment, Extrapolation
from .uv_analysis.spectrum import (absorbance_at_wavelength, absorbance_maximum, read_agilent_spectra,
                                         read_agilent_spectrum, read_nucleoside_spectra)

WAVELENGTH = Axis("Wavelength", "m", display="nm")
# The marks' looks come from the theme by their role (Style(role=...): "measured" blue crosses, "fit" grey dashed,
# "result" black, "reference" green, "unused" grey dashed, "event" grey dashed lines), as in the old viewer; a user's
# theme file can change them. Only the single nucleosides' greens are this module's own.
EXPECTED_PARTS = ["#00441b", "#74c476", "#238b45", "#a1d99b"]  # the single nucleosides, dashed
X_AXES = {"energy": "absorbed energy", "time": "irradiation time"}


def nucleoside_list(text):
    return [n for n in text.upper() if n.strip()]


# --- readers ---------------------------------------------------------------------------------------------------------
class AgilentSpectra(tesselab.Reader):
    name = "Agilent spectra file"
    help = "Reads every spectrum of an Agilent CSV file (column pairs of wavelength and absorbance), as measured."
    params = [P.Path("file", filter="Agilent CSV (*.csv)", help="The Agilent CSV file."),
              P.Int("skipfooter", 0, min=0, advanced=True, help="Lines at the end of the file to skip.")]
    outputs = ["spectra"]

    def run(self, inputs, params):
        spectra = read_agilent_spectra(params.file, None, skiprows=2, skipfooter=params.skipfooter)
        spectra.wavelength.attrs["units"] = "m"
        return {"spectra": spectra.to_dataset(name="absorbance")}


class AgilentSpectrum(tesselab.Reader):
    name = "Agilent spectrum file"
    help = ("Reads one spectrum (the first column pair) of an Agilent CSV file, e.g. a buffer. Without a file it "
            "gives nothing, and an optional input it feeds stays empty.")
    params = [P.Path("file", optional=True, filter="Agilent CSV (*.csv)", help="The Agilent CSV file (empty: none).")]
    outputs = ["spectrum"]

    def run(self, inputs, params):
        if not params.file:
            tesselab.diagnostic("info", "No file: no spectrum.", code="no-file")
            return {}
        spectrum = read_agilent_spectrum(params.file, skiprows=2, skipfooter=0)
        spectrum.wavelength.attrs["units"] = "m"
        return {"spectrum": spectrum.to_dataset(name="absorbance")}


class NucleosideSpectra(tesselab.Reader):
    name = "Nucleoside spectra file"
    help = ("Reads the absorption coefficients ε of the nucleosides: a CSV file with a column wavelength_nm and one "
            "column per nucleoside (A, U, ...) in L/(mol·cm), kept in the file's units.")
    params = [P.Path("file", filter="CSV (*.csv)", help="The nucleosides' absorption spectra.")]
    outputs = ["epsilon"]

    def run(self, inputs, params):
        return {"epsilon": read_nucleoside_spectra(params.file).to_dataset(name="epsilon")}


class PowerMeter(tesselab.Reader):
    name = "Power meter folder"
    help = ("Reads the rig's power folder: readings behind the cuvette, background (LED off) subtracted. During the "
            "irradiations (readings below 20 % of the maximum dropped), calibrations (usually a cuvette without "
            "sample, below 50 % dropped), and the LED power during the irradiations, interpolated between the "
            "calibrations. Absorbed power = LED power × 0.96 − measured power / 0.96; its integral over an "
            "irradiation is the absorbed energy.")
    params = [P.Path("folder", folder=True, help="The folder of power CSV files (calib_*, background, power*).")]
    outputs = ["irradiations", "power_traces"]
    key_results = [Result("total_irradiation_time", label="total irradiation", unit="s", display="min", digits=3)]

    def run(self, inputs, params):
        exp = IrradiationExperiment()
        exp.create_powers_from_folder(params.folder, skip_rows=1)
        exp.calc_irradiations()

        def stack(traces):
            if not traces:
                return np.zeros(0), np.zeros(0), np.zeros(0, int)
            return (np.concatenate([t.values for t in traces]), np.concatenate([t.time.values for t in traces]),
                    np.concatenate([np.full(t.sizes["time"], i) for i, t in enumerate(traces)]))

        power, time, index = stack(exp.power_t)
        led, _, _ = stack(exp.led_power_t)
        calib_power, calib_time, calib_index = stack(exp.power_calib_t)
        first = [float(t.time.values[0]) for t in exp.power_t + exp.power_calib_t]
        powers = xr.Dataset({"power": ("sample", power), "time": ("sample", time), "index": ("sample", index),
                             "led_power": ("sample", led),
                             "calib_power": ("calib_sample", calib_power), "calib_time": ("calib_sample", calib_time),
                             "calib_index": ("calib_sample", calib_index)},
                            attrs={"time_zero": min(first) if first else 0.0})
        irradiations = exp.irradiations.copy()
        irradiations["total_irradiation_time"] = float(irradiations.irradiation_time.sum())
        return {"irradiations": irradiations, "power_traces": powers}


class RigRecords(tesselab.Reader):
    name = "Rig records"
    help = ("Reads what the rig recorded about the run (second series on): steps.jsonl (every step with its start and "
            "duration) and config.json (the run's parameters), next to the power folder. When each spectrum was "
            "measured, when the stirrer ran, the interrupts (e.g. a sample taken out). Without the files (first "
            "series) it gives nothing.")
    params = [P.Path("steps_file", optional=True, filter="Rig steps (steps.jsonl)", help="steps.jsonl (empty: none)."),
              P.Path("config_file", optional=True, filter="Rig configuration (config.json)",
                     help="config.json (empty: none).")]
    outputs = ["rig_records"]

    def run(self, inputs, params):
        import json
        from .uv_analysis.run_records import RunRecords
        if not (params.steps_file and params.config_file):
            tesselab.diagnostic("info", "No records (the first series has none).", code="no-file")
            return {}
        try:  # as read_run_records: the analysis does not need them
            with open(params.steps_file) as f:
                steps = [json.loads(line) for line in f if line.strip()]
            with open(params.config_file) as f:
                records = RunRecords(steps, json.load(f))
            stirring, interrupts, times = records.stirring, records.interrupts, records.spectrum_times
        except Exception as error:
            tesselab.warning(f"The rig's records could not be read: {error}")
            return {}

        def stirring_label(s):
            if s.cycle is None:
                return "stirring before the first spectra"
            r = s.result or {}
            return (f"stirring, cycle {s.cycle} ({s.uv_seconds:.1f} s UV): {r.get('rpm', float('nan')):.0f} rpm, "
                    f"{r.get('temp_c', float('nan')):.1f} °C, {r.get('humidity_pct', float('nan')):.0f} % RH")

        return {"rig_records": xr.Dataset({
            "spectrum_time": ("record", np.asarray(times, dtype=float)),
            "stirring_start": ("stirring", [s.start for s in stirring]),
            "stirring_end": ("stirring", [s.end for s in stirring]),
            "stirring_label": ("stirring", [stirring_label(s) for s in stirring]),
            "interrupt_start": ("interrupt", [i.start for i in interrupts]),
            "interrupt_end": ("interrupt", [i.end for i in interrupts]),
            "interrupt_label": ("interrupt", [f"{i.message} ({round(i.end - i.start)} s)" for i in interrupts])},
            attrs={"has_records": True})}


# --- compute ---------------------------------------------------------------------------------------------------------
class SpectrumCorrections(tesselab.Node):
    name = "Spectrum corrections"
    help = ("The measured spectra minus the blank (the first spectrum), the buffer and a baseline; extra spectra "
            "marked; the spectra measured after one irradiation combined into one.")
    inputs = {"measured_spectra": Signature(absorbance=Var("spectrum", "wavelength"),
                                            help="Every spectrum as measured; the first is the blank."),
              "buffer": Signature(absorbance=Var("wavelength"), optional=True,
                                  help="A buffer spectrum, subtracted from every spectrum.")}
    params = [
        P.Int("combine_spectra", 0, min=0, help="Spectra measured after each irradiation, combined into one (0: none)."),
        P.Choice("combine_spectra_method", "last", options={"last": "last", "mean": "mean", "lowest": "lowest"},
                 help="How the spectra of one irradiation are combined."),
        P.Ints("extra_spectra", help="Spectra measured without an irradiation before them (numbered as in the file, "
                                     "without the blank)."),
        P.Range("baseline", optional=True, unit="m", display="nm",
                help="Subtract every spectrum's mean in this range (empty: no baseline correction; the analysis "
                     "variable baseline)."),
    ]
    outputs = ["spectra", "uncombined_spectra"]
    key_results = [Result("lambda_max", label="λmax", unit="m", display="nm", digits=4),
                   Result("absorbance_max", label="Amax", unit="OD", digits=3)]

    def run(self, inputs, params):
        exp = IrradiationExperiment()
        # as create_spectra_from_agilent_file (reference_spectrum_index=0): the first spectrum is the blank
        measured = inputs.measured_spectra.absorbance
        exp.spectra = (measured - measured.isel(spectrum=0)).drop_isel(spectrum=0)
        exp.mark_extra_spectra([])
        exp.mark_ignored_spectra([])
        # as IrradiationExperiment.load_data, the rest of its spectral part
        if "buffer" in inputs:
            exp.subtract_reference_spectrum(inputs.buffer.absorbance.reset_coords(drop=True))
        if params.baseline:
            exp.subtract_baseline(*params.baseline)
        exp.mark_extra_spectra(params.extra_spectra)
        exp.measured_spectra = exp.spectra.drop_vars("ignored").assign_coords(
            combined=("spectrum", np.arange(exp.spectra.sizes["spectrum"])))
        if params.combine_spectra:
            exp.combine_spectra(method=params.combine_spectra_method, num=params.combine_spectra)
        lambda_max, absorbance_max = exp.initial_maximum()
        spectra = exp.spectra.to_dataset(name="absorbance")
        spectra["lambda_max"], spectra["absorbance_max"] = lambda_max, absorbance_max
        spectra.wavelength.attrs["units"] = "m"
        return {"spectra": spectra, "uncombined_spectra": exp.measured_spectra.to_dataset(name="absorbance")}


def export_mean_spectrum(ctx):
    """The mean of the used spectra (not extra, not ignored; minus blank, buffer and baseline) as an Agilent CSV, so
    that a buffer measurement becomes a buffer file (as the viewer's export button)."""
    import tempfile
    from .uv_analysis.spectrum import to_agilent_csv
    spectra = ctx.out.spectra_with_totals.absorbance
    used = ~(spectra.extra.values | spectra.ignored.values)
    mean = spectra.isel(spectrum=used).reset_coords(drop=True).mean("spectrum")
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "mean.csv")
        to_agilent_csv(mean, path)
        with open(path, "rb") as f:
            return "file", f"{ctx.label}_mean_spectrum.csv", f.read()


class Steps(tesselab.Node):
    name = "Irradiation steps"
    help = ("Which irradiations lie between which used spectra: every spectrum gets the total irradiation time and "
            "absorbed energy before it; the irradiations between two used spectra form a step, with its delta "
            "spectrum.")
    inputs = {"spectra": Signature(absorbance=Var("spectrum", "wavelength"),
                                   help="The corrected spectra, one per measuring point."),
              "irradiations": Signature(irradiation_time=Var("irradiation"), absorbed_energy=Var("irradiation"),
                                        help="Every irradiation's time and absorbed energy."),
              "uncombined_spectra": Signature(absorbance=Var("spectrum", "wavelength"), optional=True,
                                              help="The spectra before combining: they get their clock times."),
              "rig_records": Signature(spectrum_time=Var("record"), optional=True,
                                       help="The rig's records: when each spectrum was measured.")}
    params = [P.Ints("ignore_spectra", help="Spectra not used (after combining): the irradiations around an "
                                            "ignored spectrum form one step (the analysis variable ignore_spectra: "
                                            "click a point in 'Absorbance at λ').")]
    outputs = ["steps", "spectra_with_totals", "uncombined_spectra", "irradiations"]
    actions = [Action("export_mean", "Export mean spectrum", export_mean_spectrum,
                      help="The mean of the used spectra as an Agilent CSV (e.g. a buffer measurement → buffer file).")]

    def run(self, inputs, params):
        exp = IrradiationExperiment()
        exp.spectra = inputs.spectra.absorbance
        exp.irradiations = inputs.irradiations[["irradiation_time", "end_time", "absorbed_energy"]] \
            if "end_time" in inputs.irradiations else inputs.irradiations[["irradiation_time", "absorbed_energy"]]
        # as load_data: ignored spectra, the totals of every spectrum, the clock times of the measured spectra
        exp.mark_ignored_spectra(params.ignore_spectra)
        exp.calc_spectrum_totals()
        out = {"spectra_with_totals": exp.spectra.to_dataset(name="absorbance")}
        if "uncombined_spectra" in inputs:
            uncombined = inputs.uncombined_spectra.absorbance
            times = np.full(uncombined.sizes["spectrum"], np.nan)
            not_extra = ~uncombined.extra.values
            if "rig_records" in inputs and inputs.rig_records.attrs.get("has_records") and \
                    inputs.rig_records.sizes["record"] == np.sum(not_extra):
                times[not_extra] = inputs.rig_records.spectrum_time.values
            out["uncombined_spectra"] = uncombined.assign_coords(time=("spectrum", times)).to_dataset(
                name="absorbance")
        # as calculate: the steps
        try:
            exp.calc_steps()
        except ValueError as error:  # e.g. spectra and irradiations do not fit (yet): the spectra are still shown
            tesselab.diagnostic("error", str(error), code="steps")
            return out
        steps = exp.steps
        steps.attrs.update(pre_irradiation_time=exp.pre_irradiation_time, pre_absorbed_energy=exp.pre_absorbed_energy)
        out.update(steps=steps, irradiations=exp.irradiations)
        return out


def sample_params():
    """The sample's numbers, which Quantum yield and φ₀(λ) both need (bound to analysis variables in the template)."""
    return [
        P.Str("nucleosides", "", help="The nucleosides of the sample, e.g. UC (their absorption coefficients are "
                                      "summed)."),
        P.Float("lambda_irradiation", 265e-9, unit="m", display="nm", label="λ irradiation",
                help="The LED's wavelength."),
        P.Float("thickness", 10e-3, unit="m", display="mm", help="Path length of the cuvette."),
        P.Float("volume", 1.5e-6, unit="m³", help="Sample volume [m³] (1.5 mL = 1.5e-6 m³)."),
        P.Int("fit_num", 5, min=2, optional=True, help="Steps used for the extrapolation to zero (empty: none)."),
    ]


STEPS_INPUT = Signature(delta_spectrum=Var("step", "wavelength"), absorbed_energy=Var("step"),
                        total_absorbed_energy=Var("step"), total_irradiation_time=Var("step"),
                        help="The irradiation steps with their delta spectra.")
EPSILON_INPUT = Signature(epsilon=Var("nucleoside", "wavelength_nm"),
                          help="The nucleosides' absorption coefficients.")


class QuantumYield(tesselab.Node):
    name = "Quantum yield"
    help = ("Per step: photoproducts (Lambert-Beer at λ spectrum) / absorbed photons (at λ irradiation); a linear fit "
            "through the first fit_num steps, extrapolated to zero, gives φ₀.")
    inputs = {"steps": STEPS_INPUT, "epsilon": EPSILON_INPUT}
    params = sample_params() + [
        P.Float("lambda_spectrum", 260e-9, unit="m", display="nm", label="λ spectrum",
                help="Where the absorbance change is read."),
    ]
    outputs = ["quantum_yields", "phi0_fit"]
    key_results = [Result("phi0_energy", label="φ₀ (energy)", error="phi0_energy_err", output="phi0_fit",
                          display="%", digits=6),
                   Result("phi0_time", label="φ₀ (time)", error="phi0_time_err", output="phi0_fit",
                          display="%", digits=6)]

    def run(self, inputs, params):
        exp = IrradiationExperiment()
        exp.steps = inputs.steps
        exp.nucleoside_spectra = inputs.epsilon.epsilon
        nucleosides = nucleoside_list(params.nucleosides)
        if not nucleosides:
            tesselab.warning("Set the nucleosides: without them there are no quantum yields.", code="nucleosides")
            return {"quantum_yields": exp.steps}
        # as IrradiationExperiment.calculate, after calc_steps
        result = exp.quantum_yields(nucleosides, params.lambda_irradiation, params.lambda_spectrum, params.thickness,
                                    params.volume)
        exp.steps = exp.steps.assign(result.drop_vars("wavelength"))
        fit_values = {}
        for suffix, total in (("time", exp.steps.total_irradiation_time), ("energy", exp.steps.total_absorbed_energy)):
            fit = Extrapolation.from_fit(total, exp.steps.quantum_yield, params.fit_num)
            fit_values[f"phi0_{suffix}"] = fit.y_axis if fit else np.nan
            fit_values[f"phi0_{suffix}_err"] = fit.y_axis_err if fit else np.nan
            fit_values[f"slope_{suffix}"] = fit.slope if fit else np.nan
            fit_values["fit_num"] = fit.num if fit else 0
        return {"quantum_yields": exp.steps, "phi0_fit": xr.Dataset({k: ((), v) for k, v in fit_values.items()})}


class Phi0Spectrum(tesselab.Node):
    name = "φ₀ vs. λ spectrum"
    help = ("φ₀ with every measured wavelength as λ spectrum, where the sample absorbs (summed absorption "
            "coefficient above 10 % of its maximum). Ideally it hardly depends on λ; a strong change suggests that "
            "photoproducts absorb there too.")
    inputs = {"steps": STEPS_INPUT, "epsilon": EPSILON_INPUT}
    params = sample_params()
    outputs = ["phi0_spectrum"]

    def run(self, inputs, params):
        nucleosides = nucleoside_list(params.nucleosides)
        if not nucleosides or not params.fit_num:
            tesselab.warning("φ0(λ) needs the nucleosides and fit_num.", code="nucleosides")
            return {}
        exp = IrradiationExperiment()
        exp.steps = inputs.steps
        exp.spectra = inputs.steps.delta_spectrum  # phi0_spectrum reads only the wavelengths of the spectra
        exp.nucleoside_spectra = inputs.epsilon.epsilon
        return {"phi0_spectrum": exp.phi0_spectrum(nucleosides, params.lambda_irradiation, params.thickness,
                                                   params.volume, params.fit_num)}


class MeanSpectrum(tesselab.Node):
    name = "Mean spectrum"
    help = "The mean of the used spectra (not extra, not ignored): link it into other analyses as their buffer."
    inputs = {"spectra": Signature(absorbance=Var("spectrum", "wavelength"), help="The spectra to average.")}
    outputs = ["spectrum"]

    def run(self, inputs, params):
        spectra = inputs.spectra.absorbance
        used = np.ones(spectra.sizes["spectrum"], bool)
        for flag in ("extra", "ignored"):
            if flag in spectra.coords:
                used &= ~spectra[flag].values
        mean = spectra.isel(spectrum=used).reset_coords(drop=True).mean("spectrum")
        return {"spectrum": mean.to_dataset(name="absorbance")}


# --- the view: the old viewer's "Quantum yield" group ------------------------------------------------------------------
def _flag(spectra, name):
    """A spectrum flag (extra, ignored) as booleans; all False when the input has none (wired to other spectra)."""
    return spectra[name].values.astype(bool) if name in spectra.coords else np.zeros(spectra.sizes["spectrum"], bool)


def _x_of(ctx):
    """(the totals' variable, its axis, the suffix in phi0_fit) for the chosen x axis (as the old viewer: absorbed
    energy in J, irradiation time in min)."""
    if ctx.params.x_axis == "time":
        return "total_irradiation_time", Axis("Irradiation time", "s", display=["min", "s", "h"]), "time"
    return "total_absorbed_energy", Axis("Absorbed energy", "J"), "energy"


def _interrupts(ctx, x_name):
    """Dashed grey lines at the interrupts of the run (a sample taken out, ...): at the total of the last spectrum
    before each, as the old viewer."""
    if not all(k in ctx.inputs for k in ("spectra", "rig_records", "irradiations")):
        return []
    records, spectra = ctx.inputs.rig_records, ctx.inputs.spectra.absorbance
    if "end_time" not in ctx.inputs.irradiations or not records.sizes.get("interrupt") or x_name not in spectra.coords \
            or "interrupt_start" not in records:
        return []
    totals = spectra[x_name].values[~_flag(spectra, "extra")]
    ends = ctx.inputs.irradiations.end_time.values
    lines = []
    for start, label in zip(records.interrupt_start.values, records.interrupt_label.values):
        before = int(np.sum(ends < start))
        if before < len(totals) and np.isfinite(totals[before]):
            lines.append(VLine(float(totals[before]), name=str(label), style=Style(role="event")))
    return lines


def _expected(ctx, spectra):
    """The expected spectrum: the absorption coefficients of the nucleosides at the measured wavelengths and their
    sum, scaled by one factor so that the sum has the maximum of the first spectrum above 230 nm."""
    counts = Counter(nucleoside_list(ctx.params.nucleosides))
    if "epsilon" not in ctx.inputs or not counts:
        return []
    exp = IrradiationExperiment()
    exp.nucleoside_spectra = ctx.inputs.epsilon.epsilon
    if not all(n in exp.nucleoside_spectra.nucleoside.values for n in counts):
        return []
    file_nm = exp.nucleoside_spectra.wavelength_nm.values
    wavelengths = spectra.wavelength.values
    wavelengths = wavelengths[(wavelengths * 1e9 >= file_nm.min()) & (wavelengths * 1e9 <= file_nm.max())]
    if not (wavelengths >= 230e-9).any():
        return []
    parts = {n: exp.absorption_coefficient([n] * count, wavelengths) for n, count in counts.items()}
    total = sum(parts.values())
    try:  # the first spectrum's maximum above 230 nm (as the old viewer); spectra on an odd grid have none
        factor = absorbance_maximum(spectra.isel(spectrum=0), 230e-9)[1] / np.max(total[wavelengths >= 230e-9])
    except (ValueError, TypeError, KeyError, IndexError):
        return []
    names = {n: f"{count} × {n}" if count > 1 else n for n, count in counts.items()}
    marks = []
    for i, n in enumerate(counts if len(counts) > 1 else []):  # one nucleoside: the same line as the sum
        marks.append(Line(wavelengths, parts[n] * factor, name=names[n],
                          style=Style(role="reference", color=EXPECTED_PARTS[i % len(EXPECTED_PARTS)], width=1.5,
                                      dash="dash")))
    marks.append(Line(wavelengths, total * factor, name=" + ".join(names.values()) + " (expected)",
                      style=Style(role="reference")))
    return marks


def draw_spectra(ctx):
    """The used spectra black to red in the order measured, the extra and ignored ones grey dashed; the baseline
    range as a grey band; the expected spectrum green (solid), the single nucleosides green dashed."""
    y_axis = Axis("Absorbance", "OD")
    if "spectra" not in ctx.inputs:
        return note_only("No spectra", x=WAVELENGTH, y=y_axis)
    spectra = ctx.inputs.spectra.absorbance
    extra = _flag(spectra, "extra")
    unused = extra | _flag(spectra, "ignored")
    marks = []
    if ctx.params.baseline:
        marks.append(Region([ctx.params.baseline[0]], [ctx.params.baseline[1]]))
    if unused.any():
        marks.append(Lines(spectra.wavelength, spectra.values[unused], style=Style(role="unused"), legend=False,
                           labels=[f"spectrum {i} ({'extra' if extra[i] else 'ignored'})"
                                   for i in np.flatnonzero(unused)]))
    marks.append(Lines(spectra.wavelength, spectra.values[~unused], colormap="black-red", legend=False,
                       labels=[f"spectrum {i}" for i in np.flatnonzero(~unused)]))
    marks += _expected(ctx, spectra)
    return PlotSpec(x=WAVELENGTH, y=y_axis, marks=marks, legend="top right")


def draw_absorbance(ctx):
    """The absorbance at λ spectrum of every spectrum with an irradiation before it, against the total absorbed
    energy or irradiation time up to it; the ignored ones grey. A click on a point ignores its spectrum or uses it
    again (Toggle on ignore_spectra)."""
    x_name, x_axis, _ = _x_of(ctx)
    y_axis = Axis(f"Absorbance at {ctx.params.marker * 1e9:.2f} nm", "OD")
    if "spectra" not in ctx.inputs:
        return note_only("No spectra", x=x_axis, y=y_axis)
    spectra = ctx.inputs.spectra.absorbance
    if x_name not in spectra.coords or np.isnan(spectra[x_name].values).all():
        return note_only("The numbers of spectra and irradiations do not fit (see Irradiation steps).",
                         x=x_axis, y=y_axis)
    wavelengths = spectra.wavelength.values
    if not wavelengths.min() <= ctx.params.marker <= wavelengths.max():
        return note_only(f"λ spectrum ({ctx.params.marker * 1e9:.1f} nm) is outside the spectra "
                         f"({wavelengths.min() * 1e9:.0f}–{wavelengths.max() * 1e9:.0f} nm).", x=x_axis, y=y_axis)
    not_extra = np.flatnonzero(~_flag(spectra, "extra"))
    x = spectra[x_name].values[not_extra]
    y = absorbance_at_wavelength(spectra.isel(spectrum=not_extra), ctx.params.marker).values
    ignored = np.isin(not_extra, ctx.params.ignore_spectra)
    marks = _interrupts(ctx, x_name)
    for chosen, role, status, click in ((~ignored, "measured", "used", "ignore"), (ignored, "unused", "ignored", "use")):
        if chosen.any():
            marks.append(Points(x[chosen], y[chosen], style=Style(role=role, symbol="x", size=11),
                                keys=not_extra[chosen],
                                labels=[f"spectrum {i} ({status}): click to {click} it" for i in not_extra[chosen]]))
    return PlotSpec(x=x_axis, y=y_axis, marks=marks)


def draw_qy(ctx):
    """The steps' quantum yields as blue crosses (in %), the fit through the first fit_num steps as a grey dashed line
    down to x = 0, φ₀ there as a black bar (± its standard error), written in the corner."""
    x_name, axis, suffix = _x_of(ctx)
    y_axis = Axis("Quantum yield", "", display="%")
    if "quantum_yields" not in ctx.inputs or "quantum_yield" not in ctx.inputs.quantum_yields:
        return note_only("Set the nucleosides (e.g. UC) to calculate the quantum yields.", x=axis, y=y_axis)
    steps = ctx.inputs.quantum_yields
    x, y = steps[x_name].values, steps.quantum_yield.values
    marks = _interrupts(ctx, x_name) + [Points(x, y, style=Style(role="measured"),
                                               labels=[f"step {i}" for i in steps.step.values])]
    if "phi0_fit" in ctx.inputs and f"slope_{suffix}" in ctx.inputs.phi0_fit:
        fit = ctx.inputs.phi0_fit
        phi0, err, slope = (float(fit[f"phi0_{suffix}"]), float(fit[f"phi0_{suffix}_err"]),
                            float(fit[f"slope_{suffix}"]))
        if np.isfinite(phi0):
            num = int(fit.fit_num)
            end = float(x[num - 1]) if num else 0.0  # from 0 to the last fitted step
            text = f"φ₀ = ({phi0 * 100:.5f} ± {err * 100:.5f}) %"
            marks.append(Line([0.0, end], [phi0, phi0 + slope * end], style=Style(role="fit")))
            marks.append(Points([0.0], [phi0], error_y=[err], style=Style(role="result"), labels=[text]))
            marks.append(Text(0.98, 0.95, text, coords="tile", anchor="right"))
    return PlotSpec(x=axis, y=y_axis, marks=marks)


def draw_phi0_spectrum(ctx):
    """φ₀ for the chosen x axis, a blue line with a light blue band of ± its standard error."""
    x_axis, y_axis = Axis("λ spectrum", "m", display="nm"), Axis("φ₀", "", display="%")
    if "phi0_spectrum" not in ctx.inputs:
        return note_only("Set the nucleosides and fit_num to calculate φ₀ for every wavelength.", x=x_axis, y=y_axis)
    if "x" not in ctx.inputs.phi0_spectrum.coords or ctx.params.x_axis not in ctx.inputs.phi0_spectrum.x.values:
        return note_only("The input is not φ₀ vs. λ spectrum.", x=x_axis, y=y_axis)
    data = ctx.inputs.phi0_spectrum.sel(x=ctx.params.x_axis)
    return PlotSpec(x=x_axis, y=y_axis, marks=[Line(data.wavelength, data.phi0.values, error_y=data.phi0_err.values,
                                                    name=f"φ₀ ({ctx.params.x_axis})",
                                                    style=Style(color="data", width=2))])


def draw_delta(ctx):
    """Per step, the spectrum before minus after, divided by the step's irradiation time (mOD/s)."""
    y_axis = Axis("Δ Absorbance / irr. time [mOD/s]")
    if "quantum_yields" not in ctx.inputs:
        return note_only("No steps (see Irradiation steps).", x=WAVELENGTH, y=y_axis)
    steps = ctx.inputs.quantum_yields
    per_time = steps.delta_spectrum_per_time.transpose("step", "wavelength")
    return PlotSpec(x=WAVELENGTH, y=y_axis, marks=[Lines(steps.wavelength, per_time.values * 1e3, colormap="black-red",
                                                         labels=[f"step {i}" for i in steps.step.values],
                                                         legend=False)])


def _traces(times, values, index):
    """One line through several traces: a gap (NaN) between two traces."""
    breaks = np.flatnonzero(np.diff(index) != 0) + 1
    return np.insert(times, breaks, np.nan), np.insert(values, breaks, np.nan)


def draw_powers(ctx):
    """Blue: during the irradiations; orange: calibrations; black: the LED power from the calibrations; grey bands:
    the stirrer running; purple bands: interrupts."""
    x_axis, y_axis = Axis("Time since the first reading", "s", display="min"), Axis("Power", "W", display="mW")
    if "power_traces" not in ctx.inputs or ctx.inputs.power_traces.sizes["sample"] == 0:
        return note_only("No power traces", x=x_axis, y=y_axis)
    powers = ctx.inputs.power_traces
    if not all(k in powers for k in ("power", "time", "index", "led_power", "calib_power")):
        return note_only("The input is not the power meter's traces.", x=x_axis, y=y_axis)
    t0 = float(powers.attrs.get("time_zero", 0.0))
    marks = []
    if "rig_records" in ctx.inputs and "stirring_start" in ctx.inputs.rig_records:
        records = ctx.inputs.rig_records
        for start, stop, labels, name, color in (
                (records.stirring_start.values, records.stirring_end.values, records.stirring_label.values,
                 "stirring", "muted"),
                (records.interrupt_start.values, records.interrupt_end.values, records.interrupt_label.values,
                 "interrupts", "#9467bd")):
            if len(start):
                marks.append(Region(start - t0, stop - t0, name=name, labels=list(labels),
                                    style=Style(color=color, alpha=0.12 if color == "muted" else 0.3)))
    for prefix, name, color in (("", "irradiation", "data"), ("calib_", "calibration", "series:1"),
                                ("led_", "LED (from calibrations)", "ink")):
        if powers.sizes[prefix + "sample" if prefix != "led_" else "sample"] == 0:
            continue
        index = powers[prefix + "index"].values if prefix != "led_" else powers.index.values
        time = (powers[prefix + "time"].values if prefix != "led_" else powers.time.values) - t0
        x, y = _traces(time, powers[prefix + "power"].values, index)
        marks.append(Line(x, y, name=name, style=Style(color=color, width=2.5)))
    return PlotSpec(x=x_axis, y=y_axis, marks=marks, legend="below")


class QuantumYieldView(tesselab.View):
    name = "Quantum yield"
    help = ("The six plots of the old viewer's quantum yield group: spectra with the expected spectrum, absorbance "
            "at λ (click a point to ignore its spectrum), quantum yield, φ₀ vs. λ spectrum, Δ spectra, power meter.")
    inputs = {"spectra": Signature(absorbance=Var("spectrum", "wavelength"),
                                   help="The spectra with their totals and which are ignored (Irradiation steps)."),
              "quantum_yields": Signature(delta_spectrum_per_time=Var("step", "wavelength"),
                                          total_absorbed_energy=Var("step"), optional=True,
                                          help="The steps with their quantum yields (without them the spectra and "
                                               "the power meter are still shown)."),
              "phi0_fit": Signature(phi0_energy=Var(), optional=True, help="The extrapolation to zero."),
              "phi0_spectrum": Signature(phi0=Var("x", "wavelength"), optional=True,
                                         help="φ₀ for every wavelength."),
              "epsilon": Signature(epsilon=Var("nucleoside", "wavelength_nm"), optional=True,
                                   help="The nucleosides' absorption coefficients (the expected spectrum)."),
              "irradiations": Signature(end_time=Var("irradiation"), optional=True,
                                        help="When each irradiation ended (to place the interrupts)."),
              "power_traces": Signature(power=Var("sample"), optional=True, help="The power meter's readings."),
              "rig_records": Signature(spectrum_time=Var("record"), optional=True,
                                       help="The rig's records: stirring and interrupts.")}
    params = [
        P.Float("marker", 260e-9, unit="m", display="nm", label="λ spectrum",
                help="The red line (the analysis variable lambda_spectrum: dragging it sets λ spectrum)."),
        P.Choice("x_axis", "energy", options=X_AXES, label="x axis",
                 help="Against the total absorbed energy or the total irradiation time (the analysis variable "
                      "x_axis)."),
        P.Str("nucleosides", "", help="For the expected spectrum (the analysis variable nucleosides)."),
        P.Ints("ignore_spectra", help="The ignored spectra, grey (the analysis variable ignore_spectra: a click on a "
                                      "point in 'Absorbance at λ' changes it)."),
        P.Range("baseline", optional=True, unit="m", display="nm",
                help="The baseline range, a grey band (the analysis variable baseline)."),
    ]
    plots = [
        Plot("spectra", "Spectra", draw=draw_spectra,
             help="Absorbance spectra minus the blank (the first in the file), the buffer and the baseline (grey "
                  "band). Coloured: the spectra used, black to red in the order measured; grey dashed: ignored or "
                  "extra spectra. Green: the expected spectrum, the sum of the nucleosides' absorption coefficients "
                  "(solid), and the single nucleosides (dashed), scaled so that the sum has the maximum of the first "
                  "spectrum above 230 nm. Drag the red line to set λ spectrum."),
        Plot("absorbance", "Absorbance at λ", draw=draw_absorbance, settings=["x_axis"],
             help="Absorbance at λ spectrum of every spectrum with an irradiation before it, against the total "
                  "absorbed energy or irradiation time up to that spectrum. Click a point to ignore its spectrum "
                  "(grey) or to use it again: the irradiations around an ignored spectrum count as one step. Dashed "
                  "grey lines: interrupts of the run."),
        Plot("qy", "Quantum yield", draw=draw_qy,
             help="Quantum yield of every step = photoproducts / absorbed photons. Photoproducts: from the absorbance "
                  "change at λ spectrum, the summed absorption coefficients of the nucleosides there, the volume and "
                  "the path length. Absorbed photons: the step's absorbed energy / the photon energy at λ "
                  "irradiation. Each cross sits at the total at the end of its step. Dashed line: linear fit through "
                  "the first fit_num steps, extrapolated to zero; φ₀ is its intercept, the black bar its standard "
                  "error."),
        Plot("phi0", "φ₀ vs. λ spectrum", draw=draw_phi0_spectrum,
             help="φ₀ calculated with every measured wavelength as λ spectrum, where the nucleosides absorb (summed "
                  "absorption coefficient above 10 % of its maximum). Band: ± standard error. Ideally φ₀ hardly "
                  "depends on λ spectrum; a strong change suggests that photoproducts also absorb there. Drag the "
                  "red line to set λ spectrum."),
        Plot("delta", "Δ spectra / irradiation time", draw=draw_delta,
             help="Spectrum before minus spectrum after each step, divided by the irradiation time of the step; one "
                  "line per step, black to red in the order of the steps. Positive: absorbance lost (e.g. the "
                  "nucleosides), negative: absorbance gained (e.g. photoproducts). A wavelength where all lines cross "
                  "zero is an isosbestic point."),
        Plot("power", "Power meter", draw=draw_powers,
             help="Power meter readings (behind the cuvette), background subtracted. Blue: during the irradiations; "
                  "orange: calibrations (usually a cuvette without sample); black: LED power during the "
                  "irradiations, interpolated between the calibrations. Absorbed power = LED power × 0.96 − measured "
                  "power / 0.96. Grey bands: the stirrer running; purple bands: interrupts (hover for details)."),
    ]
    interactions = [Cursor("marker", plots=["spectra", "phi0"], snap="wavelength", link="lambda",
                           label="λ = {value:.2f} nm"),
                    Toggle("ignore_spectra", plots=["absorbance"])]


# --- template ----------------------------------------------------------------------------------------------------------
def _path(path, project):
    if not path:
        return None
    relative = os.path.relpath(path, project)
    return path.replace("\\", "/") if relative.startswith("..") else relative.replace("\\", "/")


def _beside(folder, name):
    """A file next to a folder (the rig's records beside the power folder), or None if it is not there."""
    path = os.path.join(os.path.dirname(os.path.normpath(folder)), name) if folder else None
    return path if path and os.path.isfile(path) else None


def fill_template(folder, project):
    """The template's values for an experiment folder: from its analysis_params.json if it has one, otherwise the
    package's defaults (params.load_params)."""
    from .uv_analysis.params import load_params
    p = load_params(folder, resolve=True)
    return {"label": p.get("label") or os.path.basename(folder), "comment": p.get("comment", ""),
            "spectrum_file": _path(p["spectrum_file"], project), "buffer_file": _path(p.get("buffer_file"), project),
            "power_folder": _path(p["power_folder"], project),
            # the rig's records, next to the power folder (second series on)
            "steps_file": _path(_beside(p["power_folder"], "steps.jsonl"), project),
            "config_file": _path(_beside(p["power_folder"], "config.json"), project),
            "nucleoside_spectrum_file": _path(p["nucleoside_spectrum_file"], project),
            "skipfooter_spectra": p.get("skipfooter_spectra", 0),
            "combine_spectra": int(p.get("combine_spectra") or 0),
            "combine_spectra_method": p.get("combine_spectra_method", "last"),
            "extra_spectra": list(p.get("extra_spectra", [])), "ignore_spectra": list(p.get("ignore_spectra", [])),
            "baseline": p.get("baseline"), "nucleosides": "".join(p.get("nucleosides") or []),
            "lambda_irradiation": p["lambda_irradiation"], "lambda_spectrum": p["lambda_spectrum"],
            "thickness": p["thickness"], "volume": p["volume"], "fit_num": p.get("fit_num")}