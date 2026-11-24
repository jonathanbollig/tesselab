"""
The analysis of an irradiation experiment: quantum yields of a photoreaction from UV-Vis spectra and power meter
readings (IrradiationExperiment).

The experiment: a cuvette with the sample (nucleosides / RNA in buffer) is irradiated by an LED (λ irradiation) in
steps. After every irradiation a spectrum is measured (Agilent CSV with all spectra; its first spectrum is the blank).
A power meter behind the cuvette records every irradiation (one CSV per irradiation in the power folder; calib_*: the
LED without sample, background: LED off).

The analysis, in two calls:
- load_data(): reads the files. Spectra minus blank and buffer (no buffer if buffer_file is None), optionally minus a
  flat baseline (each spectrum's mean over a wavelength range, baseline=[start, stop]); "extra" spectra
  (measured without an irradiation before them) removed; groups of spectra combined (combine_spectra, e.g. 3 spectra
  per step: mean, last or lowest; the measured ones are kept as measured_spectra); "ignored" spectra marked. Power
  traces, and per irradiation its time and its absorbed energy (LED power interpolated between the calibrations
  × 0.96 − measured power / 0.96, integrated over time); every spectrum gets the totals of the irradiations before it,
  every measured spectrum the time it was measured (from the rig's records, second series only).
  The nucleoside spectra.
- calculate(): the irradiations are grouped into steps (between two used spectra; calc_steps). Per step the absorbance
  change at λ spectrum gives the photoproducts (Lambert-Beer: summed absorption coefficients of the nucleosides at
  λ spectrum, volume, path length), the absorbed energy gives the absorbed photons (photon energy at λ irradiation);
  quantum yield = photoproducts / photons (quantum_yields). Linear fits through the first fit_num quantum yields,
  extrapolated to zero, give φ0 (vs. time and vs. energy).
  calculate() can be called again with other parameters (e.g. λ spectrum) without reading the files again.
- phi0_spectrum() (after calculate()): φ0 with every measured wavelength as λ spectrum, all at once.
How spectra, irradiations and steps belong together is explained above the class.

Units are SI throughout (m, s, J, W, m³); the plots convert (nm, mW, %).

Usage, as in the notebooks (print_relevant and plot_relevant are in legacy.py; with a parameter file, see params.py):
    experiment = IrradiationExperiment(label="5_UC")
    experiment.load_data(spectrum_file=..., buffer_file=..., power_folder=..., nucleoside_spectrum_file=...)
    experiment.calculate(nucleosides=["U", "C"], lambda_irradiation=265e-9, lambda_spectrum=260e-9, thickness=10e-3,
                         volume=1.5e-6, fit_num=5)
    print_relevant(experiment)
    plot_relevant(experiment, save_path=figure_path)

These are scientific results: the numbers must not change unnoticed. tests/test_analysis.py checks them.
"""
import numpy as np
import xarray as xr
from .spectrum import (read_agilent_spectra, read_agilent_spectrum, read_nucleoside_spectra,
                                         absorbance_at_wavelength, absorbance_maximum)
from .power import read_power_csv, remove_below_threshold, irradiation_time, integrate_energy
from .run_records import read_run_records
import os
import re
import warnings
from dataclasses import dataclass
from typing import Literal

# exact by the definition of the SI units (2019), the same values as in scipy.constants (which took ~0.1 s to import)
c = 299792458.0  # speed of light [m/s]
h = 6.62607015e-34  # Planck constant [J s]
Avogadro = 6.02214076e23  # [1/mol]

"""
========================================================================================
Helper Functions
========================================================================================
"""

def _natural_sort_key(s: str):  # written by Opus 4.6
    """Sort key that orders numeric parts numerically (e.g. 2 before 10)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', s)]

def _linear_fit(x, y):
    """Ordinary least squares: slope, intercept and the standard error of the intercept,
    s * sqrt(1/n + mean(x)**2 / sum((x - mean(x))**2)). The same operations as scipy.stats.linregress (scipy 1.17),
    so the same results to the last digit; scipy.stats alone took ~0.75 s to import, a third of the viewer's start.
    NaN for fewer than 2 points or NaN in the data, as linregress; the standard error is 0 for 2 points."""
    n = len(x)
    if n <= 1 or np.isnan(x).any() or np.isnan(y).any():
        return np.nan, np.nan, np.nan
    if np.amax(x) == np.amin(x):
        raise ValueError("Cannot calculate a linear regression if all x values are identical")
    xmean, ymean = np.mean(x), np.mean(y)
    ssxm, ssxym, _, ssym = np.cov(x, y, bias=1).flat  # mean((x - xmean)**2), mean((x - xmean) * (y - ymean)), ...
    if ssxm == 0.0 or ssym == 0.0:
        r = np.nan if ssxym == 0 else 0.0
    else:
        r = min(max(ssxym / np.sqrt(ssxm * ssym), -1.0), 1.0)  # correlation coefficient, rounding errors cut off
    slope = ssxym / ssxm
    intercept = ymean - slope * xmean
    if n == 2:
        return slope, intercept, 0.0
    slope_stderr = np.sqrt((1 - r**2) * ssym / ssxm / (n - 2))
    return slope, intercept, slope_stderr * np.sqrt(ssxm + xmean**2)

@dataclass
class Extrapolation:
    """Linear fit through the first `num` quantum yields, extrapolated to x = 0."""
    slope: float
    y_axis: float      # intercept, i.e. the quantum yield extrapolated to x = 0
    y_axis_err: float  # standard error of the intercept
    num: int           # number of points the fit used

    @classmethod
    def from_fit(cls, x, y, num) -> "Extrapolation | None":
        """Fit the first `num` points (all of them if there are fewer). num=None means no extrapolation, and returns
        None."""
        if num is None:
            return None
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        num = min(num, len(x))
        slope, intercept, intercept_err = _linear_fit(x[:num], y[:num])  # the error needs num > 2 to be defined
        return cls(float(slope), float(intercept), float(intercept_err), num)

    def line(self, x_end):
        """x and y of the extrapolation line, from x = 0 to x_end."""
        x = np.array([0.0, x_end])
        return x, self.y_axis + self.slope * x

    def __str__(self):
        return f"({self.y_axis*100:.5f} +- {self.y_axis_err*100:.5f}) %"

"""
========================================================================================
Experiment Class
========================================================================================

How spectra and irradiations belong together:

    spectrum 0 | irradiation 0 | spectrum 1 | irradiation 1 | spectrum 2 | ... | spectrum n

Every irradiation (= one power file) lies between two spectra. Spectra that were measured
without an irradiation before them are "extra" and are removed first. They are counted as
in the file (without the blank), i.e. before combining spectra.
A "step" is the part of the experiment between two spectra that are used:
  - ignoring the first spectra: the irradiations before the first used spectrum are the
    pre-irradiation (their time and energy are added to the totals, but they are no step)
  - ignoring a spectrum in the middle: the irradiations before and after it form one step
"""

class IrradiationExperiment:
    def __init__(self, label: str | None = None) -> None:
        # All spectra (spectrum x wavelength) after subtracting the blank, the buffer and the baseline (if any), and
        # after combining (if any). Coordinates: `extra` (measured without an irradiation before it), `ignored` (not
        # used), and total_irradiation_time, total_absorbed_energy: the sums over all irradiations before the
        # spectrum (NaN if the numbers of spectra and irradiations do not fit; see calc_spectrum_totals).
        self.spectra: xr.DataArray | None = None
        # The spectra as measured, before combining, numbered as in the file (without the blank); coordinates `extra`,
        # `combined`: the index in self.spectra of the spectrum each went into, -1 for an extra spectrum removed when
        # combining, and `time`: when it was measured (epoch time [s], from the rig's records; NaN without them, see
        # calc_spectrum_times). Without combining the same spectra as self.spectra.
        self.measured_spectra: xr.DataArray | None = None

        # Power meter traces, one DataArray (time) per power file. They have different lengths, so they are lists.
        self.power_t: list[xr.DataArray] = []            # during the irradiations
        self.power_calib_t: list[xr.DataArray] = []      # calibration measurements (usually with a cuvette without sample)
        self.background_power: xr.DataArray | None = None  # LED off
        self.led_power_t: list[xr.DataArray] = []        # LED power during the irradiations, interpolated from the calibrations
        self.power_absorptions_t: list[xr.DataArray] = []  # power absorbed by the sample during the irradiations

        # absorption coefficients of the nucleosides (nucleoside x wavelength_nm), in nm and L/(mol·cm) as in the file
        self.nucleoside_spectra: xr.DataArray | None = None

        # One entry per irradiation (= power file): irradiation_time, end_time, absorbed_energy.
        # Coordinate `step`: the step the irradiation belongs to, -1 for the pre-irradiation.
        self.irradiations: xr.Dataset = xr.Dataset()
        self.pre_irradiation_time: float | None = None  # summed time of the pre-irradiation (calc_steps)
        self.pre_absorbed_energy: float | None = None   # summed absorbed energy of the pre-irradiation (calc_steps)

        # One entry per step, i.e. per delta spectrum: irradiation_time, absorbed_energy (summed over the
        # irradiations of the step), total_irradiation_time, total_absorbed_energy (at the end of each step: the
        # totals of its last spectrum, including the pre-irradiation), delta_spectrum (spectrum before minus spectrum after),
        # delta_spectrum_per_time, delta_absorbance, photoproducts, absorbed_photons, quantum_yield
        self.steps: xr.Dataset = xr.Dataset()

        # extrapolation lines in the quantum yield plots
        self.extrapolation_time: Extrapolation | None = None    # vs. irradiation time
        self.extrapolation_energy: Extrapolation | None = None  # vs. absorbed energy

        self.label = label  # the name of the experiment, e.g. the title of the overview figure
        self.baseline: list[float] | None = None  # [start, stop] of the baseline correction [m], None: none

    @property
    def relevant_spectra(self) -> xr.DataArray:
        """The used spectra: one at the start, and one at the end of every step."""
        return self.spectra.isel(spectrum=~(self.spectra.extra.values | self.spectra.ignored.values))

    def initial_maximum(self, min_wavelength=230e-9) -> tuple[float, float]:
        """λ max and the absorbance there (m, OD) of the very first spectrum (minus blank and buffer), on the hump
        above `min_wavelength` (see absorbance_maximum). After load_data()."""
        return absorbance_maximum(self.spectra.isel(spectrum=0), min_wavelength)

    """
    ----------------------------------------------------------------------------------------
    Spectrum functions
    ----------------------------------------------------------------------------------------
    """
    # removed Jasco spectrum reading, if needed, check git before 22.07.2026

    def create_spectra_from_agilent_file(self, filename, number: int | None = None, skiprows: int = 2, skipfooter: int = 0, reference_spectrum_index: int | None = 0):
        # number includes all spectra in the file (also calibration etc.)
        spectra = read_agilent_spectra(filename, number, skiprows=skiprows, skipfooter=skipfooter)
        if reference_spectrum_index is not None:
            spectra = spectra - spectra.isel(spectrum=reference_spectrum_index)
            spectra = spectra.drop_isel(spectrum=reference_spectrum_index)
        self.spectra = spectra
        self.mark_extra_spectra([])
        self.mark_ignored_spectra([])

    def combine_spectra(self, method, num):
        """Combine every `num` consecutive spectra (measured after the same irradiation) into one.
        Extra spectra are removed first. The spectra before combining are kept as measured_spectra."""
        if method not in ["last", "mean", "lowest"]:
            raise ValueError(f"Invalid method '{method}'. Choose from 'last', 'mean', 'lowest'.")
        if num <= 0:
            raise ValueError(f"Invalid num '{num}'. Must be a positive integer.")
        spectra = self.spectra.isel(spectrum=~self.spectra.extra.values).drop_vars(["extra", "ignored"])
        if spectra.sizes["spectrum"] % num != 0:
            raise ValueError(f"{spectra.sizes['spectrum']} spectra cannot be split into groups of {num}.")
        combined = []
        for i in range(spectra.sizes["spectrum"] // num):
            group = spectra.isel(spectrum=slice(i*num, (i+1)*num))
            if method == "last":
                combined.append(group.isel(spectrum=-1))
            elif method == "mean":
                combined.append(group.mean("spectrum", skipna=False))  # skipna=False: a NaN stays NaN
            elif method == "lowest":
                combined.append(group.min("spectrum", skipna=False))
        # which combined spectrum each measured one went into (-1: extra, removed)
        into = np.full(self.spectra.sizes["spectrum"], -1)
        into[~self.spectra.extra.values] = np.arange(spectra.sizes["spectrum"]) // num
        self.measured_spectra = self.spectra.drop_vars("ignored").assign_coords(combined=("spectrum", into))
        self.spectra = xr.concat(combined, dim="spectrum")
        self.mark_extra_spectra([])
        self.mark_ignored_spectra([])

    def mark_extra_spectra(self, extra_spectra: list[int]):
        """Mark spectra (by their index in self.spectra) as extra, see the explanation above the class."""
        index = np.arange(self.spectra.sizes["spectrum"])
        self.spectra = self.spectra.assign_coords(extra=("spectrum", np.isin(index, extra_spectra)))

    def mark_ignored_spectra(self, ignore_spectra: list[int]):
        """Mark spectra (by their index in self.spectra) as ignored, see the explanation above the class."""
        index = np.arange(self.spectra.sizes["spectrum"])
        self.spectra = self.spectra.assign_coords(ignored=("spectrum", np.isin(index, ignore_spectra)))

    def subtract_reference_spectrum(self, reference: xr.DataArray):
        # This function can be used to either subtract a blank spectrum (without a cuvette), usually the first spectrum
        # In that case, this function is called from "create_spectra_from_agilent_file"
        # Or it can be used to subtract a reference spectrum that is measured separately (for example Buffer spectrum)
        if not np.array_equal(reference.wavelength.values, self.spectra.wavelength.values):
            # the buffer may be measured with a different wavelength grid: cut the spectra to the range
            # of the reference, and interpolate the reference linearly onto their wavelengths
            reference_min, reference_max = reference.wavelength.min(), reference.wavelength.max()
            in_range = (self.spectra.wavelength >= reference_min) & (self.spectra.wavelength <= reference_max)
            self.spectra = self.spectra.isel(wavelength=in_range.values)
            warnings.warn(f"Reference spectrum has different wavelengths than the spectra. Cutting the spectra to "
                          f"{float(reference_min)*1e9:.1f}-{float(reference_max)*1e9:.1f} nm and interpolating the reference.")
            reference = reference.interp(wavelength=self.spectra.wavelength)
        self.spectra = self.spectra - reference

    def subtract_baseline(self, start, stop):
        """Subtract from every spectrum its mean absorbance between start and stop [m] (both included): removes flat
        offsets (e.g. gas bubbles on the window, the cuvette's position). Only valid where nothing absorbs in that
        range and the offset really is flat."""
        in_range = ((self.spectra.wavelength >= start) & (self.spectra.wavelength <= stop)).values
        if not in_range.any():
            raise ValueError(f"No measured wavelength in the baseline range {start * 1e9:g}-{stop * 1e9:g} nm (the spectra "
                             f"cover {self.spectra.wavelength.values.min() * 1e9:.1f}-{self.spectra.wavelength.values.max() * 1e9:.1f} nm).")
        self.spectra = self.spectra - self.spectra.isel(wavelength=in_range).mean("wavelength")
        self.baseline = [start, stop]

    """
    ----------------------------------------------------------------------------------------
    Power functions
    ----------------------------------------------------------------------------------------
    """

    def create_powers_from_folder(self, folder, skip_rows=0, remove_zeros=True):
        self.power_t, self.power_calib_t = [], []
        files = sorted([f for f in os.listdir(folder) if f.endswith('.csv')], key=_natural_sort_key)

        for file in files:
            power = read_power_csv(os.path.join(folder, file), skip_rows=skip_rows)
            if 'calib' in file.lower():
                if remove_zeros:
                    power = remove_below_threshold(power, 0.5)
                self.power_calib_t.append(power)
            elif 'background' in file.lower():
                self.background_power = power
            else:
                if remove_zeros:
                    power = remove_below_threshold(power, 0.2)
                self.power_t.append(power)

        if self.background_power is not None:
            background = float(np.mean(self.background_power.values))
            self.power_t = [power - background for power in self.power_t]
            self.power_calib_t = [calib - background for calib in self.power_calib_t]

    def calc_irradiations(self):
        """Irradiation time and absorbed energy of every irradiation (= power file)."""
        # LED power during each irradiation, interpolated between the mean values of the calibration measurements
        calib_times = np.array([calib.time.values.mean() for calib in self.power_calib_t])
        calib_powers = [float(np.mean(calib.values)) for calib in self.power_calib_t]
        self.led_power_t = [power.copy(data=np.interp(power.time.values, calib_times, calib_powers)) for power in self.power_t]

        # absorbed power: what enters the sample minus what leaves it.
        # Assuming 4 % reflection at both glass surfaces, and no absorbance of the buffer
        glass_t = 1 - 0.04
        self.power_absorptions_t = [led_power * glass_t - power / glass_t
                                    for power, led_power in zip(self.power_t, self.led_power_t)]

        self.irradiations = xr.Dataset({
            "irradiation_time": ("irradiation", [irradiation_time(power) for power in self.power_t]),
            "end_time": ("irradiation", [power.time.values[-1] for power in self.power_t]),
            "absorbed_energy": ("irradiation", [integrate_energy(absorption) for absorption in self.power_absorptions_t]),
        })

    def calc_spectrum_totals(self):
        """Every spectrum gets the total irradiation time and absorbed energy of all irradiations before it, as the
        coordinates total_irradiation_time and total_absorbed_energy; an extra spectrum the same as the spectrum
        before it (0 before the first one). Independent of which spectra are ignored. NaN if the numbers of spectra
        and irradiations do not fit (calc_steps then says so)."""
        not_extra = ~self.spectra.extra.values
        # the number of the last spectrum that is not extra up to each spectrum (0 for extra ones before the first)
        last = np.maximum(np.cumsum(not_extra) - 1, 0)
        for name, column in (("total_irradiation_time", "irradiation_time"), ("total_absorbed_energy", "absorbed_energy")):
            totals = np.full(self.spectra.sizes["spectrum"], np.nan)
            values = self.irradiations[column].values
            if np.sum(not_extra) == len(values) + 1:
                # spectrum i (counted without the extra ones) comes after irradiations 0 .. i-1
                totals = np.concatenate(([0.0], np.cumsum(values)))[last]
            self.spectra = self.spectra.assign_coords({name: ("spectrum", totals)})

    def calc_spectrum_times(self, power_folder):
        """Every measured spectrum gets the time the rig triggered it (epoch time [s], the clock of the power files)
        as the coordinate `time` of measured_spectra. The rig's records (steps.jsonl, from the second series on) know
        only the spectra it triggered itself, so they are given to the spectra that are not extra, in order. NaN for
        the extra spectra, and for all if there are no records or their number of spectra does not fit."""
        times = np.full(self.measured_spectra.sizes["spectrum"], np.nan)
        try:
            records = read_run_records(power_folder)
        except Exception as error:  # e.g. a broken line in steps.jsonl: the analysis does not need them
            warnings.warn(f"The rig's records could not be read: {error}")
            records = None
        not_extra = ~self.measured_spectra.extra.values
        if records is not None and len(records.spectrum_times) == np.sum(not_extra):
            times[not_extra] = records.spectrum_times
        self.measured_spectra = self.measured_spectra.assign_coords(time=("spectrum", times))

    """
    ----------------------------------------------------------------------------------------
    Steps: delta spectra, photoproducts and quantum yields
    ----------------------------------------------------------------------------------------
    """

    def calc_steps(self):
        """Group the irradiations into steps: their times and energies, and the delta spectra. Independent of the
        parameters of calculate(); depends only on which spectra are extra or ignored."""
        # irradiation i lies between spectrum i and i+1 (without the extra spectra). It belongs to the step that
        # starts at the last used spectrum before it, i.e. step = (number of used spectra up to spectrum i) - 1.
        spectra = self.spectra.isel(spectrum=~self.spectra.extra.values)
        n_irradiations = self.irradiations.sizes["irradiation"]
        if spectra.sizes["spectrum"] != n_irradiations + 1:
            raise ValueError(f"There must be one spectrum more than irradiations, but there are {spectra.sizes['spectrum']} "
                             f"spectra (without extra spectra) and {n_irradiations} irradiations.")
        if spectra.ignored.values[-1]:
            raise ValueError("The last spectrum cannot be ignored, the last irradiation would have no spectrum after it.")
        used = ~spectra.ignored.values
        self.irradiations = self.irradiations.assign_coords(step=("irradiation", np.cumsum(used)[:-1] - 1))

        pre = self.irradiations.step.values == -1
        steps = self.irradiations[["irradiation_time", "absorbed_energy"]].isel(irradiation=~pre).groupby("step").sum()

        # the totals are those of the used spectra (calc_spectrum_totals): the first one is measured after the
        # pre-irradiation, the others each at the end of a step
        relevant_spectra = self.relevant_spectra
        self.pre_irradiation_time = float(relevant_spectra.total_irradiation_time.values[0])
        self.pre_absorbed_energy = float(relevant_spectra.total_absorbed_energy.values[0])
        steps["total_irradiation_time"] = ("step", relevant_spectra.total_irradiation_time.values[1:])
        steps["total_absorbed_energy"] = ("step", relevant_spectra.total_absorbed_energy.values[1:])

        # one delta per step: the spectrum before minus the spectrum after (only the values: the coordinates of the
        # spectra, e.g. their totals, do not belong to the steps)
        relevant_spectra = relevant_spectra.reset_coords(drop=True)
        before = relevant_spectra.isel(spectrum=slice(None, -1))
        after = relevant_spectra.isel(spectrum=slice(1, None))
        steps["delta_spectrum"] = (before - after).rename(spectrum="step")
        steps["delta_spectrum_per_time"] = steps.delta_spectrum / steps.irradiation_time
        self.steps = steps

    def absorption_coefficient(self, nucleosides, wavelength):
        """Summed decadic absorption coefficient of the nucleosides [m²/mol] at `wavelength` [m], a number or an
        array; linear interpolation in the nucleoside spectra."""
        spectra = self.nucleoside_spectra
        coeffs = [np.interp(np.asarray(wavelength) * 1e9, spectra.wavelength_nm.values,
                            spectra.sel(nucleoside=nucleoside).values) for nucleoside in nucleosides]
        return sum(coeffs) * 100 / 1000  # L/(mol·cm) -> m²/mol

    def quantum_yields(self, nucleosides, lambda_irradiation, lambda_spectrum, thickness, volume) -> xr.Dataset:
        """Per step (after calc_steps): delta_absorbance, photoproducts, absorbed_photons and quantum_yield at
        lambda_spectrum. lambda_spectrum can also be an array: the result then has a wavelength dimension."""
        result = xr.Dataset()
        result["delta_absorbance"] = absorbance_at_wavelength(self.steps.delta_spectrum, lambda_spectrum)
        decadic_absorption_coeff = self.absorption_coefficient(nucleosides, lambda_spectrum)
        if np.ndim(decadic_absorption_coeff):
            decadic_absorption_coeff = xr.DataArray(decadic_absorption_coeff, dims="wavelength",
                                                    coords={"wavelength": lambda_spectrum})
        result["photoproducts"] = (result.delta_absorbance * volume * Avogadro
                                   / (decadic_absorption_coeff * thickness))
        result["absorbed_photons"] = self.steps.absorbed_energy * lambda_irradiation / (h * c)  # E = h*c/λ per photon
        result["quantum_yield"] = result.photoproducts / result.absorbed_photons
        return result

    def load_data(self, spectrum_file, buffer_file, power_folder, nucleoside_spectrum_file, skipfooter_spectra=0,
                  combine_spectra=False,  # or the number of spectra measured after each irradiation, combined into one
                  combine_spectra_method: Literal["last", "mean", "lowest"] = "last",  # how to combine them
                  ignore_spectra=[],  # spectra that are not used, see the explanation above the class
                  extra_spectra=[],  # spectra without an irradiation before them (numbered as in the file)
                  baseline=None):  # [start, stop] in m: subtract each spectrum's mean in this range; None: no correction
        """Read the files (spectra, powers, nucleoside spectra) and calculate the irradiations; then calculate()."""
        # spectral data
        self.create_spectra_from_agilent_file(spectrum_file, reference_spectrum_index=0, number=None, skiprows=2, skipfooter=skipfooter_spectra)
        if buffer_file:  # None or "": no buffer spectrum is subtracted
            buffer_spectrum = read_agilent_spectrum(buffer_file, skiprows=2, skipfooter=0)
            self.subtract_reference_spectrum(buffer_spectrum)
        if baseline:  # every measured spectrum, before combining
            self.subtract_baseline(*baseline)
        self.mark_extra_spectra(extra_spectra)  # counted as in the file, before combining
        # each measured spectrum is a spectrum of its own, unless they are combined (which sets measured_spectra again)
        self.measured_spectra = self.spectra.drop_vars("ignored").assign_coords(
            combined=("spectrum", np.arange(self.spectra.sizes["spectrum"])))
        if combine_spectra:
            self.combine_spectra(method=combine_spectra_method, num=combine_spectra)  # removes the extra spectra
        self.mark_ignored_spectra(ignore_spectra)

        # power data
        self.create_powers_from_folder(power_folder, skip_rows=1)
        self.calc_irradiations()
        self.calc_spectrum_totals()
        self.calc_spectrum_times(power_folder)

        self.nucleoside_spectra = read_nucleoside_spectra(nucleoside_spectrum_file)

    def calculate(self, nucleosides, lambda_irradiation, lambda_spectrum, thickness, volume,
                  fit_num=None):  # points of the extrapolation fits (vs. time and vs. energy), None: no extrapolation
        """Steps, photoproducts, quantum yields and extrapolations, after load_data(). Can be called again with other
        parameters (e.g. lambda_spectrum) without reading the files again. Without nucleosides only the steps (with
        the delta spectra): the quantum yields need their absorption coefficients."""
        # the steps first: the spectra may have been marked as ignored since the last call (the viewer does that)
        self.calc_steps()
        self.extrapolation_time = self.extrapolation_energy = None
        if not nucleosides:
            return
        result = self.quantum_yields(nucleosides, lambda_irradiation, lambda_spectrum, thickness, volume)
        self.steps = self.steps.assign(result.drop_vars("wavelength"))

        self.extrapolation_time = Extrapolation.from_fit(self.steps.total_irradiation_time, self.steps.quantum_yield, fit_num)
        self.extrapolation_energy = Extrapolation.from_fit(self.steps.total_absorbed_energy, self.steps.quantum_yield, fit_num)

    def phi0_spectrum(self, nucleosides, lambda_irradiation, thickness, volume, fit_num, threshold=0.1) -> xr.Dataset:
        """φ0 with every measured wavelength as λ spectrum (after calculate(), whose steps it uses), where the sample
        absorbs: summed absorption coefficient above `threshold` × its maximum. Returns phi0 and its standard error
        phi0_err (dims x, wavelength; x: "energy" and "time"), NaN without extrapolation (fit_num None)."""
        wavelengths = self.spectra.wavelength.values
        coeffs = self.absorption_coefficient(nucleosides, wavelengths)
        wavelengths = wavelengths[coeffs > threshold * coeffs.max()]
        quantum_yield = self.quantum_yields(nucleosides, lambda_irradiation, wavelengths, thickness, volume).quantum_yield
        quantum_yield = quantum_yield.transpose("step", "wavelength").values
        phi0, phi0_err = np.full((2, len(wavelengths)), np.nan), np.full((2, len(wavelengths)), np.nan)
        for i, total in enumerate((self.steps.total_absorbed_energy, self.steps.total_irradiation_time)):
            for j in range(len(wavelengths)):
                extrapolation = Extrapolation.from_fit(total, quantum_yield[:, j], fit_num)
                if extrapolation is not None:
                    phi0[i, j], phi0_err[i, j] = extrapolation.y_axis, extrapolation.y_axis_err
        return xr.Dataset({"phi0": (("x", "wavelength"), phi0), "phi0_err": (("x", "wavelength"), phi0_err)},
                          coords={"x": ["energy", "time"], "wavelength": wavelengths})
