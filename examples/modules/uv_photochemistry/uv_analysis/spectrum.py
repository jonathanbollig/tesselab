"""
Spectra as xarray DataArrays.

A single spectrum has the dimension ("wavelength",), several spectra have
("spectrum", "wavelength"). Wavelengths are stored in m, absorbances in OD.
The functions below take spectra as their first argument, e.g.
`absorbance_at_wavelength(spectra, 262e-9)`. Plots of spectra for the notebooks are in legacy.py.

Arithmetic between spectra (spectra - reference, ...) requires exactly matching
wavelengths, otherwise xarray raises a ValueError (set below).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# Fail instead of silently intersecting the wavelengths of two spectra with different grids.
# Note: this is a global xarray option, it applies to all xarray code in the same python session.
xr.set_options(arithmetic_join="exact")


def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Convert N cell centers into N+1 cell edges (midpoints, outer edges mirrored).

    Works for non-equidistant centers, which is the point: pcolormesh cells then
    have the correct individual widths.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        return np.array([])
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    mid = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate(([first], mid, [last]))

def _to_m(wavelength, unit):
    if unit == "nm":
        return wavelength * 1e-9
    if unit != "m":
        raise ValueError("Unit must be either 'm' or 'nm'.")
    return wavelength

"""
----------------------------------------------------------------------------------------
Creating spectra
----------------------------------------------------------------------------------------
"""

def spectra_from_data(wavelengths, absorbances, name: str | None = None) -> xr.DataArray:
    """Spectra from a (n_spectra, n_wavelengths) array, or a single spectrum from a 1D array."""
    absorbances = np.asarray(absorbances)
    dims = ("wavelength",) if absorbances.ndim == 1 else ("spectrum", "wavelength")
    return xr.DataArray(absorbances, dims=dims, coords={"wavelength": wavelengths}, name=name)

def read_agilent_spectrum(filename, name: str | None = None, skiprows=2, skipfooter=0) -> xr.DataArray:
    """A single spectrum from an Agilent CSV file (first two columns are wavelength and absorbance)."""
    df = pd.read_csv(filename, skiprows=skiprows, skipfooter=skipfooter, engine="python", header=None)
    df = df.iloc[np.argsort(df.iloc[:, 0].to_numpy())]  # make sure wavelengths are sorted in ascending order
    wavelengths = df.iloc[:, 0].to_numpy() / 1e9  # convert to m
    return spectra_from_data(wavelengths, df.iloc[:, 1].to_numpy(), name=name)

def read_agilent_spectra(filename, number: int | None = None, skiprows=2, skipfooter=0) -> xr.DataArray:
    """All spectra from an Agilent CSV file (column pairs of wavelength and absorbance)."""
    df = pd.read_csv(filename, skiprows=skiprows, skipfooter=skipfooter, engine="python", header=None)
    df = df.iloc[np.argsort(df.iloc[:, 0].to_numpy())]  # make sure wavelengths are sorted in ascending order
    if number is None:
        number = df.shape[1] // 2
    wavelengths = df.iloc[:, 0].to_numpy() / 1e9  # convert to m
    for i in range(number):
        if not np.array_equal(df.iloc[:, 2*i].to_numpy() / 1e9, wavelengths):
            raise ValueError(f"Spectrum {i} in {filename} has different wavelengths than spectrum 0.")
    absorbances = np.array([df.iloc[:, 2*i + 1].to_numpy() for i in range(number)])
    return spectra_from_data(wavelengths, absorbances)

def read_nucleoside_spectra(filename) -> xr.DataArray:
    """Absorption coefficients of the nucleosides (dims nucleoside x wavelength_nm) from a CSV file with a
    `wavelength_nm` column and one column per nucleoside ("A", "U", ...) in L/(mol·cm). Unlike the other spectra they
    stay in the units of the file: the analysis has always interpolated in nm, and converting first would change the
    results in the last digits."""
    data = np.genfromtxt(filename, delimiter=",", names=True)
    nucleosides = [name for name in data.dtype.names if name != "wavelength_nm"]
    spectra = xr.DataArray([data[nucleoside] for nucleoside in nucleosides], dims=("nucleoside", "wavelength_nm"),
                           coords={"nucleoside": nucleosides, "wavelength_nm": data["wavelength_nm"]})
    return spectra.sortby("wavelength_nm")

"""
----------------------------------------------------------------------------------------
Working with spectra
----------------------------------------------------------------------------------------
"""

def absorbance_at_wavelength(spectra: xr.DataArray, wavelength, unit="m") -> xr.DataArray:
    """Absorbance at `wavelength` (linear interpolation), one value per spectrum. `wavelength` can also be an array:
    the result then has a wavelength dimension."""
    wavelength = _to_m(np.asarray(wavelength), unit)
    wavelengths = spectra.wavelength.values
    outside = wavelength[(wavelength < wavelengths.min()) | (wavelength > wavelengths.max())]
    if outside.size:
        raise ValueError(f"Wavelength {outside.flat[0]*1e9:.1f} nm is outside the range of the spectrum "
                         f"({wavelengths.min()*1e9:.1f}-{wavelengths.max()*1e9:.1f} nm).")
    return spectra.interp(wavelength=wavelength)

def absorbance_maximum(spectrum: xr.DataArray, min_wavelength=230e-9) -> tuple[float, float]:
    """The highest absorbance of a single spectrum above `min_wavelength` (a measured point, no interpolation), as
    (λ max in m, absorbance in OD). Below ~230 nm the absorbance usually rises steeply again (the sample and the buffer
    absorb far in the UV), so that the overall maximum would be at the edge of the spectrum, not on the hump."""
    above = spectrum.sel(wavelength=slice(min_wavelength, None))
    wavelength = float(above.idxmax("wavelength"))
    return wavelength, float(above.sel(wavelength=wavelength))

def to_agilent_csv(spectrum: xr.DataArray, filename):
    """Save a single spectrum in Agilent format (two columns: wavelength in nm, absorbance)."""
    df = pd.DataFrame({
        "Wavelength (nm)": spectrum.wavelength.values * 1e9,  # convert to nm
        "Absorbance (OD)": spectrum.values
    })
    df.to_csv(filename, index=False)
    # add "sample1" at top of the file for consistency
    with open(filename, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write("sample1\n" + content)
