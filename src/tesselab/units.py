"""
Display units: values are stored in SI units (or the unit a module chose), and only shown in another one.

A display unit is either a prefixed form of the stored unit (m -> nm, OD -> mOD, s -> ps, J -> mJ), a fixed factor
(s -> min, s -> h), or a conversion that is not a factor (m -> cm-1 and m -> eV, for wavelengths). to_display and
from_display convert numbers and numpy arrays; both raise ValueError for a pair they do not know.
"""
import numpy as np

PREFIXES = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "µ": 1e-6, "u": 1e-6, "m": 1e-3, "c": 1e-2, "k": 1e3, "M": 1e6,
            "G": 1e9}
FACTORS = {("s", "min"): 1 / 60, ("s", "h"): 1 / 3600, ("", "%"): 100.0, ("OD", "%"): 100.0}
# h * c / e in eV·m: the photon energy of light with a wavelength of 1 m
HC_EV_M = 6.62607015e-34 * 299792458.0 / 1.602176634e-19
NONLINEAR = {("m", "cm-1"), ("m", "eV")}


def factor(unit: str, display: str) -> float | None:
    """The factor display value / stored value, or None if the conversion is no factor (or unknown)."""
    if display is None or display == unit:
        return 1.0
    if (unit, display) in FACTORS:
        return FACTORS[(unit, display)]
    if unit and display.endswith(unit) and display[:-len(unit)] in PREFIXES:
        return 1 / PREFIXES[display[:-len(unit)]]
    if unit and unit[:-1] and unit[0] in PREFIXES and display == unit[1:]:  # e.g. stored in nm, shown in m
        return PREFIXES[unit[0]]
    return None


def is_known(unit: str, display: str) -> bool:
    return factor(unit, display) is not None or (unit, display) in NONLINEAR


def to_display(value, unit: str, display: str):
    """value (stored unit) -> the display unit. Arrays stay arrays, numbers stay numbers."""
    f = factor(unit, display)
    if f is not None:
        return value * f if f != 1.0 else value
    if (unit, display) == ("m", "cm-1"):
        return 1 / (np.asarray(value) * 100) if np.ndim(value) else 1 / (value * 100)
    if (unit, display) == ("m", "eV"):
        return HC_EV_M / np.asarray(value) if np.ndim(value) else HC_EV_M / value
    raise ValueError(f"Unknown display unit {display!r} for values in {unit!r}.")


def from_display(value, unit: str, display: str):
    """value (display unit) -> the stored unit."""
    f = factor(unit, display)
    if f is not None:
        return value / f if f != 1.0 else value
    if (unit, display) == ("m", "cm-1"):
        return 1 / (np.asarray(value) * 100) if np.ndim(value) else 1 / (value * 100)
    if (unit, display) == ("m", "eV"):
        return HC_EV_M / np.asarray(value) if np.ndim(value) else HC_EV_M / value
    raise ValueError(f"Unknown display unit {display!r} for values in {unit!r}.")


def label(text: str, unit: str | None, display: str | None) -> str:
    """'Wavelength' + 'm' shown in 'nm' -> 'Wavelength [nm]'; without a unit just the text."""
    shown = display or unit
    shown = {"cm-1": "cm⁻¹"}.get(shown, shown)
    return f"{text} [{shown}]" if shown else text
