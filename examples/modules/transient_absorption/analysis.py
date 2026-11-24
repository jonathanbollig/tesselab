"""
Transient absorption: reading a ΔA matrix, the usual corrections, and a global fit with exponentials. Plain numpy and
scipy, no Tesselab, so it can be used from a notebook too. Units are SI: delays in s, wavelengths in m, ΔA in OD.
"""
import numpy as np


def read_matrix(path, delay_factor=1e-12, wavelength_factor=1e-9, signal_factor=1.0):
    """A text matrix: the first row holds the wavelengths, the first column the delays, the rest ΔA. Separated by
    commas, semicolons, tabs or spaces; the corner cell is ignored. Returns (delays [s], wavelengths [m], ΔA [OD]
    with shape (delays, wavelengths))."""
    with open(path, encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
    sep = next((d for d in (",", ";", "\t") if d in lines[0]), None)
    rows = [[cell.strip() for cell in (line.split(sep) if sep else line.split())] for line in lines]
    wavelengths = np.array([float(v) for v in rows[0][1:]]) * wavelength_factor
    delays = np.array([float(r[0]) for r in rows[1:]]) * delay_factor
    signal = np.array([[float(v) if v not in ("", "nan", "NaN") else np.nan for v in r[1:]] for r in rows[1:]])
    if signal.shape != (len(delays), len(wavelengths)):
        raise ValueError(f"The matrix is not rectangular: {signal.shape} for {len(delays)} delays and "
                         f"{len(wavelengths)} wavelengths.")
    order = np.argsort(delays)
    return delays[order], wavelengths, signal[order] * signal_factor


def subtract_background(delays, signal, window):
    """Subtract, per wavelength, the mean ΔA of the delays in window (usually before time zero)."""
    inside = (delays >= window[0]) & (delays <= window[1])
    if not inside.any():
        raise ValueError("No delay lies in the background window.")
    return signal - np.nanmean(signal[inside], axis=0)


def chirp_curve(onsets, order, wavelengths):
    """Time zero at every wavelength: a polynomial through picked onsets [(delay, wavelength), ...] (in s, m)."""
    onsets = np.asarray(onsets, dtype=float)
    if len(onsets) == 0:
        return np.zeros_like(wavelengths)
    order = min(order, len(onsets) - 1)
    scale = 1e-9  # fit in nm, which keeps the polynomial well conditioned
    coefficients = np.polyfit(onsets[:, 1] / scale, onsets[:, 0], order)
    return np.polyval(coefficients, wavelengths / scale)


def correct_chirp(delays, signal, t0):
    """Shift every wavelength's time axis by its time zero t0 (linear interpolation onto the common delays). Before
    the first measured delay there are no data (NaN); after the last one the last value is held (the signal there is
    flat), so no wavelength loses its longest delays."""
    corrected = np.full_like(signal, np.nan)
    for j in range(signal.shape[1]):
        column = signal[:, j]
        valid = np.isfinite(column)
        if valid.sum() > 1:
            corrected[:, j] = np.interp(delays, delays[valid] - t0[j], column[valid], left=np.nan,
                                        right=column[valid][-1])
    return corrected


def average_around(values, index, n):
    """The mean of values[index - n : index + n + 1] along the last axis (clipped at the ends)."""
    import warnings
    low, high = max(0, index - n), min(values.shape[-1], index + n + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all missing (e.g. a scatter region): NaN, no message
        return np.nanmean(values[..., low:high], axis=-1)


def global_fit(delays, signal, taus, fit_range, max_iterations=200):
    """ΔA(t, λ) = Σ_k DAS_k(λ)·exp(−t/τ_k) for t > 0, fitted to the delays in fit_range: the amplitudes (decay
    associated spectra) linearly, the lifetimes nonlinearly (on a log scale). Returns lifetimes, their standard
    errors, DAS (components, wavelengths), the model (delays, wavelengths) and whether the fit converged."""
    from scipy.optimize import least_squares
    taus = np.asarray(taus, dtype=float)
    inside = (delays >= fit_range[0]) & (delays <= fit_range[1]) & (delays > 0)
    columns = np.isfinite(signal[inside]).all(axis=0)
    t, data = delays[inside], signal[np.ix_(inside, columns)]
    if len(t) < len(taus) + 2 or data.size == 0:
        raise ValueError("Too few delays in the fit range for this number of lifetimes.")

    def basis(log_taus):
        return np.exp(-t[:, None] / np.exp(log_taus)[None, :])

    def residuals(log_taus):
        b = basis(log_taus)
        amplitudes, *_ = np.linalg.lstsq(b, data, rcond=None)
        return (b @ amplitudes - data).ravel()

    result = least_squares(residuals, np.log(taus), max_nfev=max_iterations, x_scale="jac")
    log_taus = result.x
    b = basis(log_taus)
    amplitudes, *_ = np.linalg.lstsq(b, data, rcond=None)
    # standard errors of the lifetimes from the Jacobian (on the log scale, then back)
    dof = max(1, data.size - len(log_taus) - amplitudes.size)
    variance = np.sum(result.fun ** 2) / dof
    try:
        covariance = np.linalg.inv(result.jac.T @ result.jac) * variance
        errors = np.exp(log_taus) * np.sqrt(np.clip(np.diag(covariance), 0, None))
    except np.linalg.LinAlgError:
        errors = np.full(len(log_taus), np.inf)
    das = np.full((len(taus), signal.shape[1]), np.nan)
    das[:, columns] = amplitudes
    model = np.where(delays[:, None] > 0, np.exp(-np.clip(delays, 0, None)[:, None] / np.exp(log_taus)[None, :]),
                     0.0) @ np.nan_to_num(das)
    order = np.argsort(np.exp(log_taus))
    return np.exp(log_taus)[order], errors[order], das[order], model, bool(result.success)


def synthetic(seed=0):
    """A synthetic TA dataset (for the module's sample): two decays (1.5 ps, 80 ps), a 0.6 ps chirp across the
    spectrum, a background offset before time zero and noise."""
    rng = np.random.default_rng(seed)
    delays = np.concatenate([np.linspace(-3e-12, 3e-12, 61), np.logspace(np.log10(3.2e-12), np.log10(2e-9), 70)])
    wavelengths = np.linspace(450e-9, 700e-9, 101)
    nm = wavelengths * 1e9
    das = np.array([8e-3 * np.exp(-((nm - 520) / 30) ** 2) - 3e-3 * np.exp(-((nm - 640) / 25) ** 2),
                    4e-3 * np.exp(-((nm - 600) / 40) ** 2)])
    taus = np.array([1.5e-12, 80e-12])
    t0 = 0.6e-12 * ((nm - 450) / 250) ** 2  # the chirp: red arrives later
    t = delays[:, None] - t0[None, :]
    signal = sum(das[k][None, :] * np.exp(-np.clip(t, 0, None) / taus[k]) * (t > 0) for k in range(2))
    signal = signal + 2e-4 + rng.normal(0, 1.5e-4, signal.shape)  # background offset and noise
    return delays, wavelengths, signal
