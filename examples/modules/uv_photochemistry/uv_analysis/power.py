"""
Power meter traces as xarray DataArrays with the dimension ("time",).
Times are in s, powers in W.
"""
import numpy as np
import xarray as xr


def read_power_csv(filename, skip_rows=0) -> xr.DataArray:
    """A power trace from a CSV file with the columns time [s] and power [mW]. Stored in W."""
    data = np.loadtxt(filename, delimiter=",", skiprows=skip_rows)
    return xr.DataArray(data[:, 1] * 1e-3, dims="time", coords={"time": data[:, 0]})

def remove_below_threshold(power: xr.DataArray, relative_threshold: float) -> xr.DataArray:
    """Keep only the points above relative_threshold * maximum (removes the time the LED was off)."""
    return power.isel(time=(power > relative_threshold * power.max()).values)

def time_step(power: xr.DataArray) -> float:
    """Mean time between two measured points."""
    return np.mean(np.diff(power.time.values))

def irradiation_time(power: xr.DataArray) -> float:
    """Duration of the trace, assuming n points with dt mean an irradiation time of n*dt (instead of (n-1)*dt)."""
    times = power.time.values
    return times[-1] - times[0] + time_step(power)

def integrate_energy(power: xr.DataArray) -> float:
    """Energy [J] of the trace, using rectangular integration: each point lasts until the next one, the last point lasts the mean dt."""
    dt = np.concatenate([np.diff(power.time.values), [time_step(power)]])
    return np.cumsum(power.values * dt)[-1]  # cumsum instead of sum keeps the summation order of the previous code
