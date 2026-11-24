"""
Analysis parameters of an experiment, stored as JSON next to the data.

The JSON holds the keyword arguments of IrradiationExperiment.load_data (LOAD_KEYS) and calculate
(CALCULATE_KEYS), with the same names and SI units, plus the `label` of the experiment, `export_paths` (where the
viewer's Save writes the PDF of all its plots, each with a JSON of the plotted data next to it) and a free-text
`comment` (shown in the viewer). Every path can be relative to the folder of the JSON file or absolute; it is kept as
it was written. The settings of the chemometric analysis (written by the
viewer) are under "svd" and "mcr": the arguments of chemometrics.svd and chemometrics.mcr; those of the viewer's
subspectra group under "subspectra". read_params leaves them out unless asked (chemometrics=True). Results are saved with them (RESULT_KEYS, written by
the viewer's Save): `lambda_max` [m] and `absorbance_max` [OD], the hump of the first spectrum
(IrradiationExperiment.initial_maximum); `phi0_energy` and its standard error `phi0_energy_err`, φ0 extrapolated
vs. absorbed energy (missing without an extrapolation). Usage:

    params = read_params(".../5_UC/analysis_params.json")
    experiment = IrradiationExperiment(label=params["label"])
    experiment.load_data(**{key: params[key] for key in LOAD_KEYS})
    experiment.calculate(**{key: params[key] for key in CALCULATE_KEYS})

An experiment folder is a folder with an analysis_params.json. Folders without one are also found
if they look like raw data: the spectra and a `powers` folder in it (the first series, e.g. data/5_UC), or
a `data` folder that contains them (the rig's layout since the second series, e.g. data/5_UC@2). Saving their
parameters makes them experiment folders.
"""
import copy
import inspect
import json
import os
import re

from .irradiation import IrradiationExperiment, _natural_sort_key

PARAMS_FILENAME = "analysis_params.json"

# used for experiments without a parameter file. Edit here if the defaults change.
DEFAULTS = {
    "buffer_file": None,  # no buffer subtracted; set one in the viewer's Files or the parameter file
    "nucleoside_spectrum_file": None,  # the original had the path of the author's own file here: choose one
    "lambda_irradiation": 265e-9,
    "lambda_spectrum": 260e-9,
    "thickness": 10e-3,
    "volume": 1.5e-6,
    "combine_spectra_method": "mean",
    "fit_num": 5,
}

# order of the keys in the JSON file, same as in the notebooks
KEYS = ["spectrum_file", "buffer_file", "power_folder", "nucleoside_spectrum_file", "nucleosides",
        "lambda_irradiation", "lambda_spectrum", "thickness", "volume", "skipfooter_spectra",
        "combine_spectra", "combine_spectra_method", "ignore_spectra", "extra_spectra", "baseline",
        "fit_num", "label", "export_paths", "comment"]
# the arguments of IrradiationExperiment.load_data and of calculate (the rest of KEYS: label, export_paths, comment)
LOAD_KEYS = ["spectrum_file", "buffer_file", "power_folder", "nucleoside_spectrum_file", "skipfooter_spectra",
             "combine_spectra", "combine_spectra_method", "ignore_spectra", "extra_spectra", "baseline"]
CALCULATE_KEYS = ["nucleosides", "lambda_irradiation", "lambda_spectrum", "thickness", "volume", "fit_num"]
# the chemometric analysis (arguments of chemometrics.svd and chemometrics.mcr), saved after KEYS
CHEMOMETRICS_DEFAULTS = {
    "svd": {"skip_wavelengths": 15, "dimensions": 3},
    "mcr": {"skip_wavelengths": 15, "species": 3, "svd_dimensions": 3, "max_iter": 200, "runs": 20,
            "sreact_constraint": True, "seed": 0},
    # the viewer's subspectra group: more wavelengths [m] in its absorbance against time (each [λ] or a range
    # [start, stop]), the number of measured wavelengths on each side a single wavelength is also averaged over
    # (0: none), what its right axis shows ("ratio": −dark / irradiation change, "dark": the dark change), the
    # wavelength ranges of its baseline monitor, the spectrum whose dark change is shown, and whether the dark-change map is
    # divided by the change of the irradiation before each spectrum
    "subspectra": {"extra_wavelengths": [], "average": 4, "right_axis": "ratio", "baseline_ranges": [[385e-9, 398e-9], [330e-9, 340e-9]],
                   "spectrum": 1, "relative": False},
}
# results of the analysis, written after KEYS (by the viewer's Save; read_params passes them on but nothing uses them)
RESULT_KEYS = ["lambda_max", "absorbance_max",  # IrradiationExperiment.initial_maximum: [m], [OD]
               "phi0_energy", "phi0_energy_err"]  # extrapolation_energy: y_axis, y_axis_err
PATH_KEYS = ["spectrum_file", "buffer_file", "power_folder", "nucleoside_spectrum_file"]
PATH_LIST_KEYS = ["export_paths"]  # lists of paths
DEFAULT_EXPORT = "analysis.pdf"  # the viewer's PDF (and analysis_data.json), next to the parameter file
OLD_KEYS = ["figure_path"]  # the overview figure of the notebooks (until 2026-11-04), dropped when read


def data_folder(folder) -> str | None:
    """The folder with the spectra and powers of an experiment folder, None if it is no experiment folder."""
    if os.path.isdir(os.path.join(folder, "powers")):
        return folder
    data = os.path.join(folder, "data")
    if os.path.isdir(data) and any(f.lower().endswith(".csv") for f in os.listdir(data)):
        return data
    return None


def find_experiments(folder) -> list[str]:
    """All experiment folders in `folder` and its subfolders (including `folder` itself): folders with a
    parameter file, and folders with raw data but no parameter file yet."""
    found = []
    for root, dirs, _ in os.walk(folder):
        if os.path.isfile(os.path.join(root, PARAMS_FILENAME)) or data_folder(root) is not None:
            found.append(root)
            dirs.clear()  # do not look inside an experiment
        else:
            dirs[:] = sorted((d for d in dirs if not d.startswith(".")), key=_natural_sort_key)  # 2_UU before 12_UA
    return found


def default_params(folder) -> dict:
    """Parameters for an experiment folder without a parameter file. Paths as they are written into a new parameter
    file: the experiment's own files relative to `folder`, the shared ones absolute (use resolve_paths to resolve)."""
    folder = os.path.abspath(folder)
    name = os.path.basename(folder)
    data = data_folder(folder)
    if data is None:
        raise ValueError(f"{folder} is no experiment folder (no 'powers' or 'data' folder).")

    # the spectra: <name>.csv if it exists, otherwise the first CSV that is no LED calibration
    csvs = sorted(f for f in os.listdir(data) if f.lower().endswith(".csv") and not f.startswith("LED_calibration"))
    spectrum_file = f"{name}.csv" if f"{name}.csv" in csvs else (csvs[0] if csvs else f"{name}.csv")

    params = dict(DEFAULTS)
    # the experiment's own files relative to its folder, the shared ones (nucleoside spectra) absolute
    data = os.path.relpath(data, folder)  # "." or "data"
    params.update(spectrum_file=os.path.normpath(os.path.join(data, spectrum_file)),
                  power_folder=os.path.normpath(os.path.join(data, "powers")),
                  nucleosides=nucleosides_from_name(name),
                  skipfooter_spectra=0, combine_spectra=False, ignore_spectra=[], extra_spectra=[], baseline=None,
                  label=name, export_paths=[DEFAULT_EXPORT], comment="")

    # settings of the rig (since the second series)
    config_file = os.path.join(folder, data, "config.json")
    if os.path.isfile(config_file):
        with open(config_file) as f:
            config = json.load(f).get("config", {})
        if config.get("spectrometer_n", 1) > 1:
            params["combine_spectra"] = int(config["spectrometer_n"])
        if "wavelength_nm" in config:
            params["lambda_irradiation"] = config["wavelength_nm"] * 1e-9
    return {key: params[key] for key in KEYS}


def nucleosides_from_name(name) -> list[str]:
    """'5_UC' -> ['U', 'C'], '12_UA2' -> ['U', 'A'], '5_UC@2' (second series) -> ['U', 'C']. Empty if the name does
    not look like that."""
    match = re.fullmatch(r"\d+_([ACGTU]+)\d*(@\d+)?", name)
    return list(match.group(1)) if match else []


def resolve_paths(params, folder) -> dict:
    """The parameters with absolute paths: relative paths are relative to `folder` (the folder of the JSON file)."""
    params = dict(params)
    absolute = lambda path: os.path.normpath(os.path.join(os.path.abspath(folder), path))
    for key in PATH_KEYS:
        if params.get(key):
            params[key] = absolute(params[key])
    for key in PATH_LIST_KEYS:
        if params.get(key):
            params[key] = [absolute(path) for path in params[key]]
    return params


def read_params(path, resolve=True, chemometrics=False) -> dict:
    """Parameters from a JSON file. Missing keys get the defaults of load_data and calculate (label: None,
    export_paths: [DEFAULT_EXPORT], comment: ""); keys of older files that are no longer used are dropped (OLD_KEYS).
    resolve=False: paths as written in the file (relative or absolute).
    chemometrics=True: also "svd", "mcr" and "subspectra" (missing settings from CHEMOMETRICS_DEFAULTS)."""
    with open(path) as f:
        params = json.load(f)
    defaults = {p.name: p.default for function in (IrradiationExperiment.load_data, IrradiationExperiment.calculate)
                for p in inspect.signature(function).parameters.values() if p.default is not inspect.Parameter.empty}
    params = {**defaults, "label": None, "export_paths": [DEFAULT_EXPORT], "comment": "", **params}
    for key in OLD_KEYS:
        params.pop(key, None)
    for key, default in CHEMOMETRICS_DEFAULTS.items():
        if chemometrics:
            params[key] = {**default, **params.get(key, {})}
        else:
            params.pop(key, None)
    return resolve_paths(params, os.path.dirname(path)) if resolve else params


def write_params(path, params):
    """Write parameters to a JSON file. Paths are written as they are (relative to its folder or absolute)."""
    params = {key: params[key] for key in KEYS + RESULT_KEYS + list(CHEMOMETRICS_DEFAULTS) if key in params}
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
        f.write("\n")


def load_params(folder, resolve=True, chemometrics=False) -> dict:
    """Parameters of an experiment folder: from its parameter file if it has one, otherwise the defaults.
    resolve=False: paths as written (relative to the folder or absolute). chemometrics: see read_params."""
    path = os.path.join(folder, PARAMS_FILENAME)
    if os.path.isfile(path):
        params = read_params(path, resolve=False, chemometrics=chemometrics)
    else:
        params = default_params(folder)
        if chemometrics:
            params.update(copy.deepcopy(CHEMOMETRICS_DEFAULTS))
    return resolve_paths(params, folder) if resolve else params
