"""
Compare the UV photochemistry module with kufner_uv_analysis on every real experiment (read-only: nothing is written
into the data folder). For each folder with an analysis_params.json: the package's load_data + calculate +
phi0_spectrum, and the module's template + nodes; every quantum yield, φ0 (energy and time, with errors) and φ0(λ)
must be identical to the last bit.

    python tools/compare_uv.py <folder that holds the experiments>

Needs kufner_uv_analysis (the author's private package) and real experiments: it is the author's check, not a test.
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "modules")


def main(root):
    import numpy as np
    from kufner_uv_analysis.irradiation import IrradiationExperiment
    from kufner_uv_analysis.params import CALCULATE_KEYS, LOAD_KEYS, read_params
    from tesselab.engine import Engine
    from tesselab.modules import Registry
    from tesselab.project import Project
    from tesselab.templates import new_analysis

    warnings.filterwarnings("ignore")
    project = Project(root)  # only read: no tesselab/ folder is made
    registry = Registry(None, extra=[EXAMPLES], user=False, entry_points=False)
    engine = Engine(registry, project)
    same = different = failed = 0
    start = time.time()
    for folder, _, files in sorted(os.walk(os.path.join(root, "data"))):
        if "analysis_params.json" not in files:
            continue
        name = os.path.relpath(folder, root)
        try:
            params = read_params(os.path.join(folder, "analysis_params.json"))
            exp = IrradiationExperiment(label=params["label"])
            exp.load_data(**{k: params[k] for k in LOAD_KEYS})
            exp.calculate(**{k: params[k] for k in CALCULATE_KEYS})
            phi0 = exp.phi0_spectrum(params["nucleosides"], params["lambda_irradiation"], params["thickness"],
                                     params["volume"], params["fit_num"]) if params["nucleosides"] and params["fit_num"] \
                else None
        except Exception as error:
            print(f"{name}: the package itself fails ({type(error).__name__}: {error}); skipped")
            continue
        analysis = new_analysis(registry, project, "kufner.uv-photochemistry/quantum-yield", folder)
        results = engine.compute(analysis)
        problems = [f"{n}: {r.error or r.status}" for n, r in results.items() if not r.ok]
        if problems:
            print(f"{name}: FAILED {problems}")
            failed += 1
            continue
        qy = results["qy"].outputs
        checks = {}
        if "quantum_yield" in exp.steps:
            checks["quantum yields"] = qy["quantum_yields"].quantum_yield.values.tobytes() == \
                exp.steps.quantum_yield.values.tobytes()
        for suffix, extrapolation in (("energy", exp.extrapolation_energy), ("time", exp.extrapolation_time)):
            if extrapolation is not None:
                checks[f"φ0 {suffix}"] = (float(qy["phi0_fit"][f"phi0_{suffix}"]) == extrapolation.y_axis or
                                         (np.isnan(extrapolation.y_axis) and np.isnan(float(qy["phi0_fit"][f"phi0_{suffix}"]))))
                checks[f"error {suffix}"] = (float(qy["phi0_fit"][f"phi0_{suffix}_err"]) == extrapolation.y_axis_err
                                            or (np.isnan(extrapolation.y_axis_err) and
                                                np.isnan(float(qy["phi0_fit"][f"phi0_{suffix}_err"]))))
        if phi0 is not None:
            checks["φ0(λ)"] = results["phi0"].outputs["phi0_spectrum"].phi0.values.tobytes() == phi0.phi0.values.tobytes()
        checks["spectra"] = results["steps"].outputs["spectra_with_totals"].absorbance.values.tobytes() == exp.spectra.values.tobytes()
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            different += 1
            print(f"{name}: DIFFERENT in {', '.join(bad)}")
        else:
            same += 1
            print(f"{name}: identical ({', '.join(checks)})")
    print(f"\n{same} identical, {different} different, {failed} failed, in {time.time() - start:.1f} s")
    return different == 0 and failed == 0


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
