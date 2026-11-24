"""
The analysis of one UV irradiation experiment that the nodes of this module call: IrradiationExperiment (load_data,
calculate, phi0_spectrum), reading Agilent spectra and power meter traces, the run records of the rig and
analysis_params.json.

This is a copy of five files of the author's own analysis package `kufner_uv_analysis` (state of 2026-11-04, commit
3a8d9f1): irradiation, spectrum, power, run_records and params. Only their imports differ (relative instead of
`kufner_uv_analysis.`) and the default path of the nucleoside spectra file in params.py (the author's own file there
became None), so that the example runs without that package. The module's numbers are tested against the
original, bit for bit, whenever it is installed (tests/test_uv_module.py, tools/compare_uv.py).
"""
