"""The UV photochemistry example module gives the same numbers as kufner_uv_analysis itself (bit for bit), on the
package's sample experiment (copied to a temporary folder)."""
import os
import shutil

import numpy as np
import pytest

kufner = pytest.importorskip("kufner_uv_analysis")

from tesselab.engine import Engine
from tesselab.modules import Registry
from tesselab.project import Project
from tesselab.templates import new_analysis

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "modules")
SAMPLE = os.path.join(os.path.dirname(os.path.dirname(kufner.__file__)), "sample")


@pytest.fixture
def uv_project(tmp_path):
    if not os.path.isdir(SAMPLE):
        pytest.skip("the kufner sample is not there")
    root = tmp_path / "uv"
    shutil.copytree(SAMPLE, root / "uridine")
    return Project.create(root)


def original(folder):
    from kufner_uv_analysis.irradiation import IrradiationExperiment
    from kufner_uv_analysis.params import CALCULATE_KEYS, LOAD_KEYS, load_params
    params = load_params(folder)
    exp = IrradiationExperiment(label=params["label"])
    exp.load_data(**{k: params[k] for k in LOAD_KEYS})
    exp.calculate(**{k: params[k] for k in CALCULATE_KEYS})
    phi0 = exp.phi0_spectrum(params["nucleosides"], params["lambda_irradiation"], params["thickness"],
                             params["volume"], params["fit_num"])
    return exp, phi0


def test_template_recognizes_and_reproduces_the_package(uv_project):
    registry = Registry(uv_project.root, extra=[EXAMPLES], user=False, entry_points=False)
    assert "kufner.uv-photochemistry" in registry.modules, registry.errors
    from tesselab.project import find_data
    found = find_data(uv_project.root, registry.templates().values())
    assert [os.path.basename(f) for f, _ in found] == ["uridine"]
    analysis = new_analysis(registry, uv_project, "kufner.uv-photochemistry/quantum-yield", found[0][0])
    engine = Engine(registry, uv_project)
    results = engine.compute(analysis)
    for name, result in results.items():
        assert result.ok, (name, result.diagnostics)
    exp, phi0 = original(found[0][0])
    steps = results["qy"].outputs["quantum_yields"]
    assert steps.quantum_yield.values.tobytes() == exp.steps.quantum_yield.values.tobytes()
    extrapolation = results["qy"].outputs["phi0_fit"]
    assert float(extrapolation.phi0_energy) == exp.extrapolation_energy.y_axis
    assert float(extrapolation.phi0_energy_err) == exp.extrapolation_energy.y_axis_err
    assert float(extrapolation.phi0_time) == exp.extrapolation_time.y_axis
    assert results["phi0"].outputs["phi0_spectrum"].phi0.values.tobytes() == phi0.phi0.values.tobytes()
    spectra = results["steps"].outputs["spectra_with_totals"].absorbance
    assert spectra.values.tobytes() == exp.spectra.values.tobytes()
    assert np.array_equal(spectra.total_absorbed_energy.values, exp.spectra.total_absorbed_energy.values,
                          equal_nan=True)
    lambda_max, _ = exp.initial_maximum()
    assert results["corrections"].key_results["lambda_max"][0] == lambda_max


def test_changing_lambda_reruns_only_the_quantum_yield(uv_project):
    """λ is an analysis variable: the quantum yield runs again; the markers and φ0(λ) only redraw."""
    registry = Registry(uv_project.root, extra=[EXAMPLES], user=False, entry_points=False)
    engine = Engine(registry, uv_project)
    analysis = new_analysis(registry, uv_project, "kufner.uv-photochemistry/quantum-yield",
                            os.path.join(uv_project.root, "uridine"))
    first = engine.compute(analysis)
    analysis.vars["lambda_spectrum"] = 265e-9  # the analysis variable: λ spectrum and both λ markers
    second = engine.compute(analysis)
    assert second["corrections"] is first["corrections"] and second["steps"] is first["steps"]  # markers only redraw
    assert second["qy"] is not first["qy"]
    assert second["phi0"] is first["phi0"]  # φ0(λ) gets the same data (early cutoff): not computed again
    assert float(second["qy"].outputs["phi0_fit"].phi0_energy) != \
        float(first["qy"].outputs["phi0_fit"].phi0_energy)


def test_without_a_buffer_file_the_buffer_input_stays_empty(uv_project):
    """The buffer reader without a file gives nothing; the corrections run without a buffer (an optional input fed
    by nothing is left out, not an error)."""
    registry = Registry(uv_project.root, extra=[EXAMPLES], user=False, entry_points=False)
    engine = Engine(registry, uv_project)
    analysis = new_analysis(registry, uv_project, "kufner.uv-photochemistry/quantum-yield",
                            os.path.join(uv_project.root, "uridine"))
    with_buffer = engine.compute(analysis)["corrections"].outputs["spectra"].absorbance
    analysis.nodes["buffer_file"]["params"]["file"] = None
    results = engine.compute(analysis)
    assert results["buffer_file"].ok and "spectrum" not in results["buffer_file"].outputs
    assert results["corrections"].ok and results["qy"].ok
    assert not np.array_equal(results["corrections"].outputs["spectra"].absorbance.values, with_buffer.values)


def test_the_view_in_the_window(uv_project):
    """The view is the card's section (its six plots), the readers and compute nodes are folded in Details, the key
    results stand in the header; a click on a point in 'Absorbance at λ' ignores that spectrum (the variable that
    Irradiation steps uses); a view switched off is not shown."""
    import time
    from PySide6 import QtWidgets
    from tesselab.gui import theme
    from tesselab.gui.graph_view import GraphDialog
    from tesselab.gui.window import Window
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app, "light")

    def settle(seconds=0.3):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)

    registry = Registry(uv_project.root, extra=[EXAMPLES], user=False, entry_points=False)
    analysis = new_analysis(registry, uv_project, "kufner.uv-photochemistry/quantum-yield",
                            os.path.join(uv_project.root, "uridine"))
    analysis.save()
    window = Window(uv_project.root, [EXAMPLES])
    window.runner.inline = True
    window.resize(1400, 900)
    window.show()
    card = window.open_analysis(analysis.path)
    settle()
    view = card.groups["view"]
    assert view.parentWidget() is not card.details_box and card.groups["qy"].parentWidget() is card.details_box
    assert not card.details_open and "φ₀ (energy)" in card.results_label.text()
    assert set(view.tiles) == {"spectra", "absorbance", "qy", "phi0", "delta", "power"}
    names = [m.name for m in view.tiles["spectra"].view.spec.panels["main"].marks]
    assert any(n and n.endswith("(expected)") for n in names)  # the green expected spectrum
    before = window.runner.results(card.analysis)["steps"].fingerprint
    view.tiles["absorbance"].view.point_clicked.emit("main", 3)
    settle()
    assert card.analysis.vars["ignore_spectra"] == [3]
    assert window.runner.results(card.analysis)["steps"].fingerprint != before
    view.tiles["absorbance"].view.point_clicked.emit("main", 3)  # and used again
    assert card.analysis.vars["ignore_spectra"] == []
    dialog = GraphDialog.show_for(card)
    dialog.switch_view("view")
    assert "view" not in card.groups and card.analysis.nodes["view"]["off"]
    dialog.switch_view("view")
    assert "view" in card.groups
    dialog.close()
    for c in list(window.cards()):
        c.saved_content = c.analysis.state()
        if c.analysis.graph is not None:
            c.analysis.graph.saved = c.analysis.graph.doc
    window.close()
