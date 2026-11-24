"""The example modules: they pass tesselab check, and the TA module recovers the synthetic sample's lifetimes."""
import os
import shutil

import numpy as np
import pytest

from tesselab.check import check_module
from tesselab.engine import Engine
from tesselab.modules import Registry
from tesselab.project import Project
from tesselab.templates import new_analysis

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "modules")
TA = os.path.join(EXAMPLES, "transient_absorption")


def test_new_module_passes_check(tmp_path):
    from tesselab.scaffold import new_module
    folder = new_module(str(tmp_path / "curve_tool"))
    report = check_module(folder, pictures=False)
    assert report.passed, report.text()
    assert not [m for level, m in report.entries if level == "warning"], report.text()


@pytest.mark.parametrize("name", ["transient_absorption", "uv_photochemistry"])
def test_example_modules_pass_check(name, tmp_path):
    folder = tmp_path / name
    shutil.copytree(os.path.join(EXAMPLES, name), folder, ignore=shutil.ignore_patterns("check", "__pycache__"))
    report = check_module(str(folder), pictures=False)
    assert report.passed, report.text()


@pytest.fixture
def ta(tmp_path):
    root = tmp_path / "ta"
    shutil.copytree(os.path.join(TA, "sample"), root / "sample")
    project = Project.create(root)
    registry = Registry(project.root, extra=[EXAMPLES], user=False, entry_points=False)
    analysis = new_analysis(registry, project, "org.tesselab.transient-absorption/global-analysis",
                            os.path.join(project.root, "sample", "ta_run"))
    return Engine(registry, project), analysis


def test_global_fit_recovers_the_lifetimes(ta):
    engine, analysis = ta
    results = engine.compute(analysis)
    fit = results["fit"].outputs["fit"]
    assert fit.tau.values == pytest.approx([1.5e-12, 80e-12], rel=0.05)


def test_chirp_correction_improves_the_fit(ta):
    """Onsets picked where the synthetic chirp puts time zero: the residual of the global fit gets smaller."""
    engine, analysis = ta
    before = np.nanstd(engine.compute(analysis)["fit"].outputs["fit"].residual.values)
    wavelengths = np.array([460e-9, 520e-9, 580e-9, 640e-9, 690e-9])
    t0 = 0.6e-12 * ((wavelengths * 1e9 - 450) / 250) ** 2
    steps = [{"kind": "background", "window": [-3e-12, -0.5e-12]}, {"kind": "chirp", "order": 2}]
    analysis.nodes["corrections"]["params"]["steps"] = steps
    analysis.vars["chirp_onsets"] = [[t, w] for t, w in zip(t0, wavelengths)]  # as picked on the map
    results = engine.compute(analysis)
    assert results["corrections"].ok, results["corrections"].diagnostics
    after = np.nanstd(results["fit"].outputs["fit"].residual.values)
    assert after < before
    assert results["corrections"].outputs["data"].t0.values == pytest.approx(
        0.6e-12 * ((results["corrections"].outputs["data"].wavelength.values * 1e9 - 450) / 250) ** 2, abs=2e-14)


def test_ta_crosshair_and_pick_in_the_window(ta, qapp_ta):
    from tesselab.gui.window import Window
    engine, analysis = ta
    analysis.save()
    window = Window(engine.project.root, [EXAMPLES])
    window.runner.inline = True
    window.resize(1400, 1000)
    window.show()
    card = window.open_analysis(analysis.path)
    for _ in range(5):
        qapp_ta.processEvents()
    group = card.groups["explorer"]
    crosshair = [item for interaction, param, item in group.interactions]
    assert len(crosshair) == 2  # the two lines of the crosshair on the map
    crosshair[0].on_change(5e-12, True)  # the delay line dragged to 5 ps (it snaps when dragged by the mouse)
    assert card.analysis.nodes["explorer"]["params"]["cursor_delay"] == 5e-12
    # a pick on the corrected map (a view) sets the variable that Corrections uses
    card.set_open("map", True)
    card.groups["map"]._pick("chirp_onsets", 1e-13, 500e-9, True)
    assert card.analysis.vars["chirp_onsets"] == [[1e-13, 500e-9]]
    for _ in range(5):
        qapp_ta.processEvents()
    assert card.engine.resolve_params(card.analysis, "corrections")[0]["chirp_onsets"] == [[1e-13, 500e-9]]
    card.undo()
    assert not card.analysis.vars["chirp_onsets"]
    for c in list(window.cards()):  # nothing to save: closing asks no question (a real dialog would block here)
        c.saved_content = c.analysis.state()
        if c.analysis.graph is not None:
            c.analysis.graph.saved = c.analysis.graph.doc
    window.close()


@pytest.fixture(scope="module")
def qapp_ta():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets
    from tesselab.gui import theme
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app, "light")
    return app
