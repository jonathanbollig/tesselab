"""The window, offscreen: opening, computing, editing, undo, saving, sessions, linking, the graph, themes, export.
The runner runs inline (no worker processes) so that the tests are deterministic; test_worker_processes checks the
real way once."""
import json
import os
import time

import numpy as np
import pytest

from tesselab.analysis import Analysis

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    from tesselab.gui import theme
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app, "light")
    yield app


def mix_analysis(project, label="mix", views=True):
    """A table and its SVD; with views (default) their views too: the plots are theirs."""
    analysis = Analysis(path=project.new_analysis_path(label))
    analysis.doc["label"] = label
    analysis.add("org.tesselab.table/csv", "table", params={"file": "data/mix.csv"})
    if views:
        analysis.add("org.tesselab.table/table-plot", "plot", inputs={"table": "table.table"})
    analysis.add("org.tesselab.chemometrics/svd", "svd", inputs={"data": "table.table"})
    if views:
        analysis.add("org.tesselab.chemometrics/svd-view", "svd_view", inputs={"svd": "svd.svd"})
    analysis.doc["view"]["open"] = ["svd_view"] if views else ["svd"]
    analysis.save()
    return analysis.path


@pytest.fixture
def window(qapp, project, monkeypatch):
    from tesselab.gui import widgets
    from tesselab.gui.window import Window
    monkeypatch.setattr(widgets, "message_box", lambda *args, **kwargs: 1)  # "Discard" / "Cancel" in dialogs
    import tesselab.gui.window as window_module
    import tesselab.gui.card as card_module
    monkeypatch.setattr(window_module, "message_box", lambda *args, **kwargs: 1)
    monkeypatch.setattr(card_module, "message_box", lambda *args, **kwargs: 0)
    w = Window(project.root)
    w.runner.inline = True
    w.resize(1300, 900)
    w.show()
    qapp.processEvents()
    yield w
    for card in list(w.cards()):
        card.saved_content = card.analysis.state()
    w.close()


def settle(qapp, seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        qapp.processEvents()
        time.sleep(0.01)


def test_open_compute_edit_undo_save(window, project, qapp):
    path = mix_analysis(project)
    card = window.open_analysis(path)
    settle(qapp)
    results = window.runner.results(card.analysis)
    assert results["svd"].ok and results["table"].ok
    group = card.groups["svd_view"]
    assert group.tiles["values"].view.spec is not None  # the open view drew its plots
    assert [t.number for t in group.tiles.values()] == [2, 3, 4, 5]  # numbered after the table's plot
    assert "residual" in card.groups["svd"].summary.text()  # the compute node's key result (in Details)
    assert "residual" in card.results_label.text()  # and in the card's header
    assert not card.is_unsaved()
    card.set_param("svd", "components", 2)
    settle(qapp)
    assert card.is_unsaved()
    assert window.runner.results(card.analysis)["svd"].outputs["svd"].sizes["component"] == 2
    card.undo()
    settle(qapp)
    assert not card.is_unsaved()
    assert window.runner.results(card.analysis)["svd"].outputs["svd"].sizes["component"] == 3
    card.redo()
    assert card.is_unsaved()
    assert window.save_card(card)
    assert not card.is_unsaved()
    with open(path) as f:
        saved = json.load(f)
    assert saved["nodes"]["svd"]["params"]["components"] == 2
    assert "residual" in saved["results"]["svd"]
    assert saved["provenance"]["nodes"]["svd"].startswith("org.tesselab.chemometrics/svd")


def test_errors_are_shown_not_raised(window, project, qapp):
    path = mix_analysis(project)
    card = window.open_analysis(path)
    settle(qapp)
    card.set_param("table", "file", "data/nothing.csv")
    settle(qapp)
    results = window.runner.results(card.analysis)
    assert results["table"].status == "error"
    assert results["svd"].status == "blocked"
    assert card.groups["table"].messages.isVisibleTo(card.groups["table"])  # (in Details, folded)


def test_add_and_remove_nodes(window, project, qapp):
    path = mix_analysis(project)
    card = window.open_analysis(path)
    settle(qapp)
    from PySide6 import QtWidgets
    menu = QtWidgets.QMenu()
    card.fill_add_menu(menu)
    texts = [a.text() for a in menu.actions()]
    assert any(t.startswith("SVD") for t in texts) and any(t.startswith("MCR-ALS") for t in texts)
    card.add_node("org.tesselab.chemometrics/mcr")
    settle(qapp)
    assert "mcr" in card.analysis.nodes
    assert card.analysis.nodes["mcr"]["inputs"]["data"] == "table.table"  # the same source as SVD
    assert window.runner.results(card.analysis)["mcr"].status == "waiting"  # runs on its button only
    card.groups["mcr"].settings_button.setChecked(True)  # with its settings open (the group stayed, grey)
    card.remove("mcr")
    assert "mcr" not in card.analysis.nodes and "mcr" not in card.groups
    settle(qapp)
    # the rebuilt groups draw their plots (they stayed "Computing …" when a new group was not visible yet)
    assert card.groups["svd_view"].tiles["values"].view.spec.panels["main"].note == ""
    card.undo()
    assert "mcr" in card.analysis.nodes and "mcr" in card.groups


def test_run_on_button(window, project, qapp):
    pytest.importorskip("pymcr")
    path = mix_analysis(project)
    card = window.open_analysis(path)
    settle(qapp)
    card.add_node("org.tesselab.chemometrics/mcr")
    card.set_input("mcr", "data", "table.table")
    card.set_param("mcr", "runs", 1)
    card.set_param("mcr", "max_iter", 20)
    settle(qapp)
    card.run_node("mcr")
    settle(qapp)
    result = window.runner.results(card.analysis)["mcr"]
    assert result.ok, result.diagnostics
    assert result.outputs["mcr"].sizes["species"] == 3


def test_variables_and_cursor(window, project, qapp):
    path = mix_analysis(project)
    card = window.open_analysis(path)
    settle(qapp)
    card.bind("svd", "components", "k")
    assert card.analysis.vars["k"] == 3
    card.set_var("k", 4)
    settle(qapp)
    assert window.runner.results(card.analysis)["svd"].outputs["svd"].sizes["component"] == 4
    card.bind("svd", "components", None)
    assert card.analysis.nodes["svd"]["params"]["components"] == 4


def test_linked_cards_share_cursors_and_folds(window, project, qapp):
    a = window.open_analysis(mix_analysis(project, "a"))
    b = window.open_analysis(mix_analysis(project, "b"))
    settle(qapp)
    a.link_button.setChecked(True)
    b.link_button.setChecked(True)
    a.set_open("table", True)
    assert "table" in b.analysis.doc["view"]["open"]
    # zoom: the same plot of the linked analyses shares its view box
    box_a = a.groups["svd_view"].tiles["values"].view.panel().box
    box_b = b.groups["svd_view"].tiles["values"].view.panel().box
    assert box_b.linkedView(0) is box_a or box_a.linkedView(0) is box_b


def test_session_round_trip(window, project, qapp):
    a = window.open_analysis(mix_analysis(project, "a"))
    tab = window.new_tab("second")
    b = window.open_analysis(mix_analysis(project, "b"), tab)
    b.link_button.setChecked(True)
    path = os.path.join(project.folder("sessions"), "w.tesselab-session")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    window.write_session(path)
    window.open_session(path)
    settle(qapp)
    assert window.tabs.count() == 2
    assert [c.analysis.label for c in window.tabs.widget(1).cards] == ["b"]
    assert window.tabs.widget(1).cards[0].link_button.isChecked()
    assert window.tabs.widget(1).name == "second"


def test_graph_view_and_theme_and_export(window, project, qapp, tmp_path):
    from tesselab.gui.graph_view import GraphDialog, layout_graph
    card = window.open_analysis(mix_analysis(project))
    settle(qapp)
    kinds = {"plot": "view", "svd_view": "view"}  # views in the last column
    assert layout_graph(card.analysis, kinds=kinds) == {"table": (0, 0), "svd": (16, 0), "plot": (32, 0),
                                                         "svd_view": (32, 5)}  # grid units
    GraphDialog.show_for(card)
    dialog = GraphDialog._open[id(card)]
    assert len([i for i in dialog.scene.items() if i.__class__.__name__ == "Box"]) == 4
    dialog.close()
    window.set_theme("dark")
    settle(qapp)
    window.set_theme("light")
    from tesselab import export
    pdf, js = export.export_analysis(window.engine, card.analysis, str(tmp_path / "out.pdf"),
                                     results=window.runner.results(card.analysis))
    with open(js) as f:
        data = json.load(f)
    assert [p["number"] for n in data["nodes"] for p in n["plots"]] == [1, 2, 3, 4, 5]
    assert os.path.getsize(pdf) > 10000


def test_graph_editing(window, project, qapp, monkeypatch):
    """Connect by dragging from an output to an input, remove a connection, remove a node: all through the card
    (undo follows), the graph redraws itself."""
    from PySide6 import QtCore
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from tesselab.gui.graph_view import GraphDialog
    card = window.open_analysis(mix_analysis(project, views=False))
    settle(qapp)
    GraphDialog.show_for(card)
    dialog = GraphDialog._open[id(card)]
    assert len(dialog.wires) == 1
    path = dialog.wires[0].path()
    name, input_name, _ = dialog.edge_at(path.pointAtPercent(0.5))
    assert (name, input_name) == ("svd", "data")
    assert dialog.edge_at(path.pointAtPercent(0.5) + QtCore.QPointF(0, 40)) is None
    dialog.disconnect_input("svd", "data", dialog.wires[0].key[2])
    settle(qapp)
    assert card.analysis.nodes["svd"]["inputs"]["data"] is None and dialog.wires == []
    table, svd = dialog.boxes["table"], dialog.boxes["svd"]
    assert dialog.begin_drag(table.port("out", "table"))
    assert dialog.drag.accepted["svd"] == ["data"]
    assert svd.input_items["data"][0].rect().width() == 12  # highlighted
    dialog.drop(svd.port("in", "data") + QtCore.QPointF(150, 0))  # dropped beside the port: nothing changes
    assert dialog.drag is None and card.analysis.nodes["svd"]["inputs"]["data"] is None
    table, svd = dialog.boxes["table"], dialog.boxes["svd"]
    view, port = dialog.view, lambda p: dialog.view.mapFromScene(p)  # by the mouse, as the user does it
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=port(table.port("out", "table")))
    end = port(svd.port("in", "data") + QtCore.QPointF(5, 5))  # near enough: snaps
    QTest.mouseMove(view.viewport(), end)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
    settle(qapp)
    assert card.analysis.nodes["svd"]["inputs"]["data"] == "table.table" and len(dialog.wires) == 1
    assert dialog.begin_drag(dialog.boxes["svd"].port("out", "svd"))
    assert "svd" not in dialog.drag.accepted  # no cycle
    dialog.redraw()
    assert not dialog.begin_drag(QtCore.QPointF(-500, -500))
    # a connection picked up at its input and dropped on empty space is removed; Ctrl+Z in the graph window undoes it
    svd = dialog.boxes["svd"]
    assert dialog.begin_drag(svd.port("in", "data"))
    assert dialog.drag.lifted == ("svd", "data") and not dialog.wires[0].isVisible()
    assert "remove the connection" in dialog.drag_hint()
    dialog.drop(QtCore.QPointF(-500, -500))
    settle(qapp)
    assert card.analysis.nodes["svd"]["inputs"]["data"] is None
    dialog.activateWindow()
    QTest.keyClick(dialog, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    settle(qapp)
    assert card.analysis.nodes["svd"]["inputs"]["data"] == "table.table"
    # picked up and put back: nothing changes (no undo step)
    steps = len(card.undo_steps)
    dialog.begin_drag(dialog.boxes["svd"].port("in", "data"))
    dialog.drop(dialog.boxes["svd"].port("in", "data"))
    assert len(card.undo_steps) == steps
    # from an input to an output, after disconnecting; Esc cancels without closing the window
    card.set_input("svd", "data", None)
    settle(qapp)
    assert dialog.begin_drag(dialog.boxes["svd"].port("in", "data"))
    assert dialog.drag.kind == "in" and dialog.drag.accepted == {"table": ["table"]}
    QTest.keyClick(dialog, Qt.Key.Key_Escape)
    assert dialog.drag is None and dialog.isVisible()
    dialog.begin_drag(dialog.boxes["svd"].port("in", "data"))
    dialog.drop(dialog.boxes["table"].port("out", "table"))
    settle(qapp)
    assert card.analysis.nodes["svd"]["inputs"]["data"] == "table.table"
    # the connection under the mouse is highlighted
    edge = dialog.wires[0]
    dialog.hover(edge.path().pointAtPercent(0.5))
    assert dialog.hovered is edge and edge.pen().widthF() > 2
    dialog.hover(QtCore.QPointF(-500, -500))
    assert dialog.hovered is None and edge.pen().widthF() < 2
    card.remove("svd")
    settle(qapp)
    assert set(dialog.boxes) == {"table"}
    card.undo()
    settle(qapp)
    assert set(dialog.boxes) == {"table", "svd"}
    # right-click menus; inside a box the node counts, not a connection crossing it
    texts = [a.text() for a in dialog.menu_at(dialog.wires[0].path().pointAtPercent(0.5)).actions()]
    assert texts[0] == "Remove connection" and "Use as row_pos" in texts  # and which variable of the table is which
    texts = [a.text() for a in dialog.menu_at(dialog.boxes["svd"].sceneBoundingRect().center()).actions()]
    assert texts[:3] == ["Show in the main window", "Rename…", "Duplicate"] and texts[-1] == "Remove"
    assert [a.text() for a in dialog.menu_at(QtCore.QPointF(-500, -500)).actions()][0] == "Add node"
    # closing the card closes its graph; results arriving later never reach the deleted window
    import sys
    caught = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: caught.append(exc))
    other = window.open_analysis(mix_analysis(project, "other"))
    settle(qapp)
    window.close_card(card)
    assert not dialog.isVisible() and id(card) not in GraphDialog._open
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    window.runner.updated.emit(other.analysis)
    settle(qapp)
    assert caught == []


def test_cancelled_tasks_and_job_answers(window, project, qapp):
    """A task cancelled by Reload modules never becomes a result; a job action's answer is checked by the
    parameters (np.int64 stored as int, an unknown parameter refused) and applied as one undoable change."""
    from concurrent.futures import Future
    from tesselab.engine import Key
    from tesselab.node import Action
    card = window.open_analysis(mix_analysis(project))
    settle(qapp)
    future = Future()
    future.cancel()
    window.runner.pending["x" * 32] = (Key(card.analysis, "svd"), {"fingerprint": "x" * 32}, future, {card.analysis})
    window.runner._collect("x" * 32)
    assert "x" * 32 not in window.engine.memory
    action = Action("fit", "Fit", run=None, job=True)
    done = Future()
    done.set_result({"components": np.int64(2)})
    card._finish_job("svd", action, done)
    value = card.analysis.nodes["svd"]["params"]["components"]
    assert value == 2 and type(value) is int
    card.undo()
    assert card.analysis.nodes["svd"]["params"].get("components") != 2
    refused = Future()
    refused.set_result({"no_such_parameter": 1})
    steps = len(card.undo_steps)
    card._finish_job("svd", action, refused)
    assert len(card.undo_steps) == steps and "no_such_parameter" not in card.analysis.nodes["svd"]["params"]
    gone = Future()
    gone.set_result({"components": 2})
    card._finish_job("removed_meanwhile", action, gone)  # no KeyError


def test_closing_a_linked_analysis_recomputes_the_ones_linking_to_it(window, project, qapp):
    """B takes its data from A by a link. A edited but closed without saving: B reads A's file again, so B must
    compute again (it kept showing results of A's discarded edit, and its warning)."""
    import shutil
    from tesselab.analysis import Source
    shutil.copy(project.absolute("data/mix.csv"), project.absolute("data/other.csv"))
    with open(project.absolute("data/other.csv"), "a") as f:
        f.write("401," + ",".join(["0.5"] * 20) + "\n")
    a = window.open_analysis(mix_analysis(project, "a"))
    b = window.open_analysis(mix_analysis(project, "b"))
    settle(qapp)
    b.set_input("svd", "data", Source("table", "table", window.engine.link_path(b.analysis, a.analysis.path)))
    window.save_card(b)
    settle(qapp)
    changed = lambda: any(d.get("code") == "link-changed" for d in window.runner.results(b.analysis)["svd"].diagnostics)
    assert not changed()
    a.set_param("table", "file", "data/other.csv")
    settle(qapp)
    assert changed()
    window.close_card(a)  # "Discard" (the fixture's answer)
    settle(qapp)
    assert not changed()


def test_new_analysis_from_template(window, project, qapp):
    window.new_from_data(os.path.join(project.root, "data"), ["org.tesselab.table/table"])
    settle(qapp)
    card = window.active_card
    assert card.analysis.label == "data"
    assert card.is_unsaved()  # new: unsaved until saved
    assert window.runner.results(card.analysis)["table"].ok


def test_worker_processes(window, project, qapp):
    """The real way: nodes in worker processes, results back through the Qt signal."""
    window.runner.inline = False
    card = window.open_analysis(mix_analysis(project))
    end = time.time() + 60
    while time.time() < end:
        qapp.processEvents()
        result = window.runner.results(card.analysis).get("svd")
        if result is not None and result.ok and not window.runner.pending:
            break
        time.sleep(0.02)
    assert window.runner.results(card.analysis)["svd"].ok
    from tesselab import worker
    worker.shutdown()


def test_same_analysis_in_two_tabs(window, project, qapp):
    """A second card of an analysis in another tab: the same analysis, one undo history, folds of its own."""
    path = mix_analysis(project)
    first = window.open_analysis(path)
    assert window.open_analysis(path) is first  # in the same tab: shown, not opened twice
    tab = window.new_tab()
    second = window.open_analysis(path, tab)
    settle(qapp)
    assert second is not first and second.analysis is first.analysis
    assert second.shared is first.shared
    first.set_param("svd", "components", 2)
    settle(qapp)
    assert second.is_unsaved()
    assert second.groups["svd_view"].tiles["values"].view.spec is not None
    assert window.runner.results(second.analysis)["svd"].outputs["svd"].sizes["component"] == 2
    second.undo()  # undoes the edit made in the first card
    assert first.analysis.nodes["svd"]["params"].get("components", 3) == 3 and not first.is_unsaved()
    second.set_open("table", True)
    assert "table" in second.open_groups and "table" not in first.open_groups  # folds: each card its own
    first.redo()
    assert window.close_card(second)  # the analysis stays open in the first card: no question, nothing lost
    assert window.engine.open.get(os.path.abspath(path)) is first.analysis
    assert first.shared.cards == [first]
    assert window.save_card(first) and not first.is_unsaved()


def test_tab_strip_widths(qapp, project, monkeypatch):
    """The + right after the last tab, and the tab shown when a saved window is opened before the window is shown
    (as at start: the tab bar kept a stale width, a gap before + or no tab at all)."""
    import tesselab.gui.window as window_module
    monkeypatch.setattr(window_module, "message_box", lambda *args, **kwargs: 1)
    path = mix_analysis(project)
    session = os.path.join(project.folder("sessions"), "w.tesselab-session")
    from tesselab.analysis import write_json
    write_json(session, {"tesselab_session": 1, "tabs": [{"name": None, "analyses": [os.path.relpath(path, os.path.dirname(session))],
                                                         "linked": []}], "current": 0})
    for start in (project.root, session):
        w = window_module.Window(start)
        w.runner.inline = True
        w.resize(1000, 500)
        w.show()
        settle(qapp)
        bar = w.tabs.bar
        assert bar.count() == 1
        assert bar.width() == bar.sizeHint().width() > 20
        assert 0 <= w.tabs.plus.geometry().x() - bar.geometry().right() <= 10
        for c in w.cards():
            c.saved_content = c.analysis.state()
        w.close()


def test_tab_drag_and_menus(window, qapp):
    """A dragged tab swaps with its neighbour once the mouse is over the place it will take, never back and forth
    between tabs of other widths (Qt's sliding tabs left their feet behind). The menus are behind the rail's ☰: their
    shortcuts must work while they are closed, so the window owns the actions too."""
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt
    first, second = window.tabs.currentWidget(), window.new_tab("a much longer tab name")
    window.show()
    settle(qapp)
    bar = window.tabs.bar
    left = Qt.MouseButton.LeftButton

    def mouse(kind, x, buttons=left):
        pos = QtCore.QPointF(x, bar.height() / 2)
        QtWidgets.QApplication.sendEvent(bar, QtGui.QMouseEvent(kind, pos, bar.mapToGlobal(pos), left, buttons,
                                                                Qt.KeyboardModifier.NoModifier))
    Type = QtCore.QEvent.Type
    mouse(Type.MouseButtonPress, bar.tabRect(0).center().x())
    mouse(Type.MouseMove, bar.tabRect(1).left() + 5)  # over the long tab, but the short one would not be under it
    assert window.tabs.widget(0) is first
    mouse(Type.MouseMove, bar.tabRect(1).right() - 5)
    assert window.tabs.widget(1) is first and window.tabs.currentWidget() is first and bar.tabText(0) == second.name
    mouse(Type.MouseButtonRelease, bar.tabRect(1).center().x(), Qt.MouseButton.NoButton)
    assert window.explorer_action in window.actions() and window.explorer_action in window.all_actions()
    window.explorer_button.click()
    assert not window.explorer.isVisibleTo(window) and not window.explorer_action.isChecked()


def test_active_card_follows_clicks_and_tabs(window, project, qapp):
    """Ctrl+S saves the active card: the one last clicked (also on a child), in the current tab; it is outlined."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    first = window.open_analysis(mix_analysis(project, "a"))
    second = window.open_analysis(mix_analysis(project, "b"))
    settle(qapp)
    assert window.active_card is second and second.property("active") and not first.property("active")
    QTest.mouseClick(first.label, Qt.MouseButton.LeftButton)  # the label takes the click, not the card
    assert window.active_card is first and first.property("active") and not second.property("active")
    third = window.open_analysis(mix_analysis(project, "c"), window.new_tab())
    assert window.active_card is third
    window.tabs.setCurrentIndex(0)
    assert window.active_card is first  # the tab's own active card, not the other tab's


def test_undo_redo_buttons_and_keys(window, project, qapp):
    """Each card's header has undo and redo buttons for its own analysis (not the rail, which acted on the active
    card), usable only when the analysis has a step to take back or do again, also right after an undo (the redo
    step is there before anything is told). Redo is Ctrl+Shift+Z and Ctrl+Y, in the main window and in the graph
    window; the graph window has the buttons too."""
    from PySide6 import QtGui, QtWidgets
    from tesselab.gui.graph_view import GraphDialog
    assert not {"undo", "redo"} & {b.property("icon_name") for b in window.rail.findChildren(QtWidgets.QToolButton)}
    first = window.open_analysis(mix_analysis(project, "a"))
    second = window.open_analysis(mix_analysis(project, "b"))
    settle(qapp)
    assert not second.undo_button.isEnabled() and not second.redo_button.isEnabled()
    second.set_param("svd", "components", 2)
    assert second.undo_button.isEnabled() and not second.redo_button.isEnabled()
    assert not first.undo_button.isEnabled()  # nothing to undo there
    window.active_card = first
    second.undo_button.click()  # its own analysis, whichever card is active
    assert second.analysis.nodes["svd"].get("params", {}).get("components") != 2
    assert not second.undo_button.isEnabled() and second.redo_button.isEnabled()
    second.redo_button.click()
    assert second.analysis.nodes["svd"]["params"]["components"] == 2
    redo = next(a for a in window.all_actions() if a.text() == "Redo")
    assert {k.toString() for k in redo.shortcuts()} == {"Ctrl+Shift+Z", "Ctrl+Y"}
    GraphDialog.show_for(second)
    dialog = GraphDialog._open[id(second)]
    keys = {s.key().toString() for s in dialog.findChildren(QtGui.QShortcut) if s.parent() is dialog}
    assert {"Ctrl+Z", "Ctrl+Shift+Z", "Ctrl+Y"} <= keys
    assert dialog.undo_button.isEnabled() and not dialog.redo_button.isEnabled()
    dialog.undo_button.click()
    assert not dialog.undo_button.isEnabled() and dialog.redo_button.isEnabled()
    assert second.redo_button.isEnabled()  # the card follows an undo made in the graph window
    dialog.close()



def test_a_theme_switch_recolors_every_icon(window, project, qapp):
    """Icons are pixmaps in the colors of the theme they were drawn in: after a switch, the buttons of the card
    header, the rail and the graph window are drawn anew (the card headers kept the light theme's dark grey on the
    dark theme's page, 2026-11-23), and the group chevrons too."""
    from PySide6 import QtCore, QtGui, QtWidgets
    from tesselab.gui import theme
    from tesselab.gui.graph_view import GraphDialog
    size = QtCore.QSize(36, 36)

    def image(icon):
        return icon.pixmap(size).toImage()
    card = window.open_analysis(mix_analysis(project, "a"))
    GraphDialog.show_for(card)
    settle(qapp)
    dialog = GraphDialog._open[id(card)]
    light_close = image(theme.icon("close"))
    try:
        window.set_theme("dark")
        settle(qapp)
        assert image(theme.icon("close")) != light_close  # the test means something
        buttons = [b for widget in (card.header, window.rail, dialog)
                   for b in widget.findChildren(QtWidgets.QToolButton)
                   if b.property("icon_name") and not b.isCheckable()]
        names = {b.property("icon_name") for b in buttons}
        assert {"undo", "redo", "graph", "save", "export", "close", "folder", "save_all"} <= names
        for button in buttons:
            name = button.property("icon_name")
            assert image(button.icon()) == image(theme.icon(name)), name
        # unavailable: a faint icon of its own (Qt's greyed one looked like the usable one on the dark page)
        assert not card.undo_button.isEnabled()
        assert card.undo_button.icon().availableSizes(QtGui.QIcon.Mode.Disabled)
        group = next(iter(card.groups.values()))
        chevron = "chevron_down" if group.name in card.open_groups else "chevron_right"
        assert image(group.chevron.icon()) == image(theme.icon(chevron))
    finally:
        dialog.close()
        window.set_theme("light")

def test_undo_keys_in_a_field(window, project, qapp):
    """Ctrl+Z right after typing a value and Enter undoes it in the analysis, the focus still in the field (the
    field's own history is empty then; Qt's line edit used to take the key and do nothing). Typing not finished
    yet is undone in the field first."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    card = window.open_analysis(mix_analysis(project))
    settle(qapp)
    window.activateWindow()
    group = card.groups["svd"]
    group._settings_toggled(True)
    settle(qapp)
    edit = group.form.fields["components"].editor
    original = edit.text()
    ctrl, shift = Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.ShiftModifier
    edit.setFocus()
    edit.selectAll()
    QTest.keyClicks(edit, "2")
    QTest.keyClick(edit, Qt.Key.Key_Return)
    settle(qapp)
    assert card.analysis.nodes["svd"]["params"]["components"] == 2 and edit.hasFocus()
    QTest.keyClick(edit, Qt.Key.Key_Z, ctrl)
    settle(qapp)
    assert card.analysis.nodes["svd"].get("params", {}).get("components") != 2 and edit.text() == original
    QTest.keyClick(edit, Qt.Key.Key_Z, ctrl | shift)
    settle(qapp)
    assert card.analysis.nodes["svd"]["params"]["components"] == 2 and edit.text() == "2"
    steps = len(card.undo_steps)
    QTest.keyClicks(edit, "5")  # typed, no Enter: Ctrl+Z takes back the typing, not the analysis's last step
    QTest.keyClick(edit, Qt.Key.Key_Z, ctrl)
    assert edit.text() == "2" and len(card.undo_steps) == steps


def test_redraw_keeps_unchanged_marks(qapp):
    """A redraw where only a marker moved keeps the items of the other marks (dragging λ redrew every spectrum)."""
    from tesselab.gui.plots import PlotView
    from tesselab.plotspec import Lines, PlotSpec, VLine
    view = PlotView()
    view.set_spec(PlotSpec(marks=[Lines(np.arange(5), np.ones((3, 5))), VLine(1.0)]))
    panel = view.panels["main"]
    spectra = panel.drawn[0].items
    view.set_spec(PlotSpec(marks=[Lines(np.arange(5), np.ones((3, 5))), VLine(2.0)]))
    assert panel.drawn[0].items is spectra and len(spectra) == 3
    assert panel.drawn[1].mark.x == 2.0
    view.set_spec(PlotSpec(marks=[Lines(np.arange(5), np.zeros((3, 5)))]))  # changed data, marker gone
    assert panel.drawn[0].items is spectra and len(panel.drawn) == 1  # the same curves, with the new data
    assert not spectra[0][0].yData.any()
    assert sum(1 for item in panel.box.addedItems if isinstance(item, type(spectra[0][0]))) == 3
    view.set_spec(PlotSpec(marks=[Lines(np.arange(5), np.zeros((4, 5)))]))  # another number of lines: drawn anew
    assert panel.drawn[0].items is not spectra
    assert sum(1 for item in panel.box.addedItems if isinstance(item, type(spectra[0][0]))) == 4


def test_heatmaps_with_missing_values_are_drawn(qapp):
    """Maps with NaN cells (a TA map shifted out of its data, a crop): a masked array crashed the process inside
    pyqtgraph (segfault); all-NaN, partly NaN, a single cell and infinities must draw."""
    from tesselab.gui.plots import render_image
    from tesselab.plotspec import Axis, Heatmap, PlotSpec
    x, y = np.linspace(-3e-12, 2e-9, 31), np.linspace(450e-9, 700e-9, 21)
    z = np.random.default_rng(0).normal(size=(31, 21))
    half = z.copy()
    half[:15] = np.nan
    one = np.full_like(z, np.nan)
    one[3, 4] = 1.0
    for values in (half, one, np.full_like(z, np.nan), np.full_like(z, np.inf)):
        spec = PlotSpec(x=Axis("Delay", unit="s", display="ps", scale="linlog", linthresh=1e-12),
                        y=Axis("Wavelength", unit="m", display="nm"), marks=[Heatmap(x, y, values, colorbar="ΔA")])
        assert render_image(spec, 320, 200).width() == 320


def test_a_rebuilt_plot_leaves_nothing_behind_in_its_scene(qapp):
    """The view box narrows (here: room on the right, as beside a partner's color bar) while the left axis stays,
    whose bounding rect (with the grid) includes the view box; then the plot is built anew. With Qt's BSP index the
    axis stayed in the index under its old rect, the scene still found it, and once it was deleted painting read
    freed memory: a segfault in about one full test run in four (2026-11-23)."""
    from PySide6 import QtCore
    from tesselab.gui.plots import PlotView
    from tesselab.plotspec import Axis, Line, PlotSpec
    view = PlotView()
    view.resize(900, 320)
    view.show()
    view.set_spec(PlotSpec(x=Axis("x"), y=Axis("y"), marks=[Line(np.arange(5.0), np.arange(5.0))]))
    settle(qapp, 0.1)
    view.repaint()
    view.panel().pad_right(600)
    settle(qapp, 0.1)
    view.repaint()
    old = view.panel().item  # kept alive: a stale entry is then found instead of crashing
    view.set_spec(PlotSpec(x=Axis("x", scale="log"), y=Axis("y"), note="No result"))  # another layout: built anew
    scene = view.scene()
    assert [item for item in scene.items(QtCore.QRectF(-1e4, -1e4, 2e4, 2e4)) if item.scene() is not scene] == []
    assert old.scene() is None


def test_new_data_go_into_the_items_of_a_mark_with_the_same_look(qapp):
    """A moved marker or new points are put into the items the mark made (dragging λ made hundreds of items per step);
    the picture is the one a plot drawn anew gives."""
    from PySide6 import QtGui
    from tesselab.gui.plots import PlotView
    from tesselab.plotspec import Axis, HLine, Line, Lines, PlotSpec, Points, Style, VLine
    x = np.linspace(0, 10, 40)

    def spec(shift):
        return PlotSpec(x=Axis("x"), y=Axis("y"), marks=[
            Line(x, np.sin(x + shift), name="a", error_y=0.1 + 0 * x), Line(x, np.cos(x + shift), style=Style(dash="dash")),
            Points(x[::4], np.sin(x[::4] + shift), error_y=0.2 + 0 * x[::4], labels=[str(i) for i in range(10)],
                   style=Style(symbol="x", size=9)),
            Lines(x, np.vstack([np.sin(x + shift + k) for k in range(4)]), labels=list("abcd"), colormap="black-red"),
            VLine(3.0 + shift), HLine(0.5 + shift)])

    def image(view):
        view.resize(500, 300)
        view.show()
        settle(qapp, 0.15)
        return view.grab().toImage()

    changed = PlotView()
    changed.set_spec(spec(0.0))
    items = [item for entry in changed.panel().drawn for item, _ in entry.items]
    changed.set_spec(spec(0.7))
    assert [item for entry in changed.panel().drawn for item, _ in entry.items] == items  # none made anew
    fresh = PlotView()
    fresh.set_spec(spec(0.7))
    a, b = image(changed), image(fresh)
    assert a.size() == b.size() and a == b
    # a different look (here a color) is drawn anew
    other = PlotSpec(x=Axis("x"), y=Axis("y"), marks=[Line(x, x, style=Style(color="muted"))])
    changed.set_spec(other)
    assert [item for entry in changed.panel().drawn for item, _ in entry.items] != items[:1]
    changed.close()
    fresh.close()


def test_a_dragged_cursor_waits_for_a_new_layout(qapp, monkeypatch):
    """The note "Computing …" (no axes: another layout) arrived while the λ line was being dragged. Building the plot
    anew then would read the mouse in the wrong coordinates (and, as long as the line was a pyqtgraph item, deleted it
    under the mouse: "Error sending hover exit event", RuntimeError: ... already deleted, 2026-11-24). Now the plot
    stays as it is until the mouse is released, and the held-back description is drawn then."""
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from tesselab.gui.plots import CursorItem, PlotView
    from tesselab.plotspec import Axis, Line, PlotSpec
    changes = []
    for widget in QtWidgets.QApplication.topLevelWidgets():  # plots other tests left shown take the mouse offscreen
        if isinstance(widget, PlotView):
            widget.close()
    view = PlotView()
    view.resize(600, 320)
    view.show()
    spec = PlotSpec(x=Axis("x"), y=Axis("y"), marks=[Line(np.arange(11.0), np.arange(11.0))])
    view.set_spec(spec)
    cursor = CursorItem(view, None, "x", 5.0, lambda value, final: changes.append((value, final)), label="x = {value:.1f}")
    settle(qapp, 0.3)
    box = view.panel().box
    start = view.mapFromScene(box.mapViewToScene(QtCore.QPointF(5.0, 5.0)))
    viewport, left = view.viewport(), Qt.MouseButton.LeftButton
    QTest.mouseMove(viewport, start)
    settle(qapp, 0.05)
    assert cursor.hover  # (the line is under the mouse)
    QTest.mousePress(viewport, left, pos=start)
    for k in range(1, 6):
        QTest.mouseMove(viewport, QtCore.QPoint(start.x() + 6 * k, start.y()))
        settle(qapp, 0.02)
    assert cursor.dragging() and view.dragging()
    view.set_spec(PlotSpec(x=Axis("x", scale="log"), y=Axis("y"), note="Computing …"))  # another layout, mid-drag
    assert view.spec is spec and view._postponed is not None
    for k in range(6, 11):
        QTest.mouseMove(viewport, QtCore.QPoint(start.x() + 6 * k, start.y()))
        settle(qapp, 0.02)
    assert cursor.value > 5.0 and changes and not changes[-1][1]
    moved = cursor.value
    QTest.mouseRelease(viewport, left, pos=QtCore.QPoint(start.x() + 60, start.y()))
    settle(qapp, 0.1)
    assert changes[-1][1] is True and view._postponed is None and not cursor.dragging()
    assert view.panel().panel.note == "Computing …"  # drawn now
    assert cursor.value == pytest.approx(moved)


def test_a_cursor_is_painted_over_the_picture_without_drawing_the_plot_again(qapp):
    """Moving the λ line only paints the line: the picture of the plot is kept (the spectra took 10 ms to draw)."""
    from PySide6 import QtGui
    from tesselab.gui.plots import CursorItem, PlotView
    from tesselab.plotspec import Axis, Line, PlotSpec
    view = PlotView()
    view.resize(600, 320)
    view.show()
    view.set_spec(PlotSpec(x=Axis("x"), y=Axis("y"), marks=[Line(np.arange(11.0), np.arange(11.0))]))
    cursor = CursorItem(view, None, "x", 3.0, lambda value, final: None, label="x = {value:.1f}")
    settle(qapp, 0.3)
    for _ in range(10):  # (until the layout has settled and the picture stays the same)
        picture = view._picture
        view.grab()
        settle(qapp, 0.1)
        if view._picture is picture:
            break
    before = view.grab().toImage()
    picture = view._picture
    cursor.set_value(8.0)
    settle(qapp, 0.1)
    after = view.grab().toImage()
    assert view._picture is picture and before != after  # the same picture, another overlay
    red = QtGui.QColor(cursor.color).rgb()
    columns = lambda image: {x for x in range(image.width()) for y in (100,) if image.pixelColor(x, y).red() > 150
                             and image.pixelColor(x, y).green() < 80}
    assert columns(before) and columns(after) and min(columns(after)) > max(columns(before))  # the line moved right
    view.close()


def two_from_template(window, project):
    """Two analyses made from the table template in the window: they share one graph file."""
    import shutil
    shutil.copytree(project.absolute("data"), project.absolute("data2"))
    window.new_from_data(project.absolute("data"), ["org.tesselab.table/table"])
    a = window.active_card
    window.new_from_data(project.absolute("data2"), ["org.tesselab.table/table"])
    return a, window.active_card


def test_shared_graph_in_the_window(window, project, qapp):
    """A node added in one analysis appears in the other; values set in one are its own until made the graph's;
    undo in one never takes back the other's graph edits; saving writes the sample and the graph file."""
    a, b = two_from_template(window, project)
    settle(qapp)
    assert a.analysis.graph is b.analysis.graph and a.analysis.is_own("table", "file")
    name = a.add_node("org.tesselab.chemometrics/svd", pos=(14, 0), scroll=False)
    settle(qapp)
    assert name in b.analysis.nodes and name in b.groups
    assert b.analysis.nodes["table"]["params"]["file"] == "data2/mix.csv"
    assert window.runner.results(b.analysis)[name].ok
    a.set_param(name, "components", 2)
    assert a.analysis.is_own(name, "components") and b.analysis.nodes[name]["params"].get("components") != 2
    a.promote_param(name, "components")
    settle(qapp)
    assert b.analysis.nodes[name]["params"]["components"] == 2 and not a.analysis.is_own(name, "components")
    b.set_param(name, "components", 4)
    b.undo()  # b's own step: a's node stays
    assert name in a.analysis.nodes and b.analysis.nodes[name]["params"]["components"] == 2
    a.undo()  # the promotion: the graph's value goes back, b follows
    assert b.analysis.nodes[name]["params"].get("components") != 2
    assert window.save_card(a)
    with open(a.analysis.path) as f:
        saved = json.load(f)
    assert "nodes" not in saved and saved["graph"].endswith(".json") and saved["params"]["table"]["file"] == "data/mix.csv"
    with open(a.analysis.graph.path) as f:
        graph = json.load(f)
    assert graph["nodes"][name]["pos"] == [14, 0] and "file" not in graph["nodes"]["table"]["params"]
    assert window.save_card(b) and not b.is_unsaved() and not a.is_unsaved()
    assert Analysis.load(b.analysis.path).content() == b.analysis.content()


def test_graph_editor_moving_adding_variables(window, project, qapp):
    """Boxes move by the mouse and snap to the grid (one undo step, every position stored); nodes are added by name
    (also for a dropped connection), removed and duplicated from the selection; variables are edited in the graph;
    the inspector edits the selected node's inputs."""
    from PySide6 import QtCore
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from tesselab.graph import GRID
    from tesselab.gui.graph_view import GraphDialog, HEADER_H
    card = window.open_analysis(mix_analysis(project, views=False))
    settle(qapp)
    dialog = GraphDialog.show_for(card)
    settle(qapp)
    view, box = dialog.view, dialog.boxes["svd"]
    x, y = round(box.pos().x() / GRID), round(box.pos().y() / GRID)
    start = view.mapFromScene(box.pos() + QtCore.QPointF(60, HEADER_H / 3))  # on its title
    end = view.mapFromScene(box.pos() + QtCore.QPointF(60 + 3 * GRID + 4, HEADER_H / 3 + 2 * GRID - 3))
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view.viewport(), end)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
    settle(qapp)
    assert card.analysis.nodes["svd"]["pos"] == [x + 3, y + 2] and "pos" in card.analysis.nodes["table"]
    card.undo()
    assert "pos" not in card.analysis.nodes["svd"]
    # by name, at a place
    popup = dialog.search(QtCore.QPointF(0, 10 * GRID), QtCore.QPoint(100, 100))
    popup.filter.setText("SVD")
    popup._pick()
    settle(qapp)
    added = [n for n in card.analysis.nodes if n not in ("table", "svd")]
    assert added == ["svd2", "svd_view"]  # SVD comes with its default view, connected to it
    assert card.analysis.nodes["svd_view"]["inputs"]["svd"] == "svd2.svd"
    added = added[:1]
    assert card.analysis.nodes[added[0]]["pos"] == [0, 10]
    assert card.analysis.nodes[added[0]]["inputs"]["data"] == "table.table" and dialog.selected_nodes() == added
    # removed from the selection without asking, one undo step; duplicated with its connection
    steps = len(card.undo_steps)
    dialog.delete_selection()
    assert added[0] not in card.analysis.nodes and len(card.undo_steps) == steps + 1
    card.undo()
    settle(qapp)
    dialog.select("svd")
    dialog.duplicate_selection()
    copies = [n for n in card.analysis.nodes if n not in ("table", "svd", added[0], "svd_view")]
    assert len(copies) == 1 and card.analysis.nodes[copies[0]]["inputs"]["data"] == "table.table"
    # a connection dropped on empty space: a node that takes it
    assert dialog.begin_drag(dialog.boxes["table"].port("out", "table"))
    dialog.drop(QtCore.QPointF(-40 * GRID, 30 * GRID), QtCore.QPoint(100, 100))
    labels = [dialog._search.list.item(i).text() for i in range(dialog._search.list.count())]
    assert any(t.startswith("SVD › data") for t in labels) and any(t.startswith("MCR-ALS › data") for t in labels)
    dialog._search.close()
    # variables: made from a parameter, edited, renamed, removed in the graph
    card.bind("svd", "components", "k")
    settle(qapp)
    assert "k" in dialog.var_table.rows and "svd" in dialog.accepted_params("k").keys() | {"svd"}
    assert dialog.var_text("k") == "3"
    dialog.select("k", "var")  # a row of the table: the inspector shows it, its parameters are highlighted
    assert dialog.selection() == [("var", "k")] and dialog.inspector.page == (("var", "k"),)
    dialog.begin_var_drag("k", QtCore.QPointF(0, 0))  # from the table: the nodes with a parameter of its kind
    assert dialog.drag.kind == "var"
    dialog.cancel_drag()
    dialog.edit_var_value("k")
    dialog._inline.setText("2")
    dialog._inline.returnPressed.emit()
    settle(qapp)
    assert card.analysis.vars["k"] == 2
    assert window.runner.results(card.analysis)["svd"].outputs["svd"].sizes["component"] == 2
    dialog.rename_var("k")
    dialog._inline.setText("comps")
    dialog._inline.returnPressed.emit()
    assert card.analysis.nodes["svd"]["params"]["components"] == {"var": "comps"}
    dialog.remove_var("comps")
    assert card.analysis.nodes["svd"]["params"]["components"] == 2 and not card.analysis.vars
    # the inspector shows the selected node; its input choice disconnects
    settle(qapp)
    dialog.select("svd")
    assert dialog.inspector.page == (("node", "svd"),) and dialog.inspector.form is not None
    combo = dialog.inspector.inputs["data"][0]
    combo.setCurrentIndex(combo.findData(""))
    dialog.inspector._input_chosen("svd", "data", combo)
    assert card.analysis.nodes["svd"]["inputs"]["data"] is None
    # a needed input left open is drawn red; the table's file row names what the table node reads
    settle(qapp)
    from tesselab.gui import theme
    circle = dialog.boxes["svd"].input_items["data"][0]
    assert circle.pen().color().name() == theme.current()["roles"]["error"]
    assert [label.text() for _, label in dialog.boxes["table"].file_items] == ["mix.csv"]


def test_param_form_units_and_columns(qapp):
    """Units in brackets after the label; short fields side by side where there is room, one column at the form's
    minimum width (a narrow inspector), a path in a row of its own; the drop-down list is not see-through."""
    from PySide6 import QtWidgets
    from tesselab import params as P
    from tesselab.gui.forms import ParamForm
    params = [P.Float("thickness", 0.01, unit="m", display="mm"), P.Float("volume", 1e-6, unit="m³"),
              P.Int("fit_num", 5), P.Choice("method", "last", options={"last": "last", "mean": "mean"}),
              P.Path("file")]
    form = ParamForm(params, {})
    assert form.labels["thickness"].text() == "thickness [mm]"
    assert form.labels["fit_num"].text() == "fit num"
    box = QtWidgets.QWidget()  # in a layout, as in a card or the inspector (a window keeps its layout's minimum)
    QtWidgets.QVBoxLayout(box).addWidget(form)
    box.resize(1200, 300)
    box.show()
    settle(qapp)
    assert form.columns == 4
    assert form.fields["fit_num"].width() < 150 and form.fields["file"].width() > 900
    box.resize(box.minimumSizeHint().width(), 300)
    settle(qapp)
    assert form.columns == 1
    assert all(label.width() >= label.sizeHint().width() for label in form.labels.values())  # nothing cut off
    combo = form.fields["method"].editor
    combo.showPopup()
    settle(qapp)
    container = combo.view().window()
    assert container.metaObject().className() == "QComboBoxPrivateContainer"
    assert "{" in container.styleSheet()  # a selector: the list inside keeps its background
    assert isinstance(combo.itemDelegate(), QtWidgets.QStyledItemDelegate)  # its items as the style sheet says
    combo.hidePopup()
    box.close()


def test_port_dimensions_and_box_widths(qapp):
    """A table's sizes after a port's name; the lengths of each table for a dataset of unrelated ones (the rig's
    records); boxes grow so that names and sizes are never cut."""
    import xarray as xr
    from tesselab.gui.graph_view import MAX_W, NODE_W, box_width, dims_text, port_dims, shape_text
    table = xr.Dataset({"absorbance": (("spectrum", "wavelength"), np.zeros((51, 160))),
                        "total": ("spectrum", np.zeros(51))})
    assert dims_text(*port_dims(dataset=table)) == "51 × 160"
    records = xr.Dataset({"spectrum_time": ("record", np.zeros(153)), "stirring_start": ("stirring", np.zeros(51))})
    assert dims_text(*port_dims(dataset=records)) == "" and shape_text(records) == "153 | 51"  # two tables
    narrow = box_width("SVD", ["data"], ["svd"], {}, [], [])
    wide = box_width("Spectrum corrections", ["measured_spectra"], ["uncombined_spectra"],
                     {("in", "measured_spectra"): "154 × 201", ("out", "uncombined_spectra"): "153 × 160"}, [], [])
    assert narrow == NODE_W and NODE_W < wide <= MAX_W


def test_graph_window_panel_and_readable_zoom(window, project, qapp):
    """Ctrl+B folds the side panel (and it stays folded the next time); the window opens at a readable zoom."""
    from tesselab.gui.graph_view import READABLE, GraphDialog
    card = window.open_analysis(mix_analysis(project))
    settle(qapp)
    dialog = GraphDialog.show_for(card)
    settle(qapp)
    assert dialog.view.transform().m11() >= READABLE - 1e-9
    assert dialog.inspector.isVisibleTo(dialog)
    dialog.panel_button.toggle()  # as Ctrl+B
    assert not dialog.inspector.isVisibleTo(dialog)
    dialog.close()
    GraphDialog._open.clear()
    again = GraphDialog.show_for(card)
    assert not again.inspector.isVisibleTo(again)
    again.panel_button.toggle()
    again.close()


# --- following changes on disk (gui/watch.py) ----------------------------------------------------------------------
def wait_for(qapp, condition, seconds=8):
    """Process events until the condition holds (the watcher waits watch.DELAY ms after the last change)."""
    end = time.time() + seconds
    while time.time() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


def rewrite(path, text):
    """Write a file in place (as many editors and an LLM's edit do)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def test_follow_module_changes(window, project, qapp):
    """The window follows a module's code on disk: an edit runs that module's nodes again (a node of another module
    is found in memory), a syntax error shows on its nodes without stopping the window, and the fixed code runs."""
    import shutil
    from tesselab.modules import user_module_folder
    from tesselab.scaffold import new_module
    folder = new_module(os.path.join(user_module_folder(), "curve"))
    window.reload_modules()  # as Modules → New module… does
    shutil.copy(os.path.join(folder, "sample", "example", "curve.csv"), os.path.join(project.root, "data"))
    analysis = Analysis(path=project.new_analysis_path("curve"))
    analysis.add("my.curve/read", "read", params={"file": "data/curve.csv"})
    analysis.add("my.curve/smooth", "smooth", inputs={"curve": "read.curve"})
    analysis.add("org.tesselab.table/csv", "table", params={"file": "data/curve.csv"})
    analysis.save()
    card = window.open_analysis(analysis.path)
    window.follow.update()
    results = lambda: window.runner.results(card.analysis)  # noqa: E731
    assert results()["smooth"].ok and results()["table"].ok
    before, table = results()["smooth"].outputs["smoothed_curve"].y.values, results()["table"]
    path = os.path.join(folder, "analysis.py")
    original = open(path, encoding="utf-8").read()
    rewrite(path, original.replace('return np.convolve(y, kernel, mode="same")',
                                   'return 2 * np.convolve(y, kernel, mode="same")'))
    doubled = lambda: results()["smooth"].ok and np.allclose(  # noqa: E731
        results()["smooth"].outputs["smoothed_curve"].y.values, 2 * before)
    assert wait_for(qapp, doubled)
    assert results()["table"] is table  # another module: found in memory, not run again
    assert "reloaded" in window.statusBar().currentMessage()
    nodes = os.path.join(folder, "nodes.py")
    good = open(nodes, encoding="utf-8").read()
    rewrite(nodes, good + "\ndef broken(:\n")
    assert wait_for(qapp, lambda: results()["smooth"].status == "error")
    assert "syntax" in results()["smooth"].error.lower()
    assert results()["table"].ok and card.groups  # the window goes on
    rewrite(nodes, good)
    assert wait_for(qapp, doubled)


def test_follow_module_changes_in_worker_processes(window, project, qapp):
    """The real way: the workers keep running when a module changes and import its new code (worker.node_class)."""
    from tesselab import worker
    from tesselab.modules import user_module_folder
    from tesselab.scaffold import new_module
    window.runner.inline = False
    folder = new_module(os.path.join(user_module_folder(), "curve"))
    window.reload_modules()
    analysis = Analysis(path=project.new_analysis_path("curve"))
    analysis.add("my.curve/read", "read", params={"file": os.path.join(folder, "sample", "example", "curve.csv")})
    analysis.add("my.curve/smooth", "smooth", inputs={"curve": "read.curve"})
    analysis.save()
    card = window.open_analysis(analysis.path)
    window.follow.update()
    smooth = lambda: window.runner.results(card.analysis).get("smooth")  # noqa: E731
    assert wait_for(qapp, lambda: smooth() is not None and smooth().ok, seconds=60)
    before, pool = smooth().outputs["smoothed_curve"].y.values, worker.pool()
    path = os.path.join(folder, "analysis.py")
    original = open(path, encoding="utf-8").read()
    rewrite(path, original.replace('return np.convolve(y, kernel, mode="same")',
                                   'return 2 * np.convolve(y, kernel, mode="same")'))
    assert wait_for(qapp, lambda: smooth() is not None and smooth().ok and np.allclose(
        smooth().outputs["smoothed_curve"].y.values, 2 * before), seconds=60)
    assert worker.pool() is pool  # not shut down
    worker.shutdown()


def test_follow_data_changes(window, project, qapp):
    """A file an open analysis reads changes: the analysis is computed again, also after a save that replaced the
    file (written anew and renamed over the old one); nothing happens while following is off."""
    card = window.open_analysis(mix_analysis(project, views=False))
    window.follow.update()
    results = lambda: window.runner.results(card.analysis)  # noqa: E731
    rows = lambda: results()["table"].outputs["table"].sizes["x"] if results()["table"].ok else None  # noqa: E731
    assert rows() == 101
    svd = results()["svd"]
    path = os.path.join(project.root, "data", "mix.csv")
    original = open(path, encoding="utf-8").read()
    with open(path, "a", encoding="utf-8") as f:
        f.write("401.0," + ",".join("0.0" for _ in range(20)) + "\n")
    assert wait_for(qapp, lambda: rows() == 102)
    assert results()["svd"] is not svd and results()["svd"].ok
    assert "mix.csv" in window.statusBar().currentMessage()
    rewrite(path + ".new", original)
    os.replace(path + ".new", path)
    assert wait_for(qapp, lambda: rows() == 101)
    with open(path, "a", encoding="utf-8") as f:  # the replaced file is watched again
        f.write("402.0," + ",".join("0.0" for _ in range(20)) + "\n")
    assert wait_for(qapp, lambda: rows() == 102)
    window.follow.set_on(False)
    rewrite(path, original)
    assert not wait_for(qapp, lambda: rows() == 101, seconds=1.5)
    window.follow.set_on(True)


FILES_MODULE = '''\
[module]
id = "test.files"
name = "Files"
version = "0.1.0"
api = 1

[[nodes]]
id = "count"
class = "nodes.py:CountFiles"
'''

FILES_NODES = '''\
import os

import xarray as xr

import tesselab
from tesselab import P


class CountFiles(tesselab.Reader):
    name = "Count files"
    params = [P.Path("folder", folder=True)]
    outputs = ["files"]

    def run(self, inputs, params):
        return {"files": xr.Dataset({"count": ((), len(os.listdir(params.folder)))})}
'''


def test_follow_a_folder(window, project, qapp):
    """A folder parameter (the rig's power folder, say): a file added to it, and a file in it written in place, run
    the reader again."""
    from tesselab.modules import user_module_folder
    module = os.path.join(user_module_folder(), "files")
    os.makedirs(module)
    rewrite(os.path.join(module, "module.toml"), FILES_MODULE)
    rewrite(os.path.join(module, "nodes.py"), FILES_NODES)
    window.reload_modules()
    powers = os.path.join(project.root, "data", "powers")
    os.makedirs(powers)
    rewrite(os.path.join(powers, "power1.csv"), "1,2\n")
    analysis = Analysis(path=project.new_analysis_path("powers"))
    analysis.add("test.files/count", "count", params={"folder": "data/powers"})
    analysis.save()
    card = window.open_analysis(analysis.path)
    window.follow.update()
    result = lambda: window.runner.results(card.analysis)["count"]  # noqa: E731
    assert int(result().outputs["files"]["count"]) == 1
    rewrite(os.path.join(powers, "power2.csv"), "3,4\n")
    assert wait_for(qapp, lambda: result().ok and int(result().outputs["files"]["count"]) == 2)
    first = result()
    with open(os.path.join(powers, "power1.csv"), "a", encoding="utf-8") as f:  # still being written
        f.write("5,6\n")
    assert wait_for(qapp, lambda: result() is not first and result().ok)
