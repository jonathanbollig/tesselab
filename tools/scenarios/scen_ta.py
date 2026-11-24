"""TA module scenarios: the Steps editor (add every kind, toggle, move, remove, odd values in step fields), picks
inside and outside the data, crosshair moves, the global fit and MCR on their buttons with real workers, removing
a node and undo, a missing file, the graph window."""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios_common import *  # noqa

if __name__ == "__main__":
    from PySide6 import QtCore, QtWidgets
    from tesselab.modules import Registry
    from tesselab.project import Project
    from tesselab.templates import new_analysis

    root = sys.argv[1]
    shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(REPO + "/examples/modules/transient_absorption/sample", os.path.join(root, "sample"))
    project = Project.create(root)
    registry = Registry(project.root, extra=[REPO + "/examples/modules"], user=False, entry_points=False)
    analysis = new_analysis(registry, project, "org.tesselab.transient-absorption/global-analysis",
                            os.path.join(project.root, "sample", "ta_run"))
    analysis.save()
    app, window = setup(project.root)
    card = window.open_analysis(analysis.path)
    settle(app, 2)
    print("nodes:", list(card.analysis.nodes))
    start = card.analysis.content()

    def report(where):
        for name, r in window.runner.results(card.analysis).items():
            if r.error or r.status == "error":
                print(f"  {where}: {name}: {(r.error or str(r.diagnostics[:1]))[:200]}")

    report("start")
    # T1: the Steps editor of the corrections node
    group = card.groups["corrections"]
    group.settings_button.setChecked(True)
    settle(app, 0.2)

    def steps_editor():
        g = card.groups["corrections"]
        return g.form.fields["steps"].editor

    kinds = list(steps_editor().param.kinds)
    print("step kinds:", kinds)
    for kind in kinds:
        steps_editor()._add(kind)
        settle(app, 0.1)
    report("after adding every kind")
    editor = steps_editor()
    for row in range(editor.list.count()):
        editor = steps_editor()
        editor.list.setCurrentRow(row)
        settle(app, 0.05)
        form = steps_editor().step_form
        if form is None:
            continue
        for pname, field in list(form.fields.items()):
            for text in ["", "abc", "-1", "0", "1e9", "1-2", "300-200", "5, 3", "nan"]:
                f = steps_editor().step_form.fields.get(pname) if steps_editor().step_form else None
                if f is None or not isinstance(f.editor, QtWidgets.QLineEdit):
                    break
                f.editor.setText(text)
                f._typed()
                settle(app, 0.03)
    report("after odd step values")
    for _ in range(3):
        editor = steps_editor()
        if editor.list.count():
            editor.list.setCurrentRow(editor.list.count() - 1)
            editor._move(-1)
            settle(app, 0.05)
            editor = steps_editor()
            item = editor.list.item(0)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked if item.checkState() == QtCore.Qt.CheckState.Checked
                               else QtCore.Qt.CheckState.Checked)
            settle(app, 0.05)
    while steps_editor().list.count():
        editor = steps_editor()
        editor.list.setCurrentRow(0)
        editor._remove()
        settle(app, 0.05)
    report("after removing every step")
    while card.undo_steps:
        card.undo()
    settle(app, 0.5)
    check(card.analysis.content() == start, "T1 undo to start")
    # T2: picks (Ctrl+click) inside and outside the data, then removed (Ctrl+right-click)
    card.set_open("map", True)  # the corrected map (a view): its picks set the variable Corrections uses
    settle(app, 0.2)
    group = card.groups["map"]
    import numpy as np
    for x, y in [(1e-13, 500e-9), (2e-12, 450e-9), (-1, 1), (1e3, 1e3), (0.0, 0.0), (5e-13, 600e-9)]:
        group._pick("chirp_onsets", x, y, True)
        settle(app, 0.05)
    report("after picks")
    for x, y in [(1e-13, 500e-9), (1e3, 1e3), (7, 7)]:
        card.groups["map"]._pick("chirp_onsets", x, y, False)
        settle(app, 0.05)
    report("after removing picks")
    # T3: the crosshair over and beyond the data
    explorer = card.groups["explorer"]
    card.set_open("explorer", True)
    settle(app, 0.2)
    lines = [item for interaction, param, item in card.groups["explorer"].interactions]
    for value in [-1e-12, 0, 1e-12, 5e-12, 1e-9, 1.0, -1.0]:
        lines[0].on_change(value, False)
        settle(app, 0.02)
    lines[0].on_change(5e-12, True)
    for value in [400e-9, 500e-9, 1e-6, 0.0]:
        lines[1].on_change(value, False)
        settle(app, 0.02)
    lines[1].on_change(500e-9, True)
    report("after crosshair moves")
    # T4: the global fit and an added MCR on their buttons, with real worker processes
    window.runner.inline = False
    window.engine.memory.clear()
    window.runner.request(card.analysis)
    settle(app, 10)
    card.add_node("org.tesselab.chemometrics/mcr")
    settle(app, 1)
    mcr = [n for n in card.analysis.nodes if n.startswith("mcr")][0]
    for name in [n for n in card.analysis.nodes if card.groups[n].run_button is not None]:
        card.run_node(name)
    settle(app, 30)
    for name, r in window.runner.results(card.analysis).items():
        print(f"  T4 {name}: {r.status} {r.error or ''} {r.seconds * 1000:.0f} ms")
    check(not window.runner.is_running(card.analysis), f"T4 still running {window.runner.is_running(card.analysis)}")
    # T5: remove the corrections node (downstream blocked), undo
    card.remove("corrections")
    settle(app, 3)
    blocked = {n: r.status for n, r in window.runner.results(card.analysis).items()}
    print("  T5 after removing corrections:", blocked)
    card.undo()
    settle(app, 5)
    report("after undo of the removal")
    # T6: a missing file, then back
    reader = next(n for n in card.analysis.nodes if not card.engine.node_class(card.analysis, n).inputs)
    path_param = next(p.name for p in card.engine.node_class(card.analysis, reader).params
                      if type(p).__name__ in ("Path", "Paths"))
    old = card.analysis.nodes[reader]["params"][path_param]
    card.set_param(reader, path_param, "sample/does_not_exist")
    settle(app, 3)
    print("  T6 missing file:", {n: (r.status, (r.error or "")[:80]) for n, r in
                                 window.runner.results(card.analysis).items()})
    card.set_param(reader, path_param, old)
    settle(app, 5)
    report("after restoring the file")
    # T7: the graph window on this analysis
    from tesselab.gui.graph_view import GraphDialog
    GraphDialog.show_for(card)
    d = GraphDialog._open[id(card)]
    settle(app, 0.3)
    n = 0
    for name in list(d.boxes):
        for output in list(d.boxes[name].outputs):
            d.begin_drag(d.boxes[name].port("out", output))
            accepted = dict(d.drag.accepted)
            d.cancel_drag()
            for target, inputs in accepted.items():
                for input_name in inputs:
                    steps = len(card.undo_steps)
                    d.begin_drag(d.boxes[name].port("out", output))
                    d.drop(d.boxes[target].port("in", input_name))
                    settle(app, 0.05)
                    if len(card.undo_steps) > steps:
                        card.undo()
                    settle(app, 0.05)
                    n += 1
    print("T7 graph connections tried:", n)
    check(set(d.boxes) == set(card.analysis.nodes), "T7 graph boxes")
    finish(window)
