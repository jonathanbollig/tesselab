"""Graph window scenarios on the UV project: every drag, pick-up, reverse drag, context menu, undo, a second card,
theme switches, closing the card with the graph open."""
import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios_common import *  # noqa

if __name__ == "__main__":
    from PySide6 import QtCore
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from tesselab.analysis import sources_of
    from tesselab.gui.graph_view import GraphDialog

    proj = sys.argv[1]
    app, window = setup(os.path.join(proj, "tesselab", "analyses", "5_UC@2.json"))
    card = list(window.cards())[0]
    settle(app, 1)
    GraphDialog.show_for(card)
    d = GraphDialog._open[id(card)]
    settle(app)

    def edges_match(where):
        settle(app, 0.05)
        expected = {(n, i, s.ref()) for n, e in card.analysis.nodes.items()
                    for i, v in e.get("inputs", {}).items() for s in sources_of(v)}
        drawn = {(w.key[0], w.key[1], w.key[2].ref()) for w in d.wires if w.style != "var"}
        check(expected == drawn, f"{where}: edges differ: {expected ^ drawn}")

    original = card.analysis.content()
    edges_match("start")
    # G1: every output to every input that accepts it (at most PER_INPUT outputs per input: each connection recomputes
    # and redraws the views, and a view takes many inputs), then undo
    PER_INPUT = 3
    tried = {}
    count, started = 0, time.time()
    for name in list(d.boxes):
        for output in list(d.boxes[name].outputs):
            d.begin_drag(d.boxes[name].port("out", output))
            accepted = dict(d.drag.accepted)
            d.cancel_drag()
            for target, inputs in accepted.items():
                for input_name in inputs:
                    if tried.get((target, input_name), 0) >= PER_INPUT:
                        continue
                    tried[(target, input_name)] = tried.get((target, input_name), 0) + 1
                    check(d.begin_drag(d.boxes[name].port("out", output)), f"drag {name}.{output}")
                    d.drag_to(d.boxes[target].port("in", input_name))
                    d.drop(d.boxes[target].port("in", input_name))
                    settle(app, 0.05)
                    value = card.analysis.nodes[target]["inputs"].get(input_name)
                    check(f"{name}.{output}" in [s.ref() for s in sources_of(value)],
                          f"{name}.{output} -> {target}.{input_name}: got {value}")
                    edges_match(f"after {name}.{output} -> {target}.{input_name}")
                    card.undo()
                    settle(app, 0.05)
                    count += 1
    check(card.analysis.content() == original, "G1: undo did not restore")
    print(f"G1 connections tried: {count} in {time.time() - started:.0f} s", flush=True)
    # G2: pick up each connected input, drop on empty space (removed), undo; pick up and move to another input
    for target, entry in list(card.analysis.nodes.items()):
        for input_name, value in list(entry.get("inputs", {}).items()):
            if not value or isinstance(value, list):
                continue
            check(d.begin_drag(d.boxes[target].port("in", input_name)), f"lift {target}.{input_name}")
            check(d.drag.lifted == (target, input_name), f"lifted {target}.{input_name}: {d.drag.lifted}")
            others = [(n, i) for n, ins in d.drag.accepted.items() for i in ins if (n, i) != (target, input_name)]
            d.drop(QtCore.QPointF(-2000, -2000))
            settle(app, 0.05)
            check(card.analysis.nodes[target]["inputs"][input_name] is None, f"G2 removed {target}.{input_name}")
            edges_match("G2 removed")
            card.undo()
            settle(app, 0.05)
            if others:
                n, i = others[0]
                before = card.analysis.nodes[n]["inputs"].get(i)
                d.begin_drag(d.boxes[target].port("in", input_name))
                d.drop(d.boxes[n].port("in", i))
                settle(app, 0.05)
                check(card.analysis.nodes[target]["inputs"][input_name] is None and
                      card.analysis.nodes[n]["inputs"][i] == value, f"G2 moved {target}.{input_name} -> {n}.{i}")
                steps = len(card.undo_steps)
                card.undo()
                settle(app, 0.05)
                check(card.analysis.nodes[target]["inputs"][input_name] == value and
                      card.analysis.nodes[n]["inputs"].get(i) == before, "G2 move undone in one step")
    check(card.analysis.content() == original, "G2: undo did not restore")
    # G3: reverse drags from every input to every fitting output
    count = 0
    for target in list(d.boxes):
        for input_name in list(d.boxes[target].inputs):
            value = card.analysis.nodes[target]["inputs"].get(input_name)
            if value and not isinstance(value, list):
                card.set_input(target, input_name, None)  # unconnected: a reverse drag
                settle(app, 0.05)
            d.begin_drag(d.boxes[target].port("in", input_name))
            if d.drag is None or d.drag.kind != "in":
                check(False, f"G3 no reverse drag from {target}.{input_name}")
                d.cancel_drag()
                continue
            accepted = dict(d.drag.accepted)
            d.cancel_drag()
            pairs = [(source, output) for source, outputs in accepted.items() for output in outputs][:PER_INPUT]
            for source, output in pairs:
                if True:
                    d.begin_drag(d.boxes[target].port("in", input_name))
                    d.drop(d.boxes[source].port("out", output))
                    settle(app, 0.05)
                    check(card.analysis.nodes[target]["inputs"].get(input_name) == f"{source}.{output}",
                          f"G3 {source}.{output} -> {target}.{input_name}")
                    card.undo()
                    settle(app, 0.05)
                    count += 1
            if value and not isinstance(value, list):
                card.undo()
                settle(app, 0.05)
    check(card.analysis.content() == original, "G3: undo did not restore")
    print("G3 reverse connections tried:", count, flush=True)
    print("G4", flush=True)
    texts = lambda m: [a.text() for a in m.actions()] if m is not None else None
    path = [w for w in d.wires if w.style != "var"][0].path()
    m = d.menu_at(path.pointAtPercent(0.5))
    check(texts(m)[0] == "Remove connection", f"edge menu {texts(m)}")
    m.actions()[0].trigger()
    settle(app, 0.05)
    check(card.analysis.content() != original, "Remove connection did nothing")
    card.undo(); settle(app, 0.05)
    box = d.boxes["qy"]
    m = d.menu_at(box.sceneBoundingRect().center() + QtCore.QPointF(0, 5))
    check(texts(m)[:3] == ["Show in the main window", "Rename…", "Duplicate"] and texts(m)[-1] == "Remove",
          f"node menu {texts(m)}")
    m.actions()[0].trigger(); settle(app, 0.05)
    m.actions()[-1].trigger(); settle(app, 0.1)
    check("qy" not in card.analysis.nodes, "Remove node from the menu")
    card.undo(); settle(app, 0.1)
    m = d.menu_at(QtCore.QPointF(-3000, -3000))
    print("add menu:", texts(m))
    check(texts(m)[0] == "Add node" and m.actions()[0].menu().actions(), f"empty-space menu {texts(m)}")
    check(card.analysis.content() == original, "menus: undo did not restore")
    # every node from the add menu, then undo
    full_ids = list(window.engine.registry.nodes())
    for full_id in full_ids:
        print("adding", full_id, flush=True)
        n_before = len(card.analysis.nodes)
        try:
            card.add_node(full_id)
        except Exception as error:  # a node that cannot be loaded
            check(False, f"add {full_id}: {error}")
            continue
        settle(app, 0.1)
        views = len(getattr(window.engine.registry.node_class(full_id), "views", []))  # added with it
        check(len(card.analysis.nodes) == n_before + 1 + views, f"add {full_id}")
        check(set(d.boxes) == set(card.analysis.nodes), f"graph after adding {full_id}")
        card.undo()
        settle(app, 0.05)
    check(card.analysis.content() == original, "G4: undo of adds did not restore")
    # remove each node through the graph's action, then undo
    for name in list(card.analysis.nodes):
        card.remove(name)
        settle(app, 0.1)
        check(name not in d.boxes, f"removed {name} still drawn")
        edges_match(f"after removing {name}")
        card.undo()
        settle(app, 0.1)
    check(card.analysis.content() == original, "G4: undo of removes did not restore")
    # G5: shortcuts in the graph window
    card.set_input("qy", "epsilon", None)
    settle(app, 0.05)
    d.activateWindow(); d.raise_(); settle(app, 0.1)
    from PySide6 import QtWidgets
    print("active window is the graph:", QtWidgets.QApplication.activeWindow() is d)
    QTest.keyClick(d, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    settle(app, 0.05)
    check(card.analysis.content() == original, "G5: Ctrl+Z in the graph window")
    QTest.keyClick(d, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    settle(app, 0.05)
    check(card.analysis.nodes["qy"]["inputs"]["epsilon"] is None, "G5: Ctrl+Shift+Z")
    card.undo()
    settle(app, 0.05)
    # G9: a redraw request during a drag keeps the drag
    d.begin_drag(d.boxes["spectra_file"].port("out", "spectra"))
    card.runner.updated.emit(card.analysis)
    card.saved_changed.emit(card)
    settle(app, 0.1)
    check(d.drag is not None, "G9: a result arriving cancelled the drag")
    QTest.keyClick(d, Qt.Key.Key_Escape)
    check(d.drag is None and d.isVisible(), "G9: Esc")
    # a node removed (by an undo elsewhere) during a drag: the drop does nothing
    card.add_node("org.tesselab.chemometrics/svd")
    settle(app, 0.1)
    added = [n for n in card.analysis.nodes if n not in original["nodes"]][0]
    d.begin_drag(d.boxes["spectra_file"].port("out", "spectra"))
    card.undo()
    settle(app, 0.1)
    d.drop(QtCore.QPointF(0, 0))
    settle(app, 0.1)
    check(card.analysis.content() == original, "drop after the target was removed")
    # G6: a second card of the same analysis (another tab); an edit there redraws this graph
    window.new_tab()
    settle(app, 0.3)
    card2 = window.open_analysis(card.analysis.path)
    settle(app, 0.5)
    check(card2 is not card and card2.analysis is card.analysis, "second card shares the analysis")
    card2.set_input("qy", "epsilon", None)
    settle(app, 0.1)
    edges_match("G6 after an edit in the other card")
    card2.undo()
    settle(app, 0.1)
    edges_match("G6 after undo in the other card")
    # G8: themes with the graph open
    for name in ("dark", "light"):
        window.set_theme(name)
        settle(app, 0.1)
        d.redraw()
    # hover over everything
    for x in range(-300, 2000, 37):
        for y in range(-300, 800, 23):
            d.hover(QtCore.QPointF(x, y))
    # G7: close the first card with its graph open
    window.close_card(card) if hasattr(window, "close_card") else card.close_requested.emit(card)
    settle(app, 0.3)
    check(id(card) not in GraphDialog._open and not d.isVisible(), "G7 graph left over")
    from PySide6 import QtCore as C
    C.QCoreApplication.sendPostedEvents(None, C.QEvent.Type.DeferredDelete)  # the card and its graph deleted
    settle(app, 0.1)
    window.runner.updated.emit(card2.analysis)  # a result for the other card: nothing calls the deleted graph
    settle(app, 0.2)
    finish(window)
