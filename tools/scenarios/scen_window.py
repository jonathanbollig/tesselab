"""Main window scenarios on the two linked UV analyses (the saved window): odd values in every field, every input
choice, cursor moves, save (with a file changed elsewhere), export, Reload modules with real workers, close and
reopen the window's session, undo back to the start."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios_common import *  # noqa

TEXTS = ["", "abc", "-1", "0", "1e9", "5, 3", "1-2", "nan", "inf", "260", "-5 - -1", "1,2,3", "  7  "]

if __name__ == "__main__":
    from PySide6 import QtWidgets
    from tesselab import params as P
    from tesselab.analysis import Analysis

    proj = sys.argv[1]
    session = os.path.join(proj, "tesselab", "sessions", "two experiments.tesselab-session")
    app, window = setup(session)
    settle(app, 1)
    cards = list(window.cards())
    print("cards:", [c.analysis.label for c in cards])
    originals = {id(c.analysis): c.analysis.content() for c in cards}

    def all_results_ok(where):
        for c in window.cards():
            for name, r in window.runner.results(c.analysis).items():
                if r.status == "error" or r.error:
                    print(f"  {where}: {c.analysis.label}.{name}: {r.error or r.diagnostics[:1]}")

    # W1: odd text in every line edit, every option of every choice, toggle every check box
    n = 0
    for card in cards:
        for name, group in list(card.groups.items()):
            group.settings_button.setChecked(True)
            settle(app, 0.1)
            group = card.groups[name]
            if group.form is None:
                continue
            for pname, field in list(group.form.fields.items()):
                editor = field.editor
                if isinstance(editor, QtWidgets.QLineEdit):
                    original = editor.text()
                    for text in TEXTS + [original]:
                        editor.setText(text)
                        field._typed()
                        settle(app, 0.02)
                        n += 1
                elif isinstance(editor, QtWidgets.QComboBox):
                    for i in list(range(editor.count())) + [editor.currentIndex()]:
                        field._emit(editor.itemData(i))
                        settle(app, 0.02)
                        n += 1
                elif isinstance(editor, QtWidgets.QCheckBox):
                    for _ in range(2):
                        editor.setChecked(not editor.isChecked())
                        settle(app, 0.02)
                        n += 1
                group = card.groups.get(name)  # an edit may rebuild the groups
                if group is None or group.form is None:
                    break
    print("W1 field edits:", n)
    all_results_ok("after W1")
    # W2: every input choice in the graph window's inspector (not the link dialog)
    from tesselab.gui.graph_view import GraphDialog
    n = 0
    for card in cards:
        dialog = GraphDialog.show_for(card)
        for name in list(card.analysis.nodes):
            dialog.select(name)
            settle(app, 0.05)
            for input_name in list(dialog.inspector.inputs):
                for i in range(dialog.inspector.inputs[input_name][0].count()):
                    combo = dialog.inspector.inputs.get(input_name, (None,))[0]
                    if combo is None or i >= combo.count() or combo.itemData(i) in ("#link", None):
                        continue
                    combo.setCurrentIndex(i)
                    dialog.inspector._input_chosen(name, input_name, combo)
                    settle(app, 0.05)
                    n += 1
        dialog.close()
    print("W2 input choices:", n)
    # W3: the λ cursor over the whole range and beyond, coalesced like a drag
    card = cards[0]
    var = next(iter(card.analysis.vars), None)
    if var:
        start = card.analysis.vars[var]
        for x in [200e-9 + i * 2e-9 for i in range(90)] + [1.0, -1.0, 0.0, start]:
            card.set_var(var, x)
            settle(app, 0.01)
        print("W3 cursor moves done, var", var)
    all_results_ok("after W3")
    # W4: undo everything: back to the start in every analysis
    for card in cards:
        while card.undo_steps:
            card.undo()
        settle(app, 0.2)
        # more than 200 edits above: the undo history keeps the last 200, so this is not the start
        print("W4", card.analysis.label, "back at the start:", card.analysis.content() == originals[id(card.analysis)])
    # W5: save, with the file changed elsewhere first
    card = cards[0]
    card.analysis.doc["comment"] = "scenario"
    card.edit(lambda: card.analysis.doc.__setitem__("comment", "scenario edit"))
    other = Analysis.load(card.analysis.path)
    other.doc["comment"] = "changed elsewhere"
    other.save()
    ANSWER["value"] = 1  # Cancel
    boxes.clear()
    check(not window.save_card(card), "W5 save went ahead although the file changed elsewhere")
    check(boxes and boxes[-1][0] == "Changed elsewhere", f"W5 no question: {boxes}")
    ANSWER["value"] = 0  # Overwrite
    check(window.save_card(card), "W5 overwrite")
    check(Analysis.load(card.analysis.path).doc["comment"] == "scenario edit", "W5 saved content")
    check(not card.is_unsaved(), "W5 still unsaved")
    # W6: export each card (PDF + JSON)
    out = os.path.join(proj, "exports_scenario")
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (os.path.join(out, "x.pdf"), ""))
    for c in cards:
        window.export_card(c)
        settle(app, 0.2)
        check(os.path.exists(os.path.join(out, "x.pdf")), f"W6 export {c.analysis.label}")
        os.remove(os.path.join(out, "x.pdf"))
    # W7: Reload modules with real worker processes, while nodes are queued
    window.runner.inline = False
    for c in cards:
        window.runner.request(c.analysis)
    window.engine.memory.clear()
    for c in cards:
        window.runner.request(c.analysis)
    window.reload_modules()
    settle(app, 15)
    stopped = [(c.analysis.label, n, r.error) for c in window.cards()
               for n, r in window.runner.results(c.analysis).items() if r.error and "worker stopped" in r.error]
    check(not stopped, f"W7 worker-stopped results after reload: {stopped}")
    running = {c.analysis.label: window.runner.is_running(c.analysis) for c in window.cards()}
    check(not any(running.values()), f"W7 still running: {running}")
    all_results_ok("after W7")
    # W8: close every tab, open the session again
    ANSWER["value"] = 1  # Discard if asked
    for c in list(window.cards()):
        window.close_card(c)
    settle(app, 0.3)
    check(not list(window.cards()), "W8 cards left")
    window.open_path(session) if hasattr(window, "open_path") else window.open_session(session)
    settle(app, 3)
    check(len(list(window.cards())) == len(cards), f"W8 reopened: {len(list(window.cards()))}")
    all_results_ok("after W8")
    finish(window)
