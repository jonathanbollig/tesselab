"""Links between analyses: a link made, the graph window with a link box, the "changed since saved" warning and
saving it away, the source analysis closed, a cycle through links, closing tabs and the window with unsaved
changes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios_common import *  # noqa

if __name__ == "__main__":
    from PySide6 import QtCore
    from tesselab.analysis import Source
    from tesselab.gui.graph_view import GraphDialog

    proj = sys.argv[1]
    session = os.path.join(proj, "tesselab", "sessions", "two experiments.tesselab-session")
    app, window = setup(session)
    settle(app, 1)
    a, b = list(window.cards())
    print("cards:", a.analysis.label, b.analysis.label)

    def diag(card, name):
        r = window.runner.results(card.analysis).get(name)
        return (r.status, [d["message"][:90] for d in r.diagnostics if d["level"] != "info"]) if r else None

    # L1: b's steps take their spectra from a (a link)
    source = Source("corrections", "spectra", window.engine.link_path(b.analysis, a.analysis.path))
    print("link:", source.ref())
    b.set_input("steps", "spectra", source)
    settle(app, 0.5)
    print("  b.steps:", diag(b, "steps"))
    check(window.runner.results(b.analysis)["steps"].ok, "L1 linked steps not ok")
    # L2: the graph of b: a dashed link box; menus; a reverse drag from the linked input; remove the link edge
    GraphDialog.show_for(b)
    d = GraphDialog._open[id(b)]
    settle(app, 0.3)
    check(len(d.link_boxes) == 1, f"L2 link boxes {len(d.link_boxes)}")
    link_box = next(iter(d.link_boxes.values()))
    check([a.text() for a in d.menu_at(link_box.sceneBoundingRect().center()).actions()] == ["Open this analysis"],
          "L2 menu on a link box")
    link_edge = [(w, *w.key) for w in d.wires if w.style == "link"]
    check(len(link_edge) == 1, "L2 one dashed edge")
    path = link_edge[0][0].path()
    point = next((path.pointAtPercent(t / 20) for t in range(1, 20)
                  if d.edge_at(path.pointAtPercent(t / 20)) is not None), None)
    menu = d.menu_at(point) if point is not None else None
    check(menu is not None and [x.text() for x in menu.actions()] == ["Remove connection"], "L2 link edge menu")
    check(d.begin_drag(d.boxes["steps"].port("in", "spectra")), "L2 drag from the linked input")
    check(d.drag.kind == "in" and d.drag.lifted is None, f"L2 a linked input starts a new connection: {d.drag.kind}")
    d.cancel_drag()
    if menu is not None:
        menu.actions()[0].trigger()
        settle(app, 0.3)
        check(b.analysis.nodes["steps"]["inputs"]["spectra"] is None, "L2 link removed")
        b.undo()
        settle(app, 0.3)
    check(b.analysis.nodes["steps"]["inputs"]["spectra"] is not None, "L2 link back after undo")
    # L3: save b (remembers the link's content), change a's spectra: b warns; saving b accepts it
    ANSWER["value"] = 0
    check(window.save_card(b), "L3 save b")
    settle(app, 0.5)
    print("  b.steps after save:", diag(b, "steps"))
    params = window.engine.node_class(a.analysis, "corrections").params
    print("  spectra params:", [(p.name, type(p).__name__) for p in params])
    changed = False
    for p in params:
        if p.name == "baseline":
            old = a.param_value("corrections", p.name)
            print("  baseline was", old)
            print("  b input after save:", b.analysis.nodes["steps"]["inputs"]["spectra"])
            h0 = window.runner.results(a.analysis)["corrections"].output_hash("spectra")
            a.set_param("corrections", p.name, [340e-9, 355e-9])
            settle(app, 1)
            h1 = window.runner.results(a.analysis)["corrections"].output_hash("spectra")
            print("  a spectra hash before/after:", h0[:8], (h1 or "None")[:8], "a.corrections:", diag(a, "corrections"))
            print("  b.steps now:", diag(b, "steps"), window.runner.results(b.analysis)["steps"].fingerprint[:8])
            settle(app, 1)
            r = window.runner.results(b.analysis)["steps"]
            if any(x.get("code") == "link-changed" for x in r.diagnostics):
                print("  changed", p.name, "-> warning:", diag(b, "steps"))
                changed = True
                break
            a.undo()
            settle(app, 0.3)
    check(changed, "L3 no link-changed warning after changing a's spectra")
    if changed:
        window.save_card(b)
        settle(app, 0.5)
        r = window.runner.results(b.analysis)["steps"]
        check(not any(x.get("code") == "link-changed" for x in r.diagnostics), "L3 warning gone after saving b")
    # L4: a cycle through the link: a's spectra input (optional buffer) from b
    a_inputs = ["buffer"]  # the optional input of a's corrections
    print("  a.spectra inputs:", list(a_inputs))
    if a_inputs:
        boxes.clear()
        back = Source("steps", "spectra_with_totals", window.engine.link_path(a.analysis, b.analysis.path))
        a.set_input("corrections", list(a_inputs)[0], back)
        settle(app, 1)
        print("  cycle through link -> boxes:", boxes[-1:], "a.corrections:", diag(a, "corrections"))
        if a.analysis.nodes["corrections"]["inputs"].get(list(a_inputs)[0]):
            a.undo()
            settle(app, 0.5)
    # L5: close a with unsaved changes (Discard): b's link now reads a from its file
    ANSWER["value"] = 1
    window.close_card(a)
    settle(app, 1)
    print("  b.steps after a closed:", diag(b, "steps"))
    check(window.runner.results(b.analysis)["steps"].ok, "L5 b's link after a closed")
    r = window.runner.results(b.analysis)["steps"]
    check(any(x.get("code") == "link-changed" for x in r.diagnostics),
          "L5 b still shows results of a's discarded changes (no link-changed warning)")
    # L6: close b's tab with unsaved changes, Cancel: stays; the window with Cancel: stays open
    b.edit(lambda: b.analysis.doc.__setitem__("comment", "unsaved"))
    ANSWER["value"] = 2  # Cancel
    check(not window.close_card(b), "L6 close with Cancel went ahead")
    window.close()
    settle(app, 0.3)
    check(window.isVisible(), "L6 window closed despite Cancel")
    finish(window)
