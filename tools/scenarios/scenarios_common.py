"""
Scenario scripts: the window driven offscreen through many user actions, every Python exception collected (also
inside Qt slots), message boxes answered automatically. They found the heatmap segfault, the graph window's lambda
on the runner and the stale results after closing a linked analysis (2026-11-22). Each works on a COPY (never the
user's data; they edit and save):

    cp -r <a project with two UV experiments> /tmp/proj
    python tools/scenarios/scen_graph.py /tmp/proj      # graph window: every drag, pick-up, menu, undo
    python tools/scenarios/scen_window.py /tmp/proj     # odd values in every field, inputs, save, export, reload
    python tools/scenarios/scen_links.py /tmp/proj      # links between analyses, warnings, closing
    python tools/scenarios/scen_actions.py /tmp/proj    # every menu action, dialogs rejected by a timer
    python tools/scenarios/scen_ta.py /tmp/taproj       # TA module (makes the project from the sample)

Use a fresh copy for each, run them under the memory watch (CLAUDE.md), and read the "N problems" summary at the end.
"""
import os
import sys
import tempfile
import time
import traceback
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
warnings.filterwarnings("ignore")
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="tesselab-scen-")
os.environ["XDG_DATA_HOME"] = tempfile.mkdtemp(prefix="tesselab-scen-data-")
os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="tesselab-scen-cache-")

errors, boxes, menus = [], [], []


def hook(kind, value, tb):
    text = "".join(traceback.format_exception(kind, value, tb))
    errors.append(text)
    print("EXCEPTION:", text, file=sys.stderr)


sys.excepthook = hook
ANSWER = {"value": 0}  # what message boxes answer (index of the button)


def setup(path, inline=True):
    from PySide6 import QtWidgets
    from tesselab.gui import theme, widgets
    import tesselab.gui.card as card_module
    import tesselab.gui.window as window_module
    import tesselab.gui.graph_view as graph_module

    def fake_box(parent, title, text, buttons=("OK",), default=0, icon="question"):
        boxes.append((title, text, buttons))
        return ANSWER["value"] if ANSWER["value"] < len(buttons) else default
    for module in (widgets, card_module, window_module, graph_module):
        module.message_box = fake_box

    def fake_exec(menu, *args, **kwargs):
        menus.append([a.text() for a in menu.actions()])
        return None
    QtWidgets.QMenu.exec = fake_exec
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app, "light")
    from tesselab.gui.window import Window
    window = Window(path, [REPO + "/examples/modules"])
    window.runner.inline = inline
    window.resize(1300, 900)
    window.show()
    settle(app, 2)
    return app, window


def settle(app, seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


def check(condition, message):
    if not condition:
        errors.append("CHECK FAILED: " + message)
        print("CHECK FAILED:", message, file=sys.stderr)


def finish(window):
    from tesselab import worker
    for c in window.cards():
        c.saved_content = c.analysis.state()
    window.close()
    worker.shutdown()
    print(f"\n{len(errors)} problems")
    for e in errors:
        print("----\n" + e)
