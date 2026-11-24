"""
Screenshots of the window, offscreen, for looking at the design (the user chooses visual changes from rendered
pictures; look at scale 1.5 and 1.25, the user's screens): the whole window, its top left corner, scrolled down,
and the graph window, each in light and dark.

    python tools/screenshots.py <a .tesselab-session or project> <output folder> [scale, default 1.5]

Only reads; never saves.
"""
import os
import sys
import tempfile
import time
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["QT_SCALE_FACTOR"] = sys.argv[3] if len(sys.argv) > 3 else "1.5"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
warnings.filterwarnings("ignore")
# the window writes settings (theme, recent projects): into a temporary folder, never the user's
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="tesselab-tool-")

if __name__ == "__main__":
    from PySide6 import QtCore, QtWidgets
    from tesselab import worker
    from tesselab.gui import theme
    from tesselab.gui.graph_view import GraphDialog
    from tesselab.gui.window import Window

    path, out = os.path.abspath(sys.argv[1]), sys.argv[2]
    os.makedirs(out, exist_ok=True)
    app = QtWidgets.QApplication([])
    theme.apply(app, "light")
    examples = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "modules")
    window = Window(path, [examples])
    window.runner.inline = True
    window.resize(1100, 700)
    window.show()

    def settle(seconds=1.0):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)

    settle(2)
    for name in ("light", "dark"):
        window.set_theme(name)
        tab = window.tabs.currentWidget()
        tab.verticalScrollBar().setValue(0)
        settle()
        window.grab().save(os.path.join(out, f"window_{name}.png"))
        window.grab(QtCore.QRect(0, 0, 520, 170)).save(os.path.join(out, f"corner_{name}.png"))
        tab.verticalScrollBar().setValue(300)
        settle()
        window.grab().save(os.path.join(out, f"scrolled_{name}.png"))
        cards = list(window.cards())
        if cards:
            cards[0]._show_graph()
            settle(0.3)
            for dialog in GraphDialog._open.values():
                dialog.grab().save(os.path.join(out, f"graph_{name}.png"))
                dialog.close()
    window.set_theme("light")
    for c in window.cards():
        c.saved_content = c.analysis.state()  # never save, never ask
    worker.shutdown()
    print("saved in", out)
