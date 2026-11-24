"""
How long one step of dragging a cursor takes in the window (offscreen, real worker processes): the part the mouse
waits for, and until the results are back. Speed of λ dragging is a priority for the user (2026-11-22: ~190 ms per
step before PanelView kept unchanged marks, ~21 ms after, with 5_UC@2 and 7_GU linked).

    python tools/measure_drag.py <a .tesselab-session> [node instance with a cursor, default phi0] [--profile]

It only reads the session and its analyses and never saves.
"""
import cProfile
import os
import pstats
import sys
import tempfile
import time
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_SCALE_FACTOR", "1.5")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
warnings.filterwarnings("ignore")
# the window writes settings (theme, recent projects): into a temporary folder, never the user's
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="tesselab-tool-")

if __name__ == "__main__":  # spawned workers import this file again
    from PySide6 import QtWidgets
    from tesselab import worker
    from tesselab.gui import theme
    from tesselab.gui.window import Window

    session, node = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "phi0")
    app = QtWidgets.QApplication([])
    theme.apply(app, "light")
    examples = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "modules")
    window = Window(os.path.abspath(session), [examples])
    window.resize(1500, 950)
    window.show()

    def wait():
        start = time.perf_counter()
        while time.perf_counter() - start < 60:
            app.processEvents()
            if not window.runner.pending and not any(window.runner.is_running(c.analysis) for c in window.cards()):
                return
            time.sleep(0.001)

    wait()
    for _ in range(200):
        app.processEvents()
        time.sleep(0.01)
    wait()
    card = next(iter(window.cards()))
    cursor = next(item for _, _, item in card.groups[node].interactions if hasattr(item, "on_change"))
    profile = cProfile.Profile()
    for k in range(1, 13):
        value = cursor.value * (1 + k * 4e-3)
        start = time.perf_counter()
        profile.enable()
        cursor.on_change(value, False)
        profile.disable()
        waited = time.perf_counter()
        wait()
        print(f"step {k}: {(waited - start) * 1000:.1f} ms in the window, results {1000 * (time.perf_counter() - waited):.1f} ms later")
    if "--profile" in sys.argv:
        pstats.Stats(profile).sort_stats("cumulative").print_stats(30)
    for c in window.cards():
        c.saved_content = c.analysis.state()  # never save, never ask
    worker.shutdown()
