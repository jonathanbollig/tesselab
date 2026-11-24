"""
Start the window: tesselab [a project folder, an analysis file or a saved window] [--modules folder ...]
"""
import os
import signal
import sys
import time

from tesselab import worker


def run(path=None, extra_modules=()) -> int:
    import pyqtgraph as pg
    from pyqtgraph.util.garbage_collector import GarbageCollector
    from PySide6 import QtCore, QtWidgets

    from tesselab.gui import theme
    from tesselab.gui.window import Window

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Tesselab")
    app.setOrganizationName("tesselab")
    settings = QtCore.QSettings("tesselab", "tesselab")
    theme.apply(app, settings.value("theme", "light"))
    app.setWindowIcon(theme.app_icon())
    # Python's garbage collection only in this (the Qt) thread, once a second: on its own it may run in the thread
    # that hands results back from the worker processes and delete Qt objects there, which hangs or crashes Qt
    app.garbage_collector = GarbageCollector(interval=1.0)
    window = Window(os.path.expanduser(path) if path else None, extra_modules)
    window.show()

    # Ctrl+C in the terminal: warn about unsaved changes once, a second Ctrl+C within 5 s stops anyway. Python only
    # sees the signal when it runs, so a timer lets it run regularly.
    warned = {"at": 0.0}

    def on_ctrl_c(signum, frame):
        unsaved = [c.analysis.label for c in window.cards() if c.is_unsaved()]
        if unsaved and time.monotonic() - warned["at"] > 5:
            warned["at"] = time.monotonic()
            print(f"\nUnsaved changes in: {', '.join(unsaved)}. Press Ctrl+C again within 5 s to stop anyway.",
                  flush=True)
            return
        app.exit(0)

    signal.signal(signal.SIGINT, on_ctrl_c)
    timer = QtCore.QTimer(interval=200, timeout=lambda: None)
    timer.start()
    code = app.exec()
    worker.shutdown()
    return code
