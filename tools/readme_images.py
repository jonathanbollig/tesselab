"""
The screenshots of the README (docs/images/: window_annotated, window_ta_dark, graph_uv_light, graph_ta_dark),
rendered offscreen from the samples that ship with the examples, so that they show nothing but those and can be
made again whenever the window changes:

    python tools/readme_images.py [output folder, default docs/images] [scale, default 1.25]

Works on a temporary project made from the UV and TA samples; never touches the user's folders or settings.
"""
import os
import shutil
import sys
import tempfile
import time
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["QT_SCALE_FACTOR"] = sys.argv[2] if len(sys.argv) > 2 else "1.25"
sys.path.insert(0, os.path.join(REPO, "src"))
warnings.filterwarnings("ignore")
for var in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):  # settings, trust and cache: never the user's
    os.environ[var] = tempfile.mkdtemp(prefix="tesselab-readme-")

EXAMPLES = os.path.join(REPO, "examples", "modules")


def make_project(root):
    from tesselab.modules import Registry
    from tesselab.project import Project
    from tesselab.templates import new_analysis

    os.makedirs(os.path.join(root, "data"))
    shutil.copytree(os.path.join(EXAMPLES, "uv_photochemistry", "sample", "uridine"),
                    os.path.join(root, "data", "uridine"))
    shutil.copytree(os.path.join(EXAMPLES, "transient_absorption", "sample", "ta_run"),
                    os.path.join(root, "data", "ta_run"))
    project = Project.create(root, "demo")
    registry = Registry(project.root, extra=[EXAMPLES], user=False, entry_points=False)
    paths = []
    for template, folder in (("kufner.uv-photochemistry/quantum-yield", "uridine"),
                             ("org.tesselab.transient-absorption/global-analysis", "ta_run")):
        analysis = new_analysis(registry, project, template, os.path.join(root, "data", folder))
        analysis.save()
        paths.append(analysis.path)
    return project, paths


def annotate(window, card, path):
    """The window with numbered badges at its parts (positions from the widgets, so they follow the layout)."""
    from PySide6 import QtCore, QtGui, QtWidgets
    from tesselab.gui.widgets import PlotTile

    def spot(widget, dx, dy):  # a point inside the window, from the widget's top left corner
        return widget.mapTo(window, QtCore.QPoint(0, 0)) + QtCore.QPoint(dx, dy)

    group = next(iter(card.groups.values()))
    tile = group.findChildren(PlotTile)[0]
    badges = [(1, spot(window.rail, window.rail.width() // 2, 205)),  # the rail (below its icons)
              (2, spot(window.explorer, window.explorer.width() - 24, 150)),  # the explorer
              (3, spot(window.tabs.tabBar(), 260, 12)),               # the tabs (after the +)
              (4, spot(card, 0, 150)),                                # the card
              (5, spot(card.header, card.header.width() - 410, card.header.height() + 5)),  # its header
              (6, spot(group, 200, 12)),                              # a view's section
              (7, spot(tile, 250, 15))]                               # a plot
    pixmap = window.grab()
    ratio = pixmap.devicePixelRatio()
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    font = painter.font()
    font.setBold(True)
    font.setPixelSize(15)
    painter.setFont(font)
    radius = 13
    for number, point in badges:
        painter.setPen(QtGui.QPen(QtGui.QColor("white"), 2))
        painter.setBrush(QtGui.QColor("#d9480f"))
        painter.drawEllipse(QtCore.QPointF(point), radius, radius)
        painter.drawText(QtCore.QRectF(point.x() - radius, point.y() - radius, 2 * radius, 2 * radius),
                         QtCore.Qt.AlignmentFlag.AlignCenter, str(number))
    painter.end()
    pixmap.save(path)


if __name__ == "__main__":
    from PySide6 import QtWidgets
    from tesselab import worker
    from tesselab.gui import theme
    from tesselab.gui.graph_view import GraphDialog
    from tesselab.gui.window import Window

    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "docs", "images"))
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="tesselab-readme-project-")
    project, paths = make_project(os.path.join(work, "demo"))
    app = QtWidgets.QApplication([])
    theme.apply(app, "light")
    window = Window(project.root, [EXAMPLES])
    window.runner.inline = True
    window.resize(1240, 800)
    window.show()

    def settle(seconds=1.0):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)

    settle(2)
    cards = [window.open_analysis(paths[0]), None]
    settle(2)
    cards[1] = window.open_analysis(paths[1], window.new_tab())
    settle(2)

    def show_tab(index):
        window.tabs.setCurrentIndex(index)
        window.tabs.currentWidget().verticalScrollBar().setValue(0)
        settle(1)

    def graph_shot(card, size, filename):
        """The graph window, wide, the whole graph in view, its side panel folded so that the nodes get the room."""
        dialog = card._show_graph()
        dialog.resize(*size)
        dialog.show_panel(False)
        settle(0.5)
        dialog.fit()
        settle(0.5)
        dialog.grab().save(os.path.join(out, filename))
        dialog.close()

    window.set_theme("light")
    show_tab(0)
    annotate(window, cards[0], os.path.join(out, "window_annotated.png"))
    graph_shot(cards[0], (1900, 880), "graph_uv_light.png")
    window.set_theme("dark")
    show_tab(1)
    window.grab().save(os.path.join(out, "window_ta_dark.png"))
    graph_shot(cards[1], (1500, 640), "graph_ta_dark.png")
    window.set_theme("light")
    for c in window.cards():
        c.saved_content = c.analysis.state()  # never save, never ask
    window.close()
    worker.shutdown()
    shutil.rmtree(work, ignore_errors=True)
    print("saved in", out)
