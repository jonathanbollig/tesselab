"""
The window: the explorer on the left, tabs of analyses side by side, menus and a command palette (Ctrl+Shift+P).

A tab holds analyses (cards) next to each other; the linked ones (link button in their header) share the zoom of
each plot kind, cursors and which groups are open. The window owns the project, the module registry, the engine and
the runner, and passes changes between them: a card changed -> the runner computes -> the cards show the results;
an analysis changed -> the analyses that link to it compute again too.

Saved windows (*.tesselab-session, JSON): the tabs, their analyses (paths relative to the session file), tab names,
linked analyses, the current tab, the open groups are in the analysis files themselves.
"""
import atexit
import json
import os
import traceback
import weakref

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from tesselab import project as projects, worker
from tesselab.analysis import Analysis, AnalysisError, write_json
from tesselab.engine import Engine, provenance
from tesselab.gui import theme
from tesselab.gui.card import AnalysisCard, Shared
from tesselab.gui.explorer import Explorer
from tesselab.gui.runner import Runner
from tesselab.gui.watch import Follow
from tesselab.gui.widgets import message_box
from tesselab.modules import Registry, user_module_folder


class Tab(QtWidgets.QScrollArea):
    """Analyses side by side; scrolls vertically (the wheel), and horizontally when they do not fit."""

    def __init__(self, window):
        super().__init__(objectName="tab_page")
        self.window_, self.name = window, None
        self.setWidgetResizable(True)
        # a rounded panel: the scrolled area is inset by the corners' radius at the top and bottom, so that what
        # scrolls by disappears below a calm margin instead of at the tabs, and never covers the round corners
        self.setViewportMargins(0, theme.PANEL_R, 0, theme.PANEL_R)
        self.content = QtWidgets.QWidget(objectName="page")
        self.row = QtWidgets.QHBoxLayout(self.content)
        self.row.setContentsMargins(10, 0, 10, 0)
        self.row.setSpacing(10)
        self.empty = QtWidgets.QLabel("Open an analysis from the explorer on the left (double-click), or make one: "
                                      "File → New analysis from a template.", objectName="hint",
                                      alignment=Qt.AlignmentFlag.AlignCenter, wordWrap=True)
        self.row.addWidget(self.empty)
        self.setWidget(self.content)
        self.cards = []
        self.viewport().setAutoFillBackground(True)
        # the cards' headers stay in view when the tab is scrolled down (AnalysisCard.place_header)
        self.verticalScrollBar().valueChanged.connect(lambda: [card.place_header() for card in self.cards])

    def add_card(self, card):
        self.empty.setVisible(False)
        self.cards.append(card)
        self.row.addWidget(card, 1, Qt.AlignmentFlag.AlignTop)

    def remove_card(self, card):
        from tesselab.gui.graph_view import GraphDialog
        self.cards.remove(card)
        self.row.removeWidget(card)
        dialog = GraphDialog._open.pop(id(card), None)
        if dialog is not None:
            dialog.close()  # now, not only when the card is deleted later
        card.setParent(None)
        card.deleteLater()
        self.empty.setVisible(not self.cards)

    def title(self):
        if self.name:
            return self.name
        labels = [c.analysis.label for c in self.cards]
        return ", ".join(labels) if labels else "Empty"

    def linked(self):
        return [c for c in self.cards if c.link_button.isChecked()]


class TabBar(QtWidgets.QTabBar):
    """Never wider than its tabs: Qt's minimum width of a tab bar (room for its scroll buttons) is larger than one
    short tab, which left a gap before the +.

    A dragged tab jumps to its new place once the mouse is far enough over a neighbour, instead of Qt's movable tabs,
    which slide with the mouse: the current tab's feet (Tabs.paintEvent) only know the tab's place, so they jumped
    while the tab slid (the user's review)."""

    _dragged = -1

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        return QtCore.QSize(min(hint.width(), self.sizeHint().width()), hint.height())

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self._dragged = self.tabAt(event.position().toPoint()) if event.button() == Qt.MouseButton.LeftButton else -1

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._dragged = -1

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        x, i = event.position().x(), self.tabAt(event.position().toPoint())
        if not event.buttons() & Qt.MouseButton.LeftButton or self._dragged < 0 or i < 0 or i == self._dragged:
            return
        width = self.tabRect(self._dragged).width()
        # move only where the moved tab will be under the mouse: tabs of other widths would swap back and forth
        if (x > self.tabRect(i).right() - width) if i > self._dragged else (x < self.tabRect(i).left() + width):
            self.moveTab(self._dragged, i)
            self._dragged = i


class Tabs(QtWidgets.QWidget):
    """The tabs: a strip with the tab bar and + right after the last tab (a QTabWidget puts it in a corner and draws
    a line under its tabs), and the pages below, in a rounded panel. Offers the parts of
    QTabWidget's interface the window uses."""
    currentChanged = QtCore.Signal(int)
    tabCloseRequested = QtCore.Signal(int)

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, theme.GAP, theme.GAP)
        layout.setSpacing(0)
        self.strip = strip = QtWidgets.QWidget(objectName="tab_strip")
        self.row = row = QtWidgets.QHBoxLayout(strip)
        # the tabs' top level with the explorer panel's top; the first tab indented, so that its left foot (paintEvent)
        # stands on the page's straight top edge, past the page's round corner, not in the chrome beside it
        row.setContentsMargins(theme.PANEL_R + theme.CARD_R, theme.GAP, 0, 0)
        row.setSpacing(4)
        self.bar = TabBar(tabsClosable=True, expanding=False, documentMode=True,
                          usesScrollButtons=True)
        self.bar.setDrawBase(False)
        self.bar.setElideMode(Qt.TextElideMode.ElideRight)
        self.bar.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        row.addWidget(self.bar, 0, Qt.AlignmentFlag.AlignBottom)
        self.plus = theme.icon_button("plus", "New tab")
        row.addWidget(self.plus, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        layout.addWidget(strip)
        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.bar.currentChanged.connect(self._current_changed)
        self.bar.tabCloseRequested.connect(self.tabCloseRequested)
        self.bar.tabMoved.connect(self._moved)

    def paintEvent(self, event):
        """The current tab stands on the page panel below and flows into it with concave corners at its foot. Drawn
        here, below the strip and the panel, and reaching a little into the panel: at a fractional screen scale the
        tab's lowest row of pixels is only partly covered by the tab and partly by the panel, which showed a seam."""
        bar = self.bar
        index = bar.currentIndex()
        if index < 0 or not bar.isVisible():
            return
        rect = QtCore.QRectF(bar.tabRect(index).translated(bar.mapTo(self, QtCore.QPoint(0, 0))))
        left, right = rect.left(), rect.right() - 2  # without the tab's margin-right (style sheet)
        bottom, r = float(self.strip.geometry().bottom() + 1), float(theme.CARD_R)
        path = QtGui.QPainterPath()
        path.addRect(QtCore.QRectF(left, bottom - r, right - left, r + 2))
        path.moveTo(left - r, bottom)  # left foot: a quarter circle hollowed out of the corner
        path.arcTo(QtCore.QRectF(left - 2 * r, bottom - 2 * r, 2 * r, 2 * r), 270, 90)
        path.lineTo(left, bottom + 2)
        path.lineTo(left - r, bottom + 2)
        path.closeSubpath()
        path.moveTo(right + r, bottom)
        path.arcTo(QtCore.QRectF(right, bottom - 2 * r, 2 * r, 2 * r), 270, -90)
        path.lineTo(right, bottom + 2)
        path.lineTo(right + r, bottom + 2)
        path.closeSubpath()
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        # a tab scrolled out of the bar: no feet
        painter.setClipRect(QtCore.QRectF(bar.geometry().translated(self.strip.pos())).adjusted(-r, 0, r, 4))
        painter.fillPath(path.simplified(), QtGui.QColor(theme.color("page")))
        painter.end()

    def update_strip(self):
        self.update(self.strip.geometry().adjusted(0, 0, 0, 4))

    def set_left_edge(self, gap):
        """The room on the left: none next to the explorer (the splitter's handle is the gap), else the chrome's."""
        margins = self.layout().contentsMargins()
        self.layout().setContentsMargins(gap, margins.top(), margins.right(), margins.bottom())

    def _changed(self):
        """Tell the layout the tab bar's new width. QTabBar does so itself only while it is visible: changes made
        before the window is shown left a stale width (a gap before +, or no tab at all when a saved window was opened
        at start)."""
        self.bar.updateGeometry()
        self.layout().invalidate()
        self.update_strip()

    def showEvent(self, event):
        super().showEvent(event)
        self._changed()

    def _current_changed(self, index):
        if 0 <= index < self.stack.count():
            self.stack.setCurrentIndex(index)
        self.update_strip()
        self.currentChanged.emit(index)

    def _moved(self, start, end):
        widget = self.stack.widget(start)
        self.stack.removeWidget(widget)
        self.stack.insertWidget(end, widget)
        self.stack.setCurrentIndex(self.bar.currentIndex())
        self._changed()

    def addTab(self, widget, text):
        self.stack.addWidget(widget)
        index = self.bar.addTab(text)
        self._changed()
        return index

    def removeTab(self, index):
        widget = self.stack.widget(index)
        self.stack.removeWidget(widget)
        self.bar.removeTab(index)
        self._changed()

    def count(self):
        return self.stack.count()

    def widget(self, index):
        return self.stack.widget(index)

    def indexOf(self, widget):
        return self.stack.indexOf(widget)

    def currentWidget(self):
        return self.stack.currentWidget()

    def currentIndex(self):
        return self.bar.currentIndex()

    def setCurrentIndex(self, index):
        self.bar.setCurrentIndex(index)
        self.stack.setCurrentIndex(index)

    def setCurrentWidget(self, widget):
        self.setCurrentIndex(self.stack.indexOf(widget))

    def setTabText(self, index, text):
        self.bar.setTabText(index, text)
        self._changed()

    def setTabToolTip(self, index, text):
        self.bar.setTabToolTip(index, text)

    def tabBar(self):
        return self.bar


def _detach(ref):
    """Before Python shuts down (atexit): Qt still passes the events of the widgets it deletes at exit through the
    application's event filters and signals, and calling Python then made shiboken allocate memory without end."""
    window = ref()
    app = QtWidgets.QApplication.instance()
    if window is not None and app is not None:
        app.removeEventFilter(window)
        try:
            app.focusChanged.disconnect(window._focus_changed)
        except (RuntimeError, TypeError):
            pass


class Window(QtWidgets.QMainWindow):
    def __init__(self, path=None, extra_modules=()):
        super().__init__()
        self._active_card = None
        self.settings = QtCore.QSettings("tesselab", "tesselab")
        self.extra_modules = list(extra_modules)
        self.advanced = self.settings.value("advanced", False, type=bool)
        self.project = None
        self.registry = Registry(None, extra=self.extra_modules)
        self.engine = Engine(self.registry)
        self.runner = Runner(self.engine)
        self.runner.updated.connect(self._results_arrived)
        self.shared = {}  # id(analysis) -> Shared: an analysis open in several tabs has one undo history
        self.setWindowTitle("Tesselab")
        self.setWindowIcon(theme.app_icon())
        self.resize(1500, 950)
        self.explorer = Explorer()
        self.explorer.open_analysis.connect(self.open_analysis)
        self.explorer.new_from_data.connect(self.new_from_data)
        self.explorer.open_session.connect(self.open_session)
        self.tabs = Tabs()
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.currentChanged.connect(lambda i: (self._tab_changed(), self._update_explorer()))
        app = QtWidgets.QApplication.instance()
        app.installEventFilter(self)
        app.focusChanged.connect(self._focus_changed)
        atexit.register(_detach, weakref.ref(self))
        self.tabs.tabBar().tabBarDoubleClicked.connect(self.rename_tab)
        self.tabs.plus.clicked.connect(lambda: self.new_tab())
        self.splitter = QtWidgets.QSplitter()
        self.splitter.setHandleWidth(theme.GAP)
        self.splitter.addWidget(self.explorer)
        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([260, 1240])
        central = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.rail = self._make_rail()
        row.addWidget(self.rail)
        row.addWidget(self.splitter, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")
        self.follow = Follow(self)
        self._make_menus()
        self.show_explorer(self.explorer_action.isChecked())
        self.follow.set_on(self.follow_action.isChecked())
        self.new_tab()
        if path:
            self.open_path(path)
        else:
            recent = self.recent_projects()
            if recent and os.path.isdir(recent[0]):
                self.open_project(recent[0])

    # --- the active card ---------------------------------------------------------------------------------------
    @property
    def active_card(self):
        """The card that Save, Undo, Export, ... act on: the one last clicked or typed in, in the current tab."""
        return self._active_card

    @active_card.setter
    def active_card(self, card):
        old, self._active_card = self._active_card, card
        for c in {old, card} - {None}:
            try:
                c.set_active(c is card)
            except RuntimeError:  # already deleted
                pass
        tab = self.tab_of(card) if card is not None else None
        if tab is not None:
            tab.active = card

    def eventFilter(self, obj, event):
        """A click anywhere in a card makes it the active one (its children take the click, the card sees nothing)."""
        if event.type() == QtCore.QEvent.Type.MouseButtonPress and isinstance(obj, QtWidgets.QWidget):
            self._activate_from(obj)
        return False

    def _focus_changed(self, old, new):
        self._activate_from(new)

    def _activate_from(self, widget):
        while widget is not None and not isinstance(widget, AnalysisCard):
            widget = widget.parentWidget()
        if widget is not None and widget is not self._active_card and self.tab_of(widget) is not None:
            self.active_card = widget

    def _tab_changed(self):
        """The current tab's own active card (or its first) becomes the active one."""
        tab = self.tabs.currentWidget()  # not current_tab(): it makes a tab when there is none (while closing all)
        if tab is None:
            return
        remembered = getattr(tab, "active", None)
        self.active_card = remembered if remembered in tab.cards else (tab.cards[0] if tab.cards else None)

    # --- menus --------------------------------------------------------------------------------------------------
    def _make_menus(self):
        """The menus, behind the ☰ button of the rail (_make_rail). Every action is also the window's own, or its
        shortcut would work only while the menu is open."""
        bar = self.menus = QtWidgets.QMenu(self)

        def add(menu, text, slot, shortcut=None, checkable=False, checked=False):
            action = QtGui.QAction(text, self, checkable=checkable)
            if checkable:
                action.setChecked(checked)
                action.toggled.connect(slot)
            else:
                action.triggered.connect(slot)
            if shortcut:  # one, or a tuple of several (the menu shows the first)
                keys = (shortcut,) if isinstance(shortcut, str) else shortcut
                action.setShortcuts([QtGui.QKeySequence(k) for k in keys])
            menu.addAction(action)
            self.addAction(action)
            return action

        file = bar.addMenu("File")
        add(file, "Open project…", self.choose_project, "Ctrl+O")
        self.recent_menu = file.addMenu("Recent projects")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        file.addSeparator()
        add(file, "New analysis from a template…", lambda: self.new_from_data(None, []), "Ctrl+N")
        add(file, "Open analysis file…", self.choose_analysis)
        add(file, "Save", self.save_active, "Ctrl+S")
        add(file, "Save all", self.save_all, "Ctrl+Shift+S")
        add(file, "Export the analysis (PDF + JSON)…", lambda: self.export_card(self.active_card) if self.active_card
            else None, "Ctrl+E")
        file.addSeparator()
        add(file, "Open saved window…", self.choose_session)
        add(file, "Save window…", self.save_session)
        file.addSeparator()
        add(file, "Quit", self.close, "Ctrl+Q")
        edit = bar.addMenu("Edit")
        add(edit, "Undo", self.undo_active, "Ctrl+Z")
        # Ctrl+Shift+Z is the redo of KDE and GNOME, Ctrl+Y that of Windows: both work
        add(edit, "Redo", self.redo_active, ("Ctrl+Shift+Z", "Ctrl+Y"))
        view = bar.addMenu("View")
        self.explorer_action = add(view, "Show the explorer", self.show_explorer, "Ctrl+B", checkable=True,
                                   checked=self.settings.value("explorer", True, type=bool))
        add(view, "Filter the explorer", self.explorer.focus_filter, "Ctrl+F")
        add(view, "Show advanced parameters", self._set_advanced, checkable=True, checked=self.advanced)
        themes = view.addMenu("Theme")
        themes.aboutToShow.connect(lambda: self._fill_themes(themes))
        add(view, "Command palette…", self.command_palette, "Ctrl+Shift+P")
        analysis = bar.addMenu("Analysis")
        add(analysis, "Add a node…", lambda: self.active_card.add_in_graph() if self.active_card else None,
            "Ctrl+Shift+A")
        add(analysis, "Show the graph", lambda: self.active_card._show_graph() if self.active_card else None, "Ctrl+G")
        add(analysis, "Recompute everything", self.recompute_all, "F5")
        add(analysis, "Clear the disk cache", self.clear_cache)
        modules = bar.addMenu("Modules")
        add(modules, "Modules…", self.show_modules)
        add(modules, "Reload modules", self.reload_modules, "Ctrl+Shift+R")
        self.follow_action = add(modules, "Follow changes on disk", self.set_follow, checkable=True,
                                 checked=self.settings.value("follow_disk", True, type=bool))
        add(modules, "Open your module folder", lambda: self._open_folder(user_module_folder()))
        add(modules, "New module…", self.new_module)
        help_ = bar.addMenu("Help")
        add(help_, "The module protocol (AGENTS.md)", self.show_protocol)
        add(help_, "About Tesselab", self.about)

    def _make_rail(self):
        """The narrow bar on the left, as in the kufner viewer (the user's choice; the window's top row is the tabs'):
        ☰ with every menu, the explorer's fold, and the actions used most. A plain column, not a QToolBar: that one
        keeps the same margin on every side, and ☰ is to be level with the tabs."""
        rail = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(rail)
        column.setContentsMargins(4, theme.GAP + 4, 0, theme.GAP)  # ☰ level with the tabs
        column.setSpacing(2)

        def add(name, tooltip, slot, checkable=False):
            button = theme.icon_button(name, tooltip, checkable)
            button.clicked.connect(slot)
            column.addWidget(button)
            return button
        menus = add("menu", "Menus", lambda: self.menus.exec(menus.mapToGlobal(menus.rect().topRight())))
        self.explorer_button = add("sidebar", "Show or fold the explorer (Ctrl+B)", self.show_explorer, checkable=True)
        column.addSpacing(10)
        add("folder", "Open project… (Ctrl+O)", lambda: self.choose_project())
        recent = add("clock", "Recent projects",
                     lambda: self.recent_menu.exec(recent.mapToGlobal(recent.rect().topRight())))
        add("save", "Save the active analysis (Ctrl+S)", lambda: self.save_active())
        add("save_all", "Save all (Ctrl+Shift+S)", lambda: self.save_all())
        column.addStretch(1)  # undo and redo are in each card's header: they act on that analysis
        return rail

    def undo_active(self):
        if self.active_card:
            self.active_card.undo()

    def redo_active(self):
        if self.active_card:
            self.active_card.redo()

    def show_explorer(self, on):
        """Show or fold the explorer (the rail's button, Ctrl+B)."""
        self.explorer.setVisible(on)
        self.tabs.set_left_edge(0 if on else theme.GAP)
        self.settings.setValue("explorer", on)
        for widget in (self.explorer_button, getattr(self, "explorer_action", None)):
            if widget is not None and widget.isChecked() != on:
                widget.blockSignals(True)
                widget.setChecked(on)
                widget.blockSignals(False)

    def all_actions(self):
        found = []

        def walk(menu):
            for action in menu.actions():
                if action.menu():
                    walk(action.menu())
                else:
                    found.append(action)
        for action in self.menus.actions():
            if action.menu():
                walk(action.menu())
        return found

    def command_palette(self):
        from tesselab.gui.dialogs import CommandPalette
        CommandPalette(self, self.all_actions()).show()

    def _fill_themes(self, menu):
        menu.clear()
        current = self.settings.value("theme", "light")
        for name in theme.available():
            action = menu.addAction(name, lambda n=name: self.set_theme(n))
            action.setCheckable(True)
            action.setChecked(name == current)

    def set_theme(self, name):
        self.settings.setValue("theme", name)
        theme.apply(QtWidgets.QApplication.instance(), name)
        theme.recolor_icons()
        for card in self.cards():
            for group in card.groups.values():
                group.update_heading()  # its chevron
                for tile in group.tiles.values():
                    tile.view.setBackground(theme.color("tile"))
                    tile.view._structure = None
                group.interactions = []
                for tile in group.tiles.values():
                    tile.view.cursors = []
                group.draw(force=True)
            card._show_details()  # its chevron
            card._shown_unsaved = None  # the save icon drawn anew
            card.update_header()
            card.update_results()
        self.explorer.update_icons()
        self.tabs.update_strip()
        self.explorer.refresh()
        self._update_tab_titles()
        from tesselab.gui.graph_view import GraphDialog
        for dialog in list(GraphDialog._open.values()):
            try:
                if dialog.isVisible():
                    dialog.redraw()  # in the new theme's colors (a hidden one is drawn again when shown)
            except RuntimeError:
                pass  # its card was deleted meanwhile

    def _set_advanced(self, on):
        self.advanced = on
        self.settings.setValue("advanced", on)
        for card in self.cards():
            card.rebuild()

    # --- projects ---------------------------------------------------------------------------------------------------
    def recent_projects(self):
        value = self.settings.value("recent_projects", [])
        return [value] if isinstance(value, str) else list(value or [])

    def _fill_recent(self):
        self.recent_menu.clear()
        for path in self.recent_projects():
            self.recent_menu.addAction(path, lambda p=path: self.open_project(p))
        if not self.recent_projects():
            self.recent_menu.addAction("(none yet)").setEnabled(False)

    def choose_project(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open project (a folder with your data)",
                                                          self.project.root if self.project else os.getcwd())
        if path:
            self.open_project(path)

    def open_path(self, path):
        """A project folder, an analysis file or a saved window."""
        path = os.path.abspath(path)
        if path.endswith(".tesselab-session"):
            root = projects.find_root(path)
            if root:
                self.open_project(root)
            self.open_session(path)
        elif path.endswith(".json"):
            root = projects.find_root(path)
            if root:
                self.open_project(root)
            self.open_analysis(path)
        else:
            self.open_project(path)

    def open_project(self, path):
        root = projects.find_root(path)
        if root is None:
            answer = message_box(self, "Make it a project?",
                                 f"{path} is no Tesselab project yet. Make it one? (A folder tesselab/ is created "
                                 f"in it for your analyses; your data are not changed.)", ("Make it a project",
                                                                                           "Cancel"))
            if answer != 0:
                return
            projects.Project.create(path)
            root = os.path.abspath(path)
        if self.project is not None and self.project.root == root:
            return
        if self.project is not None and not self.close_all_tabs():
            return
        project = projects.Project(root)
        trusted = True
        if projects.has_own_modules(root):
            state = projects.trust_state(root)
            if state is None:
                answer = message_box(self, "Modules in this project",
                                     f"The project {project.name} contains modules (tesselab/modules/). Modules are "
                                     f"Python code: run them only if you trust where the project comes from.",
                                     ("Run its modules", "Do not run them"), default=1, icon="warning")
                trusted = answer == 0
                projects.set_trust(root, trusted)
            else:
                trusted = state == "trusted"
        self.project = project
        self.registry = Registry(root, extra=self.extra_modules, trusted=trusted)
        self.engine = Engine(self.registry, project)
        self.runner.engine = self.engine
        self.explorer.set_project(project, self.registry)
        self.runner.warm(self.registry)
        recent = [root] + [p for p in self.recent_projects() if p != root]
        self.settings.setValue("recent_projects", recent[:10])
        self.setWindowTitle(f"Tesselab · {project.name}")
        problems = [f"{m.name}: {'; '.join(m.problems)}" for m in self.registry.modules.values() if m.problems]
        if self.registry.errors or problems:
            self.statusBar().showMessage("Some modules cannot be used: Modules → Modules… shows why.", 15000)
        else:
            self.statusBar().showMessage(f"Opened {root}", 5000)
        self.follow.schedule()

    # --- tabs and cards -------------------------------------------------------------------------------------------
    def new_tab(self, name=None) -> Tab:
        tab = Tab(self)
        tab.name = name
        index = self.tabs.addTab(tab, tab.title())
        self.tabs.setCurrentIndex(index)
        return tab

    def current_tab(self) -> Tab:
        tab = self.tabs.currentWidget()
        return tab if tab is not None else self.new_tab()

    def tab_of(self, card) -> Tab | None:
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            if card in tab.cards:
                return tab
        return None

    def scroll_area_of(self, card):
        return self.tab_of(card)

    def cards(self):
        for i in range(self.tabs.count()):
            yield from self.tabs.widget(i).cards

    def shared_of(self, analysis) -> Shared:
        if id(analysis) not in self.shared:
            self.shared[id(analysis)] = Shared(analysis)
        return self.shared[id(analysis)]

    def cards_of(self, analysis):
        return [c for c in self.cards() if c.analysis is analysis]

    def _update_tab_titles(self):
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            title = tab.title()
            unsaved = any(c.is_unsaved() for c in tab.cards)
            self.tabs.setTabText(i, title if len(title) < 40 else title[:38] + "…")
            self.tabs.tabBar().setTabTextColor(i, QtGui.QColor(theme.color("amber")) if unsaved else QtGui.QColor())
            self.tabs.setTabToolTip(i, title + ("  (unsaved changes)" if unsaved else ""))

    def rename_tab(self, index):
        tab = self.tabs.widget(index)
        if tab is None:
            return
        text, ok = QtWidgets.QInputDialog.getText(self, "Rename tab", "Name (empty: named after its analyses):",
                                                  text=tab.name or "")
        if ok:
            tab.name = text.strip() or None
            self._update_tab_titles()

    def close_tab(self, index):
        tab = self.tabs.widget(index)
        for card in list(tab.cards):
            if not self.close_card(card):
                return
        self.tabs.removeTab(index)
        tab.deleteLater()
        if self.tabs.count() == 0:
            self.new_tab()

    def close_all_tabs(self) -> bool:
        if not self._confirm_unsaved(list(self.cards())):
            return False
        while self.tabs.count():
            tab = self.tabs.widget(0)
            for card in list(tab.cards):
                self._drop_card(card)
            self.tabs.removeTab(0)
            tab.deleteLater()
        self.new_tab()
        return True

    def open_analysis(self, path, tab=None):
        """Open an analysis file in a tab (default: the current one). Open there already: shown; open in another
        tab: a second card of the same analysis (the same in-memory analysis, one undo history)."""
        path = os.path.abspath(path)
        target = tab or self.current_tab()
        existing = None
        for card in self.cards():
            if card.analysis.path and os.path.abspath(card.analysis.path) == path:
                if self.tab_of(card) is target:
                    self.tabs.setCurrentWidget(target)
                    self.active_card = card
                    return card
                existing = card.analysis
        if existing is not None:
            return self.show_analysis(existing, target)
        try:
            analysis = Analysis.load(path, self.engine.graphs)  # open analyses share their graph files
        except (AnalysisError, OSError) as error:
            message_box(self, "Cannot open", str(error), icon="warning")
            return None
        return self.show_analysis(analysis, target)

    def show_analysis(self, analysis, tab=None):
        card = AnalysisCard(self, analysis)
        card.changed.connect(self._card_changed)
        card.saved_changed.connect(lambda c: self._update_tab_titles())  # every edit, undo
        card.close_requested.connect(self.close_card)
        card.cursor_moved.connect(self._cursor_moved)
        card.fold_changed.connect(self._fold_changed)
        card.link_toggled.connect(lambda c, on: self.link_plots(c))
        self.engine.register(analysis)
        (tab or self.current_tab()).add_card(card)
        self.active_card = card
        self._update_tab_titles()
        self._update_explorer()
        self.runner.request(analysis)
        self.follow.schedule()
        return card

    def close_card(self, card) -> bool:
        if not card.others() and not self._confirm_unsaved([card]):  # open in another tab: nothing is lost
            return False
        self._drop_card(card)
        return True

    def _drop_card(self, card):
        tab = self.tab_of(card)
        card.shared.cards.remove(card)
        if not card.shared.cards:  # its last card: the analysis is closed
            self.shared.pop(id(card.analysis), None)
            self.engine.unregister(card.analysis)
            self.runner.forget(card.analysis)
            self._request_linking(card.analysis)
        if tab is not None:
            tab.remove_card(card)
        if self.active_card is card:
            self._tab_changed()  # another card of the current tab
        self._update_tab_titles()
        self._update_explorer()
        self.follow.schedule()

    def _confirm_unsaved(self, cards) -> bool:
        unsaved, seen = [], set()
        for c in cards:  # one card per analysis (it may be open in several tabs)
            if c.is_unsaved() and id(c.analysis) not in seen:
                seen.add(id(c.analysis))
                unsaved.append(c)
        if not unsaved:
            return True
        names = ", ".join(c.analysis.label for c in unsaved)
        answer = message_box(self, "Unsaved changes", f"Save the changes to {names}?",
                             ("Save", "Discard", "Cancel"), default=0, icon="warning")
        if answer == 0:
            return all(self.save_card(c) for c in unsaved)
        return answer == 1

    def _update_explorer(self):
        tab = self.tabs.currentWidget()
        self.explorer.set_open([c.analysis.path for c in tab.cards] if tab else [])

    # --- computing --------------------------------------------------------------------------------------------------
    def _card_changed(self, card):
        self.runner.request(card.analysis, fresh=False)  # an edit changes no file on disk
        self._request_linking(card.analysis)
        self.follow.schedule()  # a path parameter may have changed

    def graph_changed(self, analysis, recompute=True, redraw=True):
        """An edit in `analysis` changed its shared graph file: the other open analyses using it take the change
        over (keeping their own values) and are recomputed (unless the edit changes no result, e.g. a node moved)."""
        for other in self.engine.sharing(analysis):
            nodes_before = list(other.nodes)
            other.pull_graph()
            cards = self.cards_of(other)
            if list(other.nodes) != nodes_before:
                for card in cards:
                    card.rebuild()
            for i, card in enumerate(cards):
                card._after_change(recompute=recompute and i == 0, redraw=redraw)
        self._update_tab_titles()

    def _request_linking(self, analysis):
        """Recompute the open analyses that link to this one (it changed, or it was closed: its links read the
        file again, which no longer has the unsaved changes)."""
        path = analysis.path and os.path.abspath(analysis.path)
        for other in self.cards():
            if other.analysis is analysis or not path:
                continue
            for _, _, source in other.analysis.links():
                try:
                    if os.path.abspath(os.path.join(self.engine.link_folder(other.analysis), source.file)) == path:
                        self.runner.request(other.analysis, fresh=False)
                        break
                except Exception:
                    pass

    def _results_arrived(self, analysis):
        for card in self.cards_of(analysis):
            try:
                card.update_results()
            except Exception:
                traceback.print_exc()

    def recompute_all(self):
        self.engine.memory.clear()
        for card in self.cards():
            self.runner.request(card.analysis)

    def clear_cache(self):
        if self.project is None:
            return
        from tesselab import cache
        cache.clear(self.project.root)
        self.statusBar().showMessage("The disk cache is cleared; results are computed again when needed.", 5000)

    def view_changed(self, card):
        """Open groups and their order: saved with the analysis, but no unsaved change."""

    # --- linking ----------------------------------------------------------------------------------------------------
    def link_plots(self, card):
        """Plots of the same kind (node and plot) in the linked analyses of a tab zoom together."""
        tab = self.tab_of(card)
        if tab is None:
            return
        linked = tab.linked()
        for c in tab.cards:  # unlink everything first; the shared axes inside a plot stay
            for group in c.groups.values():
                for tile in group.tiles.values():
                    for panel in tile.view.panels.values():
                        panel.box.setXLink(None)
                        panel.box.setYLink(None)
                    tile.view.link_internal()
        if len(linked) < 2:
            return
        master = linked[0]
        for other in linked[1:]:
            for name, group in other.groups.items():
                node = other.analysis.nodes[name]["node"]
                twin = next((g for n, g in master.groups.items() if master.analysis.nodes[n]["node"] == node), None)
                if twin is None or group.cls is None:
                    continue
                for plot in group.cls.plots:
                    a, b = group.tiles.get(plot.id), twin.tiles.get(plot.id)
                    if a is None or b is None:
                        continue
                    for panel_name, panel in a.view.panels.items():
                        master_panel = b.view.panels.get(panel_name)
                        if master_panel is not None:
                            panel.box.setXLink(master_panel.box)
                            panel.box.setYLink(master_panel.box)

    def _cursor_moved(self, card, key, value, final):
        tab = self.tab_of(card)
        if tab is None or card not in tab.linked():
            return
        node_id, _, link = key.partition(":")
        for other in tab.linked():
            if other is card:
                continue
            for name, entry in other.analysis.nodes.items():
                if entry["node"] != node_id:
                    continue
                group = other.groups.get(name)
                if group is None or group.cls is None:
                    continue
                for interaction in group.cls.interactions:
                    if getattr(interaction, "link", None) == link:
                        other.set_param(name, interaction.param, value, final,
                                        coalesce=f"cursor:{name}:{interaction.param}", propagate=False)

    def _fold_changed(self, card, node_id, open_):
        tab = self.tab_of(card)
        if tab is None or card not in tab.linked():
            return
        for other in tab.linked():
            if other is not card:
                for name, entry in other.analysis.nodes.items():
                    if entry["node"] == node_id:
                        other.set_open(name, open_, emit=False)

    def others(self, card, node_id):
        """For plots over several analyses: (label, DrawContext) of the linked analyses' node of the same kind."""
        tab = self.tab_of(card)
        found = []
        for other in (tab.linked() if tab else []):
            results = self.runner.results(other.analysis)
            for name, entry in other.analysis.nodes.items():
                if entry["node"] == node_id and results.get(name) is not None and results[name].ok:
                    found.append((other.analysis.label, self.engine.draw_context(other.analysis, name, results[name])))
                    break
        return found

    # --- new analyses ----------------------------------------------------------------------------------------------
    def new_from_data(self, folder, templates):
        from tesselab.gui.dialogs import TemplateDialog
        from tesselab.templates import empty_analysis, new_analysis
        if self.project is None:
            message_box(self, "No project", "Open a project first (File → Open project…).", icon="info")
            return
        choices = [self.registry.template(t) for t in templates]
        if folder and len(choices) == 1:
            template_id, label = choices[0].full_id, os.path.basename(folder)
        else:
            dialog = TemplateDialog(self, self.registry, self.project.root, folder, choices or None)
            if not dialog.exec() or not dialog.choice:
                return
            template_id, folder, label = dialog.choice
        try:
            if template_id:
                analysis = new_analysis(self.registry, self.project, template_id, folder, label or None,
                                        graphs=self.engine.graphs)
            else:
                analysis = empty_analysis(self.project, label or "New analysis", graphs=self.engine.graphs)
            taken = {os.path.abspath(c.analysis.path) for c in self.cards() if c.analysis.path}
            while analysis.path in taken:  # an open analysis that is not saved yet has the same name
                analysis.path = analysis.path.removesuffix(".json") + " (new).json"
        except Exception as error:
            traceback.print_exc()
            message_box(self, "The template failed", f"{type(error).__name__}: {error}", icon="warning")
            return
        if not analysis.doc["view"].get("open"):  # the template says which groups are open; else the last node
            analysis.doc["view"]["open"] = list(analysis.doc["view"]["order"])[-1:]
        card = self.show_analysis(analysis)
        card.undo_steps.clear()
        card.saved_content = None  # new: unsaved until saved
        card.update_header()
        self._update_tab_titles()

    def choose_analysis(self):
        start = self.project.folder("analyses") if self.project else os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open analysis", start, "Analyses (*.json)")
        if path:
            self.open_path(path)

    # --- saving and export ------------------------------------------------------------------------------------------
    def save_active(self):
        if self.active_card is not None:
            self.save_card(self.active_card)

    def save_all(self):
        saved = set()
        for card in self.cards():
            if card.is_unsaved() and id(card.analysis) not in saved:
                saved.add(id(card.analysis))
                self.save_card(card)

    def save_card(self, card) -> bool:
        analysis = card.analysis
        if analysis.read_only:
            message_box(self, "Read-only", "This analysis was written by a newer Tesselab and cannot be saved here.",
                        icon="warning")
            return False
        if analysis.path is None:
            if self.project is None:
                path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save analysis", os.getcwd(), "*.json")
                if not path:
                    return False
                analysis.path = path
            else:
                analysis.path = self.project.new_analysis_path(analysis.label)
        if analysis.changed_on_disk():
            files = os.path.basename(analysis.path)
            if analysis.graph is not None and analysis.graph.changed_on_disk():
                files = f"The graph file {os.path.basename(analysis.graph.path)}"
            if message_box(self, "Changed elsewhere", f"{files} was changed outside this window since it was opened "
                           f"or saved here (another Tesselab window or a sync client). Saving overwrites those "
                           f"changes.", ("Overwrite", "Cancel"), default=1, icon="warning") != 0:
                return False
        try:
            self.engine.remember_links(analysis)
            self.engine.store_results(analysis, self.runner.results(analysis))
            analysis.doc["provenance"] = provenance(self.registry, analysis)
            analysis.save()
        except Exception as error:
            message_box(self, "Not saved", f"{type(error).__name__}: {error}", icon="warning")
            return False
        self.engine.register(analysis)
        card.mark_saved()
        for other in self.engine.sharing(analysis):  # their graph is saved now too
            for c in self.cards_of(other):
                c.update_header()
                c.saved_changed.emit(c)
        self.runner.request(analysis)  # the link warnings are gone now
        self.explorer.refresh()
        self._update_explorer()
        self.statusBar().showMessage(f"Saved {analysis.path}", 4000)
        return True

    def export_card(self, card):
        if card is None:
            return
        from tesselab import export
        default = os.path.join(self.project.root if self.project else os.getcwd(), "exports",
                               f"{card.analysis.label}.pdf")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export (PDF, with a JSON of the plotted data next to "
                                                              "it)", default, "PDF (*.pdf)")
        if not path:
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            images = {}
            for name, group in card.groups.items():  # level 2 plots: their picture
                for plot_id, tile in group.tiles.items():
                    if tile.widget is not None:
                        images[(name, plot_id)] = tile.widget.grab().toImage()
            export.export_analysis(self.engine, card.analysis, path, results=self.runner.results(card.analysis),
                                   images=images)
            self.statusBar().showMessage(f"Exported {path}", 5000)
        except Exception as error:
            traceback.print_exc()
            message_box(self, "Export failed", f"{type(error).__name__}: {error}", icon="warning")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # --- sessions ---------------------------------------------------------------------------------------------------
    def save_session(self):
        start = self.project.folder("sessions") if self.project else os.getcwd()
        os.makedirs(start, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save window", os.path.join(start, "window.tesselab-session"),
                                                        "Saved windows (*.tesselab-session)")
        if not path:
            return
        if not path.endswith(".tesselab-session"):
            path += ".tesselab-session"
        self.write_session(path)
        self.explorer.refresh()
        self.statusBar().showMessage(f"Saved the window to {path}", 4000)

    def write_session(self, path):
        base = os.path.dirname(os.path.abspath(path))
        tabs = []
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            tabs.append({"name": tab.name,
                         "analyses": [os.path.relpath(c.analysis.path, base).replace("\\", "/")
                                      for c in tab.cards if c.analysis.path],
                         "linked": [os.path.relpath(c.analysis.path, base).replace("\\", "/")
                                    for c in tab.linked() if c.analysis.path]})
        write_json(path, {"tesselab_session": 1, "tabs": tabs, "current": self.tabs.currentIndex()})

    def choose_session(self):
        start = self.project.folder("sessions") if self.project else os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open saved window", start,
                                                        "Saved windows (*.tesselab-session)")
        if path:
            self.open_path(path)

    def open_session(self, path):
        try:
            with open(path, encoding="utf-8") as f:
                session = json.load(f)
        except (OSError, json.JSONDecodeError) as error:
            message_box(self, "Cannot open", str(error), icon="warning")
            return
        if not self.close_all_tabs():
            return
        base = os.path.dirname(os.path.abspath(path))
        first = True
        for entry in session.get("tabs", []):
            tab = self.current_tab() if first else self.new_tab()
            first = False
            tab.name = entry.get("name")
            linked = {os.path.normpath(os.path.join(base, p)) for p in entry.get("linked", [])}
            for relative in entry.get("analyses", []):
                full = os.path.normpath(os.path.join(base, relative))
                if os.path.isfile(full):
                    card = self.open_analysis(full, tab)
                    if card is not None and full in linked:
                        card.link_button.setChecked(True)
        self.tabs.setCurrentIndex(min(session.get("current", 0), self.tabs.count() - 1))
        self._update_tab_titles()

    # --- modules ----------------------------------------------------------------------------------------------------
    def show_modules(self):
        from tesselab.gui.dialogs import ModulesDialog
        ModulesDialog(self, self.registry).exec()

    def reload_modules(self):
        """Start the modules afresh (Ctrl+Shift+R): new worker processes, every module read and imported again, the
        results in memory forgotten. Following changes on disk does less (modules_changed)."""
        worker.shutdown()
        self.runner.drop_pending()
        self.registry.reload()
        self.engine.memory.clear()
        self.runner.warm(self.registry)
        for card in self.cards():
            card.rebuild()
            self.runner.request(card.analysis)
        self.explorer.refresh()
        self.follow.schedule()
        self.statusBar().showMessage("Modules reloaded.", 4000)

    def set_follow(self, on):
        """Modules → Follow changes on disk (watch.py): remembered."""
        self.settings.setValue("follow_disk", on)
        self.follow.set_on(on)

    def modules_changed(self, ids):
        """Modules whose code changed on disk (Registry.refresh has read them again): the cards that use them are
        made again, and every open analysis is requested (one that links to an analysis using them depends on them
        too). The fingerprints hold each module's code hash, so only the nodes of the changed modules run again, and
        what depends on their results if those changed; the rest is found in memory. The workers keep running: they
        import a module again when a task names another version of its code (worker.node_class)."""
        self.runner.stale.clear()
        for card in self.cards():
            if any(entry["node"].rpartition("/")[0] in ids for entry in card.analysis.nodes.values()):
                card.rebuild()
        for analysis in {id(card.analysis): card.analysis for card in self.cards()}.values():
            self.runner.request(analysis)
        self.explorer.refresh()

    def data_changed(self, analyses) -> list:
        """Files that these analyses read changed on disk: they are computed again (the new file stamps give the
        nodes that read them new fingerprints), and so are the analyses that link to them. Returns those still open."""
        open_ = {id(card.analysis) for card in self.cards()}
        analyses = [a for a in analyses if id(a) in open_]
        for analysis in analyses:
            self.runner.request(analysis)
            self._request_linking(analysis)
        return analyses

    def new_module(self):
        from tesselab.scaffold import new_module
        folder = QtWidgets.QFileDialog.getSaveFileName(self, "New module (a folder)", os.path.join(
            user_module_folder(), "my_module"))[0]
        if not folder:
            return
        try:
            new_module(folder)
        except Exception as error:
            message_box(self, "Not created", str(error), icon="warning")
            return
        self.reload_modules()
        message_box(self, "Module created",
                    f"A working minimal module is in {folder}. Edit it (AGENTS.md in it explains how) and run "
                    f"'tesselab check {folder}'. The window follows your edits (Modules → Follow changes on disk; "
                    f"without it: Modules → Reload modules).", icon="info")

    def _open_folder(self, folder):
        os.makedirs(folder, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))

    def show_protocol(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AGENTS.md")
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    def about(self):
        from tesselab import __version__
        message_box(self, "About Tesselab",
                    f"Tesselab {__version__}\nA viewer for scientific data whose analyses are graphs of nodes, "
                    f"supplied by modules.", icon="info")

    # --- closing ----------------------------------------------------------------------------------------------------
    def closeEvent(self, event):
        if not self._confirm_unsaved(list(self.cards())):
            event.ignore()
            return
        self.follow.set_on(False)
        worker.shutdown()
        super().closeEvent(event)
