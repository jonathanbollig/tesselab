"""
The explorer on the left: the project's analyses, its data that a template recognizes (offered as "new"), and its
saved windows (sessions). A double click opens an analysis or a session, or makes a new analysis from a data folder;
the right mouse button offers the rest. Ctrl+F filters.
"""
import json
import os

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from tesselab import project as projects
from tesselab.gui import theme

ROLE_KIND, ROLE_PATH, ROLE_TEMPLATES = Qt.ItemDataRole.UserRole, Qt.ItemDataRole.UserRole + 1, Qt.ItemDataRole.UserRole + 2


def analysis_label(path) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("label") or os.path.splitext(os.path.basename(path))[0]
    except Exception:
        return os.path.splitext(os.path.basename(path))[0] + " (unreadable)"


class Explorer(QtWidgets.QWidget):
    open_analysis = QtCore.Signal(str)
    new_from_data = QtCore.Signal(str, list)   # folder, templates
    open_session = QtCore.Signal(str)

    def __init__(self):
        super().__init__(objectName="sidebar")
        self.project, self.registry, self.open_paths = None, None, set()
        layout = QtWidgets.QVBoxLayout(self)
        # inside the panel (paintEvent), which leaves the chrome around it (theme.GAP); right of it is the splitter
        gap = theme.GAP
        layout.setContentsMargins(gap + 10, gap + 8, 6, gap + 8)
        layout.setSpacing(6)
        layout.addWidget(QtWidgets.QLabel("PROJECT", objectName="project_label"))
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(4)
        self.title = QtWidgets.QLabel("No project", objectName="project_name")
        head.addWidget(self.title, 1)
        refresh = theme.icon_button("refresh", "Look for new analyses and data")
        refresh.clicked.connect(self.refresh)
        head.addWidget(refresh)
        layout.addLayout(head)
        self.filter = QtWidgets.QLineEdit(placeholderText="Filter (Ctrl+F)")
        self.search_icon = self.filter.addAction(theme.icon("search", theme.color("muted")),
                                                 QtWidgets.QLineEdit.ActionPosition.LeadingPosition)
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.tree.itemDoubleClicked.connect(self._activated)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        layout.addWidget(self.tree)
        self.setMinimumWidth(200)

    def paintEvent(self, event):
        """A rounded panel in the page color, its top level with the tabs' tops."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(theme.color("page")))
        gap = theme.GAP
        painter.drawRoundedRect(QtCore.QRectF(self.rect()).adjusted(gap, gap, 0, -gap), theme.PANEL_R, theme.PANEL_R)
        painter.end()

    def set_project(self, project, registry):
        self.project, self.registry = project, registry
        self.title.setText(project.name if project else "No project")
        self.title.setToolTip(project.root if project else "")
        self.refresh()

    def set_open(self, paths):
        self.open_paths = {os.path.abspath(p) for p in paths if p}
        self._mark_open()

    def update_icons(self):
        """After a change of theme (the buttons are recolored by theme.recolor_icons)."""
        self.search_icon.setIcon(theme.icon("search", theme.color("muted")))

    def refresh(self):
        self.tree.clear()
        if self.project is None:
            item = QtWidgets.QTreeWidgetItem(["File → Open project… to start"])
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.tree.addTopLevelItem(item)
            return
        bold = QtGui.QFont()
        bold.setBold(True)
        sessions = self.project.sessions()
        if sessions:
            group = QtWidgets.QTreeWidgetItem(["Saved windows"])
            group.setFont(0, bold)
            for path in sessions:
                item = QtWidgets.QTreeWidgetItem([os.path.splitext(os.path.basename(path))[0]])
                item.setData(0, ROLE_KIND, "session")
                item.setData(0, ROLE_PATH, path)
                group.addChild(item)
            self.tree.addTopLevelItem(group)
            group.setExpanded(True)
        group = QtWidgets.QTreeWidgetItem(["Analyses"])
        group.setFont(0, bold)
        for path in self.project.analyses():
            item = QtWidgets.QTreeWidgetItem([analysis_label(path)])
            item.setData(0, ROLE_KIND, "analysis")
            item.setData(0, ROLE_PATH, path)
            item.setToolTip(0, path)
            group.addChild(item)
        self.tree.addTopLevelItem(group)
        group.setExpanded(True)
        templates = list(self.registry.templates().values()) if self.registry else []
        found = projects.find_data(self.project.root, templates,
                                   detect=lambda t, folder: self.registry.function(t.module, t.detect)(folder))
        if found:
            group = QtWidgets.QTreeWidgetItem(["New data"])
            group.setFont(0, bold)
            group.setToolTip(0, "Data folders a module's template recognizes: double-click to make an analysis.")
            for folder, matching in found:
                item = QtWidgets.QTreeWidgetItem([os.path.relpath(folder, self.project.root)])
                item.setData(0, ROLE_KIND, "data")
                item.setData(0, ROLE_PATH, folder)
                item.setData(0, ROLE_TEMPLATES, [t.full_id for t in matching])
                item.setToolTip(0, "Templates: " + ", ".join(t.name for t in matching))
                item.setForeground(0, QtGui.QColor(theme.color("muted")))
                group.addChild(item)
            self.tree.addTopLevelItem(group)
            group.setExpanded(True)
        self._mark_open()
        self._apply_filter(self.filter.text())

    def _items(self):
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            for j in range(group.childCount()):
                yield group, group.child(j)

    def _mark_open(self):
        for _, item in self._items():
            path = item.data(0, ROLE_PATH)
            font = item.font(0)
            font.setBold(bool(path) and os.path.abspath(path) in self.open_paths)
            item.setFont(0, font)
            if path and os.path.abspath(path) in self.open_paths:
                item.setForeground(0, QtGui.QColor(theme.color("accent")))
            elif item.data(0, ROLE_KIND) == "analysis":
                item.setForeground(0, QtGui.QColor(theme.color("ink")))

    def _apply_filter(self, text):
        words = text.lower().split()
        for group, item in self._items():
            item.setHidden(not all(w in item.text(0).lower() for w in words))

    def focus_filter(self):
        self.filter.setFocus()
        self.filter.selectAll()

    def _activated(self, item):
        kind, path = item.data(0, ROLE_KIND), item.data(0, ROLE_PATH)
        if kind == "analysis":
            self.open_analysis.emit(path)
        elif kind == "session":
            self.open_session.emit(path)
        elif kind == "data":
            self.new_from_data.emit(path, item.data(0, ROLE_TEMPLATES) or [])

    def _menu(self, position):
        item = self.tree.itemAt(position)
        if item is None or not item.data(0, ROLE_KIND):
            return
        kind, path = item.data(0, ROLE_KIND), item.data(0, ROLE_PATH)
        menu = QtWidgets.QMenu(self)
        if kind == "analysis":
            menu.addAction("Open", lambda: self.open_analysis.emit(path))
        elif kind == "session":
            menu.addAction("Open", lambda: self.open_session.emit(path))
        elif kind == "data":
            for template_id in item.data(0, ROLE_TEMPLATES) or []:
                template = self.registry.template(template_id)
                menu.addAction(f"New analysis: {template.name}", lambda t=template_id: self.new_from_data.emit(path, [t]))
        menu.addAction("Show in the file manager",
                       lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(
                           path if os.path.isdir(path) else os.path.dirname(path))))
        menu.exec(self.tree.viewport().mapToGlobal(position))
