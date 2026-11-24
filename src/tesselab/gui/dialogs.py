"""
Dialogs: linking an input to another analysis, the command palette, the list of modules, a new analysis from a
template.
"""
import json
import os

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from tesselab.analysis import Analysis, Source


class LinkDialog(QtWidgets.QDialog):
    """Choose an output of another analysis of the project as an input (a live link)."""

    def __init__(self, card, project, analysis, engine):
        super().__init__(card)
        self.setWindowTitle("Link to another analysis")
        self.project, self.analysis, self.engine, self.source = project, analysis, engine, None
        layout = QtWidgets.QFormLayout(self)
        self.files = QtWidgets.QComboBox()
        self.instances = QtWidgets.QComboBox()
        self.outputs = QtWidgets.QComboBox()
        layout.addRow(QtWidgets.QLabel("The input always uses the current result of the other analysis; when it "
                                       "changes, this analysis shows a warning until it is saved again.",
                                       wordWrap=True, objectName="hint"))
        layout.addRow("Analysis", self.files)
        layout.addRow("Node", self.instances)
        layout.addRow("Output", self.outputs)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                             QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.others = {}
        for path in (project.analyses() if project else []):
            if analysis.path and os.path.abspath(path) == os.path.abspath(analysis.path):
                continue
            try:
                other = engine.open.get(os.path.abspath(path)) or Analysis.load(path)
            except Exception:
                continue
            self.others[path] = other
            self.files.addItem(other.label, path)
        self.files.currentIndexChanged.connect(self._fill_instances)
        self.instances.currentIndexChanged.connect(self._fill_outputs)
        self._fill_instances()

    def _fill_instances(self):
        self.instances.clear()
        other = self.others.get(self.files.currentData())
        if other is None:
            return
        for name, entry in other.nodes.items():
            try:
                cls = self.engine.registry.node_class(entry["node"])
                text = f"{cls.name} ({name})"
            except Exception:
                text = name
            self.instances.addItem(text, name)

    def _fill_outputs(self):
        self.outputs.clear()
        other = self.others.get(self.files.currentData())
        name = self.instances.currentData()
        if other is None or name is None:
            return
        try:
            cls = self.engine.registry.node_class(other.nodes[name]["node"])
            for output in cls.output_names():
                self.outputs.addItem(output, output)
        except Exception:
            pass

    def _accept(self):
        path, name, output = self.files.currentData(), self.instances.currentData(), self.outputs.currentData()
        if path and name and output:
            self.source = Source(name, output, self.engine.link_path(self.analysis, path))
            self.accept()


class CommandPalette(QtWidgets.QDialog):
    """Every command of the menus, found by typing part of its name (Ctrl+Shift+P)."""

    def __init__(self, window, actions):
        super().__init__(window, Qt.WindowType.Popup)
        self.actions = [a for a in actions if a.text() and not a.isSeparator() and not a.menu()]
        self.resize(520, 380)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # the frame inside has round corners
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        frame = QtWidgets.QFrame(objectName="palette")
        outer.addWidget(frame)
        layout = QtWidgets.QVBoxLayout(frame)
        self.filter = QtWidgets.QLineEdit(placeholderText="Type a command …")
        self.list = QtWidgets.QListWidget()
        layout.addWidget(self.filter)
        layout.addWidget(self.list)
        self.filter.textChanged.connect(self._fill)
        self.filter.returnPressed.connect(self._run)
        self.list.itemActivated.connect(lambda item: self._run())
        self._fill("")
        geometry = window.geometry()
        self.move(geometry.center().x() - 260, geometry.top() + 80)

    def _fill(self, text):
        self.list.clear()
        words = text.lower().split()
        for action in self.actions:
            name = action.text().replace("&", "")
            if all(w in name.lower() for w in words) and action.isEnabled():
                item = QtWidgets.QListWidgetItem(name + (f"    {action.shortcut().toString()}"
                                                         if not action.shortcut().isEmpty() else ""))
                item.setData(Qt.ItemDataRole.UserRole, action)
                self.list.addItem(item)
        self.list.setCurrentRow(0)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
            self.list.keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def _run(self):
        item = self.list.currentItem()
        if item is not None:
            action = item.data(Qt.ItemDataRole.UserRole)
            self.close()
            action.trigger()


class ModulesDialog(QtWidgets.QDialog):
    """The modules Tesselab found: where, which version, which nodes, and why one cannot be used."""

    def __init__(self, window, registry):
        super().__init__(window)
        self.setWindowTitle("Modules")
        self.resize(760, 480)
        layout = QtWidgets.QVBoxLayout(self)
        tree = QtWidgets.QTreeWidget()
        tree.setHeaderLabels(["Module / node", "Version", "From", "Notes"])
        tree.setColumnWidth(0, 300)
        for module in sorted(registry.modules.values(), key=lambda m: m.name):
            item = QtWidgets.QTreeWidgetItem([module.name, module.version, module.source,
                                              "; ".join(module.problems) or module.description])
            item.setToolTip(0, module.folder)
            item.setToolTip(3, "; ".join(module.problems) or module.description)
            for node in module.nodes.values():
                QtWidgets.QTreeWidgetItem(item, [node.id, "", "", node.class_spec])
            for template in module.templates.values():
                QtWidgets.QTreeWidgetItem(item, [f"template: {template.name}", "", "", template.help])
            tree.addTopLevelItem(item)
        for error in registry.errors:
            tree.addTopLevelItem(QtWidgets.QTreeWidgetItem(["(could not be read)", "", "", error]))
        layout.addWidget(tree)
        from tesselab.modules import user_module_folder
        hint = QtWidgets.QLabel(f"Your own modules go into {user_module_folder()}. Modules → Reload modules picks up "
                                f"changes without restarting.", wordWrap=True, objectName="hint")
        hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(hint)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class TemplateDialog(QtWidgets.QDialog):
    """Choose a template (and a data folder or file) for a new analysis."""

    def __init__(self, window, registry, project_root, folder=None, templates=None):
        super().__init__(window)
        self.setWindowTitle("New analysis from a template")
        self.resize(560, 300)
        self.registry, self.project_root, self.choice = registry, project_root, None
        layout = QtWidgets.QFormLayout(self)
        self.templates = QtWidgets.QComboBox()
        for template in (templates or registry.templates().values()):
            self.templates.addItem(f"{template.name}   ({template.module.name})", template.full_id)
            self.templates.setItemData(self.templates.count() - 1, template.help, Qt.ItemDataRole.ToolTipRole)
        self.templates.addItem("Empty analysis", "")
        layout.addRow("Template", self.templates)
        row = QtWidgets.QHBoxLayout()
        self.folder = QtWidgets.QLineEdit(folder or project_root or "")
        browse = QtWidgets.QToolButton(text="…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        layout.addRow("Data folder", row)
        self.label = QtWidgets.QLineEdit(os.path.basename(folder) if folder else "")
        layout.addRow("Label", self.label)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                             QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _browse(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Data folder", self.folder.text())
        if path:
            self.folder.setText(path)
            if not self.label.text():
                self.label.setText(os.path.basename(path))

    def _accept(self):
        self.choice = (self.templates.currentData(), self.folder.text().strip(), self.label.text().strip())
        self.accept()
