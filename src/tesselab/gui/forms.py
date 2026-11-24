"""
Forms built from parameter declarations (params.py): one row per parameter, its input field chosen by type.

ParamForm emits changed(name, value) when the user changes a value (the value already checked, in its stored units),
and shows the error in the field when a typed text is not valid (the value is then not changed). A parameter bound
to an analysis variable shows the variable's name next to its label; changing it changes the variable. set_values()
updates the fields without signals (after undo, a dragged cursor, ...).

With a shared graph file, a parameter this analysis sets itself (its own value, not the graph's) has an accent dot
after its label; right-click the label to take the graph's value again or to make this value the graph's.
"""
import os

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from tesselab import params as P, units
from tesselab.gui import theme


class ValueEdit(QtWidgets.QLineEdit):
    """A line edit for a value of the analysis. Undo and redo keys undo the typing while there is typing to undo,
    then reach the window (the analysis's undo). The field is given its text anew after Enter, which empties its own
    history, and Qt's line edit took the keys anyway and did nothing: Ctrl+Z right after typing a value did not
    undo it (the user, 2026-11-23).

    chars: a short field, as wide as that many digits; it does not grow, and shrinks to five digits only where there is
    no room (a narrow inspector). None: as wide as the layout gives it."""

    def __init__(self, chars=None):
        super().__init__()
        self.chars = chars
        if chars:
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)

    def _width(self, chars):
        return self.fontMetrics().horizontalAdvance("0" * chars) + 18  # and the padding

    def sizeHint(self):
        hint = super().sizeHint()
        if self.chars:
            hint.setWidth(self._width(self.chars))
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        if self.chars:
            hint.setWidth(self._width(min(self.chars, 5)))
        return hint

    def event(self, event):
        if event.type() == QtCore.QEvent.Type.ShortcutOverride and (
                event.matches(QtGui.QKeySequence.StandardKey.Undo) and not self.isUndoAvailable()
                or event.matches(QtGui.QKeySequence.StandardKey.Redo) and not self.isRedoAvailable()):
            event.ignore()  # not this field's: the shortcut of the window takes it
            return True
        return super().event(event)


class Field(QtWidgets.QWidget):
    """The input of one parameter."""
    changed = QtCore.Signal(object)

    def __init__(self, param, project_root=None):
        super().__init__()
        self.param, self.project_root, self._value = param, project_root, None
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        chars = field_chars(param)
        self.wide = chars is None  # a row of its own in the form
        self.editor = None
        if isinstance(param, P.Bool):
            self.editor = QtWidgets.QCheckBox()
            self.editor.toggled.connect(lambda on: self._emit(on))
        elif isinstance(param, P.Choice):
            self.editor = QtWidgets.QComboBox()
            for key, label in param.options.items():
                self.editor.addItem(str(label), key)
            self.editor.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
            self.editor.setMinimumWidth(self.editor.fontMetrics().horizontalAdvance("0" * chars) + 18)
            self.editor.activated.connect(lambda i: self._emit(self.editor.itemData(i)))
        elif isinstance(param, P.Steps):
            self.editor = StepsEditor(param, project_root)
            self.editor.changed.connect(self._emit)
        else:
            self.editor = ValueEdit(chars)
            self.editor.editingFinished.connect(self._typed)
        layout.addWidget(self.editor, 1)
        if isinstance(param, (P.Path, P.Paths)):
            browse = QtWidgets.QToolButton(text="…", toolTip="Choose")
            browse.clicked.connect(self._browse)
            layout.addWidget(browse)
        if not self.wide:  # as wide as its editor, never wider
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setToolTip(param.help)

    def set_value(self, value):
        self._value = value
        self.editor.blockSignals(True)
        if isinstance(self.param, P.Bool):
            self.editor.setChecked(bool(value))
        elif isinstance(self.param, P.Choice):
            index = self.editor.findData(value)
            self.editor.setCurrentIndex(max(0, index))
        elif isinstance(self.param, P.Steps):
            self.editor.set_value(value or [])
        else:
            if not self.editor.hasFocus() or self.editor.text() != self.param.to_text(value):
                self.editor.setText(self.param.to_text(value))
            self._mark_error("")
        self.editor.blockSignals(False)

    def _emit(self, value):
        try:
            value = self.param.check(value)
        except ValueError as error:
            self._mark_error(str(error))
            return
        if value != self._value:
            self._value = value
            self.changed.emit(value)

    def _typed(self):
        try:
            value = self.param.from_text(self.editor.text())
        except ValueError as error:
            self._mark_error(str(error))
            return
        self._mark_error("")
        if value != self._value:
            self._value = value
            self.changed.emit(value)
        self.editor.setText(self.param.to_text(value))

    def _mark_error(self, message):
        if not isinstance(self.editor, QtWidgets.QLineEdit):
            return
        self.editor.setStyleSheet(f"border-color: {theme.role_color('error')};" if message else "")
        self.editor.setToolTip(message or self.param.help)

    def _browse(self):
        start = self.project_root or os.getcwd()
        if isinstance(self.param, P.Path) and self.param.folder:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, self.param.label, start)
            paths = [path] if path else []
        elif isinstance(self.param, P.Path):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, self.param.label, start, self.param.filter)
            paths = [path] if path else []
        else:
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, self.param.label, start, self.param.filter)
        if not paths:
            return
        if self.project_root:
            paths = [_relative(p, self.project_root) for p in paths]
        self._emit(paths[0] if isinstance(self.param, P.Path) else paths)


def field_chars(param) -> int | None:
    """How many characters wide the field of a parameter is: numbers and choices short, a range, a short list or a
    word a bit longer; None for the wide ones (paths, lists of ranges or points, multi-line text, steps), which take
    a row of their own (the user, 2026-11-23: the fields as wide as the card were too long)."""
    if isinstance(param, (P.Path, P.Paths, P.Ranges, P.Points, P.Steps)) or isinstance(param, P.Str) and param.multiline:
        return None
    if isinstance(param, (P.Str, P.Range, P.Ints, P.Floats)):
        return 18
    return 10


def label_text(param) -> str:
    """The label of a parameter with its unit in brackets: "thickness [mm]"."""
    return units.label(param.label, getattr(param, "unit", None), getattr(param, "display", None))


def _relative(path, root):
    relative = os.path.relpath(path, root)
    return path.replace("\\", "/") if relative.startswith("..") else relative.replace("\\", "/")


class StepsEditor(QtWidgets.QWidget):
    """An ordered list of processing steps: switch each on or off, add, remove, move; the selected step's own
    parameters below the list."""
    changed = QtCore.Signal(object)

    def __init__(self, param, project_root=None):
        super().__init__()
        self.param, self.project_root, self.value = param, project_root, []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumHeight(130)
        self.list.itemChanged.connect(self._toggled)
        self.list.currentRowChanged.connect(self._show_step)
        layout.addWidget(self.list)
        buttons = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add ▸")
        menu = QtWidgets.QMenu(add)
        for kind, step in param.kinds.items():
            menu.addAction(step.label, lambda k=kind: self._add(k))
        add.setMenu(menu)
        buttons.addWidget(add)
        for text, slot in (("Remove", self._remove), ("Up", lambda: self._move(-1)), ("Down", lambda: self._move(1))):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.step_form = None
        self.step_area = QtWidgets.QVBoxLayout()
        layout.addLayout(self.step_area)

    def set_value(self, value):
        row = self.list.currentRow()
        self.value = [dict(v) for v in value]
        self.list.blockSignals(True)
        self.list.clear()
        for entry in self.value:
            item = QtWidgets.QListWidgetItem(self.param.kinds[entry["kind"]].label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if entry["on"] else Qt.CheckState.Unchecked)
            self.list.addItem(item)
        self.list.blockSignals(False)
        self.list.setCurrentRow(min(row, len(self.value) - 1) if row >= 0 else (0 if self.value else -1))
        self._show_step(self.list.currentRow())

    def _emit(self):
        self.changed.emit([dict(v) for v in self.value])

    def _toggled(self, item):
        row = self.list.row(item)
        self.value[row]["on"] = item.checkState() == Qt.CheckState.Checked
        self._emit()

    def _add(self, kind):
        self.value.append(self.param.kinds[kind].defaults())
        self._emit()

    def _remove(self):
        row = self.list.currentRow()
        if row >= 0:
            del self.value[row]
            self._emit()

    def _move(self, delta):
        row = self.list.currentRow()
        if 0 <= row + delta < len(self.value) and row >= 0:
            self.value[row], self.value[row + delta] = self.value[row + delta], self.value[row]
            self._emit()
            self.list.setCurrentRow(row + delta)

    def _show_step(self, row):
        if self.step_form is not None:
            self.step_form.setParent(None)
            self.step_form.deleteLater()
            self.step_form = None
        if row < 0 or row >= len(self.value):
            return
        entry = self.value[row]
        step = self.param.kinds[entry["kind"]]
        if not step.params:
            return
        self.step_form = ParamForm(step.params, {p.name: entry.get(p.name) for p in step.params},
                                   project_root=self.project_root)
        self.step_form.changed.connect(lambda name, value, r=row: self._step_changed(r, name, value))
        self.step_area.addWidget(self.step_form)

    def _step_changed(self, row, name, value):
        self.value[row][name] = value
        self._emit()


class ParamForm(QtWidgets.QWidget):
    """A form for a list of parameters. values: {name: value or {"var": name}}; variables: {name: value}.

    Short fields (numbers, choices, ranges) stand next to each other, label before field, in as many columns as fit,
    like text that wraps; the labels and fields of a column are aligned. Wide fields (paths, steps, ...) and section
    headings take a row of their own. The columns are chosen anew when the form's width changes (as ReflowGrid
    does): its minimum width is that of one column."""
    changed = QtCore.Signal(str, object)        # parameter name, new value
    var_changed = QtCore.Signal(str, object)    # variable name, new value
    bind_requested = QtCore.Signal(str, object)  # parameter name, variable name (None: unbind)
    reset_requested = QtCore.Signal(str)         # parameter name: take the graph's value again
    promote_requested = QtCore.Signal(str)       # parameter name: this value becomes the graph's
    SPACING, GAP = 8, 28  # between a label and its field (the label's margin), between two columns (a grid column)

    def __init__(self, params, values, variables=None, project_root=None, advanced=False, only=None):
        super().__init__()
        self.params = [p for p in params if (only is None or p.name in only)]
        self.fields, self.labels = {}, {}
        self.variables = dict(variables or {})
        self.bound = {}
        self.own = None  # with a shared graph: {parameter: (found, graph's value)} of this analysis's own values
        self.items = []  # in their order: a section heading (QLabel), or (label, field) of a shown parameter
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(0)  # the spacing is in the labels and the gap columns: no hidden spacing,
        # which Qt adds only next to columns that are not empty, so _needed knows the width exactly
        self.grid.setVerticalSpacing(4)
        group = None
        for p in self.params:
            if p.group and p.group != group:
                group = p.group
                self.items.append(QtWidgets.QLabel(group.upper(), objectName="section"))
            label = QtWidgets.QLabel(label_text(p), objectName="field", toolTip=p.help)
            label.setContentsMargins(0, 0, self.SPACING, 0)
            label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            label.customContextMenuRequested.connect(lambda pos, p=p, label=label: self._label_menu(p, label, pos))
            field = Field(p, project_root)
            field.changed.connect(lambda value, p=p: self._field_changed(p, value))
            self.fields[p.name], self.labels[p.name] = field, label
            if p.advanced and not advanced:
                for widget in (label, field):
                    widget.setParent(self)
                    widget.setVisible(False)
            else:
                self.items.append((label, field))
        self.columns = 0
        self._place(1)
        self.set_values(values, variables)

    # --- arranging the fields in columns ----------------------------------------------------------------------------
    def _place(self, columns):
        """Labels and fields into the grid: a column of the form is three of the grid (label, field, gap); the last
        gap stretches, so the fields stay as wide as they are."""
        for item in self.items:
            for widget in (item if isinstance(item, tuple) else (item,)):
                self.grid.removeWidget(widget)
        last = 3 * columns - 1
        row = column = 0
        for item in self.items:
            wide = not isinstance(item, tuple) or item[1].wide
            if wide and column:
                row, column = row + 1, 0
            if not isinstance(item, tuple):  # a section heading
                self.grid.addWidget(item, row, 0, 1, last + 1)
                row += 1
                continue
            label, field = item
            steps = isinstance(field.param, P.Steps)
            self.grid.addWidget(label, row, 3 * column, Qt.AlignmentFlag.AlignTop if steps else
                                Qt.AlignmentFlag.AlignVCenter)
            if wide:
                self.grid.addWidget(field, row, 1, 1, last)
                row += 1
                continue
            self.grid.addWidget(field, row, 3 * column + 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            column += 1
            if column == columns:
                row, column = row + 1, 0
        for c in range(max(3 * columns, 3 * self.columns)):
            self.grid.setColumnMinimumWidth(c, self.GAP if c % 3 == 2 and c < last else 0)
            self.grid.setColumnStretch(c, 1 if c == last else 0)
        self.columns = columns

    def _needed(self, columns, minimum=False) -> int:
        """The width the form takes with `columns` columns: with its fields as wide as they want to be, or (minimum)
        as narrow as they can be."""
        labels, fields, wide = [0] * columns, [0] * columns, 0
        column = 0
        for item in self.items:
            if not isinstance(item, tuple):
                column = 0
                continue
            label, field = item
            if field.wide:
                labels[0] = max(labels[0], label.sizeHint().width())
                wide = max(wide, field.minimumSizeHint().width())
                column = 0
                continue
            labels[column] = max(labels[column], label.sizeHint().width())
            hint = field.minimumSizeHint() if minimum else field.sizeHint()
            fields[column] = max(fields[column], hint.width())
            column = (column + 1) % columns
        used = sum(1 for f in fields if f)
        return max(sum(labels) + sum(fields) + max(used - 1, 0) * self.GAP, labels[0] + wide)

    def _fit(self, width):
        short = sum(1 for item in self.items if isinstance(item, tuple) and not item[1].wide)
        columns = 1
        while columns < short and self._needed(columns + 1) <= width:
            columns += 1
        if columns != self.columns:
            self._place(columns)
            self.grid.activate()
            self.updateGeometry()

    def resizeEvent(self, event):
        self._fit(event.size().width())
        super().resizeEvent(event)

    def minimumSizeHint(self):
        return QtCore.QSize(self._needed(1, minimum=True), super().minimumSizeHint().height())

    # --- values -----------------------------------------------------------------------------------------------------
    def set_values(self, values, variables=None, own=None):
        """own: None without a shared graph file, else {parameter: (found, graph's value)} of the parameters this
        analysis sets itself."""
        if variables is not None:
            self.variables = dict(variables)
        self.own = own
        accent = theme.current()["accent"]
        relabeled = False
        for p in self.params:
            value = values.get(p.name, p.default_value())
            field, label = self.fields[p.name], self.labels[p.name]
            if isinstance(value, dict) and set(value) == {"var"}:
                self.bound[p.name] = value["var"]
                text = f"{label_text(p)} → {value['var']}"
                label.setToolTip(f"Bound to the analysis variable {value['var']!r}: changing it here changes every "
                                 f"parameter bound to it. Right-click to unbind.")
                field.set_value(self.variables.get(value["var"], p.default_value()))
            elif own is not None and p.name in own:
                self.bound.pop(p.name, None)
                found, graph_value = own[p.name]
                shown = (p.to_text(graph_value) or "empty") if found else "the module's default"
                text = f"{label_text(p)} <span style='color:{accent}'>•</span>"
                label.setToolTip(f"{p.help}\n\nThis analysis's own value (the graph's: {shown}). Right-click to take "
                                 f"the graph's value or to make this one the graph's.".strip())
                field.set_value(value)
            else:
                self.bound.pop(p.name, None)
                text = label_text(p)
                label.setToolTip(p.help + ("\n\nThe graph's value (shared by the analyses using the graph); a "
                                           "change here is this analysis's own." if own is not None else ""))
                field.set_value(value)
            if label.text() != text:
                label.setText(text)
                relabeled = True
        if relabeled and self.columns:  # longer or shorter labels: maybe another number of columns fits
            self._fit(self.width())

    def _field_changed(self, p, value):
        if p.name in self.bound:
            self.var_changed.emit(self.bound[p.name], value)
        else:
            self.changed.emit(p.name, value)

    def _label_menu(self, p, label, pos):
        menu = QtWidgets.QMenu(self)
        if self.own is not None and p.name in self.own:
            menu.addAction("Use the graph's value", lambda: self.reset_requested.emit(p.name))
            menu.addAction("Make this the graph's value (for every analysis using it)",
                           lambda: self.promote_requested.emit(p.name))
            menu.addSeparator()
        if p.name in self.bound:
            menu.addAction(f"Unbind from variable {self.bound[p.name]!r}", lambda: self.bind_requested.emit(p.name, None))
        else:
            for name in self.variables:
                menu.addAction(f"Bind to variable {name!r}", lambda n=name: self.bind_requested.emit(p.name, n))
            menu.addAction("Bind to a new variable…", lambda: self._new_variable(p))
        menu.exec(label.mapToGlobal(pos))

    def _new_variable(self, p):
        name, ok = QtWidgets.QInputDialog.getText(self, "New analysis variable",
                                                  f"Name of the variable for {p.label}:", text=p.name)
        if ok and name.strip():
            self.bind_requested.emit(p.name, name.strip())
