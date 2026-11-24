"""
The inspector beside the graph (graph_view.py): what is selected there. One node: its status and messages, buttons
(show in the main window, run, its actions; a view: switch it off or on), its inputs (where each comes from, the
choices that fit, and which variable of the source to use where several fit), its parameters (the same form as in the
card; with a shared graph file the analysis's own values are marked) and, for a reader or compute node, a preview of
its outputs (preview.py: drawn from the data's shape, for checking; the views are what the analysis shows). A
variable: its value and the parameters bound to it. Nothing: the analysis's graph file (shared or not) and its
variables. Every edit goes through the AnalysisCard.
"""
import os

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from tesselab import preview, units
from tesselab.analysis import Source, sources_of
from tesselab.engine import node_kind
from tesselab.gui import theme
from tesselab.gui.forms import ParamForm, ValueEdit, field_chars
from tesselab.signature import bindings


def _section(layout, text):
    layout.addSpacing(6)
    layout.addWidget(QtWidgets.QLabel(text.upper(), objectName="section"))


def _label(text, name="hint", wrap=True):
    label = QtWidgets.QLabel(text, objectName=name, wordWrap=wrap)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _combo():
    """A combo box that fits a narrow inspector (its items may be longer than it)."""
    combo = QtWidgets.QComboBox()
    combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(8)
    combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
    return combo


class Inspector(QtWidgets.QScrollArea):
    def __init__(self, dialog):
        super().__init__()
        self.dialog, self.card = dialog, dialog.card
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # everything wraps or shrinks
        self.setMinimumWidth(260)
        self.page = None  # the selection shown
        self.form, self.inputs, self.status, self.messages, self.var_fields = None, {}, None, None, {}
        self.preview = None  # (output combo, variable combo, plot, scalars label) of a reader or compute node
        self.show_selection([])

    @property
    def analysis(self):
        return self.card.analysis

    def show_selection(self, selection):
        """Show a page for the selection ([("node", name) or ("var", name)])."""
        page = tuple(selection)
        if page == self.page and self.widget() is not None:
            return
        self.page = page
        self.form, self.inputs, self.status, self.messages, self.var_fields = None, {}, None, None, {}
        self.preview, self._previewed = None, None
        content = QtWidgets.QWidget(objectName="inspector")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(5)
        if len(page) == 1 and page[0][0] == "node" and page[0][1] in self.analysis.nodes:
            self._node_page(layout, page[0][1])
        elif len(page) == 1 and page[0][0] == "var" and page[0][1] in self.analysis.vars:
            self._var_page(layout, page[0][1])
        elif page:
            self._many_page(layout, page)
        else:
            self._analysis_page(layout)
        layout.addStretch()
        self.setWidget(content)
        self.refresh()

    def refresh(self, values=True):
        """After an edit (values) or new results: the page shows the current state."""
        page = self.page
        if any((kind == "node" and name not in self.analysis.nodes) or (kind == "var" and name not in self.analysis.vars)
               for kind, name in page):
            self.page = None
            self.show_selection([k for k in page if (k[0] == "node" and k[1] in self.analysis.nodes) or
                                 (k[0] == "var" and k[1] in self.analysis.vars)])
            return
        if len(page) == 1 and page[0][0] == "node":
            self._refresh_node(page[0][1], values)
        elif values and not page:
            self._refresh_variables()
        elif values and len(page) == 1 and page[0][0] == "var":
            self._refresh_variables()

    # --- a node ------------------------------------------------------------------------------------------------
    def _node_page(self, layout, name):
        card = self.card
        try:
            cls = card.engine.node_class(self.analysis, name)
            info = card.engine.node_info(self.analysis, name)
        except Exception as error:
            layout.addWidget(_label(f"{self.analysis.nodes[name]['node']}: {error}", "error"))
            return
        head = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel()
        head.addWidget(self.status)
        head.addWidget(_label(self.dialog._title(name), "heading"), 1)
        layout.addLayout(head)
        layout.addWidget(_label(f"'{name}' · {cls.name} ({info.module.name} {info.module.version})"))
        if cls.help:
            layout.addWidget(_label(cls.help, "field"))
        self.messages = _label("", "note")
        self.messages.setVisible(False)
        layout.addWidget(self.messages)
        first = QtWidgets.QHBoxLayout()
        go = QtWidgets.QPushButton("Show in the main window", toolTip="Open its section in the card and scroll to it "
                                                                      "(or double-click the node)")
        go.clicked.connect(lambda: self.dialog.go_to(name))
        first.addWidget(go)
        first.addStretch()
        layout.addLayout(first)
        buttons = QtWidgets.QHBoxLayout()  # a row of its own: side by side they made the page wider than the panel
        if cls.run_on == "button":
            run = QtWidgets.QPushButton("Run", objectName="primary")
            run.clicked.connect(lambda: card.run_node(name))
            buttons.addWidget(run)
        for action in cls.actions:
            button = QtWidgets.QPushButton(action.label, toolTip=action.help)
            button.clicked.connect(lambda checked=False, a=action: card.run_action(name, a))
            buttons.addWidget(button)
        if node_kind(cls) == "view":
            off = bool(self.analysis.nodes[name].get("off"))
            switch = QtWidgets.QPushButton("Switch on" if off else "Switch off",
                                           toolTip="A view switched off is not shown in the card (it stays in the "
                                                   "graph, with its connections).")
            switch.clicked.connect(lambda: self.dialog.switch_view(name))
            buttons.addWidget(switch)
        buttons.addStretch()
        layout.addLayout(buttons)
        if cls.inputs:
            _section(layout, "Inputs")
            grid = QtWidgets.QGridLayout()
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(4)
            for row, (input_name, signature) in enumerate(cls.inputs.items()):
                label = QtWidgets.QLabel(input_name, objectName="field")
                label.setToolTip(f"Takes {', '.join(signature.variables)}" + (" (optional)" if signature.optional
                                                                              else ""))
                combo = _combo()
                combo.activated.connect(lambda i, n=input_name, c=combo: self._input_chosen(name, n, c))
                binds = QtWidgets.QGridLayout()  # one row per variable with a choice
                grid.addWidget(label, 2 * row, 0)
                grid.addWidget(combo, 2 * row, 1)
                binds.setContentsMargins(12, 0, 0, 0)
                grid.addLayout(binds, 2 * row + 1, 0, 1, 2)  # across both columns: narrow panels have room for it
                self.inputs[input_name] = (combo, binds)
            grid.setColumnStretch(1, 1)
            layout.addLayout(grid)
        if cls.params:
            _section(layout, "Parameters")
            self.form = ParamForm(cls.params, self.analysis.nodes[name].get("params", {}), self.analysis.vars,
                                  card.project_root, advanced=card.window.advanced)
            card.connect_form(self.form, name)
            layout.addWidget(self.form)
            if self.analysis.graph is not None:
                layout.addWidget(_label(f"<span style='color:{theme.current()['accent']}'>•</span> this analysis's "
                                        f"own value; the others are the graph's (right-click a label to change "
                                        f"that)."))
        if node_kind(cls) != "view" and cls.output_names():
            self._preview_section(layout, name)

    # --- a preview of a reader's or compute node's outputs ---------------------------------------------------------
    def _preview_section(self, layout, name):
        from tesselab.gui.plots import PlotView
        _section(layout, "Preview")
        row = QtWidgets.QHBoxLayout()
        output, variable = _combo(), _combo()
        output.setToolTip("Which output")
        variable.setToolTip("Which of its variables")
        row.addWidget(output)
        row.addWidget(variable)
        layout.addLayout(row)
        plot = PlotView()
        plot.setMinimumSize(160, 230)
        plot.setMaximumHeight(280)
        # as wide as the panel, never wider (a plot's own size hint would push the page past the panel's edge)
        plot.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        layout.addWidget(plot)
        numbers = _label("")
        layout.addWidget(numbers)
        layout.addWidget(_label("Drawn from the data's shape, to check the node; the views show the results."))
        self.preview = (output, variable, plot, numbers)
        output.activated.connect(lambda i: self._fill_variables(name))
        variable.activated.connect(lambda i: self._draw_preview(name))

    def _preview_result(self, name):
        result = self.card.runner.results(self.analysis).get(name)
        return result if result is not None and result.ok else None

    def _update_preview(self, name):
        """New results: the outputs and variables offered again, the plot drawn again (once per result)."""
        result = self._preview_result(name)
        if self.preview is None or result is self._previewed:
            return
        self._previewed = result
        output = self.preview[0]
        current = output.currentText()
        output.clear()
        for key in (result.outputs if result else {}):
            output.addItem(key)
        if current and output.findText(current) >= 0:
            output.setCurrentIndex(output.findText(current))
        self._fill_variables(name)

    def _fill_variables(self, name):
        output, variable, plot, numbers = self.preview
        result = self._preview_result(name)
        dataset = result.outputs.get(output.currentText()) if result else None
        current = variable.currentText()
        variable.clear()
        for key in (preview.variables(dataset) if dataset is not None else []):
            variable.addItem(key)
        if current and variable.findText(current) >= 0:
            variable.setCurrentIndex(variable.findText(current))
        found = preview.scalars(dataset) if dataset is not None else []
        numbers.setText("   ".join(f"{k} = {v}" for k, v in found[:12]))
        numbers.setVisible(bool(found))
        self._draw_preview(name)

    def _draw_preview(self, name):
        output, variable, plot, numbers = self.preview
        result = self._preview_result(name)
        dataset = result.outputs.get(output.currentText()) if result else None
        if dataset is None:
            spec = preview.note_only("No result yet")
        elif not variable.count():
            spec = preview.note_only("Nothing to draw: no numbers over one or two dimensions")
        else:
            try:
                spec = preview.preview_spec(dataset, variable.currentText())
            except Exception as error:
                spec = preview.note_only(f"No preview: {error}")
        plot.set_spec(spec)
        plot.auto_range()

    def _refresh_node(self, name, values):
        card = self.card
        results = card.runner.results(self.analysis)
        result = results.get(name)
        if self.status is not None:
            level = self.dialog._level(name, results, card.runner.is_running(self.analysis))
            self.status.setPixmap(theme.dot_icon(level).pixmap(12, 12))
            self.status.setToolTip(level or "fine")
        if self.messages is not None:
            lines = [f"{d['level']}: {d['message']}" for d in (result.diagnostics if result else [])
                     if d["level"] != "info"]
            if result is not None and result.error:
                lines.insert(0, result.error)
            self.messages.setText("\n".join(lines[:6]))
            self.messages.setVisible(bool(lines))
        if self.preview is not None:
            self._update_preview(name)
        if not values:
            return
        if self.form is not None:
            card.update_form(self.form, name)
        entry = self.analysis.nodes[name]
        for input_name, (combo, binds) in self.inputs.items():
            current = entry.get("inputs", {}).get(input_name)
            sources = sources_of(current)
            combo.blockSignals(True)
            combo.clear()
            refs = []
            if len(sources) == 1:
                combo.addItem(card.source_text(sources[0]), sources[0].ref())
                refs.append(sources[0].ref())
            elif len(sources) > 1:
                combo.addItem(f"{len(sources)} sources", None)
            try:
                choices = card.engine.input_choices(self.analysis, name, input_name, results)
            except Exception:
                choices = []
            for source, _ in choices:
                if source.ref() not in refs:
                    combo.addItem(card.source_text(source), source.ref())
                    refs.append(source.ref())
            combo.addItem("(not connected)", "")
            combo.addItem("Link to another analysis…", "#link")
            combo.setCurrentIndex(0 if sources else combo.findData(""))
            combo.blockSignals(False)
            self._bindings(name, input_name, binds, sources[0] if len(sources) == 1 else None, results)

    def _bindings(self, name, input_name, row, source, results):
        """Where several variables of the source fit one of the input's: which one to use."""
        while row.count():
            item = row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        line = 0
        if source is None or source.file:
            return
        upstream = results.get(source.instance)
        if upstream is None or not upstream.ok or source.output not in upstream.outputs:
            return
        signature = self.card.engine.node_class(self.analysis, name).inputs[input_name]
        options = bindings(signature, upstream.outputs[source.output])
        for var in signature.variables:
            values = []
            for option in options:
                if option.get(var) not in values:
                    values.append(option.get(var))
            if len(values) > 1:
                label = QtWidgets.QLabel(f"{var} =", objectName="hint", toolTip="Several variables of the source fit "
                                         "this one of the input: which to use.")
                combo = _combo()
                for value in values:
                    combo.addItem(str(value) if value is not None else "(none)", value)
                combo.setCurrentIndex(max(0, combo.findData(source.bind.get(var, values[0]))))
                combo.activated.connect(lambda i, c=combo, var=var: self.card.set_binding(name, input_name, var,
                                                                                            c.itemData(i)))
                row.addWidget(label, line, 0)
                row.addWidget(combo, line, 1)
                line += 1
        row.setColumnStretch(1, 1)

    def _input_chosen(self, name, input_name, combo):
        data = combo.currentData()
        if data == "#link":
            self.card.choose_link(name, input_name)
        elif data == "":
            self.card.set_input(name, input_name, None)
        elif data is not None:
            source = Source.parse(data)
            self.dialog.connect_ports(source.instance, source.output, name, input_name)

    # --- variables, the analysis ---------------------------------------------------------------------------------
    def _var_row(self, layout, var):
        p = self.dialog.var_param(var)
        row = QtWidgets.QHBoxLayout()
        name = QtWidgets.QLabel(units.label(var, getattr(p, "unit", None), getattr(p, "display", None)),
                                objectName="field")
        row.addWidget(name)
        field = ValueEdit(field_chars(p) if p is not None else 18)  # short as in the forms
        field.setToolTip("The variable's value: every parameter bound to it takes it.")
        field.editingFinished.connect(lambda: self._var_typed(var, field))
        row.addWidget(field, 0 if field.chars else 1)
        if field.chars:
            row.addStretch()
        layout.addLayout(row)
        targets = ", ".join(f"{self.dialog._title(n)} › {k}" for n, k in
                            self.dialog_bound().get(var, [])) or "no parameter yet"
        layout.addWidget(_label(f"bound: {targets}"))
        self.var_fields[var] = (field, p)

    def dialog_bound(self):
        from tesselab.gui.graph_view import bound_params
        return bound_params(self.analysis)

    def _var_typed(self, var, field):
        p = self.var_fields.get(var, (None, None))[1]
        try:
            import json
            value = p.from_text(field.text()) if p is not None else json.loads(field.text())
        except (ValueError, TypeError) as error:
            field.setToolTip(str(error))
            field.setProperty("error", True)
            return
        if value != self.analysis.vars.get(var):
            self.card.set_var(var, value)

    def _refresh_variables(self):
        for var, (field, p) in self.var_fields.items():
            if var in self.analysis.vars and not field.hasFocus():
                value = self.analysis.vars[var]
                field.setText(p.to_text(value) if p is not None else str(value))

    def _var_page(self, layout, var):
        layout.addWidget(_label(f"Variable {var}", "heading"))
        layout.addWidget(_label("An analysis variable: every parameter bound to it takes its value, so one cursor "
                                "moves all of them. Its value is this analysis's own; which parameters follow it "
                                "belongs to the graph.", "field"))
        _section(layout, "Value")
        self._var_row(layout, var)
        buttons = QtWidgets.QHBoxLayout()
        rename = QtWidgets.QPushButton("Rename…")
        rename.clicked.connect(lambda: self.dialog.rename_var(var))
        remove = QtWidgets.QPushButton("Remove", toolTip="Its parameters keep its value.")
        remove.clicked.connect(lambda: self.dialog.remove_var(var))
        buttons.addWidget(rename)
        buttons.addWidget(remove)
        buttons.addStretch()
        layout.addLayout(buttons)

    def _many_page(self, layout, page):
        layout.addWidget(_label(f"{len(page)} selected", "heading"))
        buttons = QtWidgets.QHBoxLayout()
        for text, slot in (("Duplicate", self.dialog.duplicate_selection), ("Remove", self.dialog.delete_selection)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)

    def _analysis_page(self, layout):
        analysis = self.analysis
        layout.addWidget(_label(analysis.label, "heading"))
        if analysis.doc.get("comment"):
            layout.addWidget(_label(analysis.doc["comment"], "comment"))
        _section(layout, "Graph file")
        graph = analysis.graph
        if graph is None:
            layout.addWidget(_label("The graph is saved in the analysis file, for this analysis alone.", "field"))
            buttons = (("Share it in a file of its own…", self.card.share_graph),
                       ("Use a graph file…", self.card.choose_graph_file))
        else:
            others = [a.label for a in self.card.engine.sharing(analysis)]
            text = (f"{os.path.basename(graph.path)}: the analyses that use it share its nodes, connections and "
                    f"values; values set here (marked •) are this analysis's own.")
            if others:
                text += f" Open now and sharing it: {', '.join(others)}."
            layout.addWidget(_label(text, "field"))
            buttons = (("Use a separate copy…", self.card.share_graph),
                       ("Use another graph file…", self.card.choose_graph_file))
        row = QtWidgets.QHBoxLayout()
        for text, slot in buttons:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        _section(layout, "Variables")
        if not analysis.vars:
            layout.addWidget(_label("None yet. Right-click a node → Bind to a variable, or empty space → New variable "
                                    "from, to make one."))
        for var in analysis.vars:
            self._var_row(layout, var)
        _section(layout, "Nodes")
        layout.addWidget(_label("Click a node to see its inputs and parameters here; double-click it to show it in the "
                                "main window. Right-click anything for its menu; the ? above lists the keys."))
